#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_query.py — поиск координат по фото (zero-friction, сильные дефолты).

Улучшения версии 2.0:
  - Автопоиск эмбеддингов (embs.npy, clip_embeddings.npy, embeddings.npy)
  - Автосборка HNSW из эмбеддингов если индекс отсутствует
  - OCR текстовый re-ranking (если доступен TF-IDF)
  - Улучшенная геометрическая верификация (LightGlue + ORB с мягкими параметрами)
  - Агрегация по панорамам (несколько кропов одной панорамы → max score)
  - Полная детерминируемость (fixed seed)

Минимальный запуск:
  python scripts/05_query.py --image samples/query.jpg

С отладкой:
  python scripts/05_query.py --image samples/query.jpg --n 5 --debug
"""

import os
import sys
import json
import argparse
import inspect
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import hnswlib
import cv2

# OCR текстовый поиск (опционально)
HAS_TEXT_SEARCH = False
try:
    import joblib
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_TEXT_SEARCH = True
except ImportError:
    pass

# LightGlue (опционально)
USE_LIGHTGLUE = False
try:
    from lightglue import LightGlue, SuperPoint
    import kornia as K
    USE_LIGHTGLUE = True
except Exception:
    pass

# ========================= Константы =========================
SEED = 42
DEFAULT_TILE_SIZE = 336
DEFAULT_TILE_STRIDE = 224
DEFAULT_EF = 256
DEFAULT_TOPK = 400
DEFAULT_VERIFY_K = 80
DEFAULT_GEOM_WEIGHT = 0.30
DEFAULT_TEXT_WEIGHT = 0.15

# Геометрия
ORB_RATIO = 0.80  # Было 0.75 — слишком строго
RANSAC_THRESH = 4.0  # Было 3.0 — увеличиваем толерантность
MAX_KEYPOINTS = 2200

# ========================= Утилиты =========================

def pick_device():
    """Автоопределение устройства"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_from_index(index_dir: Path, image_size_fallback=336):
    """
    Загружает OpenCLIP и препроцесс с учётом различий API между версиями.
    Читает model.json для получения правильной модели.
    """
    import open_clip
    from safetensors.torch import load_file as safetensors_load

    index_dir = Path(index_dir)
    model_name = "ViT-L-14"
    pretrained = "openai"
    ckpt_path = None

    # Читаем метаданные модели
    meta_json = index_dir / "model.json"
    if meta_json.exists():
        try:
            meta = json.loads(meta_json.read_text())
            model_name = meta.get("model", model_name)
            pretrained = meta.get("pretrained", pretrained)
            print(f"[i] Загрузка модели из model.json: {model_name} ({pretrained})")
        except Exception as e:
            print(f"[!] Ошибка чтения model.json: {e}, используем дефолт")
    else:
        print(f"[i] model.json не найден, используем дефолт: {model_name} ({pretrained})")

    # Ищем веса safetensors (не обязательно)
    for cand in [
        index_dir / "open_clip_model.safetensors",
        index_dir / "clip_model.safetensors",
        Path("open_clip_model.safetensors"),
    ]:
        if cand.exists():
            ckpt_path = str(cand)
            print(f"[i] Найдены веса: {ckpt_path}")
            break

    # Совместимость по сигнатуре open_clip
    sig = inspect.signature(open_clip.create_model_and_transforms)
    try:
        if "image_size" in sig.parameters:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, image_size=image_size_fallback
            )
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
    except Exception as e:
        print(f"[!] Ошибка загрузки модели: {e}")
        sys.exit(1)

    # Загрузка весов из safetensors
    if ckpt_path:
        try:
            state = safetensors_load(ckpt_path, device="cpu")
            model.load_state_dict(state, strict=False)
            print("[✓] Веса загружены из safetensors")
        except Exception as e:
            print(f"[!] Не удалось загрузить веса: {e}")

    model.eval()
    return model, preprocess


def read_image_pil(path):
    """Загрузка изображения в RGB"""
    return Image.open(path).convert("RGB")


def tile_image_pil(pil_img: Image.Image, size=336, stride=224):
    """
    Тайлинг изображения с перекрытием.
    Возвращает список PIL.Image размера (size, size).
    """
    W, H = pil_img.size
    tiles = []
    
    for y in range(0, max(1, H - size + 1), stride):
        for x in range(0, max(1, W - size + 1), stride):
            tile = pil_img.crop((x, y, x + size, y + size))
            tiles.append(tile)
    
    if not tiles:
        # Если изображение меньше size — ресайзим
        tiles = [pil_img.resize((size, size), Image.BICUBIC)]
    
    return tiles


def cosine_to_sim(dist: np.ndarray) -> np.ndarray:
    """Конвертация cosine distance в similarity"""
    return 1.0 - dist


# ===================== Загрузка / сборка HNSW =====================

def _find_embs_file(index_dir: Path) -> Path | None:
    """
    Ищет файл с эмбеддингами в index_dir.
    Поддерживает разные названия для обратной совместимости.
    """
    candidates = [
        index_dir / "embs.npy",
        index_dir / "embeddings.npy",
        index_dir / "clip_embeddings.npy",
        index_dir / "clip_embs.npy",
    ]
    
    for c in candidates:
        if c.exists():
            return c
    
    # Последний шанс — любой *.npy с 'emb' в имени
    for c in index_dir.glob("*.npy"):
        if "emb" in c.name.lower() and "ids" not in c.name.lower():
            return c
    
    return None


def build_hnsw_from_embs(
    index_dir: Path, space: str = "cosine", M: int = 32, efC: int = 200
) -> hnswlib.Index:
    """
    Строит HNSW из сохранённых эмбеддингов и сохраняет index/hnsw.bin.
    """
    index_dir = Path(index_dir)
    embs_path = _find_embs_file(index_dir)
    
    if not embs_path or not embs_path.exists():
        raise FileNotFoundError(
            f"Не найдены эмбеддинги в {index_dir}. "
            f"Ожидались файлы: embs.npy / embeddings.npy / clip_embeddings.npy\n"
            f"Сначала запустите: python scripts/04_build_index.py"
        )

    print(f"[i] Загрузка эмбеддингов: {embs_path.name}")
    embs = np.load(embs_path)
    
    if embs.ndim != 2:
        raise RuntimeError(f"Неверная форма эмбеддингов: {embs.shape}, ожидалось [N, D]")

    # Нормализуем (для cosine space)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    embs = (embs / norms).astype(np.float32, copy=False)

    N, D = embs.shape
    print(f"[i] Построение HNSW: {N} векторов, dim={D}, M={M}, efC={efC}")
    
    index = hnswlib.Index(space=space, dim=D)
    index.init_index(max_elements=N, M=M, ef_construction=efC, random_seed=SEED)
    
    # Добавляем порциями для больших датасетов
    batch_size = 10000
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        index.add_items(embs[i:end], np.arange(i, end))
    
    # Сохраняем для будущих запусков
    out = index_dir / "hnsw.bin"
    index.save_index(str(out))
    print(f"[✓] HNSW сохранён: {out}")
    
    return index


def load_or_build_hnsw(index_dir: Path, dim_hint: int | None = None) -> hnswlib.Index:
    """
    Пытается загрузить HNSW из index/hnsw.bin или index/hnsw_clip.bin.
    Если не найден — строит из эмбеддингов и сохраняет.
    """
    index_dir = Path(index_dir)
    
    # Пробуем загрузить существующий
    for fname in ("hnsw.bin", "hnsw_clip.bin", "hnsw.index"):
        f = index_dir / fname
        if f.exists():
            print(f"[i] Загрузка HNSW: {f.name}")
            index = hnswlib.Index(space="cosine", dim=dim_hint or 16)
            index.load_index(str(f))
            return index
    
    # Индекса нет — строим
    print("[i] HNSW индекс не найден, собираем из эмбеддингов...")
    return build_hnsw_from_embs(index_dir)


# ==================== Текстовый поиск ====================

def load_text_index(index_dir: Path):
    """
    Загружает TF-IDF индекс если он есть.
    Возвращает (vectorizer, tfidf_matrix, ocr_texts) или (None, None, None).
    """
    if not HAS_TEXT_SEARCH:
        return None, None, None
    
    vect_path = index_dir / "tfidf_vectorizer.joblib"
    matrix_path = index_dir / "tfidf_matrix.npz"
    texts_path = index_dir / "ocr_texts.txt"
    
    if not (vect_path.exists() and matrix_path.exists() and texts_path.exists()):
        return None, None, None
    
    try:
        vect = joblib.load(vect_path)
        matrix = sparse.load_npz(matrix_path)
        
        with open(texts_path, "r", encoding="utf-8") as f:
            texts = [line.rstrip("\n") for line in f]
        
        print(f"[✓] Текстовый индекс загружен: {len(texts)} текстов, vocab={len(vect.vocabulary_)}")
        return vect, matrix, texts
    except Exception as e:
        print(f"[!] Ошибка загрузки текстового индекса: {e}")
        return None, None, None


def text_search(query_text: str, vectorizer, tfidf_matrix, top_k: int = 100):
    """
    Поиск по текстовому запросу через TF-IDF.
    Возвращает [(idx, score), ...].
    """
    if not query_text.strip():
        return []
    
    q_vec = vectorizer.transform([query_text])
    scores = (tfidf_matrix @ q_vec.T).toarray().flatten()
    
    # Топ-K индексов
    top_indices = np.argsort(-scores)[:top_k]
    results = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
    
    return results


# ================== Геометрическая верификация ==================

class GeomVerifier:
    """
    LightGlue (если доступен) или ORB+RANSAC.
    Возвращает скор [0..1] на основе числа инлаеров.
    """

    def __init__(
        self,
        device,
        use_lightglue=USE_LIGHTGLUE,
        max_kp=MAX_KEYPOINTS,
        orb_ratio=ORB_RATIO,
        ransac_thresh=RANSAC_THRESH,
    ):
        self.device = device
        self.use_lightglue = bool(use_lightglue)
        self.ransac_thresh = float(ransac_thresh)
        self.orb_ratio = float(orb_ratio)
        self.max_kp = int(max_kp)
        self._init_backends()

    def _init_backends(self):
        """Инициализация LightGlue или ORB"""
        if self.use_lightglue:
            try:
                self.sp = SuperPoint(max_num_keypoints=self.max_kp).eval().to(self.device)
                self.lg = LightGlue(features="superpoint").eval().to(self.device)
                self.backend = "lightglue"
                print(f"[✓] Геометрия: LightGlue (max_kp={self.max_kp})")
                return
            except Exception as e:
                print(f"[!] LightGlue недоступен: {e}")
        
        # Fallback на ORB
        self.backend = "orb"
        self.orb = cv2.ORB_create(nfeatures=self.max_kp)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        print(f"[i] Геометрия: ORB (ratio={self.orb_ratio}, ransac={self.ransac_thresh})")

    def _score_from_inliers(self, mask):
        """Конвертация числа инлаеров в нормализованный скор [0..1]"""
        if mask is None:
            return 0.0
        ninl = int(mask.sum())
        # Нормируем: 150 инлаеров = 1.0
        return float(min(ninl / 150.0, 1.0))

    def verify(self, img_q_pil: Image.Image, img_db_pil: Image.Image) -> float:
        """
        Геометрическая верификация двух изображений.
        Возвращает скор [0..1].
        """
        if self.backend == "lightglue":
            return self._verify_lightglue(img_q_pil, img_db_pil)
        else:
            return self._verify_orb(img_q_pil, img_db_pil)

    def _verify_lightglue(self, img_q_pil, img_db_pil):
        """LightGlue верификация"""
        iq = np.array(img_q_pil)
        idb = np.array(img_db_pil)

        def to_tensor(img_np):
            t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            t = t.unsqueeze(0)
            return K.color.rgb_to_grayscale(t).to(self.device)

        tq = to_tensor(iq)
        td = to_tensor(idb)

        with torch.inference_mode():
            fq = self.sp({"image": tq})
            fd = self.sp({"image": td})
            matches = self.lg({"image0": fq, "image1": fd})

        if "matches" not in matches or matches["matches"][0].numel() == 0:
            return 0.0

        m = matches["matches"][0].detach().cpu().numpy()
        k0 = matches["keypoints0"][0].detach().cpu().numpy()
        k1 = matches["keypoints1"][0].detach().cpu().numpy()

        pts0 = k0[m[:, 0]]
        pts1 = k1[m[:, 1]]

        if len(pts0) < 6:
            return 0.0

        try:
            H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, self.ransac_thresh)
        except Exception:
            return 0.0

        return self._score_from_inliers(mask)

    def _verify_orb(self, img_q_pil, img_db_pil):
        """ORB+RANSAC верификация"""
        gq = cv2.cvtColor(np.array(img_q_pil), cv2.COLOR_RGB2GRAY)
        gd = cv2.cvtColor(np.array(img_db_pil), cv2.COLOR_RGB2GRAY)

        kq, dq = self.orb.detectAndCompute(gq, None)
        kd, dd = self.orb.detectAndCompute(gd, None)

        if dq is None or dd is None or len(kq) < 8 or len(kd) < 8:
            return 0.0

        matches = self.bf.knnMatch(dq, dd, k=2)
        
        # Lowe's ratio test с более мягким порогом
        good = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.orb_ratio * n.distance:
                    good.append(m)

        if len(good) < 8:
            return 0.0

        src = np.float32([kq[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kd[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        try:
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_thresh)
        except Exception:
            return 0.0

        return self._score_from_inliers(mask)


# ========================= Main =========================

def main():
    p = argparse.ArgumentParser(
        description="Поиск координат по фото (CLIP + геометрия + OCR)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Обязательные параметры
    p.add_argument("--image", required=True, help="Путь к запросу (jpg/png)")
    
    # Пути
    p.add_argument("--index-dir", default="index", help="Папка индекса")
    p.add_argument("--crops-meta", default="meta/crops.csv", help="CSV с кропами")
    
    # Параметры тайлинга
    p.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    p.add_argument("--tile-stride", type=int, default=DEFAULT_TILE_STRIDE)
    
    # HNSW параметры
    p.add_argument("--ef", type=int, default=DEFAULT_EF, help="HNSW ef для поиска")
    p.add_argument("--topk", type=int, default=DEFAULT_TOPK, help="Кандидатов из HNSW")
    p.add_argument("--verify-k", type=int, default=DEFAULT_VERIFY_K, 
                   help="Сколько кандидатов проверять геометрией")
    
    # Веса для re-ranking
    p.add_argument("--geom-weight", type=float, default=DEFAULT_GEOM_WEIGHT,
                   help="Вес геометрии в финальном скоре")
    p.add_argument("--text-weight", type=float, default=DEFAULT_TEXT_WEIGHT,
                   help="Вес текстового совпадения (если есть OCR)")
    
    # Опции
    p.add_argument("--no-geom", action="store_true", help="Отключить геометрию")
    p.add_argument("--text-query", type=str, default="",
                   help="Текстовый запрос для OCR поиска (опционально)")
    
    # Вывод
    p.add_argument("--n", type=int, default=1, help="Сколько координат вывести")
    p.add_argument("--debug", action="store_true", help="Показать таблицу результатов")
    
    args = p.parse_args()

    # Фиксируем seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Проверка входного изображения
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[!] Файл не найден: {image_path}")
        sys.exit(1)

    device = pick_device()
    print(f"[i] Устройство: {device}")

    # ============= 1. Модель и препроцесс =============
    print(f"\n[1/6] Загрузка CLIP модели")
    model, preprocess = load_model_from_index(args.index_dir, args.tile_size)
    model.to(device)

    # ============= 2. Метаданные =============
    print(f"\n[2/6] Загрузка метаданных")
    
    # Пробуем parquet (быстрее) или csv
    meta_parquet = Path(args.index_dir) / "crops.parquet"
    if meta_parquet.exists():
        meta = pd.read_parquet(meta_parquet)
    else:
        meta = pd.read_csv(args.crops_meta)
    
    for col in ["path", "pano_id", "crop_id", "lat", "lon"]:
        if col not in meta.columns:
            raise RuntimeError(f"В метаданных нет колонки '{col}'")
    
    print(f"[✓] Загружено {len(meta)} кропов")

    # ============= 3. Размерность эмбеддинга =============
    with torch.inference_mode():
        dummy = preprocess(Image.new("RGB", (args.tile_size, args.tile_size))).unsqueeze(0).to(device)
        dim = model.encode_image(dummy).shape[-1]
    print(f"[i] Размерность эмбеддингов: {dim}")

    # ============= 4. HNSW индекс =============
    print(f"\n[3/6] Загрузка HNSW индекса")
    index = load_or_build_hnsw(Path(args.index_dir), dim_hint=dim)
    index.set_ef(args.ef)
    print(f"[✓] HNSW готов (ef={args.ef})")

    # ============= 5. Текстовый индекс (опционально) =============
    vectorizer, tfidf_matrix, ocr_texts = load_text_index(Path(args.index_dir))
    use_text = vectorizer is not None and args.text_query.strip()

    # ============= 6. Тайлинг запроса и эмбеддинг =============
    print(f"\n[4/6] Обработка запроса")
    img_q = read_image_pil(image_path)
    tiles = tile_image_pil(img_q, size=args.tile_size, stride=args.tile_stride)
    print(f"[i] Тайлов: {len(tiles)}")

    embeds = []
    with torch.inference_mode():
        for t in tiles:
            ten = preprocess(t).unsqueeze(0).to(device)
            e = model.encode_image(ten)
            e = torch.nn.functional.normalize(e, dim=-1)
            embeds.append(e)
    
    E = torch.stack(embeds, dim=0).squeeze(1)  # [T, D]
    q_emb = torch.amax(E, dim=0).detach().cpu().numpy()
    
    # Дополнительная нормализация
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)

    # ============= 7. Визуальный поиск =============
    print(f"\n[5/6] Поиск кандидатов (topk={args.topk})")
    k = min(args.topk, len(meta))
    labels, dists = index.knn_query(q_emb, k=k)
    labels = labels[0]
    sims = cosine_to_sim(dists[0])

    # ============= 8. Дедупликация по pano_id =============
    seen_panos = set()
    filtered = []
    
    for lab, sim in zip(labels, sims):
        row = meta.iloc[int(lab)]
        pano = str(row["pano_id"])
        
        if pano in seen_panos:
            continue
        
        seen_panos.add(pano)
        filtered.append((int(lab), float(sim)))
        
        # Ограничиваем для геометрии
        if len(filtered) >= max(args.verify_k, args.n * 10):
            break
    
    print(f"[i] После дедупликации: {len(filtered)} уникальных панорам")

    # ============= 9. Геометрическая верификация =============
    geom_scores = np.zeros(len(filtered), dtype=np.float32)
    
    if not args.no_geom:
        print(f"\n[6/6] Геометрическая верификация (verify_k={min(len(filtered), args.verify_k)})")
        verifier = GeomVerifier(
            device=device,
            use_lightglue=USE_LIGHTGLUE,
            max_kp=MAX_KEYPOINTS,
            orb_ratio=ORB_RATIO,
            ransac_thresh=RANSAC_THRESH,
        )
        
        verify_limit = min(len(filtered), args.verify_k)
        for i in tqdm(range(verify_limit), desc="Геометрия", unit="img", disable=not args.debug):
            lab, _ = filtered[i]
            row = meta.iloc[lab]
            
            try:
                db_img = Image.open(row["path"]).convert("RGB")
                geom_scores[i] = verifier.verify(img_q, db_img)
            except Exception as e:
                if args.debug:
                    print(f"[!] Ошибка геометрии для {row['path']}: {e}")
                geom_scores[i] = 0.0
    else:
        print("\n[6/6] Геометрия отключена (--no-geom)")

    # ============= 10. Текстовый re-ranking (опционально) =============
    text_scores = np.zeros(len(filtered), dtype=np.float32)
    
    if use_text:
        print(f"[i] Текстовый поиск: '{args.text_query}'")
        text_results = text_search(args.text_query, vectorizer, tfidf_matrix, top_k=200)
        text_dict = dict(text_results)
        
        for i, (lab, _) in enumerate(filtered):
            if lab in text_dict:
                text_scores[i] = text_dict[lab]

    # ============= 11. Финальный скор и сортировка =============
    sims_arr = np.array([s for _, s in filtered], dtype=np.float32)
    
    final_scores = (
        sims_arr
        + (0.0 if args.no_geom else args.geom_weight * geom_scores)
        + (args.text_weight * text_scores if use_text else 0.0)
    )
    
    order = np.argsort(-final_scores)

    # ============= 12. Агрегация по панорамам =============
    # Берём лучший скор среди всех кропов одной панорамы
    pano_best = {}  # pano_id -> (best_score, best_idx)
    
    for i in order:
        lab, _ = filtered[i]
        row = meta.iloc[lab]
        pano = str(row["pano_id"])
        
        if pano not in pano_best or final_scores[i] > pano_best[pano][0]:
            pano_best[pano] = (final_scores[i], i)

    # Сортируем панорамы по лучшему скору
    sorted_panos = sorted(pano_best.items(), key=lambda x: x[1][0], reverse=True)

    # ============= 13. Формирование результатов =============
    results = []
    for pano_id, (score, i) in sorted_panos:
        lab, _ = filtered[i]
        row = meta.iloc[lab]
        
        results.append({
            "rank": len(results) + 1,
            "score": float(score),
            "visual_sim": float(sims_arr[i]),
            "geom_score": float(geom_scores[i]),
            "text_score": float(text_scores[i]) if use_text else 0.0,
            "pano_id": str(row["pano_id"]),
            "crop_id": str(row["crop_id"]),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "path": str(row["path"]),
        })
        
        if len(results) >= args.n * 20:
            break

    # ============= 14. Вывод =============
    if args.debug:
        import shutil
        cols = shutil.get_terminal_size((120, 30)).columns
        sep = "-" * min(cols, 180)
        
        print("\n" + sep)
        if use_text:
            print(f"{'rank':>4}  {'score':>8}  {'visual':>8}  {'geom':>6}  {'text':>6}  "
                  f"{'pano_id':<28} {'lat':>10} {'lon':>10}")
        else:
            print(f"{'rank':>4}  {'score':>8}  {'visual':>8}  {'geom':>6}  "
                  f"{'pano_id':<28} {'lat':>10} {'lon':>10}")
        print(sep)
        
        for r in results[:20]:
            if use_text:
                print(f"{r['rank']:>4}  {r['score']:>8.6f}  {r['visual_sim']:>8.6f}  "
                      f"{r['geom_score']:>6.3f}  {r['text_score']:>6.3f}  "
                      f"{r['pano_id']:<28} {r['lat']:>10.6f} {r['lon']:>10.6f}")
            else:
                print(f"{r['rank']:>4}  {r['score']:>8.6f}  {r['visual_sim']:>8.6f}  "
                      f"{r['geom_score']:>6.3f}  "
                      f"{r['pano_id']:<28} {r['lat']:>10.6f} {r['lon']:>10.6f}")
        print(sep)

    # Топ-N координат в stdout
    for r in results[: args.n]:
        print(f"{r['lat']:.6f},{r['lon']:.6f}")


if __name__ == "__main__":
    main()