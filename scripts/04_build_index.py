#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_build_index.py — визуальный (CLIP+HNSW) и текстовый (OCR+TFIDF) индексы по кропам.

Улучшения:
  - ViT-L/14 по умолчанию (вместо ViT-B-32)
  - Сохраняет model.json с метаданными
  - Правильные имена файлов (embs.npy вместо clip_embeddings.npy)
  - Детерминизм (fixed seed)
  - Улучшенная обработка ошибок

Артефакты:
  index/embs.npy              # эмбеддинги (N, D), float32, L2-normalized
  index/hnsw.bin              # HNSW индекс
  index/model.json            # метаданные модели
  index/crops.parquet         # копия метаданных
  index/ocr_texts.txt         # OCR текст (если --ocr)
  index/tfidf_vectorizer.joblib
  index/tfidf_matrix.npz
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import warnings
from typing import List
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import open_clip
import hnswlib

# OCR + текстовый индекс
try:
    import easyocr
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy import sparse
    import joblib
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

from multiprocessing import Pool, cpu_count

# Спрячем бесполезные варнинги PyTorch про pin_memory на MPS
warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*")

# ========================= Константы по умолчанию =========================
DEFAULT_MODEL = "ViT-L-14"
DEFAULT_PRETRAINED = "openai"
DEFAULT_BATCH = 64
DEFAULT_HNSW_M = 32
DEFAULT_HNSW_EFC = 200
SEED = 42

# ========================= Utils =========================

def device_auto() -> torch.device:
    """Автоопределение устройства: MPS > CUDA > CPU"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_clip(model_name: str, pretrained: str, device: torch.device):
    """Загрузка CLIP модели через open_clip"""
    print(f"[i] Загрузка модели: {model_name}, pretrained: {pretrained}")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
    except Exception as e:
        print(f"[!] Ошибка загрузки модели: {e}")
        print(f"[i] Попытка без указания device...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model = model.to(device)
    
    model.eval()
    
    # Определяем размерность эмбеддингов
    try:
        embed_dim = model.visual.output_dim
    except AttributeError:
        # Fallback для разных версий
        embed_dim = model.text_projection.shape[-1]
    
    print(f"[✓] Модель загружена, размерность: {embed_dim}")
    return model, preprocess, embed_dim


def batched(iterable, n: int = 64):
    """Разбивает итерируемый объект на батчи размера n"""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def normalize_embeddings(embs: np.ndarray) -> np.ndarray:
    """L2-нормализация эмбеддингов по строкам"""
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    return (embs / norms).astype(np.float32)


# ========================= OCR Functions =========================

def _init_reader(langs):
    """Инициализация EasyOCR reader (вызывается один раз на процесс)"""
    global _READER
    _READER = easyocr.Reader(list(langs), gpu=False, verbose=False)


def _ocr_one(path: str) -> str:
    """OCR одного изображения"""
    try:
        res = _READER.readtext(path, detail=0, paragraph=True, batch_size=16)
        return " ".join([t for t in res if isinstance(t, str)]).strip()
    except Exception:
        return ""


def run_ocr(
    paths: List[str],
    lang=("ru", "en"),
    workers: int = 0,
    chunk: int = 32,
    out_txt_path: str | None = None,
    append: bool = False,
) -> List[str]:
    """
    Параллельный OCR. Возвращает список текстов в том же порядке, что и paths.
    
    Args:
        paths: Список путей к изображениям
        lang: Языки для распознавания
        workers: Число процессов (0 = auto)
        chunk: Размер чанка для multiprocessing
        out_txt_path: Путь для сохранения текстов (опционально)
        append: Дописывать к существующему файлу (для резюма)
    
    Returns:
        Список распознанных текстов
    """
    workers = workers or max(1, cpu_count() // 2)
    texts: List[str] = []

    if not append and out_txt_path and os.path.exists(out_txt_path):
        os.remove(out_txt_path)

    if workers <= 1:
        # Однопоточный режим
        reader = easyocr.Reader(list(lang), gpu=False, verbose=False)
        for p in tqdm(paths, desc="OCR", unit="img"):
            try:
                res = reader.readtext(p, detail=0, paragraph=True, batch_size=16)
                txt = " ".join([t for t in res if isinstance(t, str)]).strip()
            except Exception:
                txt = ""
            texts.append(txt)
            if out_txt_path:
                with open(out_txt_path, "a", encoding="utf-8") as f:
                    f.write((txt or "") + "\n")
        return texts

    # Многопоточный режим
    with Pool(processes=workers, initializer=_init_reader, initargs=(list(lang),)) as pool:
        for txt in tqdm(
            pool.imap(_ocr_one, paths, chunksize=chunk),
            total=len(paths),
            desc="OCR",
            unit="img",
        ):
            texts.append(txt)
            if out_txt_path:
                with open(out_txt_path, "a", encoding="utf-8") as f:
                    f.write((txt or "") + "\n")
    return texts


# ========================= Main =========================

def main():
    ap = argparse.ArgumentParser(
        description="Построение CLIP+HNSW индекса для геолокации",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Основные параметры
    ap.add_argument("--crops-meta", default="meta/crops.csv",
                    help="CSV с метаданными кропов")
    ap.add_argument("--outdir", default="index",
                    help="Папка для сохранения индекса")
    
    # CLIP модель
    ap.add_argument("--clip-model", default=DEFAULT_MODEL,
                    help=f"Название модели CLIP (дефолт: {DEFAULT_MODEL})")
    ap.add_argument("--clip-ckpt", default=DEFAULT_PRETRAINED,
                    help=f"Pretrained веса (дефолт: {DEFAULT_PRETRAINED})")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                    help="Размер батча для эмбеддингов")
    
    # HNSW параметры
    ap.add_argument("--hnsw-M", dest="hnsw_M", type=int, default=DEFAULT_HNSW_M,
                    help="HNSW: число связей (больше = точнее, но медленнее)")
    ap.add_argument("--hnsw-efC", dest="hnsw_efC", type=int, default=DEFAULT_HNSW_EFC,
                    help="HNSW: ef при построении")
    ap.add_argument("--no-hnsw", action="store_true",
                    help="Не строить HNSW (только эмбеддинги)")
    
    # OCR
    ap.add_argument("--ocr", action="store_true",
                    help="Выполнить OCR и построить TF-IDF индекс")
    ap.add_argument("--ocr-workers", type=int, default=0,
                    help="Число процессов для OCR (0=auto)")
    ap.add_argument("--ocr-chunk", type=int, default=32,
                    help="Размер чанка для OCR multiprocessing")
    ap.add_argument("--resume", action="store_true",
                    help="Продолжить OCR с места остановки")
    
    args = ap.parse_args()
    
    # Проверка OCR
    if args.ocr and not HAS_OCR:
        print("[!] OCR запрошен, но не установлены зависимости:")
        print("    pip install easyocr scikit-learn joblib scipy")
        sys.exit(1)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # ============= 1. Загрузка метаданных =============
    print(f"\n[1/4] Загрузка метаданных из {args.crops_meta}")
    
    if not os.path.exists(args.crops_meta):
        print(f"[!] Не найден файл: {args.crops_meta}")
        print("    Сначала запустите: python scripts/03_prepare_dataset.py")
        sys.exit(1)
    
    df = pd.read_csv(args.crops_meta)
    needed = {"path", "crop_id", "pano_id", "lat", "lon"}
    missing = needed - set(df.columns)
    if missing:
        print(f"[!] В {args.crops_meta} отсутствуют колонки: {missing}")
        sys.exit(1)
    
    # Проверка существования файлов
    print("[i] Проверка существования файлов кропов...")
    valid_mask = df["path"].apply(os.path.exists)
    n_missing = (~valid_mask).sum()
    if n_missing > 0:
        print(f"[!] Предупреждение: {n_missing} файлов не найдено, они будут пропущены")
        df = df[valid_mask].reset_index(drop=True)
    
    print(f"[✓] Загружено {len(df)} кропов")
    
    # Сохраняем копию метаданных
    df.to_parquet(os.path.join(args.outdir, "crops.parquet"), index=False)
    
    paths = df["path"].tolist()
    
    # ============= 2. Вычисление CLIP эмбеддингов =============
    print(f"\n[2/4] Вычисление CLIP эмбеддингов ({args.clip_model})")
    
    device = device_auto()
    print(f"[i] Устройство: {device}")
    
    # Фиксируем seed для детерминизма
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    model, preprocess, embed_dim = load_clip(args.clip_model, args.clip_ckpt, device)
    
    embs = np.zeros((len(paths), embed_dim), dtype=np.float32)
    
    with torch.no_grad():
        idx = 0
        batches = list(batched(paths, args.batch))
        
        for batch_paths in tqdm(batches, desc="CLIP", unit="batch"):
            ims = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    ims.append(preprocess(img))
                except Exception as e:
                    # Если изображение битое — чёрный квадрат
                    print(f"[!] Ошибка загрузки {p}: {e}")
                    ims.append(preprocess(Image.new("RGB", (224, 224), (0, 0, 0))))
            
            ims_tensor = torch.stack(ims, dim=0).to(device)
            feats = model.encode_image(ims_tensor)
            
            # Нормализация на GPU перед переносом на CPU
            feats = torch.nn.functional.normalize(feats, dim=-1)
            feats_np = feats.detach().cpu().numpy().astype(np.float32)
            
            embs[idx : idx + len(batch_paths)] = feats_np
            idx += len(batch_paths)
    
    # Дополнительная нормализация (на случай численных ошибок)
    embs = normalize_embeddings(embs)
    
    # Сохранение эмбеддингов
    embs_path = os.path.join(args.outdir, "embs.npy")
    np.save(embs_path, embs)
    print(f"[✓] Эмбеддинги сохранены: {embs_path} (shape: {embs.shape})")
    
    # Сохранение метаданных модели
    model_meta = {
        "model": args.clip_model,
        "pretrained": args.clip_ckpt,
        "embed_dim": int(embed_dim),
        "tile_size": 336,
        "tile_stride": 224,
        "seed": SEED,
        "version": "2.0",
    }
    model_json_path = os.path.join(args.outdir, "model.json")
    with open(model_json_path, "w") as f:
        json.dump(model_meta, f, indent=2)
    print(f"[✓] Метаданные сохранены: {model_json_path}")
    
    # ============= 3. Построение HNSW индекса =============
    if not args.no_hnsw:
        print(f"\n[3/4] Построение HNSW индекса (M={args.hnsw_M}, efC={args.hnsw_efC})")
        
        index = hnswlib.Index(space="cosine", dim=embed_dim)
        index.init_index(
            max_elements=embs.shape[0],
            M=args.hnsw_M,
            ef_construction=args.hnsw_efC,
            random_seed=SEED,
        )
        
        # Добавляем эмбеддинги порциями (для больших датасетов)
        batch_size = 10000
        for i in tqdm(range(0, len(embs), batch_size), desc="HNSW", unit="batch"):
            end = min(i + batch_size, len(embs))
            index.add_items(embs[i:end], np.arange(i, end, dtype=np.int64))
        
        hnsw_path = os.path.join(args.outdir, "hnsw.bin")
        index.save_index(hnsw_path)
        print(f"[✓] HNSW индекс сохранён: {hnsw_path}")
    else:
        print("\n[3/4] Пропуск построения HNSW (--no-hnsw)")
    
    # ============= 4. OCR + TF-IDF (опционально) =============
    if args.ocr:
        print(f"\n[4/4] OCR и построение TF-IDF индекса")
        
        ocr_txt_path = os.path.join(args.outdir, "ocr_texts.txt")
        texts: List[str] | None = None
        
        # Проверка резюма
        if args.resume and os.path.isfile(ocr_txt_path):
            with open(ocr_txt_path, "r", encoding="utf-8") as f:
                lines = [ln.rstrip("\n") for ln in f]
            
            if len(lines) == len(paths):
                print("[i] OCR уже выполнен, используем существующий файл")
                texts = lines
            else:
                print(f"[i] Найдено {len(lines)}/{len(paths)} OCR, продолжаем...")
                base = lines
                to_process = paths[len(lines) :]
                new_texts = run_ocr(
                    to_process,
                    lang=("ru", "en"),
                    workers=args.ocr_workers,
                    chunk=args.ocr_chunk,
                    out_txt_path=ocr_txt_path,
                    append=True,
                )
                texts = base + new_texts
        
        if texts is None:
            print("[i] Выполнение OCR с нуля...")
            texts = run_ocr(
                paths,
                lang=("ru", "en"),
                workers=args.ocr_workers,
                chunk=args.ocr_chunk,
                out_txt_path=ocr_txt_path,
                append=False,
            )
        
        # Синхронизация файла
        if not os.path.isfile(ocr_txt_path) or sum(1 for _ in open(ocr_txt_path, "r")) != len(texts):
            with open(ocr_txt_path, "w", encoding="utf-8") as f:
                for line in texts:
                    f.write((line or "") + "\n")
        
        print(f"[✓] OCR завершён: {ocr_txt_path}")
        
        # Построение TF-IDF
        print("[i] Построение TF-IDF индекса...")
        vect = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            token_pattern=r"(?u)\b[\w\-]{2,}\b",
            ngram_range=(1, 2),
            max_features=200_000,
            min_df=1,
            max_df=0.95,
        )
        X = vect.fit_transform(texts)  # [N, V], CSR sparse matrix
        
        joblib.dump(vect, os.path.join(args.outdir, "tfidf_vectorizer.joblib"))
        sparse.save_npz(os.path.join(args.outdir, "tfidf_matrix.npz"), X)
        
        print(f"[✓] TF-IDF индекс сохранён (vocabulary: {len(vect.vocabulary_)})")
    else:
        print("\n[4/4] Пропуск OCR (не указан флаг --ocr)")
    
    print("\n" + "=" * 60)
    print("✅ Индексация завершена успешно!")
    print("=" * 60)
    print(f"Индекс сохранён в: {args.outdir}/")
    print(f"  - embs.npy ({embs.shape[0]} эмбеддингов, dim={embs.shape[1]})")
    if not args.no_hnsw:
        print(f"  - hnsw.bin (M={args.hnsw_M}, efC={args.hnsw_efC})")
    print(f"  - model.json (метаданные)")
    if args.ocr:
        print(f"  - ocr_texts.txt + TF-IDF индекс")
    print("\nТеперь можно запустить поиск:")
    print("  python scripts/05_query.py --image samples/query.jpg")


if __name__ == "__main__":
    main()