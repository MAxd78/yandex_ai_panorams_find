#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_query_improved.py — поиск координат с fine-tuned моделью

Использует:
- Fine-tuned CLIP + GeM embeddings
- TF-IDF текстовый re-ranking
- OSM semantic boosting
- Геометрическая верификация

Использование:
python scripts/07_query_improved.py \
    --image samples/query.jpg \
    --index-dir index \
    --top-k 10
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import hnswlib
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from scipy import sparse

# ========================= Model Loading =========================

class GeM(torch.nn.Module):
    def __init__(self, p=3.0, eps=1e-6, learn_p=True):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p) if learn_p else p
        self.eps = eps
        
    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p).squeeze(-1).squeeze(-1)

class CLIPGeM(torch.nn.Module):
    def __init__(self, clip_model, gem_p=3.0):
        super().__init__()
        self.visual = clip_model.visual
        self.gem = GeM(p=gem_p, learn_p=True)
        
    def forward(self, x):
        x = self.visual.conv1(x)
        x = x.reshape(x.shape, x.shape, -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.visual.class_embedding.to(x.dtype) + \
                      torch.zeros(x.shape, 1, x.shape[-1], dtype=x.dtype, device=x.device), 
                      x], dim=1)
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)
        
        x = x.permute(1, 0, 2)
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)
        
        x = x[:, 1:, :]
        B, HW, C = x.shape
        H = W = int(np.sqrt(HW))
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        pooled = self.gem(x)
        return F.normalize(pooled, p=2, dim=1)

# ========================= Geometric Verification =========================

class GeomVerifier:
    def __init__(self, max_kp=2000, orb_ratio=0.75, ransac_thresh=4.0):
        self.orb = cv2.ORB_create(nfeatures=max_kp)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.orb_ratio = orb_ratio
        self.ransac_thresh = ransac_thresh
    
    def verify(self, img_q, img_db):
        """ORB + RANSAC verification"""
        gq = cv2.cvtColor(np.array(img_q), cv2.COLOR_RGB2GRAY)
        gd = cv2.cvtColor(np.array(img_db), cv2.COLOR_RGB2GRAY)
        
        kq, dq = self.orb.detectAndCompute(gq, None)
        kd, dd = self.orb.detectAndCompute(gd, None)
        
        if dq is None or dd is None or len(kq) < 8 or len(kd) < 8:
            return 0.0
        
        matches = self.bf.knnMatch(dq, dd, k=2)
        
        # Lowe's ratio test
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
            if mask is None:
                return 0.0
            ninl = int(mask.sum())
            return min(ninl / 150.0, 1.0)
        except:
            return 0.0

# ========================= OSM Semantic Boost =========================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Distance in meters"""
    from math import radians, cos, sin, asin, sqrt
    R = 6371000  # Радиус Земли в метрах
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    return R * c

def semantic_boost(crop_lat, crop_lon, query_ocr_text, osm_spatial_data, radius_m=50):
    """Boost score if OCR matches nearby POI"""
    if osm_spatial_data is None or not query_ocr_text:
        return 0.0
    
    tree = osm_spatial_data['tree']
    pois = osm_spatial_data['pois']
    coords = osm_spatial_data['coords']
    
    # Найти POI в радиусе
    query_rad = np.radians([[crop_lat, crop_lon]])
    radius_rad = radius_m / 6371000
    
    indices = tree.query_radius(query_rad, r=radius_rad)
    
    if len(indices) == 0:
        return 0.0
    
    boost = 0.0
    query_lower = query_ocr_text.lower()
    
    for idx in indices:
        poi = pois[idx]
        
        # Проверка токенов
        for token in poi.get('tokens', []):
            if token.lower() in query_lower:
                boost += 0.2
        
        # Проверка типа POI
        kind = poi.get('kind', '')
        if kind.split(':') in query_lower:
            boost += 0.1
    
    return min(boost, 0.5)  # Cap на 50%

# ========================= Main Query Logic =========================

def main():
    parser = argparse.ArgumentParser(
        description="Поиск координат по изображению (improved)"
    )
    
    # Входные данные
    parser.add_argument("--image", required=True,
                       help="Путь к query изображению")
    parser.add_argument("--index-dir", default="index",
                       help="Папка с индексами")
    
    # Параметры поиска
    parser.add_argument("--top-k", type=int, default=10,
                       help="Количество результатов (default: 10)")
    parser.add_argument("--ef", type=int, default=256,
                       help="HNSW ef parameter (default: 256)")
    parser.add_argument("--verify-k", type=int, default=50,
                       help="Количество для геом. верификации (default: 50)")
    
    # Веса для scoring
    parser.add_argument("--w-visual", type=float, default=0.6,
                       help="Вес visual similarity (default: 0.6)")
    parser.add_argument("--w-geom", type=float, default=0.3,
                       help="Вес геометрии (default: 0.3)")
    parser.add_argument("--w-text", type=float, default=0.1,
                       help="Вес текста (default: 0.1)")
    
    # Опции
    parser.add_argument("--no-geom", action='store_true',
                       help="Отключить геом. верификацию")
    parser.add_argument("--debug", action='store_true',
                       help="Показать debug таблицу")
    
    args = parser.parse_args()
    
    # Проверка входного изображения
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[!] Файл не найден: {image_path}")
        sys.exit(1)
    
    index_dir = Path(args.index_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[i] Device: {device}")
    
    # ========== 1. Загрузка конфига ==========
    print(f"\n[1/7] Загрузка конфигурации")
    config_path = index_dir / "index_config.json"
    if not config_path.exists():
        print(f"[!] Config не найден: {config_path}")
        sys.exit(1)
    
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"[✓] Model: {config['model_name']}, Crops: {config['n_crops']}")
    
    # ========== 2. Загрузка модели ==========
    print(f"\n[2/7] Загрузка модели")
    import open_clip
    
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        config['model_name'], 
        pretrained=config['pretrained']
    )
    
    # Загрузка fine-tuned весов
    model_path = Path(config['model_path'])
    checkpoint = torch.load(model_path, map_location='cpu')
    gem_p = checkpoint.get('gem_p', 3.0)
    
    model = CLIPGeM(clip_model, gem_p=gem_p)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"[✓] CLIPGeM loaded (p={gem_p:.2f})")
    
    # ========== 3. Загрузка метаданных ==========
    print(f"\n[3/7] Загрузка метаданных")
    crops_df = pd.read_csv(index_dir / "crops.csv")
    print(f"[✓] {len(crops_df)} crops")
    
    # ========== 4. Загрузка HNSW индекса ==========
    print(f"\n[4/7] Загрузка HNSW индекса")
    hnsw_path = index_dir / "hnsw_gem.bin"
    
    dim = config['embedding_dim']
    hnsw_index = hnswlib.Index(space='cosine', dim=dim)
    hnsw_index.load_index(str(hnsw_path))
    hnsw_index.set_ef(args.ef)
    
    print(f"[✓] HNSW loaded (ef={args.ef})")
    
    # ========== 5. Загрузка TF-IDF (опционально) ==========
    tfidf_vectorizer = None
    tfidf_matrix = None
    
    vect_path = index_dir / "tfidf_vectorizer.joblib"
    if vect_path.exists():
        print(f"\n[5/7] Загрузка TF-IDF индекса")
        tfidf_vectorizer = joblib.load(vect_path)
        tfidf_matrix = sparse.load_npz(index_dir / "tfidf_matrix.npz")
        print(f"[✓] TF-IDF loaded (vocab={len(tfidf_vectorizer.vocabulary_)})")
    else:
        print(f"\n[5/7] TF-IDF индекс не найден, пропускаем")
    
    # ========== 6. Загрузка OSM (опционально) ==========
    osm_spatial = None
    osm_path = index_dir / "osm_spatial.pkl"
    
    if osm_path.exists():
        print(f"\n[6/7] Загрузка OSM spatial index")
        osm_spatial = joblib.load(osm_path)
        print(f"[✓] OSM loaded ({len(osm_spatial['pois'])} POI)")
    else:
        print(f"\n[6/7] OSM spatial index не найден, пропускаем")
    
    # ========== 7. Обработка query ==========
    print(f"\n[7/7] Обработка query изображения")
    query_img = Image.open(image_path).convert('RGB')
    
    # Extract embedding
    with torch.no_grad():
        query_tensor = preprocess(query_img).unsqueeze(0).to(device)
        query_emb = model(query_tensor).cpu().numpy()
    
    # OCR (если доступен EasyOCR)
    query_text = ""
    try:
        import easyocr
        reader = easyocr.Reader(['ru', 'en'], gpu=torch.cuda.is_available())
        results = reader.readtext(np.array(query_img), detail=1)
        query_text = ' '.join([text for (_, text, conf) in results if conf > 0.4])
        if query_text:
            print(f"[i] Query OCR: '{query_text}'")
    except:
        pass
    
    # ========== 8. Visual search ==========
    print(f"\n[8/9] Visual search (HNSW)")
    k = min(args.top_k * 10, len(crops_df))
    labels, dists = hnsw_index.knn_query(query_emb, k=k)
    
    labels = labels
    visual_scores = 1.0 - dists  # Cosine similarity
    
    # ========== 9. Геометрическая верификация ==========
    geom_scores = np.zeros(len(labels))
    
    if not args.no_geom:
        print(f"\n[9/9] Геометрическая верификация (top {args.verify_k})")
        verifier = GeomVerifier()
        
        for i in range(min(args.verify_k, len(labels))):
            idx = labels[i]
            row = crops_df.iloc[idx]
            
            try:
                db_img = Image.open(row['path']).convert('RGB')
                geom_scores[i] = verifier.verify(query_img, db_img)
            except:
                geom_scores[i] = 0.0
    else:
        print("\n[9/9] Геометрия отключена")
    
    # ========== 10. Текстовый re-ranking ==========
    text_scores = np.zeros(len(labels))
    
    if tfidf_vectorizer is not None and query_text:
        query_tfidf = tfidf_vectorizer.transform([query_text])
        for i, idx in enumerate(labels):
            doc_tfidf = tfidf_matrix[idx]
            text_scores[i] = cosine_similarity(query_tfidf, doc_tfidf)[0, 0]
    
    # ========== 11. OSM semantic boost ==========
    osm_boosts = np.zeros(len(labels))
    
    if osm_spatial is not None and query_text:
        for i, idx in enumerate(labels):
            row = crops_df.iloc[idx]
            osm_boosts[i] = semantic_boost(
                row['lat'], row['lon'], query_text, osm_spatial
            )
    
    # ========== 12. Final scoring ==========
    final_scores = (
        args.w_visual * visual_scores +
        args.w_geom * geom_scores +
        args.w_text * text_scores +
        osm_boosts  # Additive boost
    )
    
    # Сортировка
    order = np.argsort(-final_scores)
    
    # Дедупликация по pano_id
    seen_panos = set()
    results = []
    
    for i in order:
        idx = labels[i]
        row = crops_df.iloc[idx]
        pano = row['pano_id']
        
        if pano in seen_panos:
            continue
        seen_panos.add(pano)
        
        results.append({
            'rank': len(results) + 1,
            'pano_id': pano,
            'lat': row['lat'],
            'lon': row['lon'],
            'final_score': final_scores[i],
            'visual': visual_scores[i],
            'geom': geom_scores[i],
            'text': text_scores[i],
            'osm_boost': osm_boosts[i],
        })
        
        if len(results) >= args.top_k:
            break
    
    # ========== 13. Output ==========
    if args.debug:
        print("\n" + "="*100)
        print(f"{'Rank':>4} {'Score':>8} {'Visual':>7} {'Geom':>6} {'Text':>6} {'OSM':>5} {'PanoID':<24} {'Coordinates':<20}")
        print("="*100)
        
        for r in results:
            print(f"{r['rank']:>4} {r['final_score']:>8.4f} {r['visual']:>7.4f} "
                  f"{r['geom']:>6.3f} {r['text']:>6.3f} {r['osm_boost']:>5.2f} "
                  f"{r['pano_id']:<24} {r['lat']:>10.6f},{r['lon']:>10.6f}")
        print("="*100)
    
    # Вывод топ координат
    print(f"\nТоп-{args.top_k} координаты:")
    for r in results:
        print(f"{r['lat']:.6f},{r['lon']:.6f}")
    
    print(f"\n[✓] Лучший результат: {results['lat']:.6f}, {results['lon']:.6f}")
    print(f"    Score: {results['final_score']:.4f} (visual={results['visual']:.3f}, geom={results['geom']:.3f})")

if __name__ == "__main__":
    main()
