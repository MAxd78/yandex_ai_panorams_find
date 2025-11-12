#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_build_index.py — построение индексов для геолокализации

Создаёт:
1. HNSW индекс из fine-tuned embeddings
2. TF-IDF индекс из OCR текстов
3. OSM spatial index (BallTree) для POI

Использование:
python scripts/06_build_index.py \
    --model-path models/clip_gem/best_model.pt \
    --crops-meta meta/crops_with_ocr.csv \
    --osm-data data/osm_places.jsonl \
    --output-dir index
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
from PIL import Image
import hnswlib
import open_clip
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import BallTree
from scipy import sparse
import joblib

# ========================= GeM Model Loading =========================

class GeM(torch.nn.Module):
    """GeM pooling (копия из 05_train_model.py)"""
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
    """CLIP + GeM (копия из 05_train_model.py)"""
    def __init__(self, clip_model, gem_p=3.0):
        super().__init__()
        self.visual = clip_model.visual
        self.gem = GeM(p=gem_p, learn_p=True)
        
    def forward(self, x):
        # Feature extraction
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
        
        # Reshape для GeM
        x = x[:, 1:, :]
        B, HW, C = x.shape
        H = W = int(np.sqrt(HW))
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        pooled = self.gem(x)
        return F.normalize(pooled, p=2, dim=1)

def load_finetuned_model(checkpoint_path, model_name="ViT-L-14", pretrained="openai", device='cuda'):
    """Загрузка fine-tuned модели"""
    print(f"[i] Загрузка модели: {checkpoint_path}")
    
    # Загружаем базовый CLIP
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    
    # Создаём CLIPGeM
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    gem_p = checkpoint.get('gem_p', 3.0)
    
    model = CLIPGeM(clip_model, gem_p=gem_p)
    
    # Загружаем веса
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"[✓] Модель загружена (GeM p={gem_p:.2f})")
    return model, preprocess

# ========================= Embeddings Extraction =========================

@torch.no_grad()
def extract_embeddings(model, crops_df, preprocess, device, batch_size=128):
    """Извлечение embeddings для всех кропов"""
    model.eval()
    
    embeddings = []
    valid_indices = []
    
    # Батч-процессинг
    n_batches = (len(crops_df) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(n_batches), desc="Extracting embeddings", unit="batch"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(crops_df))
        batch_df = crops_df.iloc[start_idx:end_idx]
        
        batch_images = []
        batch_valid = []
        
        for idx, row in batch_df.iterrows():
            try:
                img = Image.open(row['path']).convert('RGB')
                img_tensor = preprocess(img)
                batch_images.append(img_tensor)
                batch_valid.append(idx)
            except Exception as e:
                print(f"[!] Error loading {row['path']}: {e}")
                continue
        
        if not batch_images:
            continue
        
        # Forward pass
        batch_tensor = torch.stack(batch_images).to(device)
        batch_embs = model(batch_tensor)
        
        embeddings.append(batch_embs.cpu().numpy())
        valid_indices.extend(batch_valid)
    
    # Concatenate
    embeddings = np.vstack(embeddings)
    
    print(f"[✓] Extracted {len(embeddings)} embeddings (dim={embeddings.shape})")
    return embeddings, valid_indices

# ========================= HNSW Index =========================

def build_hnsw_index(embeddings, output_path, M=32, efC=200):
    """Построение HNSW индекса"""
    print(f"\n[i] Построение HNSW индекса...")
    print(f"    Vectors: {len(embeddings)}, Dim: {embeddings.shape}")
    print(f"    M={M}, efConstruction={efC}")
    
    N, D = embeddings.shape
    
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    embeddings = (embeddings / norms).astype(np.float32)
    
    # Create index
    index = hnswlib.Index(space='cosine', dim=D)
    index.init_index(max_elements=N, M=M, ef_construction=efC, random_seed=42)
    
    # Add items in batches
    batch_size = 10000
    for i in tqdm(range(0, N, batch_size), desc="Adding to HNSW", unit="batch"):
        end = min(i + batch_size, N)
        index.add_items(embeddings[i:end], np.arange(i, end))
    
    # Save
    index.save_index(str(output_path))
    print(f"[✓] HNSW saved: {output_path}")
    
    return index

# ========================= TF-IDF Index =========================

def build_tfidf_index(crops_df, output_dir):
    """Построение TF-IDF индекса из OCR текстов"""
    print(f"\n[i] Построение TF-IDF индекса...")
    
    if 'ocr_text' not in crops_df.columns:
        print("[!] Нет колонки ocr_text, пропускаем TF-IDF")
        return None, None
    
    texts = crops_df['ocr_text'].fillna('').tolist()
    
    # Подсчёт непустых текстов
    n_nonempty = sum(1 for t in texts if len(t.strip()) > 0)
    print(f"    Texts: {len(texts)}, Non-empty: {n_nonempty}")
    
    if n_nonempty < 10:
        print("[!] Слишком мало текстов, пропускаем TF-IDF")
        return None, None
    
    # Build vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        lowercase=True,
        token_pattern=r'\b\w+\b'
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    print(f"[✓] TF-IDF: vocab size={len(vectorizer.vocabulary_)}, matrix shape={tfidf_matrix.shape}")
    
    # Save
    vectorizer_path = output_dir / "tfidf_vectorizer.joblib"
    matrix_path = output_dir / "tfidf_matrix.npz"
    texts_path = output_dir / "ocr_texts.txt"
    
    joblib.dump(vectorizer, vectorizer_path)
    sparse.save_npz(matrix_path, tfidf_matrix)
    
    with open(texts_path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')
    
    print(f"[✓] TF-IDF saved:")
    print(f"    Vectorizer: {vectorizer_path}")
    print(f"    Matrix: {matrix_path}")
    
    return vectorizer, tfidf_matrix

# ========================= OSM Spatial Index =========================

def build_osm_spatial_index(osm_data_path, output_path):
    """Построение spatial index для OSM POI"""
    print(f"\n[i] Построение OSM spatial index...")
    
    if not Path(osm_data_path).exists():
        print(f"[!] OSM data не найдена: {osm_data_path}")
        print("    Пропускаем OSM index")
        return None
    
    # Загрузка OSM POI
    pois = []
    with open(osm_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                pois.append(json.loads(line))
    
    print(f"[i] Загружено {len(pois)} POI")
    
    if len(pois) == 0:
        print("[!] Нет POI данных")
        return None
    
    # Extract coordinates
    coords = np.array([[poi['lat'], poi['lon']] for poi in pois])
    
    # Build BallTree для быстрого радиусного поиска
    tree = BallTree(np.radians(coords), metric='haversine')
    
    print(f"[✓] BallTree построен: {len(coords)} points")
    
    # Save
    spatial_data = {
        'tree': tree,
        'pois': pois,
        'coords': coords
    }
    joblib.dump(spatial_data, output_path)
    
    print(f"[✓] OSM spatial index saved: {output_path}")
    
    return spatial_data

# ========================= Main =========================

def main():
    parser = argparse.ArgumentParser(
        description="Построение индексов для геолокализации"
    )
    
    # Входные данные
    parser.add_argument("--model-path", required=True,
                       help="Path к fine-tuned модели (.pt файл)")
    parser.add_argument("--crops-meta", required=True,
                       help="CSV с метаданными кропов")
    parser.add_argument("--osm-data", default=None,
                       help="JSONL файл с OSM POI (optional)")
    
    # Выходная папка
    parser.add_argument("--output-dir", default="index",
                       help="Папка для сохранения индексов")
    
    # Параметры модели
    parser.add_argument("--model-name", default="ViT-L-14",
                       help="CLIP модель (default: ViT-L-14)")
    parser.add_argument("--pretrained", default="openai",
                       help="Pretrained source")
    
    # Параметры индексов
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Batch size для извлечения embeddings")
    parser.add_argument("--hnsw-M", type=int, default=32,
                       help="HNSW parameter M")
    parser.add_argument("--hnsw-efC", type=int, default=200,
                       help="HNSW efConstruction")
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[i] Device: {device}")
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== 1. Загрузка модели ==========
    print("\n[1/5] Загрузка fine-tuned модели")
    model, preprocess = load_finetuned_model(
        args.model_path,
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=device
    )
    
    # ========== 2. Загрузка метаданных ==========
    print(f"\n[2/5] Загрузка метаданных")
    crops_df = pd.read_csv(args.crops_meta)
    
    # Фильтрация существующих файлов
    existing_mask = crops_df['path'].apply(lambda p: Path(p).exists())
    crops_df = crops_df[existing_mask].reset_index(drop=True)
    
    print(f"[✓] {len(crops_df)} crops")
    
    # ========== 3. Извлечение embeddings ==========
    print(f"\n[3/5] Извлечение embeddings")
    embeddings, valid_indices = extract_embeddings(
        model, crops_df, preprocess, device, batch_size=args.batch_size
    )
    
    # Обновляем crops_df только валидными
    crops_df = crops_df.iloc[valid_indices].reset_index(drop=True)
    
    # Save embeddings
    embs_path = output_dir / "clip_gem_embeddings.npy"
    np.save(embs_path, embeddings)
    print(f"[✓] Embeddings saved: {embs_path}")
    
    # Save metadata
    meta_path = output_dir / "crops.csv"
    crops_df.to_csv(meta_path, index=False)
    print(f"[✓] Metadata saved: {meta_path}")
    
    # ========== 4. Построение HNSW индекса ==========
    print(f"\n[4/5] Построение HNSW индекса")
    hnsw_path = output_dir / "hnsw_gem.bin"
    build_hnsw_index(embeddings, hnsw_path, M=args.hnsw_M, efC=args.hnsw_efC)
    
    # ========== 5. Построение TF-IDF индекса ==========
    build_tfidf_index(crops_df, output_dir)
    
    # ========== 6. Построение OSM spatial index ==========
    if args.osm_data:
        osm_path = output_dir / "osm_spatial.pkl"
        build_osm_spatial_index(args.osm_data, osm_path)
    
    # ========== Сохранение конфигурации ==========
    config = {
        'model_name': args.model_name,
        'pretrained': args.pretrained,
        'model_path': str(args.model_path),
        'embedding_dim': int(embeddings.shape),
        'n_crops': len(crops_df),
        'hnsw_M': args.hnsw_M,
        'hnsw_efC': args.hnsw_efC,
    }
    
    config_path = output_dir / "index_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✅ ПОСТРОЕНИЕ ИНДЕКСОВ ЗАВЕРШЕНО")
    print(f"{'='*60}")
    print(f"Индексы сохранены в: {output_dir}/")
    print(f"  - HNSW: hnsw_gem.bin")
    print(f"  - Embeddings: clip_gem_embeddings.npy")
    print(f"  - TF-IDF: tfidf_*.joblib/npz")
    print(f"  - OSM: osm_spatial.pkl")
    print(f"  - Config: index_config.json")
    
    print("\n🎯 Следующий шаг:")
    print(f"   python scripts/07_query_improved.py --image samples/test.jpg")

if __name__ == "__main__":
    main()
