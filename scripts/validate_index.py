#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_index.py — валидация индекса с метриками и диагностикой

Проверяет:
  1. Top-K accuracy (модель находит свои фото?)
  2. Mean Reciprocal Rank (MRR)
  3. Mean Average Precision (MAP)
  4. Распределение расстояний до правильного match
  5. Preprocessing consistency (query vs index)
  6. Визуализация ближайших соседей

Использование:
  # Базовая валидация
  python scripts/validate_index.py --test-size 100
  
  # Полная диагностика с визуализацией
  python scripts/validate_index.py --test-size 500 --visualize --save-failures
  
  # Только проверка preprocessing
  python scripts/validate_index.py --check-preprocessing
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import hnswlib

# ========================= Константы =========================
SEED = 42
DEFAULT_TILE_SIZE = 336
DEFAULT_TILE_STRIDE = 224
DEFAULT_EF = 256

# ========================= Утилиты =========================

def pick_device():
    """Автоопределение устройства"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_from_index(index_dir: Path):
    """Загрузка CLIP модели из индекса"""
    import open_clip
    
    index_dir = Path(index_dir)
    model_name = "ViT-L-14"
    pretrained = "openai"
    
    meta_json = index_dir / "model.json"
    if meta_json.exists():
        try:
            meta = json.loads(meta_json.read_text())
            model_name = meta.get("model", model_name)
            pretrained = meta.get("pretrained", pretrained)
        except Exception as e:
            print(f"[!] Ошибка чтения model.json: {e}")
    
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
    except Exception as e:
        print(f"[!] Ошибка загрузки модели: {e}")
        sys.exit(1)
    
    model.eval()
    return model, preprocess, model_name


def tile_image_pil(pil_img: Image.Image, size=336, stride=224):
    """Тайлинг изображения с перекрытием"""
    W, H = pil_img.size
    tiles = []
    
    for y in range(0, max(1, H - size + 1), stride):
        for x in range(0, max(1, W - size + 1), stride):
            tile = pil_img.crop((x, y, x + size, y + size))
            tiles.append(tile)
    
    if not tiles:
        tiles = [pil_img.resize((size, size), Image.BICUBIC)]
    
    return tiles


def compute_query_embedding(
    img_path: str,
    model,
    preprocess,
    device,
    tile_size=336,
    tile_stride=224,
    aggregation="max"
):
    """Вычисление эмбеддинга запроса (как в 05_query.py)"""
    img = Image.open(img_path).convert("RGB")
    tiles = tile_image_pil(img, size=tile_size, stride=tile_stride)
    
    embeds = []
    with torch.inference_mode():
        for t in tiles:
            ten = preprocess(t).unsqueeze(0).to(device)
            e = model.encode_image(ten)
            e = torch.nn.functional.normalize(e, dim=-1)
            embeds.append(e)
    
    E = torch.stack(embeds, dim=0).squeeze(1)  # [T, D]
    
    if aggregation == "max":
        q_emb = torch.amax(E, dim=0)
    elif aggregation == "mean":
        q_emb = torch.mean(E, dim=0)
    elif aggregation == "first":
        q_emb = E[0]
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    q_emb = q_emb.detach().cpu().numpy()
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    
    return q_emb


# ========================= Метрики =========================

def compute_metrics(results: List[Dict], verbose=True):
    """
    Вычисление метрик валидации
    
    Args:
        results: Список результатов с полями:
            - query_idx: индекс запроса
            - ground_truth: список правильных индексов
            - retrieved: список найденных индексов (отсортированы по релевантности)
    
    Returns:
        Dict с метриками
    """
    topk_acc = {1: 0, 5: 0, 10: 0, 50: 0, 100: 0}
    reciprocal_ranks = []
    avg_precisions = []
    distances = []  # Расстояния до правильного match
    
    for res in results:
        gt_set = set(res["ground_truth"])
        retrieved = res["retrieved"]
        
        # Top-K accuracy
        for k in topk_acc.keys():
            if any(idx in gt_set for idx in retrieved[:k]):
                topk_acc[k] += 1
        
        # Reciprocal Rank
        rank = None
        for i, idx in enumerate(retrieved):
            if idx in gt_set:
                rank = i + 1
                break
        
        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
            distances.append(rank)
        else:
            reciprocal_ranks.append(0.0)
            distances.append(len(retrieved) + 1)
        
        # Average Precision
        hits = 0
        precisions = []
        for i, idx in enumerate(retrieved):
            if idx in gt_set:
                hits += 1
                precisions.append(hits / (i + 1))
        
        if precisions:
            avg_precisions.append(np.mean(precisions))
        else:
            avg_precisions.append(0.0)
    
    # Нормализация
    n = len(results)
    for k in topk_acc.keys():
        topk_acc[k] = (topk_acc[k] / n) * 100.0
    
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    map_score = np.mean(avg_precisions) if avg_precisions else 0.0
    
    metrics = {
        "top1_acc": topk_acc[1],
        "top5_acc": topk_acc[5],
        "top10_acc": topk_acc[10],
        "top50_acc": topk_acc[50],
        "top100_acc": topk_acc[100],
        "mrr": mrr,
        "map": map_score,
        "mean_distance": np.mean(distances),
        "median_distance": np.median(distances),
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("📊 МЕТРИКИ ВАЛИДАЦИИ")
        print("=" * 60)
        print(f"Top-1 Accuracy:   {metrics['top1_acc']:>6.2f}%")
        print(f"Top-5 Accuracy:   {metrics['top5_acc']:>6.2f}%")
        print(f"Top-10 Accuracy:  {metrics['top10_acc']:>6.2f}%")
        print(f"Top-50 Accuracy:  {metrics['top50_acc']:>6.2f}%")
        print(f"Top-100 Accuracy: {metrics['top100_acc']:>6.2f}%")
        print(f"Mean Reciprocal Rank (MRR): {metrics['mrr']:.4f}")
        print(f"Mean Average Precision (MAP): {metrics['map']:.4f}")
        print(f"Mean Distance to GT: {metrics['mean_distance']:.1f}")
        print(f"Median Distance to GT: {metrics['median_distance']:.1f}")
        print("=" * 60)
        
        # Интерпретация
        if metrics['top1_acc'] >= 90:
            print("✅ ОТЛИЧНО! Модель находит свои фото.")
        elif metrics['top1_acc'] >= 70:
            print("⚠️  ПРИЕМЛЕМО, но есть потенциал для улучшения.")
        elif metrics['top1_acc'] >= 50:
            print("⚠️  ПРОБЛЕМА! Top-1 accuracy слишком низкая.")
        else:
            print("🔥 КРИТИЧЕСКАЯ ПРОБЛЕМА! Модель не находит свои фото!")
            print("   Возможные причины:")
            print("   1. Preprocessing mismatch (query ≠ index)")
            print("   2. Неправильная нормализация")
            print("   3. Tile aggregation слишком агрессивная")
            print("   4. Проблемы с моделью/весами")
    
    return metrics


# ========================= Валидация =========================

def validate_index(
    index_dir: Path,
    crops_meta: Path,
    test_size: int = 100,
    ef: int = 256,
    topk: int = 100,
    tile_size: int = 336,
    tile_stride: int = 224,
    aggregation: str = "max",
    save_failures: bool = False,
    visualize: bool = False,
):
    """Валидация индекса"""
    
    print(f"[i] Загрузка метаданных...")
    meta_parquet = index_dir / "crops.parquet"
    if meta_parquet.exists():
        meta = pd.read_parquet(meta_parquet)
    else:
        meta = pd.read_csv(crops_meta)
    
    print(f"[✓] Загружено {len(meta)} кропов")
    
    # Выборка тестовых изображений
    random.seed(SEED)
    valid_indices = [i for i in range(len(meta)) if os.path.exists(meta.iloc[i]["path"])]
    
    if len(valid_indices) < test_size:
        print(f"[!] Недостаточно валидных изображений ({len(valid_indices)}), уменьшаю test_size")
        test_size = len(valid_indices)
    
    test_indices = random.sample(valid_indices, test_size)
    print(f"[i] Тестовая выборка: {test_size} изображений")
    
    # Загрузка модели
    print(f"\n[i] Загрузка CLIP модели...")
    device = pick_device()
    model, preprocess, model_name = load_model_from_index(index_dir)
    model.to(device)
    print(f"[✓] Модель загружена: {model_name} на {device}")
    
    # Загрузка индекса
    print(f"\n[i] Загрузка HNSW индекса...")
    index_path = index_dir / "hnsw.bin"
    if not index_path.exists():
        print(f"[!] Не найден индекс: {index_path}")
        sys.exit(1)
    
    embs_path = index_dir / "embs.npy"
    embs = np.load(embs_path)
    dim = embs.shape[1]
    
    index = hnswlib.Index(space="cosine", dim=dim)
    index.load_index(str(index_path))
    index.set_ef(ef)
    print(f"[✓] HNSW готов (ef={ef})")
    
    # Валидация
    print(f"\n[i] Запуск валидации...")
    results = []
    
    for test_idx in tqdm(test_indices, desc="Валидация", unit="img"):
        row = meta.iloc[test_idx]
        img_path = row["path"]
        pano_id = row["pano_id"]
        
        try:
            # Вычисление эмбеддинга запроса
            q_emb = compute_query_embedding(
                img_path, model, preprocess, device,
                tile_size=tile_size, tile_stride=tile_stride,
                aggregation=aggregation
            )
            
            # Поиск
            labels, dists = index.knn_query(q_emb, k=topk)
            retrieved = labels[0].tolist()
            
            # Ground truth — все кропы этой панорамы
            gt_indices = meta[meta["pano_id"] == pano_id].index.tolist()
            
            results.append({
                "query_idx": test_idx,
                "pano_id": pano_id,
                "ground_truth": gt_indices,
                "retrieved": retrieved,
                "distances": dists[0].tolist(),
            })
            
        except Exception as e:
            print(f"[!] Ошибка для {img_path}: {e}")
            continue
    
    # Вычисление метрик
    metrics = compute_metrics(results, verbose=True)
    
    # Сохранение результатов
    results_path = index_dir / "validation_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "metrics": metrics,
            "test_size": test_size,
            "config": {
                "ef": ef,
                "topk": topk,
                "tile_size": tile_size,
                "tile_stride": tile_stride,
                "aggregation": aggregation,
            },
        }, f, indent=2)
    
    print(f"\n[✓] Результаты сохранены: {results_path}")
    
    # Визуализация
    if visualize:
        visualize_results(results, meta, index_dir)
    
    # Сохранение failures
    if save_failures:
        save_failure_cases(results, meta, index_dir)
    
    return metrics, results


def visualize_results(results: List[Dict], meta: pd.DataFrame, index_dir: Path):
    """Визуализация результатов валидации"""
    print(f"\n[i] Создание визуализаций...")
    
    vis_dir = index_dir / "validation_viz"
    vis_dir.mkdir(exist_ok=True)
    
    # 1. Распределение рангов правильного match
    ranks = []
    for res in results:
        gt_set = set(res["ground_truth"])
        for i, idx in enumerate(res["retrieved"]):
            if idx in gt_set:
                ranks.append(i + 1)
                break
        else:
            ranks.append(len(res["retrieved"]) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.hist(ranks, bins=50, edgecolor='black')
    plt.xlabel("Rank of Ground Truth")
    plt.ylabel("Frequency")
    plt.title("Distribution of Ground Truth Ranks")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.savefig(vis_dir / "ranks_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # 2. Распределение cosine distances
    distances = []
    for res in results:
        distances.extend(res["distances"][:10])  # Top-10
    
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50, edgecolor='black')
    plt.xlabel("Cosine Distance")
    plt.ylabel("Frequency")
    plt.title("Distribution of Cosine Distances (Top-10)")
    plt.grid(True, alpha=0.3)
    plt.savefig(vis_dir / "distances_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"[✓] Визуализации сохранены: {vis_dir}")


def save_failure_cases(results: List[Dict], meta: pd.DataFrame, index_dir: Path):
    """Сохранение случаев где модель ошиблась"""
    print(f"\n[i] Сохранение failure cases...")
    
    failures_dir = index_dir / "validation_failures"
    failures_dir.mkdir(exist_ok=True)
    
    failures = []
    for res in results:
        gt_set = set(res["ground_truth"])
        top1_idx = res["retrieved"][0]
        
        if top1_idx not in gt_set:
            failures.append({
                "query_idx": res["query_idx"],
                "query_pano": res["pano_id"],
                "retrieved_idx": top1_idx,
                "retrieved_pano": meta.iloc[top1_idx]["pano_id"],
                "distance": res["distances"][0],
            })
    
    if failures:
        failures_path = failures_dir / "failures.json"
        with open(failures_path, "w") as f:
            json.dump(failures, f, indent=2)
        
        print(f"[✓] Сохранено {len(failures)} failures: {failures_path}")
    else:
        print("[✓] Нет failures!")


# ========================= Проверка preprocessing =========================

def check_preprocessing_consistency(index_dir: Path, crops_meta: Path):
    """
    Проверяет consistency между preprocessing при индексации и query
    """
    print("\n" + "=" * 60)
    print("🔍 ПРОВЕРКА PREPROCESSING CONSISTENCY")
    print("=" * 60)
    
    # Загрузка
    meta_parquet = index_dir / "crops.parquet"
    if meta_parquet.exists():
        meta = pd.read_parquet(meta_parquet)
    else:
        meta = pd.read_csv(crops_meta)
    
    device = pick_device()
    model, preprocess, model_name = load_model_from_index(index_dir)
    model.to(device)
    
    embs_path = index_dir / "embs.npy"
    index_embs = np.load(embs_path)
    
    # Выбираем несколько случайных изображений
    random.seed(SEED)
    test_indices = random.sample(range(len(meta)), min(10, len(meta)))
    
    diffs = []
    for idx in test_indices:
        row = meta.iloc[idx]
        img_path = row["path"]
        
        if not os.path.exists(img_path):
            continue
        
        try:
            # Вычисляем эмбеддинг "как при query"
            img = Image.open(img_path).convert("RGB")
            with torch.inference_mode():
                ten = preprocess(img).unsqueeze(0).to(device)
                e = model.encode_image(ten)
                e = torch.nn.functional.normalize(e, dim=-1)
                query_emb = e.detach().cpu().numpy()[0]
            
            # Нормализация
            query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-9)
            
            # Сравниваем с индексированным
            index_emb = index_embs[idx]
            
            # Cosine similarity
            cos_sim = np.dot(query_emb, index_emb)
            diff = 1.0 - cos_sim
            
            diffs.append(diff)
            
            print(f"[{idx:>5}] Cosine diff: {diff:.6f} (similarity: {cos_sim:.6f})")
            
        except Exception as e:
            print(f"[!] Ошибка для {img_path}: {e}")
            continue
    
    if diffs:
        mean_diff = np.mean(diffs)
        max_diff = np.max(diffs)
        
        print("\n" + "-" * 60)
        print(f"Mean preprocessing diff: {mean_diff:.6f}")
        print(f"Max preprocessing diff:  {max_diff:.6f}")
        print("-" * 60)
        
        if mean_diff < 0.001:
            print("✅ ИДЕАЛЬНО! Preprocessing полностью совпадает.")
        elif mean_diff < 0.01:
            print("✅ ХОРОШО! Preprocessing почти идентичен.")
        elif mean_diff < 0.05:
            print("⚠️  ВНИМАНИЕ! Есть небольшие различия в preprocessing.")
        else:
            print("🔥 ПРОБЛЕМА! Preprocessing сильно отличается!")
            print("   Возможные причины:")
            print("   1. Разные параметры resize/crop")
            print("   2. Разная нормализация")
            print("   3. Разные версии библиотек (PIL, torchvision)")
    
    print("=" * 60)


# ========================= Main =========================

def main():
    parser = argparse.ArgumentParser(
        description="Валидация индекса с метриками и диагностикой"
    )
    
    # Основные параметры
    parser.add_argument("--index-dir", default="index", help="Папка с индексом")
    parser.add_argument("--crops-meta", default="meta/crops.csv", help="Метаданные кропов")
    parser.add_argument("--test-size", type=int, default=100, help="Размер тестовой выборки")
    
    # HNSW параметры
    parser.add_argument("--ef", type=int, default=DEFAULT_EF, help="HNSW ef parameter")
    parser.add_argument("--topk", type=int, default=100, help="Сколько соседей искать")
    
    # Query параметры
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--tile-stride", type=int, default=DEFAULT_TILE_STRIDE)
    parser.add_argument("--aggregation", choices=["max", "mean", "first"], default="max",
                       help="Метод агрегации тайлов")
    
    # Дополнительно
    parser.add_argument("--visualize", action="store_true", help="Создать визуализации")
    parser.add_argument("--save-failures", action="store_true", help="Сохранить failure cases")
    parser.add_argument("--check-preprocessing", action="store_true",
                       help="Проверить preprocessing consistency")
    
    args = parser.parse_args()
    
    index_dir = Path(args.index_dir)
    crops_meta = Path(args.crops_meta)
    
    if not index_dir.exists():
        print(f"[!] Не найдена папка индекса: {index_dir}")
        sys.exit(1)
    
    # Проверка preprocessing
    if args.check_preprocessing:
        check_preprocessing_consistency(index_dir, crops_meta)
        return
    
    # Валидация
    validate_index(
        index_dir=index_dir,
        crops_meta=crops_meta,
        test_size=args.test_size,
        ef=args.ef,
        topk=args.topk,
        tile_size=args.tile_size,
        tile_stride=args.tile_stride,
        aggregation=args.aggregation,
        save_failures=args.save_failures,
        visualize=args.visualize,
    )


if __name__ == "__main__":
    main()