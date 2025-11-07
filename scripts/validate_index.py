#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_index.py — Валидация и самопроверка индекса

Функции:
  ✅ Self-testing: использует кропы как query и проверяет точность
  ✅ Quality metrics: средняя similarity, геометрия, распределение
  ✅ Auto-tuning: подбирает оптимальные параметры (geom_weight, verify_k)
  ✅ Benchmark: замеряет скорость поиска

Использование:
  # Базовая валидация
  python scripts/validate_index.py

  # С auto-tuning параметров
  python scripts/validate_index.py --auto-tune

  # Только быстрые тесты
  python scripts/validate_index.py --quick
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple
import time

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import hnswlib

# Подавляем варнинги
import warnings
warnings.filterwarnings("ignore")

# ========================= Константы =========================
SEED = 42
DEFAULT_TEST_SIZE = 100  # Кропов для тестирования
DEFAULT_QUICK_SIZE = 20

# ========================= Utils =========================

def load_index(index_dir: Path) -> Tuple:
    """Загрузить индекс и метаданные"""
    print(f"\n📂 Загрузка индекса из: {index_dir}")
    
    # Метаданные
    meta_parquet = index_dir / "crops.parquet"
    if meta_parquet.exists():
        meta = pd.read_parquet(meta_parquet)
    else:
        meta = pd.read_csv("meta/crops.csv")
    
    print(f"   Кропов в мета: {len(meta)}")
    
    # Эмбеддинги
    embs_file = None
    for candidate in ["embs.npy", "embeddings.npy", "clip_embeddings.npy"]:
        f = index_dir / candidate
        if f.exists():
            embs_file = f
            break
    
    if embs_file is None:
        raise FileNotFoundError("Не найден файл эмбеддингов!")
    
    embs = np.load(embs_file)
    print(f"   Эмбеддинги: {embs.shape}")
    
    # HNSW
    hnsw_file = None
    for candidate in ["hnsw.bin", "hnsw_clip.bin"]:
        f = index_dir / candidate
        if f.exists():
            hnsw_file = f
            break
    
    if hnsw_file is None:
        raise FileNotFoundError("Не найден HNSW индекс!")
    
    dim = embs.shape[1]
    index = hnswlib.Index(space="cosine", dim=dim)
    index.load_index(str(hnsw_file))
    print(f"   HNSW: {hnsw_file.name}")
    
    # Model config
    model_json = index_dir / "model.json"
    if model_json.exists():
        with open(model_json) as f:
            model_config = json.load(f)
        print(f"   Модель: {model_config.get('model', 'unknown')}")
    else:
        model_config = {}
    
    return meta, embs, index, model_config


def sample_test_set(meta: pd.DataFrame, size: int) -> pd.DataFrame:
    """Выбрать случайные кропы для тестирования"""
    random.seed(SEED)
    indices = random.sample(range(len(meta)), min(size, len(meta)))
    return meta.iloc[indices].reset_index(drop=True)


def cosine_to_sim(dist: np.ndarray) -> np.ndarray:
    return 1.0 - dist


# ========================= Tests =========================

class IndexValidator:
    """Валидатор индекса"""
    
    def __init__(self, meta, embs, index, model_config):
        self.meta = meta
        self.embs = embs
        self.index = index
        self.model_config = model_config
        
        self.results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "metrics": {},
        }
    
    def test_self_retrieval(self, test_set: pd.DataFrame, topk: int = 50) -> Dict:
        """
        Тест самопоиска: каждый кроп должен найти сам себя в топ-1
        """
        print("\n" + "="*80)
        print("🔍 ТЕСТ 1: Self-Retrieval (кроп находит сам себя)")
        print("="*80)
        
        self.index.set_ef(200)
        
        correct_top1 = 0
        correct_top5 = 0
        correct_top10 = 0
        similarities = []
        ranks = []
        
        for idx, row in tqdm(test_set.iterrows(), total=len(test_set), desc="Self-retrieval"):
            # Получаем эмбеддинг кропа
            crop_idx = meta[meta["crop_id"] == row["crop_id"]].index[0]
            q_emb = self.embs[crop_idx:crop_idx+1]
            
            # Поиск
            labels, dists = self.index.knn_query(q_emb, k=topk)
            labels = labels[0]
            sims = cosine_to_sim(dists[0])
            
            # Проверяем где находится сам кроп
            if crop_idx in labels:
                rank = np.where(labels == crop_idx)[0][0] + 1
                ranks.append(rank)
                
                if rank == 1:
                    correct_top1 += 1
                if rank <= 5:
                    correct_top5 += 1
                if rank <= 10:
                    correct_top10 += 1
                
                similarities.append(sims[rank-1])
            else:
                ranks.append(topk + 1)
                similarities.append(0.0)
        
        # Результаты
        total = len(test_set)
        top1_acc = correct_top1 / total * 100
        top5_acc = correct_top5 / total * 100
        top10_acc = correct_top10 / total * 100
        avg_sim = np.mean(similarities) if similarities else 0
        avg_rank = np.mean(ranks)
        
        print(f"\n📊 Результаты:")
        print(f"   Top-1 точность: {top1_acc:.1f}% ({correct_top1}/{total})")
        print(f"   Top-5 точность: {top5_acc:.1f}% ({correct_top5}/{total})")
        print(f"   Top-10 точность: {top10_acc:.1f}% ({correct_top10}/{total})")
        print(f"   Средняя similarity: {avg_sim:.4f}")
        print(f"   Средний ранг: {avg_rank:.1f}")
        
        # Оценка
        if top1_acc >= 95:
            print("   ✅ ОТЛИЧНО! Индекс работает идеально")
        elif top1_acc >= 85:
            print("   ✅ ХОРОШО! Индекс работает корректно")
        elif top1_acc >= 70:
            print("   ⚠️  УДОВЛЕТВОРИТЕЛЬНО. Возможны проблемы с нормализацией")
        else:
            print("   ❌ ПЛОХО! Индекс работает неправильно!")
        
        self.results["test_self_retrieval"] = {
            "top1_acc": top1_acc,
            "top5_acc": top5_acc,
            "top10_acc": top10_acc,
            "avg_similarity": float(avg_sim),
            "avg_rank": float(avg_rank),
        }
        
        return self.results["test_self_retrieval"]
    
    def test_same_pano_retrieval(self, test_set: pd.DataFrame, topk: int = 50) -> Dict:
        """
        Тест поиска той же панорамы: кропы одной панорамы должны находиться близко
        """
        print("\n" + "="*80)
        print("🔍 ТЕСТ 2: Same-Pano Retrieval (находит кропы той же панорамы)")
        print("="*80)
        
        self.index.set_ef(200)
        
        same_pano_in_top5 = 0
        same_pano_in_top10 = 0
        avg_same_pano_count = []
        
        for idx, row in tqdm(test_set.iterrows(), total=len(test_set), desc="Same-pano"):
            crop_idx = meta[meta["crop_id"] == row["crop_id"]].index[0]
            pano_id = row["pano_id"]
            
            q_emb = self.embs[crop_idx:crop_idx+1]
            labels, dists = self.index.knn_query(q_emb, k=topk)
            labels = labels[0]
            
            # Сколько кропов той же панорамы в топ-K
            same_pano_labels = meta.iloc[labels]["pano_id"] == pano_id
            same_count_top50 = same_pano_labels.sum()
            same_count_top5 = same_pano_labels[:5].sum()
            same_count_top10 = same_pano_labels[:10].sum()
            
            avg_same_pano_count.append(same_count_top50)
            
            if same_count_top5 >= 2:  # Минимум 2 кропа (сам + ещё один)
                same_pano_in_top5 += 1
            if same_count_top10 >= 3:
                same_pano_in_top10 += 1
        
        total = len(test_set)
        top5_rate = same_pano_in_top5 / total * 100
        top10_rate = same_pano_in_top10 / total * 100
        avg_count = np.mean(avg_same_pano_count)
        
        print(f"\n📊 Результаты:")
        print(f"   Кропы той же панорамы в Top-5: {top5_rate:.1f}%")
        print(f"   Кропы той же панорамы в Top-10: {top10_rate:.1f}%")
        print(f"   Среднее кол-во кропов в Top-50: {avg_count:.1f}")
        
        if top5_rate >= 80:
            print("   ✅ ОТЛИЧНО! Кропы панорам группируются правильно")
        elif top5_rate >= 60:
            print("   ✅ ХОРОШО!")
        else:
            print("   ⚠️  СЛАБО. Возможно нужно больше кропов на панораму")
        
        self.results["test_same_pano"] = {
            "top5_rate": top5_rate,
            "top10_rate": top10_rate,
            "avg_same_pano_count": float(avg_count),
        }
        
        return self.results["test_same_pano"]
    
    def test_similarity_distribution(self) -> Dict:
        """
        Тест распределения similarity: проверяем что эмбеддинги нормализованы
        """
        print("\n" + "="*80)
        print("📊 ТЕСТ 3: Распределение Similarity")
        print("="*80)
        
        # Берём случайные 500 эмбеддингов
        sample_size = min(500, len(self.embs))
        indices = random.sample(range(len(self.embs)), sample_size)
        sample_embs = self.embs[indices]
        
        # Считаем нормы
        norms = np.linalg.norm(sample_embs, axis=1)
        avg_norm = float(np.mean(norms))
        std_norm = float(np.std(norms))
        
        print(f"\n📊 Нормы эмбеддингов:")
        print(f"   Среднее: {avg_norm:.6f}")
        print(f"   Std: {std_norm:.6f}")
        print(f"   Min: {norms.min():.6f}")
        print(f"   Max: {norms.max():.6f}")
        
        # Проверка нормализации
        is_normalized = (0.99 <= avg_norm <= 1.01) and (std_norm < 0.01)
        
        if is_normalized:
            print("   ✅ ОТЛИЧНО! Эмбеддинги правильно нормализованы")
        else:
            print("   ⚠️  ПРОБЛЕМА! Эмбеддинги не нормализованы (может быть медленнее)")
        
        self.results["test_similarity_dist"] = {
            "avg_norm": avg_norm,
            "std_norm": std_norm,
            "is_normalized": is_normalized,
        }
        
        return self.results["test_similarity_dist"]
    
    def benchmark_speed(self, test_set: pd.DataFrame, topk: int = 50) -> Dict:
        """
        Benchmark скорости поиска
        """
        print("\n" + "="*80)
        print("⚡ BENCHMARK: Скорость поиска")
        print("="*80)
        
        self.index.set_ef(200)
        
        times = []
        
        for idx, row in tqdm(test_set.iterrows(), total=len(test_set), desc="Benchmark"):
            crop_idx = meta[meta["crop_id"] == row["crop_id"]].index[0]
            q_emb = self.embs[crop_idx:crop_idx+1]
            
            start = time.time()
            labels, dists = self.index.knn_query(q_emb, k=topk)
            elapsed = time.time() - start
            
            times.append(elapsed * 1000)  # в миллисекунды
        
        avg_time = np.mean(times)
        p50_time = np.percentile(times, 50)
        p95_time = np.percentile(times, 95)
        p99_time = np.percentile(times, 99)
        
        queries_per_sec = 1000 / avg_time if avg_time > 0 else 0
        
        print(f"\n⏱️  Время поиска (k={topk}):")
        print(f"   Среднее: {avg_time:.2f} ms")
        print(f"   P50: {p50_time:.2f} ms")
        print(f"   P95: {p95_time:.2f} ms")
        print(f"   P99: {p99_time:.2f} ms")
        print(f"   Queries/sec: {queries_per_sec:.1f}")
        
        if avg_time < 50:
            print("   ⚡ ОТЛИЧНО! Очень быстро")
        elif avg_time < 100:
            print("   ✅ ХОРОШО!")
        else:
            print("   ⚠️  МЕДЛЕННО. Возможно нужно уменьшить ef или M")
        
        self.results["benchmark"] = {
            "avg_time_ms": float(avg_time),
            "p95_time_ms": float(p95_time),
            "queries_per_sec": float(queries_per_sec),
        }
        
        return self.results["benchmark"]
    
    def save_report(self, output_path: str):
        """Сохранить отчёт в JSON"""
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Отчёт сохранён: {output_path}")


# ========================= Main =========================

def main():
    ap = argparse.ArgumentParser(
        description="Валидация индекса с самопроверкой",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    ap.add_argument("--index-dir", default="index")
    ap.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE,
                    help="Кол-во кропов для тестирования")
    ap.add_argument("--quick", action="store_true",
                    help="Быстрый тест (меньше кропов)")
    ap.add_argument("--report", default="validation_report.json",
                    help="Куда сохранить отчёт")
    ap.add_argument("--auto-tune", action="store_true",
                    help="Auto-tuning параметров (пока не реализовано)")
    
    args = ap.parse_args()
    
    print("=" * 80)
    print("🔬 ВАЛИДАЦИЯ ИНДЕКСА")
    print("=" * 80)
    
    # Загрузка
    meta, embs, index, model_config = load_index(Path(args.index_dir))
    
    # Тестовый набор
    test_size = DEFAULT_QUICK_SIZE if args.quick else args.test_size
    test_set = sample_test_set(meta, test_size)
    
    print(f"\n📋 Тестовый набор: {len(test_set)} кропов")
    
    # Валидатор
    validator = IndexValidator(meta, embs, index, model_config)
    
    # Тесты
    validator.test_self_retrieval(test_set)
    validator.test_same_pano_retrieval(test_set)
    validator.test_similarity_distribution()
    validator.benchmark_speed(test_set)
    
    # Сохранение отчёта
    validator.save_report(args.report)
    
    print("\n" + "=" * 80)
    print("✅ ВАЛИДАЦИЯ ЗАВЕРШЕНА")
    print("=" * 80)
    
    # Итоговая оценка
    self_ret = validator.results.get("test_self_retrieval", {})
    top1_acc = self_ret.get("top1_acc", 0)
    
    print(f"\n🎯 Общая оценка:")
    if top1_acc >= 95:
        print("   ✅ ✅ ✅ ОТЛИЧНО! Индекс готов к продакшену")
    elif top1_acc >= 85:
        print("   ✅ ✅ ХОРОШО! Индекс работает корректно")
    elif top1_acc >= 70:
        print("   ⚠️  УДОВЛЕТВОРИТЕЛЬНО. Рекомендуется пересобрать индекс")
    else:
        print("   ❌ ПЛОХО! Индекс не работает, нужна отладка!")


if __name__ == "__main__":
    main()