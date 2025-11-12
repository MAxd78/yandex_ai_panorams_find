#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_extract_ocr.py — извлечение текста из кропов через EasyOCR

Обрабатывает все кропы и добавляет колонку ocr_text в crops.csv

Использование:
python scripts/04_extract_ocr.py \
    --crops-meta meta/crops.csv \
    --output-meta meta/crops_with_ocr.csv \
    --workers 4
"""

import os
import sys
import argparse
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import easyocr

# ========================= EasyOCR Setup =========================

class OCRExtractor:
    def __init__(self, languages=['ru', 'en'], gpu=True):
        """
        Инициализация EasyOCR reader
        
        Args:
            languages: Список языков для распознавания
            gpu: Использовать GPU (требует CUDA)
        """
        print(f"[i] Инициализация EasyOCR: {languages}, GPU={gpu}")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        print("[✓] EasyOCR готов")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Предобработка изображения для улучшения OCR
        
        Returns:
            Preprocessed image (grayscale, enhanced contrast)
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Не удалось загрузить {image_path}")
        
        # Конвертация в grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # CLAHE для улучшения контраста
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def extract_text(self, image_path: str, confidence_threshold: float = 0.4) -> str:
        """
        Извлечение текста из изображения
        
        Args:
            image_path: Путь к изображению
            confidence_threshold: Минимальный порог уверенности
        
        Returns:
            Извлечённый текст (объединённый через пробел)
        """
        try:
            # Предобработка
            enhanced = self.preprocess_image(image_path)
            
            # OCR (detail=1 → bbox, text, confidence)
            results = self.reader.readtext(enhanced, detail=1)
            
            # Фильтрация по confidence
            texts = [
                text.strip() 
                for (bbox, text, conf) in results 
                if conf > confidence_threshold and text.strip()
            ]
            
            return ' '.join(texts)
        
        except Exception as e:
            print(f"[!] Ошибка OCR для {image_path}: {e}")
            return ""

# ========================= Batch Processing =========================

def process_batch(ocr_extractor: OCRExtractor, crop_paths: list, confidence: float = 0.4) -> list:
    """
    Обработка батча изображений
    
    Returns:
        Список извлечённых текстов
    """
    results = []
    for path in crop_paths:
        text = ocr_extractor.extract_text(path, confidence_threshold=confidence)
        results.append(text)
    return results

# ========================= Main =========================

def main():
    parser = argparse.ArgumentParser(
        description="Извлечение текста из кропов через EasyOCR"
    )
    
    # Входные/выходные данные
    parser.add_argument("--crops-meta", required=True, 
                       help="CSV с метаданными кропов (из 03_prepare_dataset.py)")
    parser.add_argument("--output-meta", required=True,
                       help="CSV с добавленной колонкой ocr_text")
    
    # Параметры OCR
    parser.add_argument("--languages", nargs='+', default=['ru', 'en'],
                       help="Языки для распознавания (default: ru en)")
    parser.add_argument("--confidence", type=float, default=0.4,
                       help="Минимальная уверенность для текста (default: 0.4)")
    parser.add_argument("--gpu", action='store_true', default=True,
                       help="Использовать GPU (default: True)")
    
    # Производительность
    parser.add_argument("--workers", type=int, default=1,
                       help="Количество параллельных процессов (default: 1)")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Размер батча для сохранения (default: 100)")
    
    args = parser.parse_args()
    
    # Загрузка метаданных
    crops_meta_path = Path(args.crops_meta)
    if not crops_meta_path.exists():
        print(f"[!] Файл не найден: {crops_meta_path}")
        sys.exit(1)
    
    print(f"[i] Загрузка метаданных: {crops_meta_path}")
    crops_df = pd.read_csv(crops_meta_path)
    
    if 'path' not in crops_df.columns:
        print("[!] В метаданных нет колонки 'path'")
        sys.exit(1)
    
    print(f"[✓] Загружено {len(crops_df)} кропов")
    
    # Проверка существования файлов
    existing_mask = crops_df['path'].apply(lambda p: Path(p).exists())
    n_missing = (~existing_mask).sum()
    if n_missing > 0:
        print(f"[!] Не найдено {n_missing} файлов, пропускаем...")
        crops_df = crops_df[existing_mask].reset_index(drop=True)
    
    # Инициализация OCR
    ocr_extractor = OCRExtractor(languages=args.languages, gpu=args.gpu)
    
    # Обработка кропов
    print(f"\n[i] Извлечение текста...")
    print(f"    Языки: {args.languages}")
    print(f"    Порог уверенности: {args.confidence}")
    print(f"    Workers: {args.workers}\n")
    
    ocr_texts = []
    
    if args.workers > 1:
        # Параллельная обработка
        print("[!] Внимание: EasyOCR не поддерживает multiprocessing с GPU")
        print("    Используется последовательная обработка")
        args.workers = 1
    
    # Последовательная обработка с прогресс-баром
    for idx, row in tqdm(crops_df.iterrows(), total=len(crops_df), 
                         desc="OCR extraction", unit="crop"):
        text = ocr_extractor.extract_text(row['path'], args.confidence)
        ocr_texts.append(text)
        
        # Периодическое сохранение (на случай сбоя)
        if (idx + 1) % args.batch_size == 0:
            temp_df = crops_df.iloc[:idx+1].copy()
            temp_df['ocr_text'] = ocr_texts
            temp_path = Path(args.output_meta).with_suffix('.temp.csv')
            temp_df.to_csv(temp_path, index=False)
            print(f"[i] Промежуточное сохранение: {idx+1}/{len(crops_df)}")
    
    # Добавление колонки
    crops_df['ocr_text'] = ocr_texts
    
    # Сохранение результатов
    output_path = Path(args.output_meta)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crops_df.to_csv(output_path, index=False)
    
    # Удаление временного файла
    temp_path = output_path.with_suffix('.temp.csv')
    if temp_path.exists():
        temp_path.unlink()
    
    # Статистика
    print("\n" + "=" * 60)
    print("✅ ИЗВЛЕЧЕНИЕ ТЕКСТА ЗАВЕРШЕНО")
    print("=" * 60)
    print(f"Обработано кропов: {len(crops_df)}")
    print(f"С текстом: {(crops_df['ocr_text'].str.len() > 0).sum()}")
    print(f"Без текста: {(crops_df['ocr_text'].str.len() == 0).sum()}")
    print(f"Средняя длина текста: {crops_df['ocr_text'].str.len().mean():.1f} символов")
    print(f"\nРезультаты сохранены: {output_path}")
    
    print("\n🎯 Следующий шаг:")
    print(f"   python scripts/05_train_model.py --crops-meta {output_path}")

if __name__ == "__main__":
    main()
