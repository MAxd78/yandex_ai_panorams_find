#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_prepare_dataset.py — нарезка 360° панорам на кропы (фрагменты).

Берёт каждую панораму и создаёт из неё множество кропов с разными углами обзора.
Каждый кроп — это отдельное "view" из панорамы под определённым углом.

Использование:
  python scripts/03_prepare_dataset.py \
    --panos-dir data/panos_raw \
    --panos-meta meta/panos_bbox.csv \
    --output-dir data/crops \
    --output-meta meta/crops.csv \
    --yaw-step 15 \
    --pitch 3 \
    --fov 80
"""

from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from tqdm import tqdm
from PIL import Image
import cv2

# ========================= Константы =========================
DEFAULT_YAW_STEP = 15  # Шаг по горизонтали (градусы)
DEFAULT_PITCH = 3  # Угол наклона вверх (градусы)
DEFAULT_FOV = 80  # Field of view (градусы)
DEFAULT_OUTPUT_WIDTH = 640  # Ширина выходного кропа
DEFAULT_OUTPUT_HEIGHT = 640  # Высота выходного кропа

# ========================= Equirectangular to Perspective =========================

def equirectangular_to_perspective(
    equirect_img: np.ndarray,
    yaw: float,
    pitch: float,
    fov: float,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    """
    Конвертирует equirectangular панораму в perspective view (кроп).
    
    Args:
        equirect_img: Equirectangular изображение [H, W, 3]
        yaw: Угол поворота по горизонтали (градусы, 0-360)
        pitch: Угол наклона по вертикали (градусы, -90 до 90)
        fov: Field of view (градусы)
        output_width: Ширина выходного изображения
        output_height: Высота выходного изображения
    
    Returns:
        Perspective view [output_height, output_width, 3]
    """
    H, W = equirect_img.shape[:2]
    
    # Перевод углов в радианы
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    fov_rad = np.radians(fov)
    
    # Создаём сетку координат для выходного изображения
    x = np.linspace(-1, 1, output_width)
    y = np.linspace(-1, 1, output_height)
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Вычисляем фокусное расстояние
    f = 1.0 / np.tan(fov_rad / 2.0)
    
    # 3D координаты в camera space
    z = f * np.ones_like(x_grid)
    
    # Нормализация
    norm = np.sqrt(x_grid**2 + y_grid**2 + z**2)
    x_cam = x_grid / norm
    y_cam = y_grid / norm
    z_cam = z / norm
    
    # Поворот по pitch (вертикаль)
    cos_pitch = np.cos(pitch_rad)
    sin_pitch = np.sin(pitch_rad)
    
    y_rot = y_cam * cos_pitch - z_cam * sin_pitch
    z_rot = y_cam * sin_pitch + z_cam * cos_pitch
    x_rot = x_cam
    
    # Поворот по yaw (горизонталь)
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)
    
    x_final = x_rot * cos_yaw - z_rot * sin_yaw
    z_final = x_rot * sin_yaw + z_rot * cos_yaw
    y_final = y_rot
    
    # Конвертация в equirectangular координаты
    lon = np.arctan2(x_final, z_final)
    lat = np.arcsin(np.clip(y_final, -1.0, 1.0))
    
    # Маппинг в пиксели исходного изображения
    u = ((lon + np.pi) / (2 * np.pi) * W).astype(np.float32)
    v = ((np.pi / 2 - lat) / np.pi * H).astype(np.float32)
    
    # Билинейная интерполяция
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)
    
    # Используем cv2.remap для быстрой интерполяции
    perspective = cv2.remap(
        equirect_img,
        u,
        v,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP
    )
    
    return perspective


# ========================= Crop Generation =========================

def generate_crops_from_panorama(
    pano_path: str,
    pano_id: str,
    lat: float,
    lon: float,
    output_dir: Path,
    yaw_step: float = DEFAULT_YAW_STEP,
    pitch: float = DEFAULT_PITCH,
    fov: float = DEFAULT_FOV,
    output_width: int = DEFAULT_OUTPUT_WIDTH,
    output_height: int = DEFAULT_OUTPUT_HEIGHT,
) -> List[dict]:
    """
    Создать кропы из одной панорамы
    
    Returns:
        Список словарей с метаданными кропов
    """
    # Загрузка панорамы
    try:
        pano_img = cv2.imread(pano_path)
        if pano_img is None:
            raise RuntimeError(f"Не удалось загрузить {pano_path}")
        pano_img = cv2.cvtColor(pano_img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"[!] Ошибка загрузки {pano_path}: {e}")
        return []
    
    crops_meta = []
    
    # Генерируем кропы по разным yaw углам
    yaw_angles = np.arange(0, 360, yaw_step)
    
    for yaw in yaw_angles:
        try:
            # Генерация перспективного view
            crop = equirectangular_to_perspective(
                pano_img,
                yaw=yaw,
                pitch=pitch,
                fov=fov,
                output_width=output_width,
                output_height=output_height,
            )
            
            # Формирование имени файла
            crop_id = f"{pano_id}_yaw{int(yaw)}_pitch{int(pitch)}"
            crop_filename = f"{crop_id}.jpg"
            crop_path = output_dir / crop_filename
            
            # Сохранение кропа
            crop_pil = Image.fromarray(crop)
            crop_pil.save(crop_path, quality=95, optimize=True)
            
            # Метаданные
            crops_meta.append({
                "crop_id": crop_id,
                "pano_id": pano_id,
                "path": str(crop_path),
                "yaw": yaw,
                "pitch": pitch,
                "fov": fov,
                "lat": lat,
                "lon": lon,
            })
            
        except Exception as e:
            print(f"[!] Ошибка создания кропа yaw={yaw}: {e}")
            continue
    
    return crops_meta


# ========================= Main =========================

def main():
    parser = argparse.ArgumentParser(
        description="Нарезка 360° панорам на кропы"
    )
    
    # Входные данные
    parser.add_argument("--panos-dir", required=True, help="Папка с панорамами")
    parser.add_argument("--panos-meta", required=True, help="CSV с метаданными панорам")
    
    # Выходные данные
    parser.add_argument("--output-dir", required=True, help="Папка для кропов")
    parser.add_argument("--output-meta", required=True, help="CSV для метаданных кропов")
    
    # Параметры кропов
    parser.add_argument("--yaw-step", type=float, default=DEFAULT_YAW_STEP,
                       help=f"Шаг по yaw (градусы, default: {DEFAULT_YAW_STEP})")
    parser.add_argument("--pitch", type=float, default=DEFAULT_PITCH,
                       help=f"Угол pitch (градусы, default: {DEFAULT_PITCH})")
    parser.add_argument("--fov", type=float, default=DEFAULT_FOV,
                       help=f"Field of view (градусы, default: {DEFAULT_FOV})")
    parser.add_argument("--output-width", type=int, default=DEFAULT_OUTPUT_WIDTH,
                       help=f"Ширина кропа (default: {DEFAULT_OUTPUT_WIDTH})")
    parser.add_argument("--output-height", type=int, default=DEFAULT_OUTPUT_HEIGHT,
                       help=f"Высота кропа (default: {DEFAULT_OUTPUT_HEIGHT})")
    
    # Дополнительно
    parser.add_argument("--skip-existing", action="store_true",
                       help="Пропускать уже созданные кропы")
    
    args = parser.parse_args()
    
    # Проверка входных данных
    panos_dir = Path(args.panos_dir)
    panos_meta_path = Path(args.panos_meta)
    
    if not panos_dir.exists():
        print(f"[!] Не найдена папка с панорамами: {panos_dir}")
        sys.exit(1)
    
    if not panos_meta_path.exists():
        print(f"[!] Не найден файл метаданных: {panos_meta_path}")
        sys.exit(1)
    
    # Создание выходной папки
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Загрузка метаданных панорам
    print(f"[i] Загрузка метаданных панорам...")
    panos_df = pd.read_csv(panos_meta_path)
    
    if "pano_id" not in panos_df.columns or "lat" not in panos_df.columns or "lon" not in panos_df.columns:
        print("[!] В метаданных должны быть колонки: pano_id, lat, lon")
        sys.exit(1)
    
    print(f"[✓] Загружено {len(panos_df)} панорам")
    
    # Генерация кропов
    print(f"\n[i] Генерация кропов...")
    print(f"    Параметры: yaw_step={args.yaw_step}°, pitch={args.pitch}°, fov={args.fov}°")
    print(f"    Размер кропа: {args.output_width}x{args.output_height}")
    
    n_crops_per_pano = int(360 / args.yaw_step)
    print(f"    Ожидается ~{n_crops_per_pano} кропов на панораму")
    print(f"    Всего кропов: ~{n_crops_per_pano * len(panos_df)}\n")
    
    all_crops_meta = []
    success_count = 0
    error_count = 0
    
    for _, row in tqdm(panos_df.iterrows(), total=len(panos_df), desc="Панорамы", unit="pano"):
        pano_id = str(row["pano_id"])
        lat = float(row["lat"])
        lon = float(row["lon"])
        
        # Поиск файла панорамы
        # Формат имени: {pano_id}_z0.jpg
        pano_filename = f"{pano_id}_z0.jpg"
        pano_path = panos_dir / pano_filename
        
        if not pano_path.exists():
            # Пробуем другие варианты имени
            candidates = list(panos_dir.glob(f"{pano_id}*.jpg"))
            if not candidates:
                print(f"[!] Не найдена панорама: {pano_id}")
                error_count += 1
                continue
            pano_path = candidates[0]
        
        # Проверка skip-existing
        if args.skip_existing:
            first_crop_path = output_dir / f"{pano_id}_yaw0_pitch{int(args.pitch)}.jpg"
            if first_crop_path.exists():
                # Предполагаем что все кропы уже созданы
                # Загружаем их метаданные
                yaw_angles = np.arange(0, 360, args.yaw_step)
                for yaw in yaw_angles:
                    crop_id = f"{pano_id}_yaw{int(yaw)}_pitch{int(args.pitch)}"
                    crop_path = output_dir / f"{crop_id}.jpg"
                    if crop_path.exists():
                        all_crops_meta.append({
                            "crop_id": crop_id,
                            "pano_id": pano_id,
                            "path": str(crop_path),
                            "yaw": yaw,
                            "pitch": args.pitch,
                            "fov": args.fov,
                            "lat": lat,
                            "lon": lon,
                        })
                success_count += 1
                continue
        
        # Генерация кропов
        crops_meta = generate_crops_from_panorama(
            pano_path=str(pano_path),
            pano_id=pano_id,
            lat=lat,
            lon=lon,
            output_dir=output_dir,
            yaw_step=args.yaw_step,
            pitch=args.pitch,
            fov=args.fov,
            output_width=args.output_width,
            output_height=args.output_height,
        )
        
        if crops_meta:
            all_crops_meta.extend(crops_meta)
            success_count += 1
        else:
            error_count += 1
    
    # Сохранение метаданных
    print(f"\n[i] Сохранение метаданных...")
    crops_df = pd.DataFrame(all_crops_meta)
    
    output_meta_path = Path(args.output_meta)
    output_meta_path.parent.mkdir(parents=True, exist_ok=True)
    crops_df.to_csv(output_meta_path, index=False)
    
    # Статистика
    print("\n" + "=" * 60)
    print("✅ СОЗДАНИЕ КРОПОВ ЗАВЕРШЕНО")
    print("=" * 60)
    print(f"Обработано панорам: {success_count}/{len(panos_df)}")
    print(f"Ошибок: {error_count}")
    print(f"Создано кропов: {len(all_crops_meta)}")
    print(f"Метаданные сохранены: {output_meta_path}")
    print(f"Кропы сохранены в: {output_dir}/")
    print("\n📊 Статистика:")
    print(f"   Средних кропов на панораму: {len(all_crops_meta) / max(success_count, 1):.1f}")
    print(f"   Размер одного кропа: ~{args.output_width * args.output_height * 3 / 1024 / 1024:.1f} MB")
    print(f"   Общий размер кропов: ~{len(all_crops_meta) * args.output_width * args.output_height * 3 / 1024 / 1024 / 1024:.1f} GB")
    print("\n🎯 Следующий шаг:")
    print("   python scripts/04_build_index.py --crops-meta meta/crops.csv")


if __name__ == "__main__":
    main()