#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_download_panoramas.py — тонкая обёртка над zer0-dev/yandex-pano-downloader.
Позволяет скачивать 360° панорамы Яндекса по одиночной координате или из файла.
Делает параллельные загрузки, ретраи, пропускает уже скачанные файлы.

Примеры:
  # одна точка
  python scripts/01_download_panoramas.py --coords "55.751244,37.618423" --zoom 0

  # много точек из CSV/TSV/TXT (формат: lat,lon в каждой строке)
  python scripts/01_download_panoramas.py --coords-file meta/coords_myrayon.csv --zoom 0 --workers 2

Зависимости:
  - внешний инструмент в external/yandex-pano-downloader (см. README)
"""

from __future__ import annotations
import argparse
import csv
import os
import sys
import time
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Yandex panoramas via zer0-dev/yandex-pano-downloader")
    p.add_argument("--coords", nargs="*", default=[],
                   help='Координаты в формате "lat,lon". Можно указать несколько.')
    p.add_argument("--coords-file", type=str, default=None,
                   help="Путь к файлу со списком координат (lat,lon в строке). CSV/TSV/TXT.")
    p.add_argument("--zoom", type=int, default=0,
                   help="Зум: 0 — максимальный размер панорамы (дефолт). Больше — меньше размер. ")
    p.add_argument("--adjust-aspect", action="store_true",
                   help="Добавить флаг -a (привести к 2:1, у авто-панорам снизу появится чёрный круг).")
    p.add_argument("--outdir", type=str, default="data/panos_raw",
                   help="Куда сохранять .jpg (дефолт: data/panos_raw)")
    p.add_argument("--workers", type=int, default=2,
                   help="Сколько параллельных загрузок (дефолт: 2).")
    p.add_argument("--delay", type=float, default=0.75,
                   help="Пауза между задачами (сек, дефолт: 0.75), чтобы не ловить лимиты.")
    p.add_argument("--retries", type=int, default=2,
                   help="Сколько ретраев на одну панораму (дефолт: 2).")
    p.add_argument("--pano-script", type=str,
                   default="external/yandex-pano-downloader/pano.py",
                   help="Путь до pano.py из внешнего репо (дефолт: external/yandex-pano-downloader/pano.py)")
    return p.parse_args()

def load_coords_from_file(path: str) -> List[Tuple[float, float]]:
    coords: List[Tuple[float, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        # пытаемся распарсить как CSV/TSV/простой текст
        try:
            sniffer = csv.Sniffer()
            sample = f.read(4096)
            f.seek(0)
            dialect = sniffer.sniff(sample)
            reader = csv.reader(f, dialect)
        except csv.Error:
            f.seek(0)
            reader = (line.strip().split(",") for line in f)

        for row in reader:
            if not row:
                continue
            if len(row) == 1:
                row = row[0].split(",")
            if len(row) < 2:
                continue
            lat_str, lon_str = row[0].strip(), row[1].strip()
            try:
                lat, lon = float(lat_str), float(lon_str)
            except ValueError:
                continue
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                continue
            coords.append((lat, lon))
    return coords

def parse_coords_args(values: List[str]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for v in values:
        parts = [s.strip() for s in v.split(",")]
        if len(parts) != 2:
            continue
        try:
            lat, lon = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
            out.append((lat, lon))
    return out

def sanitize_filename(lat: float, lon: float, zoom: int) -> str:
    lat_s = f"{lat:+.6f}".replace("+","").replace("-", "m").replace(".", "_")
    lon_s = f"{lon:+.6f}".replace("+","").replace("-", "m").replace(".", "_")
    return f"pano_lat{lat_s}_lon{lon_s}_z{zoom}.jpg"

def run_download(pano_script: str, lat: float, lon: float, zoom: int, out_path: str,
                 adjust: bool, retries: int) -> bool:
    cmd = ["python3", pano_script, "-c", f"{lat},{lon}", "-z", str(zoom), "-o", out_path]
    if adjust:
        cmd.append("-a")
    attempt = 0
    while True:
        attempt += 1
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  check=True, text=True)
            if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
                return True
            else:
                raise RuntimeError("No output file produced.")
        except subprocess.CalledProcessError as e:
            # <<< ключевая строка — отдаём stderr загрузчика на экран
            print(f"[pano.py stderr]\n{e.stderr.strip()}\n", file=sys.stderr)
            if attempt > retries + 1:
                print(f"[FAIL] lat={lat:.6f}, lon={lon:.6f}, z={zoom}: {e}")
                return False
            wait = min(2.0 * attempt, 6.0)
            print(f"[RETRY {attempt}/{retries}] ({lat:.6f},{lon:.6f}) ошибка. Ждём {wait:.1f}s")
            time.sleep(wait)
        except Exception as e:
            if attempt > retries + 1:
                print(f"[FAIL] lat={lat:.6f}, lon={lon:.6f}, z={zoom}: {e}")
                return False
            wait = min(2.0 * attempt, 6.0)
            print(f"[RETRY {attempt}/{retries}] ({lat:.6f},{lon:.6f}) ошибка: {e}. Ждём {wait:.1f}s")
            time.sleep(wait)


def main():
    args = parse_args()

    if not os.path.isfile(args.pano_script):
        print(f"❌ Не найден pano.py по пути: {args.pano_script}")
        print("Склонируй репозиторий в external/yandex-pano-downloader и установи зависимости, см. инструкцию в сообщении.")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    coords: List[Tuple[float, float]] = []
    coords += parse_coords_args(args.coords)
    if args.coords_file:
        coords += load_coords_from_file(args.coords_file)

    # Убираем дубликаты (округлим до 6 знаков)
    seen = set()
    uniq_coords: List[Tuple[float, float]] = []
    for lat, lon in coords:
        key = (round(lat, 6), round(lon, 6))
        if key in seen:
            continue
        seen.add(key)
        uniq_coords.append(key)

    if not uniq_coords:
        print("❌ Нет валидных координат. Укажи --coords \"lat,lon\" или --coords-file path.")
        sys.exit(2)

    print(f"Нужно скачать: {len(uniq_coords)} панорам; zoom={args.zoom}; outdir={args.outdir}; workers={args.workers}")

    def task(coord: Tuple[float, float]) -> Tuple[Tuple[float, float], bool, str]:
        lat, lon = coord
        filename = sanitize_filename(lat, lon, args.zoom)
        out_path = os.path.join(args.outdir, filename)
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            return coord, True, "skip"
        ok = run_download(args.pano_script, lat, lon, args.zoom, out_path, args.adjust_aspect, args.retries)
        # небольшая пауза между задачами, чтобы не долбить сервис
        time.sleep(args.delay)
        return coord, ok, out_path

    done = ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(task, c) for c in uniq_coords]
        for fut in as_completed(futs):
            coord, success, info = fut.result()
            done += 1
            ok += int(success)
            lat, lon = coord
            status = "OK" if success else "ERR"
            print(f"[{status}] {done}/{len(uniq_coords)} — ({lat:.6f},{lon:.6f}) → {info}")

    print(f"\nГотово: {ok}/{len(uniq_coords)} успешно.")

if __name__ == "__main__":
    main()
