#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_scan_bbox_yandex.py — поисковик панорам Яндекс в пределах bbox.
Ищет ближайшую панораму к каждой точке сетки (по шагу в метрах), дедуплит по pano_id,
сохраняет meta/panos_bbox.csv и по желанию скачивает через external/yandex-pano-downloader/pano.py.

Примеры:
  # просто найти и сохранить список (шаг 1 м — осторожно!)
  python scripts/02_scan_bbox_yandex.py --bbox "55.7385,37.4800:55.7427,37.4870" --step-m 1

  # найти и сразу скачать (максимальный zoom, приводить к 2:1), 4 параллелизма
  python scripts/02_scan_bbox_yandex.py --bbox "55.7385,37.4800:55.7427,37.4870" \
    --step-m 2 --download --zoom 0 --adjust-aspect --workers 4
"""

from __future__ import annotations
import argparse, math, os, sys, asyncio, csv
from typing import Tuple, Set, Dict
from tqdm import tqdm

# streetlevel
from streetlevel import yandex
from aiohttp import ClientSession

def parse_bbox(bbox_str: str) -> Tuple[float,float,float,float]:
    try:
        a,b = bbox_str.split(":")
        lat1,lon1 = [float(x) for x in a.split(",")]
        lat2,lon2 = [float(x) for x in b.split(",")]
    except Exception:
        raise SystemExit("Неверный формат --bbox. Ожидается \"lat1,lon1:lat2,lon2\"")
    lat_min, lat_max = sorted([lat1, lat2])
    lon_min, lon_max = sorted([lon1, lon2])
    return lat_min, lon_min, lat_max, lon_max

def deg_per_meter(lat_deg: float) -> Tuple[float,float]:
    # 1 градус широты ~ 111_320 м; долготы — 111_320 * cos(phi)
    lat_rad = math.radians(lat_deg)
    dlat = 1.0 / 111320.0
    dlon = 1.0 / (111320.0 * max(0.1, math.cos(lat_rad)))  # защита от деления на 0
    return dlat, dlon

async def find_pano(session: ClientSession, lat: float, lon: float):
    # Возвращает YandexPanorama или None
    try:
        return await yandex.find_panorama_async(lat, lon, session)
    except Exception:
        return None

async def scan_grid(lat_min: float, lon_min: float, lat_max: float, lon_max: float,
                    step_m: float, concurrency: int, progress: bool=True):
    lat_center = (lat_min + lat_max) / 2.0
    dlat_deg, dlon_deg = deg_per_meter(lat_center)
    dlat = step_m * dlat_deg
    dlon = step_m * dlon_deg

    sem = asyncio.Semaphore(concurrency)
    found: Dict[str, Tuple[float,float,str]] = {}  # pano_id -> (lat, lon, date_str)

    async with ClientSession() as session:
        tasks = []
        lat = lat_min
        total = 0
        # Предварительно прикинем количество точек (для прогресса)
        est_rows = max(1, int((lat_max - lat_min) / max(1e-8, dlat)))
        est_cols = max(1, int((lon_max - lon_min) / max(1e-8, dlon)))
        est_total = est_rows * est_cols

        pbar = tqdm(total=est_total, disable=not progress, desc="scan", unit="pt")

        while lat <= lat_max + 1e-12:
            lon = lon_min
            while lon <= lon_max + 1e-12:
                total += 1
                async def worker(la=lat, lo=lon):
                    async with sem:
                        pano = await find_pano(session, la, lo)
                        if pano is not None:
                            # Дедуп по pano.id
                            if pano.id not in found:
                                found[pano.id] = (pano.lat, pano.lon, str(pano.date) if hasattr(pano, "date") else "")
                    pbar.update(1)
                tasks.append(asyncio.create_task(worker()))
                lon += dlon
            lat += dlat
        await asyncio.gather(*tasks)
        pbar.close()

    return found

def save_csv(path: str, found: Dict[str, Tuple[float,float,str]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pano_id","lat","lon","date"])
        for pid,(la,lo,dt) in found.items():
            w.writerow([pid, f"{la:.8f}", f"{lo:.8f}", dt])

def build_outname(pid: str, zoom: int) -> str:
    return f"{pid}_z{zoom}.jpg"

def run_pano_download(pano_script: str, lat: float, lon: float, out_path: str,
                      zoom: int, adjust: bool) -> bool:
    import subprocess, time
    cmd = ["python3", pano_script, "-c", f"{lat},{lon}", "-z", str(zoom), "-o", out_path]
    if adjust:
        cmd.append("-a")
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              check=True, text=True)
        if not (os.path.isfile(out_path) and os.path.getsize(out_path) > 0):
            raise RuntimeError("No output file produced.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[pano.py stderr]\n{e.stderr.strip()}\n", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[download error] {e}", file=sys.stderr)
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", required=True, help='Формат: "lat1,lon1:lat2,lon2"')
    ap.add_argument("--step-m", type=float, default=1.0, help="Шаг сетки в метрах (дефолт 1 м)")
    ap.add_argument("--concurrency", type=int, default=32, help="Параллельные запросы поиска")
    ap.add_argument("--outlist", default="meta/panos_bbox.csv", help="Куда сохранить список панорам")
    ap.add_argument("--download", action="store_true", help="Сразу скачать найденные панорамы")
    ap.add_argument("--outdir", default="data/panos_raw", help="Папка для изображений")
    ap.add_argument("--zoom", type=int, default=0, help="Зум для скачивания (0 — максимум)")
    ap.add_argument("--adjust-aspect", action="store_true", help="Флаг -a (2:1)")
    ap.add_argument("--pano-script", default="external/yandex-pano-downloader/pano.py", help="Путь к pano.py")
    args = ap.parse_args()

    lat_min, lon_min, lat_max, lon_max = parse_bbox(args.bbox)
    print(f"[i] bbox: lat[{lat_min}, {lat_max}] lon[{lon_min}, {lon_max}], step={args.step_m} м, conc={args.concurrency}")

    found = asyncio.run(scan_grid(lat_min, lon_min, lat_max, lon_max, args.step_m, args.concurrency))
    print(f"[i] найдено уникальных панорам: {len(found)}")
    save_csv(args.outlist, found)
    print(f"[i] сохранено: {args.outlist}")

    if args.download and len(found) > 0:
        os.makedirs(args.outdir, exist_ok=True)
        ok = 0
        for pid,(la,lo,dt) in tqdm(found.items(), desc="download"):
            out_path = os.path.join(args.outdir, build_outname(pid, args.zoom))
            # пропускаем уже скачанное
            if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
                ok += 1
                continue
            if run_pano_download(args.pano_script, la, lo, out_path, args.zoom, args.adjust_aspect):
                ok += 1
        print(f"[i] скачано: {ok}/{len(found)} → {args.outdir}")

if __name__ == "__main__":
    main()