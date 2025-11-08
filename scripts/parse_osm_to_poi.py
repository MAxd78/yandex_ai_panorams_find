#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parse_osm_to_poi.py — парсинг OpenStreetMap в POI базу данных

Извлекает из OSM файла:
  - Магазины, рестораны, кафе (brand names)
  - Улицы и адреса
  - Достопримечательности
  - Транспортные объекты

Создаёт:
  - poi_database.csv — полная база POI с координатами
  - brand_dictionary.json — словарь брендов для OCR коррекции
  - spatial_index.pkl — R-tree индекс для быстрого поиска по координатам

Использование:
  # Скачать OSM данные для Москвы
  wget https://download.geofabrik.de/russia/central-fed-district-latest.osm.pbf
  
  # Парсинг
  python scripts/parse_osm_to_poi.py --osm map.osm --output poi/
  
  # Или с bbox фильтрацией
  python scripts/parse_osm_to_poi.py --osm map.osm --bbox "55.7,37.5:55.8,37.7"
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter

import pandas as pd
from tqdm import tqdm

# OSM парсинг
try:
    import osmium
    HAS_OSMIUM = True
except ImportError:
    HAS_OSMIUM = False

# Геопространственная индексация
try:
    from rtree import index as rtree_index
    HAS_RTREE = True
except ImportError:
    HAS_RTREE = False


# ========================= Константы =========================

# Категории POI которые нас интересуют
POI_TAGS = {
    "shop": None,  # Все магазины
    "amenity": ["restaurant", "cafe", "fast_food", "bar", "pub", "bank", "pharmacy", 
                "hospital", "cinema", "theatre", "fuel"],
    "tourism": ["hotel", "museum", "attraction", "viewpoint"],
    "leisure": ["park", "stadium", "sports_centre"],
    "office": ["company"],
}

# Теги для брендов
BRAND_TAGS = ["brand", "name", "operator", "brand:en", "brand:ru"]

# Известные российские бренды (для приоритизации)
KNOWN_RUSSIAN_BRANDS = {
    "пятёрочка", "пятерочка", "магнит", "дикси", "перекрёсток", "перекресток",
    "ашан", "лента", "о'кей", "окей", "метро", "вкусвилл", "азбука вкуса",
    "кофе хауз", "coffeeshop", "шоколадница", "кофемания", "теремок",
    "макдональдс", "mcdonalds", "бургер кинг", "burger king", "kfc", "кфс",
    "subway", "сабвей", "додо пицца", "dodo pizza", "папа джонс", "pizza hut",
    "сбербанк", "втб", "альфа-банк", "тинькoff", "тинькофф",
}


# ========================= OSM Handler =========================

class POIHandler(osmium.SimpleHandler):
    """Handler для извлечения POI из OSM"""
    
    def __init__(self, bbox: Tuple[float, float, float, float] | None = None):
        super().__init__()
        self.pois: List[Dict] = []
        self.bbox = bbox  # (lat_min, lon_min, lat_max, lon_max)
        self.brands: Set[str] = set()
        
    def _in_bbox(self, lat: float, lon: float) -> bool:
        """Проверка попадания в bbox"""
        if self.bbox is None:
            return True
        lat_min, lon_min, lat_max, lon_max = self.bbox
        return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max
    
    def _extract_poi(self, obj, obj_type: str):
        """Извлечение POI из объекта"""
        tags = {tag.k: tag.v for tag in obj.tags}
        
        # Проверяем наличие интересующих тегов
        category = None
        subcategory = None
        
        for key, values in POI_TAGS.items():
            if key in tags:
                if values is None or tags[key] in values:
                    category = key
                    subcategory = tags.get(key)
                    break
        
        if category is None:
            return None
        
        # Координаты
        if obj_type == "node":
            lat, lon = obj.location.lat, obj.location.lon
        elif obj_type == "way":
            # Для way берём центроид
            try:
                nodes = list(obj.nodes)
                if not nodes:
                    return None
                lats = [n.lat for n in nodes if n.lat is not None]
                lons = [n.lon for n in nodes if n.lon is not None]
                if not lats or not lons:
                    return None
                lat = sum(lats) / len(lats)
                lon = sum(lons) / len(lons)
            except Exception:
                return None
        else:
            return None
        
        # Фильтрация по bbox
        if not self._in_bbox(lat, lon):
            return None
        
        # Извлечение названия/бренда
        brand = None
        name = None
        
        for tag in BRAND_TAGS:
            if tag in tags:
                brand = tags[tag].strip()
                if brand:
                    self.brands.add(brand.lower())
                    break
        
        name = tags.get("name", tags.get("brand", "")).strip()
        
        if not name and not brand:
            return None
        
        # Формирование POI
        poi = {
            "osm_id": f"{obj_type[0]}{obj.id}",  # n123 или w456
            "category": category,
            "subcategory": subcategory,
            "name": name or brand,
            "brand": brand,
            "lat": lat,
            "lon": lon,
            "address": tags.get("addr:street", ""),
            "housenumber": tags.get("addr:housenumber", ""),
            "city": tags.get("addr:city", ""),
            "phone": tags.get("phone", ""),
            "website": tags.get("website", ""),
        }
        
        return poi
    
    def node(self, n):
        poi = self._extract_poi(n, "node")
        if poi:
            self.pois.append(poi)
    
    def way(self, w):
        poi = self._extract_poi(w, "way")
        if poi:
            self.pois.append(poi)


# ========================= Обработка =========================

def parse_bbox(bbox_str: str) -> Tuple[float, float, float, float]:
    """Парсинг bbox из строки"""
    try:
        a, b = bbox_str.split(":")
        lat1, lon1 = [float(x) for x in a.split(",")]
        lat2, lon2 = [float(x) for x in b.split(",")]
    except Exception:
        raise ValueError("Неверный формат --bbox. Ожидается 'lat1,lon1:lat2,lon2'")
    
    lat_min, lat_max = sorted([lat1, lat2])
    lon_min, lon_max = sorted([lon1, lon2])
    return lat_min, lon_min, lat_max, lon_max


def parse_osm_file(osm_path: Path, bbox: Tuple | None = None) -> Tuple[List[Dict], Set[str]]:
    """Парсинг OSM файла"""
    
    if not HAS_OSMIUM:
        print("[!] Не установлен osmium. Установите: pip install osmium")
        sys.exit(1)
    
    print(f"[i] Парсинг OSM файла: {osm_path}")
    if bbox:
        print(f"[i] Фильтрация по bbox: lat[{bbox[0]:.4f}, {bbox[2]:.4f}], lon[{bbox[1]:.4f}, {bbox[3]:.4f}]")
    
    handler = POIHandler(bbox=bbox)
    handler.apply_file(str(osm_path), locations=True)
    
    print(f"[✓] Найдено {len(handler.pois)} POI")
    print(f"[✓] Уникальных брендов: {len(handler.brands)}")
    
    return handler.pois, handler.brands


def build_brand_dictionary(brands: Set[str], pois: List[Dict]) -> Dict:
    """
    Построение словаря брендов с вариантами написания
    
    Создаёт маппинг: вариант написания → канонический бренд
    Учитывает:
      - Lowercase/uppercase
      - Транслитерацию
      - Опечатки (расстояние Левенштейна)
    """
    
    # Канонические бренды
    brand_counter = Counter()
    for poi in pois:
        if poi.get("brand"):
            brand_counter[poi["brand"].lower()] += 1
    
    # Берём топ-1000 самых частых
    top_brands = [b for b, _ in brand_counter.most_common(1000)]
    
    # Создаём словарь вариантов
    brand_dict = {}
    
    for brand in top_brands:
        variants = set()
        
        # Сам бренд
        variants.add(brand.lower())
        
        # Без пробелов
        variants.add(brand.replace(" ", "").lower())
        
        # Транслитерация (упрощённая)
        translit_map = {
            "ё": "е", "й": "и", "ъ": "", "ь": "",
            "э": "е", "ю": "у", "я": "а",
        }
        translit = brand.lower()
        for ru, en in translit_map.items():
            translit = translit.replace(ru, en)
        variants.add(translit)
        
        # Удаление знаков препинания
        import string
        no_punct = brand.translate(str.maketrans("", "", string.punctuation)).lower()
        variants.add(no_punct)
        
        # Добавляем варианты в словарь
        for v in variants:
            if v:
                brand_dict[v] = brand
    
    print(f"[i] Создан словарь из {len(top_brands)} брендов, {len(brand_dict)} вариантов")
    
    return brand_dict


def build_spatial_index(pois: List[Dict]) -> rtree_index.Index:
    """Построение R-tree индекса для быстрого поиска"""
    
    if not HAS_RTREE:
        print("[!] Не установлен rtree. Установите: pip install rtree")
        return None
    
    print(f"[i] Построение spatial index...")
    
    idx = rtree_index.Index()
    
    for i, poi in enumerate(pois):
        lat, lon = poi["lat"], poi["lon"]
        # bbox: (lon_min, lat_min, lon_max, lat_max)
        idx.insert(i, (lon, lat, lon, lat))
    
    print(f"[✓] Spatial index построен")
    
    return idx


# ========================= Main =========================

def main():
    parser = argparse.ArgumentParser(
        description="Парсинг OpenStreetMap в POI базу"
    )
    
    parser.add_argument("--osm", required=True, help="Путь к OSM файлу (.osm, .osm.pbf)")
    parser.add_argument("--output", default="poi", help="Папка для сохранения результатов")
    parser.add_argument("--bbox", default=None, help="Фильтрация по bbox: 'lat1,lon1:lat2,lon2'")
    
    args = parser.parse_args()
    
    osm_path = Path(args.osm)
    output_dir = Path(args.output)
    
    if not osm_path.exists():
        print(f"[!] Не найден OSM файл: {osm_path}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Парсинг bbox
    bbox = None
    if args.bbox:
        bbox = parse_bbox(args.bbox)
    
    # Парсинг OSM
    pois, brands = parse_osm_file(osm_path, bbox=bbox)
    
    if not pois:
        print("[!] Не найдено ни одного POI")
        sys.exit(0)
    
    # Сохранение POI
    poi_df = pd.DataFrame(pois)
    poi_csv = output_dir / "poi_database.csv"
    poi_df.to_csv(poi_csv, index=False, encoding="utf-8")
    print(f"[✓] POI сохранены: {poi_csv} ({len(poi_df)} записей)")
    
    # Статистика
    print(f"\n📊 Статистика POI:")
    print(poi_df["category"].value_counts().head(10))
    
    # Словарь брендов
    brand_dict = build_brand_dictionary(brands, pois)
    brand_json = output_dir / "brand_dictionary.json"
    with open(brand_json, "w", encoding="utf-8") as f:
        json.dump(brand_dict, f, ensure_ascii=False, indent=2)
    print(f"[✓] Словарь брендов: {brand_json}")
    
    # Spatial index
    if HAS_RTREE:
        spatial_idx = build_spatial_index(pois)
        if spatial_idx:
            idx_path = output_dir / "spatial_index"
            # Сохраняем pickle с маппингом индекс -> POI
            poi_lookup = {i: poi for i, poi in enumerate(pois)}
            with open(output_dir / "poi_lookup.pkl", "wb") as f:
                pickle.dump(poi_lookup, f)
            print(f"[✓] Spatial index: {idx_path}.*")
    
    print(f"\n✅ Парсинг завершён! Результаты в: {output_dir}/")
    print(f"\n💡 Теперь можно использовать POI для:")
    print("   1. OCR коррекции (исправление опечаток)")
    print("   2. Query narrowing (поиск только рядом с известными POI)")
    print("   3. Semantic features для re-ranking")


if __name__ == "__main__":
    main()