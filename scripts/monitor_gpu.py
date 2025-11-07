#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monitor_gpu.py — Real-time мониторинг GPU и процесса индексации

Использование:
  # Мониторинг в real-time
  python scripts/monitor_gpu.py

  # С интервалом 10 сек
  python scripts/monitor_gpu.py --interval 10

  # В другом терминале запущена индексация
"""

import os
import sys
import time
import argparse
from datetime import datetime
from typing import Dict, Optional

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False
    print("⚠️  Установите py3nvml для полного мониторинга:")
    print("   pip install py3nvml")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

import torch

# Цвета
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def clear_screen():
    """Очистка экрана"""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_gpu_info_torch() -> Dict:
    """Получить информацию о GPU через PyTorch"""
    if not torch.cuda.is_available():
        return {"available": False}
    
    info = {
        "available": True,
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
    }
    
    # Память
    try:
        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        
        info["mem_allocated_gb"] = mem_allocated
        info["mem_reserved_gb"] = mem_reserved
    except:
        pass
    
    return info


def get_gpu_info_nvml() -> Dict:
    """Получить детальную информацию о GPU через NVML"""
    if not HAS_NVML:
        return {}
    
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # Память
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        info = {
            "vram_total": mem_info.total / 1024**3,
            "vram_used": mem_info.used / 1024**3,
            "vram_free": mem_info.free / 1024**3,
            "vram_percent": (mem_info.used / mem_info.total * 100),
        }
        
        # Утилизация
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            info["gpu_util"] = util.gpu
            info["mem_util"] = util.memory
        except:
            pass
        
        # Температура
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            info["temperature"] = temp
        except:
            pass
        
        # Power
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Watts
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            info["power_usage"] = power
            info["power_limit"] = power_limit
            info["power_percent"] = (power / power_limit * 100) if power_limit > 0 else 0
        except:
            pass
        
        # Clock speeds
        try:
            sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            info["sm_clock"] = sm_clock
            info["mem_clock"] = mem_clock
        except:
            pass
        
        pynvml.nvmlShutdown()
        return info
    
    except Exception as e:
        return {"error": str(e)}


def get_system_info() -> Dict:
    """Получить информацию о системе"""
    if not HAS_PSUTIL:
        return {}
    
    info = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_percent": psutil.virtual_memory().percent,
        "ram_used_gb": psutil.virtual_memory().used / 1024**3,
        "ram_total_gb": psutil.virtual_memory().total / 1024**3,
    }
    
    # Диск
    try:
        disk = psutil.disk_usage('/')
        info["disk_percent"] = disk.percent
        info["disk_used_gb"] = disk.used / 1024**3
        info["disk_total_gb"] = disk.total / 1024**3
    except:
        pass
    
    return info


def format_bar(percent: float, width: int = 30, full_char: str = "█", empty_char: str = "░") -> str:
    """Создать прогресс-бар"""
    filled = int(width * percent / 100)
    bar = full_char * filled + empty_char * (width - filled)
    return bar


def get_color_by_percent(percent: float) -> str:
    """Выбрать цвет по проценту"""
    if percent < 50:
        return Colors.OKGREEN
    elif percent < 80:
        return Colors.WARNING
    else:
        return Colors.FAIL


def print_gpu_monitor(torch_info: Dict, nvml_info: Dict, sys_info: Dict):
    """Напечатать мониторинг"""
    clear_screen()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(Colors.BOLD + "=" * 80 + Colors.ENDC)
    print(Colors.BOLD + "🎮  GPU MONITOR" + Colors.ENDC + f" | {timestamp}")
    print(Colors.BOLD + "=" * 80 + Colors.ENDC)
    print()
    
    # GPU общая информация
    if torch_info.get("available"):
        print(Colors.HEADER + f"GPU: {torch_info['device_name']}" + Colors.ENDC)
        print()
    else:
        print(Colors.FAIL + "❌ GPU не доступен!" + Colors.ENDC)
        print()
        return
    
    # VRAM
    if "vram_total" in nvml_info:
        vram_used = nvml_info["vram_used"]
        vram_total = nvml_info["vram_total"]
        vram_percent = nvml_info["vram_percent"]
        
        color = get_color_by_percent(vram_percent)
        bar = format_bar(vram_percent)
        
        print(f"💾 VRAM:  {color}{bar}{Colors.ENDC} {vram_percent:>5.1f}%")
        print(f"         {vram_used:>6.2f} / {vram_total:.2f} GB")
        print()
    
    # GPU Utilization
    if "gpu_util" in nvml_info:
        gpu_util = nvml_info["gpu_util"]
        
        color = get_color_by_percent(gpu_util)
        bar = format_bar(gpu_util)
        
        print(f"⚡ GPU:   {color}{bar}{Colors.ENDC} {gpu_util:>5.1f}%")
        print()
    
    # Temperature
    if "temperature" in nvml_info:
        temp = nvml_info["temperature"]
        
        if temp < 70:
            color = Colors.OKGREEN
        elif temp < 80:
            color = Colors.WARNING
        else:
            color = Colors.FAIL
        
        bar = format_bar(min(temp / 90 * 100, 100))  # Max 90°C
        
        print(f"🌡️  Temp:  {color}{bar}{Colors.ENDC} {temp:>5.0f}°C")
        print()
    
    # Power
    if "power_usage" in nvml_info:
        power = nvml_info["power_usage"]
        power_limit = nvml_info["power_limit"]
        power_percent = nvml_info["power_percent"]
        
        color = get_color_by_percent(power_percent)
        bar = format_bar(power_percent)
        
        print(f"⚡ Power: {color}{bar}{Colors.ENDC} {power_percent:>5.1f}%")
        print(f"         {power:>6.1f} / {power_limit:.1f} W")
        print()
    
    # Clock speeds
    if "sm_clock" in nvml_info:
        sm_clock = nvml_info["sm_clock"]
        mem_clock = nvml_info["mem_clock"]
        
        print(f"🔧 Clocks: SM {sm_clock} MHz | Memory {mem_clock} MHz")
        print()
    
    # Система
    if sys_info:
        print(Colors.HEADER + "─" * 80 + Colors.ENDC)
        print(Colors.HEADER + "💻 СИСТЕМА" + Colors.ENDC)
        print()
        
        # CPU
        if "cpu_percent" in sys_info:
            cpu_percent = sys_info["cpu_percent"]
            color = get_color_by_percent(cpu_percent)
            bar = format_bar(cpu_percent)
            
            print(f"🧠 CPU:   {color}{bar}{Colors.ENDC} {cpu_percent:>5.1f}%")
            print()
        
        # RAM
        if "ram_percent" in sys_info:
            ram_percent = sys_info["ram_percent"]
            ram_used = sys_info["ram_used_gb"]
            ram_total = sys_info["ram_total_gb"]
            
            color = get_color_by_percent(ram_percent)
            bar = format_bar(ram_percent)
            
            print(f"💾 RAM:   {color}{bar}{Colors.ENDC} {ram_percent:>5.1f}%")
            print(f"         {ram_used:>6.2f} / {ram_total:.2f} GB")
            print()
        
        # Disk
        if "disk_percent" in sys_info:
            disk_percent = sys_info["disk_percent"]
            disk_used = sys_info["disk_used_gb"]
            disk_total = sys_info["disk_total_gb"]
            
            color = get_color_by_percent(disk_percent)
            bar = format_bar(disk_percent)
            
            print(f"💽 Disk:  {color}{bar}{Colors.ENDC} {disk_percent:>5.1f}%")
            print(f"         {disk_used:>6.1f} / {disk_total:.1f} GB")
            print()
    
    print(Colors.BOLD + "=" * 80 + Colors.ENDC)
    print(f"{Colors.OKCYAN}Нажмите Ctrl+C для выхода{Colors.ENDC}")


def main():
    ap = argparse.ArgumentParser(
        description="Real-time GPU мониторинг",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    ap.add_argument("--interval", type=float, default=2.0,
                    help="Интервал обновления (секунды)")
    
    args = ap.parse_args()
    
    print("🚀 Запуск GPU мониторинга...")
    time.sleep(1)
    
    try:
        while True:
            torch_info = get_gpu_info_torch()
            nvml_info = get_gpu_info_nvml()
            sys_info = get_system_info()
            
            print_gpu_monitor(torch_info, nvml_info, sys_info)
            
            time.sleep(args.interval)
    
    except KeyboardInterrupt:
        print("\n\n✅ Мониторинг завершён")
        sys.exit(0)


if __name__ == "__main__":
    main()