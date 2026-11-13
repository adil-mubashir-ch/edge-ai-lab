from __future__ import annotations
import os, platform, sys, time

def read_text(path):
    try:
        return open(path, encoding="utf-8", errors="ignore").read().strip()
    except Exception:
        return None

def cpu_temperature_c():
    raw = read_text("/sys/class/thermal/thermal_zone0/temp")
    if raw:
        try:
            return float(raw) / 1000
        except ValueError:
            pass
    try:
        import subprocess
        out = subprocess.check_output(["vcgencmd", "measure_temp"], text=True, timeout=1)
        return float(out.split("=")[1].split("'")[0])
    except Exception:
        return None

def hardware_snapshot():
    data = {
        "timestamp_unix": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
        "cpu_temperature_c": cpu_temperature_c()
    }
    try:
        import psutil
        vm = psutil.virtual_memory()
        data["ram_total_mb"] = vm.total / 1024**2
        data["ram_available_mb"] = vm.available / 1024**2
        data["process_rss_mb"] = psutil.Process().memory_info().rss / 1024**2
    except Exception:
        pass
    model = read_text("/proc/device-tree/model")
    if model:
        data["model"] = model[:500]
    return data
