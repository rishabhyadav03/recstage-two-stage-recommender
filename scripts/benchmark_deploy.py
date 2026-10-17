"""
Benchmark cold start time and peak RSS memory in torch-free deploy venv.
"""
import ctypes
import ctypes.wintypes
import json
import random
import sys
import threading
import time
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import uvicorn

# Set up Windows GetProcessMemoryInfo with proper 64-bit ctypes signatures
class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ('cb', ctypes.wintypes.DWORD),
        ('PageFaultCount', ctypes.wintypes.DWORD),
        ('PeakWorkingSetSize', ctypes.c_size_t),
        ('WorkingSetSize', ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
        ('PagefileUsage', ctypes.c_size_t),
        ('PeakPagefileUsage', ctypes.c_size_t),
    ]

GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
GetProcessMemoryInfo.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), ctypes.wintypes.DWORD]
GetProcessMemoryInfo.restype = ctypes.wintypes.BOOL


def get_current_memory_mb():
    h = ctypes.windll.kernel32.GetCurrentProcess()
    p = PROCESS_MEMORY_COUNTERS()
    p.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    GetProcessMemoryInfo(h, ctypes.byref(p), p.cb)
    return p.WorkingSetSize / (1024 * 1024), p.PeakWorkingSetSize / (1024 * 1024)


def http_get(url: str, timeout: float = 10.0):
    req = urllib.request.Request(url, headers={"User-Agent": "Benchmark"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8")


def main():
    port = 8027
    base_url = f"http://127.0.0.1:{port}"
    print(f"Python interpreter: {sys.executable}")
    print(f"Verifying torch is not installed...")
    try:
        import torch
        print("WARNING: torch is installed in this environment!")
    except ImportError:
        print("CONFIRMED: torch is NOT installed in this environment (torch-free serving).")

    # Measure cold start time
    print(f"\nStarting uvicorn server on {base_url}...")
    t0 = time.perf_counter()

    config = uvicorn.Config(
        "api.main:app",
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run)
    server_thread.daemon = True
    server_thread.start()

    # Poll /api/health until status == 'ok'
    startup_time = None
    health_data = None
    for _ in range(100):
        time.sleep(0.05)
        try:
            status, body = http_get(f"{base_url}/api/health", timeout=1.0)
            if status == 200:
                data = json.loads(body)
                if data.get("status") == "ok":
                    startup_time = time.perf_counter() - t0
                    health_data = data
                    break
        except Exception:
            pass

    if startup_time is None:
        server.should_exit = True
        raise TimeoutError("Server failed to respond to /api/health within 10s")

    print(f"Cold Start Time: {startup_time:.2f}s")
    print(f"Health Response: {json.dumps(health_data, indent=2)}")

    rss_after_startup, peak_after_startup = get_current_memory_mb()
    print(f"\nRSS after startup: {rss_after_startup:.2f} MB")
    print(f"Peak RSS after startup: {peak_after_startup:.2f} MB")

    # Call required endpoints
    required_endpoints = [
        "/api/health",
        "/api/recommend/1",
        "/api/compare/1",
        "/api/map/user/1",
        "/api/metrics",
    ]
    print("\nCalling required endpoints:")
    for ep in required_endpoints:
        t_req = time.perf_counter()
        status, body = http_get(f"{base_url}{ep}")
        latency_ms = (time.perf_counter() - t_req) * 1000
        data = json.loads(body)
        print(f"  {ep} -> {status} OK ({latency_ms:.1f}ms, response keys: {list(data.keys())[:5]})")

    # Send 200 mixed requests
    print("\nSending 200 mixed requests across users...")
    random.seed(42)
    t_bench_start = time.perf_counter()
    req_count = 200
    for i in range(req_count):
        u_id = (i % 500) + 1
        item_id = (i % 200) + 1
        ep_type = i % 5
        if ep_type == 0:
            ep = f"/api/recommend/{u_id}?k=10"
        elif ep_type == 1:
            ep = f"/api/compare/{u_id}?k=10"
        elif ep_type == 2:
            ep = f"/api/map/user/{u_id}"
        elif ep_type == 3:
            ep = f"/api/users/{u_id}"
        else:
            ep = f"/api/similar/{item_id}"

        status, _ = http_get(f"{base_url}{ep}")
        if status != 200:
            print(f"Request {i} failed for {ep} with status {status}")

    total_bench_time = time.perf_counter() - t_bench_start
    qps = req_count / total_bench_time

    rss_after_requests, peak_after_requests = get_current_memory_mb()
    print(f"\n200 mixed requests completed in {total_bench_time:.2f}s ({qps:.1f} req/s)")
    print(f"RSS after 200 requests: {rss_after_requests:.2f} MB")
    print(f"Peak RSS after 200 requests: {peak_after_requests:.2f} MB")

    # Shutdown
    server.should_exit = True
    server_thread.join(timeout=3)
    print("Server cleanly stopped.")

    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY (TORCH-FREE SERVING)")
    print("=" * 50)
    print(f"Cold start time: {startup_time:.2f}s")
    print(f"RSS at startup: {rss_after_startup:.1f} MB")
    print(f"Peak RSS at startup: {peak_after_startup:.1f} MB")
    print(f"RSS after 200 requests: {rss_after_requests:.1f} MB")
    print(f"Peak RSS after 200 requests: {peak_after_requests:.1f} MB")
    print(f"Target threshold: <= 350 MB (Actual Peak: {peak_after_requests:.1f} MB)")
    assert peak_after_requests <= 350, f"Peak RSS {peak_after_requests} exceeds 350 MB"
    print("Peak RSS assertion PASSED!")


if __name__ == "__main__":
    main()
