import time
import json
import requests
import threading
import numpy as np
import argparse

# Configuration
URL = "http://localhost:8080/generate"
PAYLOAD = {
    "input_ids": [1, 2, 3, 4, 5],
    "max_tokens": 20
}

latencies = []
errors = 0
lock = threading.Lock()

def worker(num_requests):
    global errors
    for _ in range(num_requests):
        start = time.time()
        try:
            resp = requests.post(URL, json=PAYLOAD, timeout=10)
            if resp.status_code == 200:
                dur = (time.time() - start) * 1000 # ms
                with lock:
                    latencies.append(dur)
            else:
                with lock:
                    errors += 1
        except Exception:
            with lock:
                errors += 1

def run_load_test(concurrency, total_requests):
    print(f"Starting Load Test: {total_requests} requests with {concurrency} threads...")
    threads = []
    
    reqs_per_thread = total_requests // concurrency
    
    start_time = time.time()
    for _ in range(concurrency):
        t = threading.Thread(target=worker, args=(reqs_per_thread,))
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
        
    total_dur = time.time() - start_time
    
    if latencies:
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        avg = np.mean(latencies)
        
        print("\n" + "="*40)
        print("  Load Test Results")
        print("="*40)
        print(f"  Total Duration:   {total_dur:.2f} s")
        print(f"  Throughput:       {len(latencies) / total_dur:.2f} req/sec")
        print(f"  Error Rate:       {errors / total_requests * 100:.2f}%")
        print(f"  Latency P50:      {p50:.2f} ms")
        print(f"  Latency P95:      {p95:.2f} ms")
        print(f"  Latency P99:      {p99:.2f} ms")
        print("="*40)
    else:
        print("No successful requests.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests", type=int, default=100)
    args = parser.parse_args()
    
    # Check if server matches connectivity
    try:
        requests.get("http://localhost:8080", timeout=1)
    except:
        print("Warning: Could not connect to localhost:8080. Make sure 'python serving/server.py' is running.")
    
    run_load_test(args.concurrency, args.requests)
