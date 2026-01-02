"""
Benchmark script for Stable Softmax kernels.
Tests baseline, NumPy, and Numba implementations.
"""

import sys
import os
import time
from pathlib import Path

# Set thread limits BEFORE importing NumPy to prevent BLAS thread contention
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

# Add project root to path (two levels up from this file)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.kernels.softmax_baseline import softmax_baseline
from src.kernels.softmax_numpy import softmax_numpy
from src.kernels.softmax_numba import softmax_numba


def benchmark_softmax(configs, num_warmup=3, num_runs=10):
    """
    Benchmark softmax kernels.
    
    Args:
        configs: List of tuples (batch_size, seq_len) representing input dimensions
        num_warmup: Number of warmup runs for JIT compilation
        num_runs: Number of timed runs
        
    Returns:
        DataFrame with benchmark results
    """
    results = []
    
    for batch_size, seq_len in configs:
        print(f"\nBenchmarking Softmax: batch_size={batch_size}, seq_len={seq_len}")
        
        # Generate test data - ensure contiguous float32 arrays
        np.random.seed(42)
        x = np.ascontiguousarray(np.random.randn(batch_size, seq_len).astype(np.float32), dtype=np.float32)
        
        # Reference for correctness check
        x_ref = x.copy()
        exp_x = np.exp(x_ref - np.max(x_ref, axis=1, keepdims=True))
        output_ref = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        # Compute theoretical operations
        # Each element: max (1 op), exp (1 op), sum (1 op), div (1 op)
        # Approximate: 4 ops per element
        ops = 4 * batch_size * seq_len
        bytes_moved = (batch_size * seq_len) * 4 * 2  # input + output, float32
        
        # 1. Baseline (pure Python)
        print("  Testing baseline (pure Python)...")
        try:
            # Only test small inputs for baseline (it's very slow)
            if batch_size * seq_len < 1e5:
                times = []
                for _ in range(num_warmup):
                    _ = softmax_baseline(x)
                
                for _ in range(num_runs):
                    t_start = time.perf_counter()
                    output = softmax_baseline(x)
                    t_end = time.perf_counter()
                    times.append(t_end - t_start)
                
                times = np.array(times)
                latency_ms = np.mean(times) * 1000
                latency_p50 = np.percentile(times, 50) * 1000
                latency_p95 = np.percentile(times, 95) * 1000
                latency_p99 = np.percentile(times, 99) * 1000
                
                throughput_mops = (ops / 1e6) / np.mean(times)
                
                results.append({
                    'kernel': 'baseline',
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'ops': ops,
                    'bytes_moved': bytes_moved,
                    'latency_ms': latency_ms,
                    'latency_p50_ms': latency_p50,
                    'latency_p95_ms': latency_p95,
                    'latency_p99_ms': latency_p99,
                    'throughput_mops': throughput_mops,
                })
            else:
                print(f"    Skipping baseline (problem too large)")
        except Exception as e:
            print(f"    Error: {e}")
        
        # 2. NumPy
        print("  Testing NumPy (vectorized)...")
        try:
            times = []
            for _ in range(num_warmup):
                _ = softmax_numpy(x)
            
            for _ in range(num_runs):
                t_start = time.perf_counter()
                output = softmax_numpy(x)
                t_end = time.perf_counter()
                times.append(t_end - t_start)
            
            times = np.array(times)
            latency_ms = np.mean(times) * 1000
            latency_p50 = np.percentile(times, 50) * 1000
            latency_p95 = np.percentile(times, 95) * 1000
            latency_p99 = np.percentile(times, 99) * 1000
            
            throughput_mops = (ops / 1e6) / np.mean(times)
            
            # Verify correctness
            assert np.allclose(output, output_ref, rtol=1e-4, atol=1e-5), "NumPy correctness check failed"
            
            results.append({
                'kernel': 'numpy',
                'batch_size': batch_size,
                'seq_len': seq_len,
                'ops': ops,
                'bytes_moved': bytes_moved,
                'latency_ms': latency_ms,
                'latency_p50_ms': latency_p50,
                'latency_p95_ms': latency_p95,
                'latency_p99_ms': latency_p99,
                'throughput_mops': throughput_mops,
            })
        except Exception as e:
            print(f"    Error: {e}")
        
        # 3. Numba
        print("  Testing Numba (JIT optimized)...")
        try:
            times = []
            for _ in range(num_warmup):
                _ = softmax_numba(x)
            
            for _ in range(num_runs):
                t_start = time.perf_counter()
                output = softmax_numba(x)
                t_end = time.perf_counter()
                times.append(t_end - t_start)
            
            times = np.array(times)
            latency_ms = np.mean(times) * 1000
            latency_p50 = np.percentile(times, 50) * 1000
            latency_p95 = np.percentile(times, 95) * 1000
            latency_p99 = np.percentile(times, 99) * 1000
            
            throughput_mops = (ops / 1e6) / np.mean(times)
            
            # Verify correctness
            # fastmath=True can cause small numerical differences
            assert np.allclose(output, output_ref, rtol=1e-3, atol=1e-3), "Numba correctness check failed"
            
            results.append({
                'kernel': 'numba',
                'batch_size': batch_size,
                'seq_len': seq_len,
                'ops': ops,
                'bytes_moved': bytes_moved,
                'latency_ms': latency_ms,
                'latency_p50_ms': latency_p50,
                'latency_p95_ms': latency_p95,
                'latency_p99_ms': latency_p99,
                'throughput_mops': throughput_mops,
            })
        except Exception as e:
            print(f"    Error: {e}")
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Benchmark configurations
    # Format: (batch_size, seq_len)
    configs = [
        (32, 128),      # Small
        (64, 256),      # Medium-small
        (128, 512),     # Medium
        (256, 1024),    # Medium-large
        (512, 2048),    # Large
        (1024, 4096),   # Very large
    ]
    
    print("=" * 60)
    print("Softmax Benchmark Suite")
    print("=" * 60)
    
    df = benchmark_softmax(configs, num_warmup=3, num_runs=10)
    
    # Save results
    output_dir = Path(__file__).parent.parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "softmax_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(df.to_string(index=False))

