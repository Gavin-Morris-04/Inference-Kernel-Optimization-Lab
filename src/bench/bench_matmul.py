"""
Benchmark script for GEMM (Matrix Multiply) kernels.
Tests baseline, NumPy, and Numba implementations.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kernels.matmul_baseline import matmul_baseline
from kernels.matmul_numpy import matmul_numpy
from kernels.matmul_numba import matmul_numba


def benchmark_matmul(configs, num_warmup=3, num_runs=10):
    """
    Benchmark matrix multiplication kernels.
    
    Args:
        configs: List of tuples (M, K, N) representing matrix dimensions
        num_warmup: Number of warmup runs for JIT compilation
        num_runs: Number of timed runs
        
    Returns:
        DataFrame with benchmark results
    """
    results = []
    
    for M, K, N in configs:
        print(f"\nBenchmarking GEMM: M={M}, K={K}, N={N}")
        
        # Generate test data
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        
        # Reference for correctness check
        C_ref = A @ B
        
        # Compute theoretical FLOPs: 2 * M * K * N
        flops = 2 * M * K * N
        bytes_moved = (M * K + K * N + M * N) * 4  # float32 = 4 bytes
        
        # 1. Baseline (pure Python)
        print("  Testing baseline (pure Python)...")
        try:
            # Only test small matrices for baseline (it's very slow)
            if M * K * N < 1e6:  # Only for small problems
                times = []
                for _ in range(num_warmup):
                    _ = matmul_baseline(A, B)
                
                for _ in range(num_runs):
                    t_start = time.perf_counter()
                    C = matmul_baseline(A, B)
                    t_end = time.perf_counter()
                    times.append(t_end - t_start)
                
                times = np.array(times)
                latency_ms = np.mean(times) * 1000
                latency_p50 = np.percentile(times, 50) * 1000
                latency_p95 = np.percentile(times, 95) * 1000
                latency_p99 = np.percentile(times, 99) * 1000
                
                throughput_gflops = (flops / 1e9) / np.mean(times)
                
                results.append({
                    'kernel': 'baseline',
                    'M': M, 'K': K, 'N': N,
                    'flops': flops,
                    'bytes_moved': bytes_moved,
                    'latency_ms': latency_ms,
                    'latency_p50_ms': latency_p50,
                    'latency_p95_ms': latency_p95,
                    'latency_p99_ms': latency_p99,
                    'throughput_gflops': throughput_gflops,
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
                _ = matmul_numpy(A, B)
            
            for _ in range(num_runs):
                t_start = time.perf_counter()
                C = matmul_numpy(A, B)
                t_end = time.perf_counter()
                times.append(t_end - t_start)
            
            times = np.array(times)
            latency_ms = np.mean(times) * 1000
            latency_p50 = np.percentile(times, 50) * 1000
            latency_p95 = np.percentile(times, 95) * 1000
            latency_p99 = np.percentile(times, 99) * 1000
            
            throughput_gflops = (flops / 1e9) / np.mean(times)
            
            # Verify correctness
            assert np.allclose(C, C_ref, rtol=1e-4), "NumPy correctness check failed"
            
            results.append({
                'kernel': 'numpy',
                'M': M, 'K': K, 'N': N,
                'flops': flops,
                'bytes_moved': bytes_moved,
                'latency_ms': latency_ms,
                'latency_p50_ms': latency_p50,
                'latency_p95_ms': latency_p95,
                'latency_p99_ms': latency_p99,
                'throughput_gflops': throughput_gflops,
            })
        except Exception as e:
            print(f"    Error: {e}")
        
        # 3. Numba
        print("  Testing Numba (JIT optimized)...")
        try:
            times = []
            for _ in range(num_warmup):
                _ = matmul_numba(A, B)
            
            for _ in range(num_runs):
                t_start = time.perf_counter()
                C = matmul_numba(A, B)
                t_end = time.perf_counter()
                times.append(t_end - t_start)
            
            times = np.array(times)
            latency_ms = np.mean(times) * 1000
            latency_p50 = np.percentile(times, 50) * 1000
            latency_p95 = np.percentile(times, 95) * 1000
            latency_p99 = np.percentile(times, 99) * 1000
            
            throughput_gflops = (flops / 1e9) / np.mean(times)
            
            # Verify correctness
            assert np.allclose(C, C_ref, rtol=1e-3), "Numba correctness check failed"
            
            results.append({
                'kernel': 'numba',
                'M': M, 'K': K, 'N': N,
                'flops': flops,
                'bytes_moved': bytes_moved,
                'latency_ms': latency_ms,
                'latency_p50_ms': latency_p50,
                'latency_p95_ms': latency_p95,
                'latency_p99_ms': latency_p99,
                'throughput_gflops': throughput_gflops,
            })
        except Exception as e:
            print(f"    Error: {e}")
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Benchmark configurations
    # Format: (M, K, N) - matrix dimensions
    configs = [
        (64, 128, 64),      # Small
        (128, 256, 128),    # Medium-small
        (256, 512, 256),    # Medium
        (512, 1024, 512),   # Medium-large
        (1024, 2048, 1024), # Large
        (2048, 4096, 2048), # Very large
    ]
    
    print("=" * 60)
    print("GEMM Benchmark Suite")
    print("=" * 60)
    
    df = benchmark_matmul(configs, num_warmup=3, num_runs=10)
    
    # Save results
    output_dir = Path(__file__).parent.parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "matmul_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(df.to_string(index=False))

