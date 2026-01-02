"""
Benchmark script for LayerNorm kernels.
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

from kernels.layernorm_baseline import layernorm_baseline
from kernels.layernorm_numpy import layernorm_numpy
from kernels.layernorm_numba import layernorm_numba


def benchmark_layernorm(configs, num_warmup=3, num_runs=10):
    """
    Benchmark LayerNorm kernels.
    
    Args:
        configs: List of tuples (batch_size, hidden_dim) representing input dimensions
        num_warmup: Number of warmup runs for JIT compilation
        num_runs: Number of timed runs
        
    Returns:
        DataFrame with benchmark results
    """
    results = []
    
    for batch_size, hidden_dim in configs:
        print(f"\nBenchmarking LayerNorm: batch_size={batch_size}, hidden_dim={hidden_dim}")
        
        # Generate test data
        np.random.seed(42)
        x = np.random.randn(batch_size, hidden_dim).astype(np.float32)
        gamma = np.ones(hidden_dim, dtype=np.float32)
        beta = np.zeros(hidden_dim, dtype=np.float32)
        
        # Reference for correctness check
        mean = np.mean(x, axis=1, keepdims=True)
        variance = np.var(x, axis=1, keepdims=True)
        output_ref = gamma * (x - mean) / np.sqrt(variance + 1e-5) + beta
        
        # Compute theoretical operations
        # Each element: mean (1 op), var (2 ops), sqrt (1 op), norm (2 ops), scale+shift (2 ops)
        # Approximate: 8 ops per element
        ops = 8 * batch_size * hidden_dim
        bytes_moved = (batch_size * hidden_dim * 3 + hidden_dim * 2) * 4  # x, output, temp, gamma, beta
        
        # 1. Baseline (pure Python)
        print("  Testing baseline (pure Python)...")
        try:
            # Only test small inputs for baseline (it's very slow)
            if batch_size * hidden_dim < 1e5:
                times = []
                for _ in range(num_warmup):
                    _ = layernorm_baseline(x, gamma, beta)
                
                for _ in range(num_runs):
                    t_start = time.perf_counter()
                    output = layernorm_baseline(x, gamma, beta)
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
                    'hidden_dim': hidden_dim,
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
                _ = layernorm_numpy(x, gamma, beta)
            
            for _ in range(num_runs):
                t_start = time.perf_counter()
                output = layernorm_numpy(x, gamma, beta)
                t_end = time.perf_counter()
                times.append(t_end - t_start)
            
            times = np.array(times)
            latency_ms = np.mean(times) * 1000
            latency_p50 = np.percentile(times, 50) * 1000
            latency_p95 = np.percentile(times, 95) * 1000
            latency_p99 = np.percentile(times, 99) * 1000
            
            throughput_mops = (ops / 1e6) / np.mean(times)
            
            # Verify correctness
            assert np.allclose(output, output_ref, rtol=1e-4), "NumPy correctness check failed"
            
            results.append({
                'kernel': 'numpy',
                'batch_size': batch_size,
                'hidden_dim': hidden_dim,
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
                _ = layernorm_numba(x, gamma, beta)
            
            for _ in range(num_runs):
                t_start = time.perf_counter()
                output = layernorm_numba(x, gamma, beta)
                t_end = time.perf_counter()
                times.append(t_end - t_start)
            
            times = np.array(times)
            latency_ms = np.mean(times) * 1000
            latency_p50 = np.percentile(times, 50) * 1000
            latency_p95 = np.percentile(times, 95) * 1000
            latency_p99 = np.percentile(times, 99) * 1000
            
            throughput_mops = (ops / 1e6) / np.mean(times)
            
            # Verify correctness
            assert np.allclose(output, output_ref, rtol=1e-3), "Numba correctness check failed"
            
            results.append({
                'kernel': 'numba',
                'batch_size': batch_size,
                'hidden_dim': hidden_dim,
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
    # Format: (batch_size, hidden_dim)
    configs = [
        (32, 512),      # Small
        (64, 1024),     # Medium-small
        (128, 2048),    # Medium
        (256, 4096),    # Medium-large
        (512, 8192),    # Large
        (1024, 16384),  # Very large
    ]
    
    print("=" * 60)
    print("LayerNorm Benchmark Suite")
    print("=" * 60)
    
    df = benchmark_layernorm(configs, num_warmup=3, num_runs=10)
    
    # Save results
    output_dir = Path(__file__).parent.parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "layernorm_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(df.to_string(index=False))

