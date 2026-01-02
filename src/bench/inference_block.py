"""
End-to-end inference block chaining kernels together.
Tests: Projection (GEMM) → Softmax → LayerNorm
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

# Add project root to path (two levels up from this file)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.kernels.matmul_numba import matmul_numba
from src.kernels.softmax_numba import softmax_numba
from src.kernels.layernorm_numba import layernorm_numba


def inference_block(x, W_proj, gamma, beta):
    """
    End-to-end inference block:
    1. Projection (GEMM): x @ W_proj
    2. Softmax: softmax(projection)
    3. LayerNorm: layernorm(softmax_output)
    
    Args:
        x: Input tensor (batch_size, input_dim), float32
        W_proj: Projection weight matrix (input_dim, hidden_dim), float32
        gamma: LayerNorm scale (hidden_dim,), float32
        beta: LayerNorm shift (hidden_dim,), float32
        
    Returns:
        output: Final output (batch_size, hidden_dim), float32
        timings: Dict with individual kernel timings
    """
    timings = {}
    
    # 1. Projection (GEMM)
    # Transpose W_proj for cache-friendly access
    W_proj_T = np.ascontiguousarray(W_proj.T, dtype=np.float32)
    t_start = time.perf_counter()
    proj_output = matmul_numba(x, W_proj_T, block=32)
    t_end = time.perf_counter()
    timings['projection_ms'] = (t_end - t_start) * 1000
    
    # 2. Softmax
    t_start = time.perf_counter()
    softmax_output = softmax_numba(proj_output)
    t_end = time.perf_counter()
    timings['softmax_ms'] = (t_end - t_start) * 1000
    
    # 3. LayerNorm
    t_start = time.perf_counter()
    output = layernorm_numba(softmax_output, gamma, beta)
    t_end = time.perf_counter()
    timings['layernorm_ms'] = (t_end - t_start) * 1000
    
    timings['total_ms'] = timings['projection_ms'] + timings['softmax_ms'] + timings['layernorm_ms']
    
    return output, timings


def benchmark_inference_block(configs, num_warmup=3, num_runs=10):
    """
    Benchmark end-to-end inference block.
    
    Args:
        configs: List of tuples (batch_size, input_dim, hidden_dim)
        num_warmup: Number of warmup runs
        num_runs: Number of timed runs
        
    Returns:
        List of timing dictionaries
    """
    results = []
    
    for batch_size, input_dim, hidden_dim in configs:
        print(f"\nBenchmarking Inference Block: batch={batch_size}, input={input_dim}, hidden={hidden_dim}")
        
        # Generate test data - ensure contiguous arrays
        np.random.seed(42)
        x = np.ascontiguousarray(np.random.randn(batch_size, input_dim).astype(np.float32), dtype=np.float32)
        W_proj = np.ascontiguousarray(np.random.randn(input_dim, hidden_dim).astype(np.float32), dtype=np.float32)
        gamma = np.ascontiguousarray(np.ones(hidden_dim, dtype=np.float32), dtype=np.float32)
        beta = np.ascontiguousarray(np.zeros(hidden_dim, dtype=np.float32), dtype=np.float32)
        
        # Warmup: compile with small subset first, then full size
        W_proj_small = W_proj[:min(32, input_dim), :min(32, hidden_dim)]
        W_proj_small_T = np.ascontiguousarray(W_proj_small.T, dtype=np.float32)
        _ = matmul_numba(x[:min(32, batch_size), :min(32, input_dim)], 
                         W_proj_small_T, block=32)
        for _ in range(num_warmup - 1):
            _, _ = inference_block(x, W_proj, gamma, beta)
        
        # Timed runs
        all_timings = []
        for _ in range(num_runs):
            _, timings = inference_block(x, W_proj, gamma, beta)
            all_timings.append(timings)
        
        # Aggregate statistics
        proj_times = [t['projection_ms'] for t in all_timings]
        softmax_times = [t['softmax_ms'] for t in all_timings]
        layernorm_times = [t['layernorm_ms'] for t in all_timings]
        total_times = [t['total_ms'] for t in all_timings]
        
        results.append({
            'batch_size': batch_size,
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'projection_ms_mean': np.mean(proj_times),
            'projection_ms_p50': np.percentile(proj_times, 50),
            'projection_ms_p95': np.percentile(proj_times, 95),
            'softmax_ms_mean': np.mean(softmax_times),
            'softmax_ms_p50': np.percentile(softmax_times, 50),
            'softmax_ms_p95': np.percentile(softmax_times, 95),
            'layernorm_ms_mean': np.mean(layernorm_times),
            'layernorm_ms_p50': np.percentile(layernorm_times, 50),
            'layernorm_ms_p95': np.percentile(layernorm_times, 95),
            'total_ms_mean': np.mean(total_times),
            'total_ms_p50': np.percentile(total_times, 50),
            'total_ms_p95': np.percentile(total_times, 95),
        })
        
        # Print breakdown
        print(f"  Projection: {np.mean(proj_times):.3f} ms ({100*np.mean(proj_times)/np.mean(total_times):.1f}%)")
        print(f"  Softmax:    {np.mean(softmax_times):.3f} ms ({100*np.mean(softmax_times)/np.mean(total_times):.1f}%)")
        print(f"  LayerNorm:  {np.mean(layernorm_times):.3f} ms ({100*np.mean(layernorm_times)/np.mean(total_times):.1f}%)")
        print(f"  Total:      {np.mean(total_times):.3f} ms")
    
    return results


if __name__ == "__main__":
    # Benchmark configurations
    # Format: (batch_size, input_dim, hidden_dim)
    configs = [
        (32, 512, 512),
        (64, 1024, 1024),
        (128, 2048, 2048),
        (256, 4096, 4096),
    ]
    
    print("=" * 60)
    print("End-to-End Inference Block Benchmark")
    print("=" * 60)
    
    results = benchmark_inference_block(configs, num_warmup=3, num_runs=10)
    
    # Save results
    import pandas as pd
    output_dir = Path(__file__).parent.parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(results)
    output_path = output_dir / "inference_block_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(df.to_string(index=False))

