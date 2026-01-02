"""
Benchmark script for GEMM (Matrix Multiply) kernels.
Tests baseline, NumPy, and Numba implementations.
"""

import sys
import os
import time
from pathlib import Path

# Set thread limits BEFORE importing NumPy to prevent BLAS thread contention
# This ensures fair comparison and stable results
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

# Add project root to path (two levels up from this file)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.kernels.matmul_baseline import matmul_baseline
from src.kernels.matmul_numpy import matmul_numpy
from src.kernels.matmul_numba import matmul_numba

# Control Numba threads (must be set before JIT compilation)
# This will be set in main() based on mode

# Control Numba threads if specified (must be before JIT compilation)
if 'numba_threads' in globals() and numba_threads is not None:
    from numba import set_num_threads
    set_num_threads(numba_threads)


def benchmark_matmul(configs, num_warmup=3, num_runs=10, verbose_sweep=False, numba_threads=None):
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
        
        # Adjust number of runs based on problem size for better statistics
        # Very small sizes need many runs to reduce variance
        # BUT: Keep large problem runs fast (don't do hundreds of iterations)
        problem_size = M * K * N
        if problem_size < 1_000_000:  # Very small problems (64×128×64)
            actual_runs = max(num_runs, 1000)
        elif problem_size < 10_000_000:  # Small problems
            actual_runs = max(num_runs, 500)
        elif problem_size < 100_000_000:  # Medium problems
            actual_runs = max(num_runs, 30)  # Reduced from 50 to minimize overhead
        elif problem_size < 1_000_000_000:  # Large problems
            actual_runs = num_runs  # Default (10 runs)
        else:  # Very large problems (can reduce further)
            actual_runs = max(5, num_runs // 2)  # 5 runs for very large
        
        # Generate test data - ensure contiguous float32 arrays
        np.random.seed(42)
        A = np.ascontiguousarray(np.random.randn(M, K).astype(np.float32), dtype=np.float32)
        B = np.ascontiguousarray(np.random.randn(K, N).astype(np.float32), dtype=np.float32)
        
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
                
                for _ in range(actual_runs):
                    t_start = time.perf_counter()
                    C = matmul_baseline(A, B)
                    t_end = time.perf_counter()
                    times.append(t_end - t_start)
                
                times = np.array(times)
                # Use median for primary metric (more robust to outliers)
                latency_ms = np.median(times) * 1000
                latency_p50 = np.percentile(times, 50) * 1000
                latency_p95 = np.percentile(times, 95) * 1000
                latency_p99 = np.percentile(times, 99) * 1000
                
                throughput_gflops = (flops / 1e9) / np.median(times)
                
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
            
            for _ in range(actual_runs):
                t_start = time.perf_counter()
                C = matmul_numpy(A, B)
                t_end = time.perf_counter()
                times.append(t_end - t_start)
            
            times = np.array(times)
            # Use median for primary metric (more robust to outliers)
            latency_ms = np.median(times) * 1000
            latency_p50 = np.percentile(times, 50) * 1000
            latency_p95 = np.percentile(times, 95) * 1000
            latency_p99 = np.percentile(times, 99) * 1000
            
            throughput_gflops = (flops / 1e9) / np.median(times)
            
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
        
        # 3. Numba - test multiple block sizes and pick best
        print("  Testing Numba (JIT optimized)...")
        try:
            # Set Numba threads if specified (must be before first call)
            if numba_threads is not None and M == configs[0][0]:
                from numba import set_num_threads
                set_num_threads(numba_threads)
                print(f"    Numba threads set to: {numba_threads}")
            
            # Force contiguous float32 right before Numba call
            A_nb = np.ascontiguousarray(A, dtype=np.float32)
            B_nb = np.ascontiguousarray(B, dtype=np.float32)
            
            # Transpose B once for cache-friendly access pattern
            # This allows BT[j, k] = B[k, j], making both A and B row-major access
            BT_nb = np.ascontiguousarray(B_nb.T, dtype=np.float32)
            
            # Check parallelization (once per benchmark run, not per size)
            if M == configs[0][0]:  # Only print once for first config
                try:
                    from numba import get_num_threads
                    num_threads = get_num_threads()
                    if numba_threads is None:
                        print(f"    Numba threads: {num_threads} (default)")
                    elif num_threads != numba_threads:
                        print(f"    Warning: Requested {numba_threads} threads but got {num_threads}")
                except:
                    pass
            
            # Test multiple block sizes and find optimal
            # Minimal sweep - just enough to pick the best block size
            block_sizes = [16, 32, 64]
            best_time = float('inf')
            best_block = 32
            sweep_results = {}  # Store results for logging
            
            # Block size sweep with stable timing methodology
            # Use median (not mean) for robustness against outliers
            for block_size in block_sizes:
                # Warmup for this block size
                BT_small = np.ascontiguousarray(B_nb[:min(64, K), :min(64, N)].T, dtype=np.float32)
                _ = matmul_numba(A_nb[:min(64, M), :min(64, K)], BT_small, block=block_size)
                # Additional warmup with full size
                for _ in range(2):
                    _ = matmul_numba(A_nb, BT_nb, block=block_size)
                
                # Stable timing: 25 runs, use median for robustness
                sweep_runs = 25  # Enough runs for stable median
                test_times = []
                for _ in range(sweep_runs):
                    t_start = time.perf_counter()
                    _ = matmul_numba(A_nb, BT_nb, block=block_size)
                    t_end = time.perf_counter()
                    test_times.append(t_end - t_start)
                
                # Use median for robust measurement (less sensitive to outliers)
                median_time = np.median(test_times)
                median_latency_ms = median_time * 1000
                sweep_results[block_size] = median_latency_ms
                
                # Diagnostic print for first size to verify sweep is working
                if M == configs[0][0]:  # Only for first config
                    mean_ms = np.mean(test_times) * 1000
                    std_ms = np.std(test_times) * 1000
                    print(f"      block={block_size}: median={median_latency_ms:.3f} ms, mean={mean_ms:.3f} ms, std={std_ms:.3f} ms")
                
                if median_time < best_time:
                    best_time = median_time
                    best_block = block_size
            
            # Print optimal block size and optionally sweep table
            if verbose_sweep or M == configs[0][0] or M == configs[-1][0]:
                print(f"    Optimal block size: {best_block}")
                if verbose_sweep or M == configs[0][0]:
                    # Always show sweep results for first size to verify sweep is working
                    print(f"    Block size sweep (latency ms): {sweep_results}")
            
            # Full warmup with best block size (fresh warmup after sweep)
            # Add extra warmup runs to stabilize CPU state after sweep
            # This helps clear any thermal/cache state from the sweep
            extra_warmup = 3 if problem_size >= 100_000_000 else 2  # More warmup for large problems
            for _ in range(num_warmup + extra_warmup):
                _ = matmul_numba(A_nb, BT_nb, block=best_block)
            
            # Small delay for CPU state to stabilize (especially for large problems)
            if problem_size >= 100_000_000:
                time.sleep(0.1)  # 100ms pause for large problems
            
            # Final timing - only time the kernel call itself
            # Make sure we're measuring pure kernel execution time
            times = []
            for _ in range(actual_runs):
                t_start = time.perf_counter()
                C = matmul_numba(A_nb, BT_nb, block=best_block)
                t_end = time.perf_counter()
                times.append(t_end - t_start)
            
            # Compute results from timing array
            times = np.array(times)
            # Use median for primary metric (more robust to outliers)
            latency_ms = np.median(times) * 1000
            latency_p50 = np.percentile(times, 50) * 1000
            latency_p95 = np.percentile(times, 95) * 1000
            latency_p99 = np.percentile(times, 99) * 1000
            
            throughput_gflops = (flops / 1e9) / np.median(times)
            
            # Verify correctness
            # fastmath=True can cause small numerical differences due to operation reordering
            # Use lenient tolerance for float32: rtol=1e-2, atol=1e-2
            if not np.allclose(C, C_ref, rtol=1e-2, atol=1e-2):
                max_diff = np.max(np.abs(C - C_ref))
                rel_error = max_diff / (np.max(np.abs(C_ref)) + 1e-10)
                raise AssertionError(f"Numba correctness check failed: max_diff={max_diff:.6e}, rel_error={rel_error:.6e}")
            
            results.append({
                'kernel': 'numba',
                'M': M, 'K': K, 'N': N,
                'block_size': best_block,
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark GEMM kernels')
    parser.add_argument('--verbose-sweep', action='store_true',
                        help='Print block size sweep results for all sizes')
    args = parser.parse_args()
    
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
    
    print("=" * 70)
    print("GEMM Benchmark Suite - SINGLE-THREADED MODE")
    print("=" * 70)
    print("Mode: Single-threaded kernel efficiency comparison")
    print("  - NumPy BLAS: 1 thread (limited via env vars set before import)")
    print("  - Numba: 1 thread (set via set_num_threads)")
    print("  - Purpose: Fair kernel quality comparison (no threading effects)")
    print("\n  This mode focuses on kernel implementation quality:")
    print("  - Cache blocking effectiveness")
    print("  - Memory access patterns")
    print("  - Algorithm efficiency")
    print("=" * 70)
    
    # Set Numba threads to 1 before any JIT compilation
    from numba import set_num_threads
    set_num_threads(1)
    
    df = benchmark_matmul(configs, num_warmup=3, num_runs=10, 
                         verbose_sweep=args.verbose_sweep, numba_threads=1)
    df['mode'] = 'single_threaded'
    
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

