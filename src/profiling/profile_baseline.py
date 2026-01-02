"""
Profiling script for baseline kernels.
Uses cProfile to identify bottlenecks in pure Python implementations.
"""

import sys
import cProfile
import pstats
from pathlib import Path
import numpy as np

# Add project root to path (two levels up from this file)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.kernels.matmul_baseline import matmul_baseline
from src.kernels.softmax_baseline import softmax_baseline
from src.kernels.layernorm_baseline import layernorm_baseline


def profile_matmul():
    """Profile baseline matrix multiplication."""
    print("Profiling baseline matmul...")
    np.random.seed(42)
    M, K, N = 64, 128, 64  # Small size for profiling
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    
    profiler = cProfile.Profile()
    profiler.enable()
    C = matmul_baseline(A, B)
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    print("\nTop 10 functions by cumulative time:")
    stats.print_stats(10)
    
    return stats


def profile_softmax():
    """Profile baseline softmax."""
    print("\nProfiling baseline softmax...")
    np.random.seed(42)
    batch_size, seq_len = 32, 128
    x = np.random.randn(batch_size, seq_len).astype(np.float32)
    
    profiler = cProfile.Profile()
    profiler.enable()
    output = softmax_baseline(x)
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    print("\nTop 10 functions by cumulative time:")
    stats.print_stats(10)
    
    return stats


def profile_layernorm():
    """Profile baseline LayerNorm."""
    print("\nProfiling baseline layernorm...")
    np.random.seed(42)
    batch_size, hidden_dim = 32, 512
    x = np.random.randn(batch_size, hidden_dim).astype(np.float32)
    gamma = np.ones(hidden_dim, dtype=np.float32)
    beta = np.zeros(hidden_dim, dtype=np.float32)
    
    profiler = cProfile.Profile()
    profiler.enable()
    output = layernorm_baseline(x, gamma, beta)
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    print("\nTop 10 functions by cumulative time:")
    stats.print_stats(10)
    
    return stats


if __name__ == "__main__":
    print("=" * 60)
    print("Baseline Kernel Profiling")
    print("=" * 60)
    
    profile_matmul()
    profile_softmax()
    profile_layernorm()
    
    print("\n" + "=" * 60)
    print("Profiling complete!")
    print("=" * 60)

