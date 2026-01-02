"""
Numba JIT optimized GEMM implementation.
Features:
- Block tiling for cache locality
- Parallel outer loops
- Fastmath optimizations
- Local accumulators to minimize memory access
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def matmul_numba(A, B, tile_size=32):
    """
    Compute matrix multiplication C = A @ B using Numba JIT with tiling.
    
    Block tiling strategy:
    - Divide matrices into tiles of size tile_size x tile_size
    - Process tiles to maximize cache reuse
    - Use parallel outer loops for thread-level parallelism
    
    Args:
        A: numpy array of shape (M, K), dtype float32
        B: numpy array of shape (K, N), dtype float32
        tile_size: tile size for blocking (default 32)
        
    Returns:
        C: numpy array of shape (M, N), dtype float32
    """
    M, K = A.shape
    N = B.shape[1]
    
    C = np.zeros((M, N), dtype=np.float32)
    
    # Tiled matrix multiplication
    # Parallelize over output tile rows
    num_tile_rows = (M + tile_size - 1) // tile_size
    num_tile_cols = (N + tile_size - 1) // tile_size
    num_tile_mid = (K + tile_size - 1) // tile_size
    
    # Parallel loop over tile rows
    for tile_i in prange(num_tile_rows):
        i_start = tile_i * tile_size
        i_end = min(i_start + tile_size, M)
        
        for tile_j in range(num_tile_cols):
            j_start = tile_j * tile_size
            j_end = min(j_start + tile_size, N)
            
            # Local accumulator for this tile
            # This keeps intermediate results in registers/L1 cache
            tile_C = np.zeros((i_end - i_start, j_end - j_start), dtype=np.float32)
            
            # Inner product loop over K dimension
            for tile_k in range(num_tile_mid):
                k_start = tile_k * tile_size
                k_end = min(k_start + tile_size, K)
                
                # Compute tile contribution
                for i in range(i_end - i_start):
                    for j in range(j_end - j_start):
                        acc = tile_C[i, j]
                        for k in range(k_end - k_start):
                            acc += A[i_start + i, k_start + k] * B[k_start + k, j_start + j]
                        tile_C[i, j] = acc
            
            # Write tile to output
            for i in range(i_end - i_start):
                for j in range(j_end - j_start):
                    C[i_start + i, j_start + j] = tile_C[i, j]
    
    return C


# Non-tiled version for comparison (still parallelized)
@njit(parallel=True, fastmath=True, cache=True)
def matmul_numba_simple(A, B):
    """
    Simpler parallelized version without explicit tiling.
    Numba's compiler may still apply optimizations.
    """
    M, K = A.shape
    N = B.shape[1]
    
    C = np.zeros((M, N), dtype=np.float32)
    
    # Parallelize over output rows
    for i in prange(M):
        for j in range(N):
            acc = 0.0
            for k in range(K):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc
    
    return C


def verify_correctness(A, B, C_result, rtol=1e-4):
    """Verify that C_result matches A @ B (reference implementation)."""
    C_ref = A @ B
    return np.allclose(C_result, C_ref, rtol=rtol)


if __name__ == "__main__":
    # Test with small matrices
    np.random.seed(42)
    M, K, N = 128, 256, 64
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    
    print("Running Numba matmul (tiled)...")
    # Warmup
    _ = matmul_numba(A, B)
    
    C = matmul_numba(A, B)
    
    if verify_correctness(A, B, C):
        print("✓ Correctness check passed!")
    else:
        print("✗ Correctness check failed!")
        print(f"Max diff: {np.max(np.abs(C - A @ B)):.6e}")

