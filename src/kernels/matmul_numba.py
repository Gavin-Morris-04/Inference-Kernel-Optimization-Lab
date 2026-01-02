"""
Numba JIT optimized GEMM implementation.
Features:
- Block tiling for cache locality
- Parallel outer loops
- Fastmath optimizations
- Local accumulators to minimize memory access
- B matrix pre-transposed for cache-friendly access
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def matmul_numba(A, BT, block=32):
    """
    Compute matrix multiplication C = A @ BT.T using Numba JIT with cache-blocked tiling.
    
    Optimized kernel-style implementation:
    - Block tiling for cache locality (block x block tiles)
    - Parallel outer tile loops using prange
    - Local accumulator tile to minimize memory traffic (stays in L1 cache)
    - Cache-friendly access patterns: B is pre-transposed (BT) for row-major access
    - 4-way loop unrolling for better SIMD opportunities
    
    Cache strategy:
    - B is pre-transposed to BT, so BT[j, k] = B[k, j]
    - Both A[i, k] and BT[j, k] are now row-major access (cache-friendly)
    - Tiling order: ii (rows), kk (shared), jj (cols) maximizes reuse
    - Local accumulator tile stays in L1 cache
    
    Args:
        A: numpy array of shape (M, K), dtype float32, must be contiguous
        BT: numpy array of shape (N, K), dtype float32, must be contiguous (B transposed)
        block: tile size for blocking (default 32, try 16, 32, 64 for best performance)
        
    Returns:
        C: numpy array of shape (M, N), dtype float32
    """
    M, K = A.shape
    N, K2 = BT.shape
    
    if K != K2:
        raise ValueError("Dimension mismatch: A.shape[1] != BT.shape[1]")
    
    C = np.zeros((M, N), dtype=np.float32)
    
    # Calculate number of tiles for parallelization
    num_tiles_i = (M + block - 1) // block
    
    # Parallelize over output tile rows (ii)
    # Better load balancing than 2D tile space for cache efficiency
    for tile_i in prange(num_tiles_i):
        ii = tile_i * block
        iimax = min(ii + block, M)
        tile_height = iimax - ii
        
        # Process all columns for this row tile
        for jj in range(0, N, block):
            jjmax = min(jj + block, N)
            tile_width = jjmax - jj
            
            # Local accumulator tile - keeps data in L1 cache
            tile_C = np.zeros((tile_height, tile_width), dtype=np.float32)
            
            # Accumulate over K dimension in blocks
            # Order: ii (rows), kk (shared), jj (cols) maximizes A block reuse
            for kk in range(0, K, block):
                kkmax = min(kk + block, K)
                
                # Compute tile contribution: tile_C += A[ii:iimax, kk:kkmax] @ BT[jj:jjmax, kk:kkmax].T
                # Both A[i, k] and BT[j, k] are now row-major (cache-friendly)
                for i_local in range(tile_height):
                    i = ii + i_local
                    for j_local in range(tile_width):
                        j = jj + j_local
                        acc = tile_C[i_local, j_local]
                        # Unroll 4-way for better SIMD opportunities
                        k = kk
                        while k + 3 < kkmax:
                            acc += A[i, k] * BT[j, k]
                            acc += A[i, k+1] * BT[j, k+1]
                            acc += A[i, k+2] * BT[j, k+2]
                            acc += A[i, k+3] * BT[j, k+3]
                            k += 4
                        # Handle remainder
                        while k < kkmax:
                            acc += A[i, k] * BT[j, k]
                            k += 1
                        tile_C[i_local, j_local] = acc
            
            # Write accumulated tile to output (single write per output tile)
            for i_local in range(tile_height):
                i = ii + i_local
                for j_local in range(tile_width):
                    j = jj + j_local
                    C[i, j] = tile_C[i_local, j_local]
    
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


def verify_correctness(A, B, C_result, rtol=1e-2, atol=1e-2):
    """Verify that C_result matches A @ B (reference implementation)."""
    C_ref = A @ B
    return np.allclose(C_result, C_ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    # Test with small matrices
    np.random.seed(42)
    M, K, N = 128, 256, 64
    A = np.ascontiguousarray(np.random.randn(M, K).astype(np.float32), dtype=np.float32)
    B = np.ascontiguousarray(np.random.randn(K, N).astype(np.float32), dtype=np.float32)
    BT = np.ascontiguousarray(B.T, dtype=np.float32)
    
    print("Running Numba matmul (tiled with B transpose)...")
    # Warmup
    _ = matmul_numba(A, BT, block=32)
    
    C = matmul_numba(A, BT, block=32)
    
    if verify_correctness(A, B, C, rtol=1e-2):
        print("Correctness check passed!")
    else:
        print("Correctness check failed!")
        print(f"Max diff: {np.max(np.abs(C - A @ B)):.6e}")
