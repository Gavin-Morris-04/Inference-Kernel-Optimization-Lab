"""
Baseline GEMM implementation using pure Python loops.
Intentionally slow to establish a performance baseline.
"""

import numpy as np


def matmul_baseline(A, B):
    """
    Compute matrix multiplication C = A @ B using pure Python loops.
    
    Args:
        A: numpy array of shape (M, K), dtype float32
        B: numpy array of shape (K, N), dtype float32
        
    Returns:
        C: numpy array of shape (M, N), dtype float32
    """
    M, K = A.shape
    K_check, N = B.shape
    
    if K != K_check:
        raise ValueError(f"Dimension mismatch: A.shape[1]={K} != B.shape[0]={K_check}")
    
    C = np.zeros((M, N), dtype=np.float32)
    
    # Triple nested loop - intentionally slow
    for i in range(M):
        for j in range(N):
            accumulator = 0.0
            for k in range(K):
                accumulator += A[i, k] * B[k, j]
            C[i, j] = accumulator
    
    return C


def verify_correctness(A, B, C_result, rtol=1e-5):
    """Verify that C_result matches A @ B (reference implementation)."""
    C_ref = A @ B
    return np.allclose(C_result, C_ref, rtol=rtol)


if __name__ == "__main__":
    # Test with small matrices
    np.random.seed(42)
    M, K, N = 128, 256, 64
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    
    print("Running baseline matmul...")
    C = matmul_baseline(A, B)
    
    if verify_correctness(A, B, C):
        print("✓ Correctness check passed!")
    else:
        print("✗ Correctness check failed!")

