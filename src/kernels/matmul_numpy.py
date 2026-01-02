"""
NumPy vectorized GEMM implementation.
Uses @ operator (BLAS-backed) for optimal performance.
"""

import numpy as np


def matmul_numpy(A, B):
    """
    Compute matrix multiplication C = A @ B using NumPy vectorization.
    
    Args:
        A: numpy array of shape (M, K), dtype float32
        B: numpy array of shape (K, N), dtype float32
        
    Returns:
        C: numpy array of shape (M, N), dtype float32
    """
    return A @ B


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
    
    print("Running NumPy matmul...")
    C = matmul_numpy(A, B)
    
    if verify_correctness(A, B, C):
        print("✓ Correctness check passed!")
    else:
        print("✗ Correctness check failed!")

