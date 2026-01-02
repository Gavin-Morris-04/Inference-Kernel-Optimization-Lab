"""
Numba JIT optimized Stable Softmax implementation.
Features:
- Parallel row processing
- Fused operations to minimize memory passes
- Fastmath optimizations
- Numerical stability via max subtraction
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def softmax_numba(x):
    """
    Compute stable softmax: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    
    For 2D input, applies softmax along the last dimension (row-wise).
    Parallelized over rows.
    
    Args:
        x: numpy array of shape (M, N), dtype float32
        
    Returns:
        Output of same shape as x, dtype float32
    """
    M, N = x.shape
    output = np.zeros_like(x)
    
    # Parallel loop over rows
    for i in prange(M):
        # First pass: find max in row
        row_max = x[i, 0]
        for j in range(1, N):
            if x[i, j] > row_max:
                row_max = x[i, j]
        
        # Second pass: compute exp(x - max) and accumulate sum
        sum_exp = 0.0
        for j in range(N):
            exp_val = np.exp(x[i, j] - row_max)
            output[i, j] = exp_val
            sum_exp += exp_val
        
        # Third pass: normalize
        for j in range(N):
            output[i, j] /= sum_exp
    
    return output


@njit(parallel=True, fastmath=True, cache=True)
def softmax_numba_fused(x):
    """
    Optimized softmax with better cache behavior.
    Uses a single pass for exp computation and sum.
    """
    M, N = x.shape
    output = np.zeros_like(x)
    
    # Parallel loop over rows
    for i in prange(M):
        # Find max in row
        row_max = x[i, 0]
        for j in range(1, N):
            if x[i, j] > row_max:
                row_max = x[i, j]
        
        # Compute exp and sum in one pass
        sum_exp = 0.0
        for j in range(N):
            exp_val = np.exp(x[i, j] - row_max)
            output[i, j] = exp_val
            sum_exp += exp_val
        
        # Normalize (divide by sum)
        inv_sum = 1.0 / sum_exp
        for j in range(N):
            output[i, j] *= inv_sum
    
    return output


def verify_correctness(x, output, rtol=1e-4):
    """Verify that output matches reference softmax implementation."""
    # Reference implementation using NumPy
    x_ref = np.asarray(x, dtype=np.float32)
    exp_x = np.exp(x_ref - np.max(x_ref, axis=1, keepdims=True))
    ref_output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    return np.allclose(output, ref_output, rtol=rtol)


if __name__ == "__main__":
    # Test with small input
    np.random.seed(42)
    batch_size, seq_len = 32, 128
    x = np.random.randn(batch_size, seq_len).astype(np.float32)
    
    print("Running Numba softmax...")
    # Warmup
    _ = softmax_numba(x)
    
    output = softmax_numba(x)
    
    # Check that rows sum to 1
    row_sums = np.sum(output, axis=1)
    print(f"Row sums (should be ~1.0): min={np.min(row_sums):.6f}, max={np.max(row_sums):.6f}")
    
    if verify_correctness(x, output):
        print("✓ Correctness check passed!")
    else:
        print("✗ Correctness check failed!")

