"""
Numba JIT optimized LayerNorm implementation.
Features:
- Fused operations to minimize memory passes
- Parallel row processing
- Fastmath optimizations
- Memory-access optimized (contiguous reads/writes)
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def layernorm_numba(x, gamma, beta, eps=1e-5):
    """
    Compute LayerNorm along the last dimension using fused operations.
    
    LayerNorm(x) = gamma * (x - mean) / sqrt(variance + eps) + beta
    
    Optimizations:
    - Single pass for mean computation
    - Single pass for variance computation
    - Single pass for normalization
    - Parallelized over rows
    
    Args:
        x: numpy array of shape (M, D), dtype float32
        gamma: scale parameter, shape (D,), dtype float32
        beta: shift parameter, shape (D,), dtype float32
        eps: small constant for numerical stability, default 1e-5
        
    Returns:
        Output of same shape as x, dtype float32
    """
    M, D = x.shape
    output = np.zeros_like(x)
    
    # Parallel loop over rows
    for i in prange(M):
        # First pass: compute mean
        mean = 0.0
        for j in range(D):
            mean += x[i, j]
        mean /= D
        
        # Second pass: compute variance
        variance = 0.0
        for j in range(D):
            diff = x[i, j] - mean
            variance += diff * diff
        variance /= D
        
        # Third pass: normalize and scale
        std = np.sqrt(variance + eps)
        inv_std = 1.0 / std
        for j in range(D):
            normalized = (x[i, j] - mean) * inv_std
            output[i, j] = gamma[j] * normalized + beta[j]
    
    return output


@njit(parallel=True, fastmath=True, cache=True)
def layernorm_numba_two_pass(x, gamma, beta, eps=1e-5):
    """
    Two-pass LayerNorm: mean in first pass, variance and normalization in second.
    Slightly better cache behavior for very large D.
    """
    M, D = x.shape
    means = np.zeros(M, dtype=np.float32)
    output = np.zeros_like(x)
    
    # First pass: compute means (parallel)
    for i in prange(M):
        mean = 0.0
        for j in range(D):
            mean += x[i, j]
        means[i] = mean / D
    
    # Second pass: compute variance and normalize (parallel)
    for i in prange(M):
        mean = means[i]
        
        # Compute variance
        variance = 0.0
        for j in range(D):
            diff = x[i, j] - mean
            variance += diff * diff
        variance /= D
        
        # Normalize and scale
        std = np.sqrt(variance + eps)
        inv_std = 1.0 / std
        for j in range(D):
            normalized = (x[i, j] - mean) * inv_std
            output[i, j] = gamma[j] * normalized + beta[j]
    
    return output


def verify_correctness(x, gamma, beta, output, eps=1e-5, rtol=1e-4):
    """Verify that output matches reference LayerNorm implementation."""
    # Reference implementation using NumPy
    x_ref = np.asarray(x, dtype=np.float32)
    mean = np.mean(x_ref, axis=1, keepdims=True)
    variance = np.var(x_ref, axis=1, keepdims=True)
    ref_output = gamma * (x_ref - mean) / np.sqrt(variance + eps) + beta
    
    return np.allclose(output, ref_output, rtol=rtol)


if __name__ == "__main__":
    # Test with small input
    np.random.seed(42)
    batch_size, hidden_dim = 32, 512
    x = np.random.randn(batch_size, hidden_dim).astype(np.float32)
    gamma = np.ones(hidden_dim, dtype=np.float32)
    beta = np.zeros(hidden_dim, dtype=np.float32)
    
    print("Running Numba layernorm...")
    # Warmup
    _ = layernorm_numba(x, gamma, beta)
    
    output = layernorm_numba(x, gamma, beta)
    
    # Check that output has zero mean and unit variance (approximately)
    output_mean = np.mean(output, axis=1)
    output_std = np.std(output, axis=1)
    print(f"Output mean (should be ~0.0): min={np.min(output_mean):.6f}, max={np.max(output_mean):.6f}")
    print(f"Output std (should be ~1.0): min={np.min(output_std):.6f}, max={np.max(output_std):.6f}")
    
    if verify_correctness(x, gamma, beta, output):
        print("✓ Correctness check passed!")
    else:
        print("✗ Correctness check failed!")

