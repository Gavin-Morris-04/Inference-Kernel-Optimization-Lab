"""
NumPy vectorized LayerNorm implementation.
Uses vectorized mean/variance computation.
"""

import numpy as np


def layernorm_numpy(x, gamma, beta, eps=1e-5):
    """
    Compute LayerNorm along the last dimension using vectorized operations.
    
    LayerNorm(x) = gamma * (x - mean) / sqrt(variance + eps) + beta
    
    Args:
        x: numpy array of shape (..., D), dtype float32
        gamma: scale parameter, shape (D,), dtype float32
        beta: shift parameter, shape (D,), dtype float32
        eps: small constant for numerical stability, default 1e-5
        
    Returns:
        Output of same shape as x, dtype float32
    """
    x = np.asarray(x, dtype=np.float32)
    gamma = np.asarray(gamma, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)
    eps = float(eps)
    
    if x.ndim == 1:
        # 1D case
        mean = np.mean(x)
        variance = np.var(x)
        std = np.sqrt(variance + eps)
        normalized = (x - mean) / std
        return gamma * normalized + beta
    
    elif x.ndim == 2:
        # 2D case: normalize each row independently
        mean = np.mean(x, axis=1, keepdims=True)  # (M, 1)
        variance = np.var(x, axis=1, keepdims=True)  # (M, 1)
        std = np.sqrt(variance + eps)  # (M, 1)
        normalized = (x - mean) / std  # Broadcasting: (M, N) - (M, 1) / (M, 1)
        return gamma * normalized + beta  # Broadcasting: (M, N) * (D,) + (D,)
    
    else:
        raise ValueError(f"Unsupported input dimension: {x.ndim}")


def verify_correctness(x, gamma, beta, output, eps=1e-5, rtol=1e-5):
    """Verify that output matches reference LayerNorm implementation."""
    # Reference implementation (same as this one, but double-check)
    x_ref = np.asarray(x, dtype=np.float32)
    if x_ref.ndim == 2:
        mean = np.mean(x_ref, axis=1, keepdims=True)
        variance = np.var(x_ref, axis=1, keepdims=True)
        ref_output = gamma * (x_ref - mean) / np.sqrt(variance + eps) + beta
    else:
        mean = np.mean(x_ref)
        variance = np.var(x_ref)
        ref_output = gamma * (x_ref - mean) / np.sqrt(variance + eps) + beta
    
    return np.allclose(output, ref_output, rtol=rtol)


if __name__ == "__main__":
    # Test with small input
    np.random.seed(42)
    batch_size, hidden_dim = 32, 512
    x = np.random.randn(batch_size, hidden_dim).astype(np.float32)
    gamma = np.ones(hidden_dim, dtype=np.float32)
    beta = np.zeros(hidden_dim, dtype=np.float32)
    
    print("Running NumPy layernorm...")
    output = layernorm_numpy(x, gamma, beta)
    
    # Check that output has zero mean and unit variance (approximately)
    output_mean = np.mean(output, axis=1)
    output_std = np.std(output, axis=1)
    print(f"Output mean (should be ~0.0): min={np.min(output_mean):.6f}, max={np.max(output_mean):.6f}")
    print(f"Output std (should be ~1.0): min={np.min(output_std):.6f}, max={np.max(output_std):.6f}")
    
    if verify_correctness(x, gamma, beta, output):
        print("✓ Correctness check passed!")
    else:
        print("✗ Correctness check failed!")

