"""
Baseline LayerNorm implementation using pure Python loops.
Computes: (x - mean) / sqrt(variance + eps) * gamma + beta
"""

import numpy as np


def layernorm_baseline(x, gamma, beta, eps=1e-5):
    """
    Compute LayerNorm along the last dimension.
    
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
        D = x.shape[0]
        
        # Compute mean
        mean = 0.0
        for i in range(D):
            mean += x[i]
        mean /= D
        
        # Compute variance
        variance = 0.0
        for i in range(D):
            variance += (x[i] - mean) ** 2
        variance /= D
        
        # Normalize and scale
        output = np.zeros_like(x)
        std = np.sqrt(variance + eps)
        for i in range(D):
            output[i] = gamma[i] * (x[i] - mean) / std + beta[i]
        
        return output
    
    elif x.ndim == 2:
        # 2D case: normalize each row independently
        M, D = x.shape
        output = np.zeros_like(x, dtype=np.float32)
        
        for i in range(M):
            # Compute mean for row i
            mean = 0.0
            for j in range(D):
                mean += x[i, j]
            mean /= D
            
            # Compute variance for row i
            variance = 0.0
            for j in range(D):
                variance += (x[i, j] - mean) ** 2
            variance /= D
            
            # Normalize and scale
            std = np.sqrt(variance + eps)
            for j in range(D):
                output[i, j] = gamma[j] * (x[i, j] - mean) / std + beta[j]
        
        return output
    
    else:
        raise ValueError(f"Unsupported input dimension: {x.ndim}")


def verify_correctness(x, gamma, beta, output, eps=1e-5, rtol=1e-5):
    """Verify that output matches reference LayerNorm implementation."""
    # Reference implementation using NumPy
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
    
    print("Running baseline layernorm...")
    output = layernorm_baseline(x, gamma, beta)
    
    # Check that output has zero mean and unit variance (approximately)
    output_mean = np.mean(output, axis=1)
    output_std = np.std(output, axis=1)
    print(f"Output mean (should be ~0.0): min={np.min(output_mean):.6f}, max={np.max(output_mean):.6f}")
    print(f"Output std (should be ~1.0): min={np.min(output_std):.6f}, max={np.max(output_std):.6f}")
    
    if verify_correctness(x, gamma, beta, output):
        print("✓ Correctness check passed!")
    else:
        print("✗ Correctness check failed!")

