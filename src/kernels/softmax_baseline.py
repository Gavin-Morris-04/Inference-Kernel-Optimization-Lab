"""
Baseline Stable Softmax implementation using pure Python loops.
Implements numerical stability via max subtraction.
"""

import numpy as np


def softmax_baseline(x):
    """
    Compute stable softmax: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    
    For 2D input, applies softmax along the last dimension (row-wise).
    
    Args:
        x: numpy array of shape (..., N), dtype float32
        
    Returns:
        Output of same shape as x, dtype float32
    """
    x = np.asarray(x, dtype=np.float32)
    
    if x.ndim == 1:
        # 1D case
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        sum_exp = np.sum(exp_x)
        return exp_x / sum_exp
    
    elif x.ndim == 2:
        # 2D case: apply softmax row-wise
        M, N = x.shape
        output = np.zeros_like(x, dtype=np.float32)
        
        # Loop over rows
        for i in range(M):
            # Find max in row
            row_max = x[i, 0]
            for j in range(1, N):
                if x[i, j] > row_max:
                    row_max = x[i, j]
            
            # Compute exp(x - max)
            sum_exp = 0.0
            for j in range(N):
                exp_val = np.exp(x[i, j] - row_max)
                output[i, j] = exp_val
                sum_exp += exp_val
            
            # Normalize
            for j in range(N):
                output[i, j] /= sum_exp
        
        return output
    
    else:
        raise ValueError(f"Unsupported input dimension: {x.ndim}")


def verify_correctness(x, output, rtol=1e-5):
    """Verify that output matches reference softmax implementation."""
    # Reference implementation using NumPy
    x_ref = np.asarray(x, dtype=np.float32)
    if x_ref.ndim == 2:
        exp_x = np.exp(x_ref - np.max(x_ref, axis=1, keepdims=True))
        ref_output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    else:
        exp_x = np.exp(x_ref - np.max(x_ref))
        ref_output = exp_x / np.sum(exp_x)
    
    return np.allclose(output, ref_output, rtol=rtol)


if __name__ == "__main__":
    # Test with small input
    np.random.seed(42)
    batch_size, seq_len = 32, 128
    x = np.random.randn(batch_size, seq_len).astype(np.float32)
    
    print("Running baseline softmax...")
    output = softmax_baseline(x)
    
    # Check that rows sum to 1
    row_sums = np.sum(output, axis=1)
    print(f"Row sums (should be ~1.0): min={np.min(row_sums):.6f}, max={np.max(row_sums):.6f}")
    
    if verify_correctness(x, output):
        print("✓ Correctness check passed!")
    else:
        print("✗ Correctness check failed!")

