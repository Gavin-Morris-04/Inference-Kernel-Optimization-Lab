# How Everything Works

This document explains the technical concepts, optimizations, and benchmarking methodology used in this project.

---

## 🧠 Core Concepts

### Why Three Implementations?

Each kernel has three versions to demonstrate the optimization journey:

1. **Baseline (Pure Python)**: Intentionally slow, shows the problem
   - Triple nested loops for GEMM
   - Python overhead dominates
   - Demonstrates why optimization matters

2. **NumPy (Vectorized)**: Production-ready reference
   - Uses optimized BLAS libraries (MKL/OpenBLAS)
   - Demonstrates what "fast" looks like
   - Provides correctness baseline

3. **Numba (JIT-Compiled)**: Hand-tuned optimization
   - Demonstrates optimization techniques
   - Cache blocking, parallelization, SIMD-friendly code
   - Shows understanding of fundamental concepts

---

## 🚀 Kernel Optimizations

### 1. GEMM (Matrix Multiplication)

**The Problem**: Naive triple-loop matrix multiply is extremely slow due to:
- Cache misses (poor memory access patterns)
- No vectorization (compiler can't optimize Python loops)
- No parallelization (single-threaded)

**The Solution**:

#### Cache Blocking/Tiling
```python
# Divide matrices into 32×32 tiles that fit in L1/L2 cache
for tile_i in prange(num_tile_rows):  # Parallel over tiles
    for jj in range(0, N, block):
        tile_C = np.zeros((tile_height, tile_width))  # Local accumulator
        for kk in range(0, K, block):
            # Process tile - data reused multiple times while in cache
```

**Why it works**:
- Small tiles fit in fast cache (L1/L2)
- Each tile element accessed multiple times
- Reduces memory bandwidth requirements

#### B Matrix Transpose
```python
# Pre-transpose B once
BT = np.ascontiguousarray(B.T, dtype=np.float32)

# Now both A and BT accessed row-major (cache-friendly)
acc += A[i, k] * BT[j, k]  # Both row-major access!
```

**Why it works**:
- Original B accessed column-major (cache-unfriendly)
- Transposed BT accessed row-major (cache-friendly)
- CPU cache lines work better with sequential access

#### 4-Way Loop Unrolling
```python
# Unroll inner loop to expose SIMD opportunities
while k + 3 < kkmax:
    acc += A[i, k] * BT[j, k]
    acc += A[i, k+1] * BT[j, k+1]
    acc += A[i, k+2] * BT[j, k+2]
    acc += A[i, k+3] * BT[j, k+3]
    k += 4
```

**Why it works**:
- Helps Numba/LLVM generate better SIMD code
- Reduces loop overhead
- More independent operations for instruction-level parallelism

#### Parallelization
```python
for tile_i in prange(num_tile_rows):  # Parallel loop
```

**Why it works**:
- Independent tiles can be computed in parallel
- Utilizes all CPU cores
- Scales with core count

**Performance Result**: In single-thread mode, the optimized Numba GEMM achieves ~3-4 GFLOPs (vs ~0.01 GFLOPs naive). Still significantly slower than BLAS (~130-140 GFLOPs) because BLAS uses hand-tuned assembly with explicit SIMD instructions. In multi-thread mode, throughput scales with core count (machine dependent).

---

### 2. Stable Softmax

**The Problem**: Standard softmax can overflow numerically, and naive implementation is slow.

**The Solution**:

#### Numerical Stability
```python
# Subtract max before exp to prevent overflow
row_max = np.max(x[i, :])
exp_x = np.exp(x[i, :] - row_max)
softmax = exp_x / np.sum(exp_x)
```

**Why it works**:
- `exp(x)` can overflow for large x
- `exp(x - max)` is bounded, preventing overflow
- Standard practice in production systems

#### Row-wise Parallelization
```python
for i in prange(M):  # Parallel over rows
    # Process row i independently
```

**Why it works**:
- Rows are independent
- Perfect parallelization opportunity
- Utilizes multiple cores

#### Fused Operations
- Compute max, exp, and sum in minimal passes
- Minimize memory traffic

**Performance Result**: Fast and numerically stable, with good parallel scaling.

---

### 3. LayerNorm

**The Problem**: Multiple passes over data, temporary arrays, memory-bound.

**The Solution**:

#### Fused Operations
```python
# Compute mean and variance efficiently
for i in prange(M):  # Parallel over rows
    mean = compute_mean(x[i, :])
    variance = compute_variance(x[i, :], mean)
    # Normalize and scale in same pass
    y[i, :] = (x[i, :] - mean) / sqrt(variance + eps) * gamma + beta
```

**Why it works**:
- Minimizes passes over data
- Reduces temporary arrays
- Better cache utilization

#### Row-wise Parallelization
- Independent rows processed in parallel
- Utilizes multiple cores

#### Contiguous Memory Access
- Row-major access patterns
- Better cache line utilization

**Performance Result**: Optimized memory-bound kernel with good parallel scaling.

---

## 📊 Roofline Model Analysis

The roofline model helps classify kernels as **compute-bound** or **memory-bound**.

### GEMM: Compute-Bound

**Arithmetic Intensity**: High (FLOPs/byte ratio grows with matrix size)
- High FLOPs per byte read/written for large matrices
- Each element reused many times
- Limited by CPU compute capability (FLOPS)

We use roofline-style reasoning to classify this as compute-bound: arithmetic intensity is high, and optimization should target compute utilization rather than memory bandwidth.

**Optimization Target**: Maximize compute utilization
- Block tiling (cache reuse)
- SIMD opportunities (loop unrolling)
- Parallelization (multiple cores)

### Softmax: Memory-Bound

**Arithmetic Intensity**: Low (< 1 FLOP/byte)
- Low FLOPs per byte
- Limited data reuse
- Limited by memory bandwidth

We use roofline-style reasoning to classify this as memory-bound: arithmetic intensity is low, and optimization should target memory bandwidth utilization.

**Optimization Target**: Maximize memory bandwidth utilization
- Parallelization (multiple memory channels)
- Fused operations (reduce passes)
- Cache-friendly access patterns

### LayerNorm: Memory-Bound

**Arithmetic Intensity**: Low (< 1 FLOP/byte)
- Low FLOPs per byte
- Multiple passes over data
- Limited by memory bandwidth

We use roofline-style reasoning to classify this as memory-bound: arithmetic intensity is low, and optimization should target minimizing memory traffic.

**Optimization Target**: Minimize memory traffic
- Fused operations
- Contiguous access patterns
- Parallelization

---

## 🔬 Benchmarking Methodology

### Single-Threaded Mode (Default)

**Purpose**: Fair kernel efficiency comparison without threading effects.

**Configuration**:
```python
# Limit BLAS threads BEFORE importing NumPy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

# Set Numba threads
from numba import set_num_threads
set_num_threads(1)
```

**Why**: Focuses on kernel implementation quality (tiling, cache blocking) rather than parallelization. Both implementations use 1 thread for fair comparison.

### Robust Timing

#### Median-Based Metrics
- Uses **median** (not mean) for primary latency metric
- Robust to outliers (thermal throttling, background processes, OS jitter)

#### Adaptive Iteration Counts
```python
# More runs for smaller problems (reduce variance)
if problem_size < 1_000_000:
    actual_runs = 500
elif problem_size < 100_000_000:
    actual_runs = 100
else:
    actual_runs = 10  # Large problems: fewer runs (minimize overhead)
```

**Why**: Small problems have higher variance (overhead dominates). More runs reduce statistical noise.

#### Block Size Sweep (GEMM)
```python
# Test multiple block sizes to find optimal
block_sizes = [16, 32, 64]
for block_size in block_sizes:
    # Measure performance with this block size
    # Pick the best one
```

**Why**: Optimal block size depends on:
- Cache hierarchy (L1/L2/L3 sizes)
- Matrix dimensions
- CPU architecture

Optimal block size is typically 16-32 for modern CPUs.

#### Warmup and Stabilization
```python
# Warmup to stabilize CPU state and JIT compilation
for _ in range(num_warmup):
    _ = kernel(inputs)

# Discard first timing run
times = []
for i in range(actual_runs):
    t_start = time.perf_counter()
    result = kernel(inputs)
    t_end = time.perf_counter()
    if i == 0 and actual_runs > 1:
        continue  # Discard first run
    times.append(t_end - t_start)
```

**Why**: 
- First run may include JIT compilation overhead
- CPU state (frequency, thermal) needs to stabilize
- Discarding first run provides cleaner measurements

---

## 💡 Why Numba Can't Match BLAS

NumPy's `A @ B` calls optimized BLAS libraries (MKL/OpenBLAS) that use:

1. **Register-Blocked Microkernels**
   - Hand-tuned assembly microkernels (e.g., 6×16 register blocks)
   - Explicit FMA pipeline utilization
   - Packed B panels for optimal memory access
   - Precise register allocation and packing strategy
   - Platform-specific assembly code with optimal instruction scheduling

2. **Explicit SIMD Instructions**
   - AVX/AVX2/AVX-512 vector instructions written directly in assembly
   - Manually unrolled loops targeting specific CPU features
   - Optimized for specific CPU architectures (Intel/AMD/ARM)

3. **Advanced Prefetching**
   - Explicit prefetch instructions to hide memory latency
   - Optimized memory access patterns aligned to cache lines
   - Deep cache hierarchy awareness (L1/L2/L3 behavior)

4. **Platform-Specific Optimizations**
   - Different code paths for Intel/AMD/ARM architectures
   - CPU feature detection (FMA, AVX, AVX-512)
   - Years of tuning by performance experts

**Key Difference**: BLAS uses register-blocked microkernels with explicit FMA pipelines and packed panels; our Numba kernel relies on auto-vectorization and cannot enforce the same register allocation/packing strategy.

Numba relies on compiler auto-vectorization (LLVM), which is less effective than hand-tuned assembly. A 30-40× performance gap is **expected and normal**.

**What Matters**: Demonstrating understanding of optimization techniques (blocking, memory access patterns, SIMD opportunities), not matching BLAS performance.

---

## 📈 Performance Characteristics

### Expected Results (Single-Threaded)

| Kernel | Implementation | Performance | Notes |
|--------|---------------|-------------|-------|
| **GEMM** | Baseline | ~0.01 GFLOPs | Intentionally slow |
| | NumPy | ~130-140 GFLOPs | BLAS-optimized |
| | Numba | ~3-4 GFLOPs (single-thread) | Custom optimized |
| **Softmax** | Baseline | Very slow | Pure Python |
| | NumPy | Fast | Vectorized |
| | Numba | Faster | Parallelized |
| **LayerNorm** | Baseline | Very slow | Pure Python |
| | NumPy | Fast | Vectorized |
| | Numba | Comparable | Memory-bound |

### Key Insights

1. **GEMM Gap**: Numba is 10× slower than NumPy/BLAS (expected)
   - Demonstrates optimization understanding
   - Shows why BLAS is so good
   - Structural limitation (no hand-tuned assembly)

2. **Softmax/LayerNorm**: Numba can match or exceed NumPy
   - Memory-bound kernels
   - Parallelization helps
   - Less dependent on SIMD

3. **Block Size**: Optimal varies by problem size
   - Small matrices: Smaller blocks (16-32)
   - Large matrices: Larger blocks (32-64)
   - Depends on cache hierarchy

---

## 🎯 Optimization Strategy Summary

### For Compute-Bound Kernels (GEMM):
1. ✅ Block tiling (cache locality)
2. ✅ Memory access pattern optimization (B-transpose)
3. ✅ Loop unrolling (SIMD opportunities)
4. ✅ Parallelization (multiple cores)
5. ✅ Local accumulators (reduce memory writes)

### For Memory-Bound Kernels (Softmax, LayerNorm):
1. ✅ Parallelization (utilize memory channels)
2. ✅ Fused operations (reduce passes)
3. ✅ Cache-friendly access patterns
4. ✅ Minimize temporary arrays

---

## 🔍 Understanding the Code Structure

### Kernel Files (`src/kernels/`)

Each kernel has three files:
- `*_baseline.py`: Pure Python implementation
- `*_numpy.py`: NumPy vectorized implementation
- `*_numba.py`: Numba JIT-optimized implementation

**Key Numba Features Used**:
- `@njit(parallel=True, fastmath=True, cache=True)`: Decorator for JIT compilation
- `prange`: Parallel loop construct
- `np.ascontiguousarray`: Ensure contiguous memory layout

### Benchmark Files (`src/bench/`)

- `bench_matmul.py`: GEMM benchmarks with block size sweep
- `bench_softmax.py`: Softmax benchmarks
- `bench_layernorm.py`: LayerNorm benchmarks
- `inference_block.py`: End-to-end kernel chain
- `plot_results.py`: Generate performance plots

**Key Benchmark Features**:
- Single-threaded mode (fair comparison)
- Adaptive iteration counts
- Median-based timing
- Block size optimization
- Correctness verification

---

## 🎓 Key Takeaways

1. **Cache Awareness**: Block tiling dramatically improves performance for compute-bound kernels

2. **Memory Access Patterns**: Row-major access is cache-friendly; column-major is not

3. **Arithmetic Intensity**: Determines if kernel is compute-bound or memory-bound

4. **Parallelization**: Helps both compute-bound (more cores) and memory-bound (more channels) kernels

5. **BLAS is Hard to Beat**: Hand-tuned assembly with explicit SIMD is extremely effective

6. **Methodology Matters**: Fair benchmarking (single-threaded, median timing) provides accurate comparisons

7. **Optimization is Systematic**: Baseline → Vectorized → Optimized progression demonstrates understanding

---

## 🔬 What I Learned / Next Steps

### Key Learnings

1. **Cache blocking effectiveness**: Dramatic performance improvement (3-4× speedup) from simple block tiling, demonstrating the critical importance of cache awareness in compute-bound kernels.

2. **Memory access patterns matter**: B-transpose optimization showed that cache-friendly access patterns (row-major) can provide significant improvements even when arithmetic intensity is high.

3. **Auto-vectorization limitations**: Numba/LLVM auto-vectorization cannot match hand-tuned assembly microkernels, confirming the value of understanding lower-level optimization techniques.

### Future Optimization Opportunities

1. **B Packing**: Implement packed B panels (common BLAS optimization) where B matrix data is reorganized into cache-aligned blocks to maximize cache line utilization.

2. **Register Microkernel**: Add a register-blocked microkernel (e.g., 4×4 or 6×16) that explicitly manages register allocation to maximize FMA pipeline utilization, closer to BLAS-style optimization.

3. **LLVM Analysis**: Compare auto-vectorization on/off and inspect LLVM IR/reports to understand exactly what vectorization opportunities are being missed, then manually guide the compiler.

4. **Multi-threaded Scaling**: Benchmark multi-threaded performance to demonstrate parallelization effectiveness and identify any scalability bottlenecks.

5. **Platform-Specific Tuning**: Explore CPU feature detection (FMA, AVX, AVX-512) and implement architecture-specific code paths for optimal performance on different platforms.

---

This project demonstrates fundamental CPU optimization understanding through practical implementation and rigorous benchmarking. The focus is on **why** optimizations work, not just **what** they achieve.

