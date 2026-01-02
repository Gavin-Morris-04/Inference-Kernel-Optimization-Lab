# Inference Kernel Optimization Lab

A comprehensive CPU-first optimization study of three critical Transformer inference kernels using Python, NumPy, and Numba JIT compilation.

## 🎯 Project Overview

This project optimizes three kernels that appear in every Transformer inference path:
1. **GEMM (Matrix Multiply)** - Compute-bound, cache-sensitive
2. **Stable Softmax** - Memory-heavy, numerically sensitive
3. **LayerNorm** - Strongly memory-bound, perfect for fusion

Each kernel is implemented in three versions:
- **Baseline**: Pure Python loops (intentionally slow)
- **NumPy**: Vectorized operations
- **Numba**: JIT-compiled with optimizations (tiling, parallelization, fastmath)

### Why This Project Matters

AMD reviewers want to see fundamental performance understanding, not framework usage. This project demonstrates:
- **Cache awareness** through block tiling
- **SIMD thinking** through vectorization
- **Memory-bound optimization** through fused operations
- **Parallel computation** through thread-level parallelism
- **Roofline model analysis** to understand compute vs memory bottlenecks

## 🛠 Tech Stack

- **Python 3.8+**
- **NumPy** - Vectorized operations
- **Numba** - CPU JIT compilation with parallel support
- **pandas** - Results analysis
- **matplotlib** - Visualization
- **line_profiler** - Line-by-line profiling
- **cProfile** - Function-level profiling

**Explicitly NOT using**: CUDA, PyTorch, Triton, or any GPU dependencies.

## 📁 Repository Structure

```
inference-kernel-optimization-lab/
├── README.md
├── requirements.txt
├── src/
│   ├── kernels/
│   │   ├── matmul_baseline.py      # Pure Python GEMM
│   │   ├── matmul_numpy.py          # NumPy vectorized GEMM
│   │   ├── matmul_numba.py          # Numba JIT optimized GEMM
│   │   ├── softmax_baseline.py      # Pure Python softmax
│   │   ├── softmax_numpy.py         # NumPy vectorized softmax
│   │   ├── softmax_numba.py         # Numba JIT optimized softmax
│   │   ├── layernorm_baseline.py    # Pure Python LayerNorm
│   │   ├── layernorm_numpy.py       # NumPy vectorized LayerNorm
│   │   └── layernorm_numba.py       # Numba JIT optimized LayerNorm
│   ├── bench/
│   │   ├── bench_matmul.py          # GEMM benchmarking
│   │   ├── bench_softmax.py         # Softmax benchmarking
│   │   ├── bench_layernorm.py       # LayerNorm benchmarking
│   │   └── inference_block.py       # End-to-end inference chain
│   └── profiling/
│       ├── profile_baseline.py      # Baseline profiling
│       └── roofline_analysis.md     # Roofline model analysis
├── results/
│   ├── matmul_results.csv
│   ├── softmax_results.csv
│   ├── layernorm_results.csv
│   ├── inference_block_results.csv
│   └── plots/                       # Generated plots
└── scripts/
    ├── run_all_benchmarks.sh        # Linux/Mac runner
    └── run_all_benchmarks.bat       # Windows runner
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -U pip
pip install -r requirements.txt
```

### 2. Run Benchmarks

```bash
# Run all benchmarks (Linux/Mac)
bash scripts/run_all_benchmarks.sh

# Run all benchmarks (Windows)
scripts\run_all_benchmarks.bat

# Or run individual benchmarks
python src/bench/bench_matmul.py
python src/bench/bench_softmax.py
python src/bench/bench_layernorm.py
python src/bench/inference_block.py
```

### 3. Test Individual Kernels

```bash
# Test baseline implementations
python src/kernels/matmul_baseline.py
python src/kernels/softmax_baseline.py
python src/kernels/layernorm_baseline.py

# Test NumPy implementations
python src/kernels/matmul_numpy.py
python src/kernels/softmax_numpy.py
python src/kernels/layernorm_numpy.py

# Test Numba implementations (first run compiles JIT)
python src/kernels/matmul_numba.py
python src/kernels/softmax_numba.py
python src/kernels/layernorm_numba.py
```

## 🧩 Kernel Breakdown

### 1. GEMM (Matrix Multiply)

**Operation**: `C = A @ B` where A is (M×K), B is (K×N), C is (M×N)

**Baseline**: Triple nested loop in pure Python
- O(M×K×N) iterations
- No vectorization, no SIMD
- Extremely slow but correct

**NumPy**: Uses `@` operator
- BLAS-backed (often OpenBLAS or MKL)
- Highly optimized via vendor libraries
- Good baseline for comparison

**Numba Optimizations**:
- **Block Tiling (32×32)**: Divides matrices into cache-friendly tiles
- **Local Accumulators**: Keeps intermediate results in registers/L1 cache
- **Parallel Outer Loops**: `prange` for thread-level parallelism
- **FastMath**: Aggressive floating-point optimizations

**Key Insight**: High arithmetic intensity (O(N) FLOPs per byte for square matrices). This is a **compute-bound** kernel.

### 2. Stable Softmax

**Operation**: `softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`

**Baseline**: Pure Python loops with three passes
- Pass 1: Find max per row
- Pass 2: Compute exp(x - max) and accumulate sum
- Pass 3: Normalize by sum

**NumPy**: Vectorized row-wise operations
- `np.max(axis=1, keepdims=True)` for stability
- Broadcasting for efficient computation

**Numba Optimizations**:
- **Row-wise Parallelization**: `prange` over rows
- **Fused Operations**: Minimize memory passes
- **Numerical Stability**: Max subtraction prevents overflow
- **Cache-Friendly Access**: Contiguous row access patterns

**Key Insight**: Low arithmetic intensity (~0.33 FLOPs/byte). This is a **memory-bound** kernel with compute components.

### 3. LayerNorm

**Operation**: `(x - mean) / sqrt(variance + eps) * gamma + beta`

**Baseline**: Pure Python loops with three passes
- Pass 1: Compute mean per row
- Pass 2: Compute variance per row
- Pass 3: Normalize and scale

**NumPy**: Vectorized using `np.mean` and `np.var`
- Efficient reduction operations
- Broadcasting for gamma and beta

**Numba Optimizations**:
- **Fused Operations**: Minimize temporary arrays
- **Parallel Row Processing**: `prange` over rows
- **Contiguous Memory Access**: Row-major patterns
- **Avoid Multiple Passes**: Where possible, fuse operations

**Key Insight**: Low arithmetic intensity (~0.5 FLOPs/byte). This is a **strongly memory-bound** kernel.

## 📊 Optimization Strategies

### GEMM Optimizations

1. **Block Tiling (Cache Locality)**
   ```python
   # Divide into 32×32 tiles
   for tile_i in prange(num_tile_rows):
       for tile_j in range(num_tile_cols):
           # Process tile in L1/L2 cache
   ```
   - Maximizes cache reuse
   - Reduces memory traffic
   - Each tile element accessed multiple times

2. **Local Accumulators**
   ```python
   tile_C = np.zeros((tile_size, tile_size))
   # Accumulate in tile_C before writing to output
   ```
   - Minimizes writes to main memory
   - Keeps data in fast cache levels

3. **Parallelization**
   ```python
   for tile_i in prange(num_tile_rows):  # Parallel
   ```
   - Utilizes all CPU cores
   - Independent tiles can be computed in parallel

### Softmax Optimizations

1. **Row-wise Parallelization**
   ```python
   for i in prange(M):  # Parallel over rows
       # Process row i independently
   ```
   - Rows are independent
   - Perfect parallelization opportunity

2. **Numerical Stability**
   ```python
   row_max = max(x[i, :])  # Subtract max before exp
   exp_x = exp(x - row_max)  # Prevents overflow
   ```
   - Critical for numerical correctness
   - Standard practice in production systems

3. **Fused Operations**
   - Compute exp and sum in one pass
   - Normalize with inverse multiplication

### LayerNorm Optimizations

1. **Minimize Memory Passes**
   - Compute mean and variance efficiently
   - Fuse normalization and scaling

2. **Contiguous Access Patterns**
   - Row-major iteration
   - Cache line utilization

3. **Avoid Temporary Arrays**
   - Reuse memory locations
   - In-place operations where possible

## 📈 Benchmark Results

### Expected Performance Characteristics

After running benchmarks, you should see results like:

#### GEMM Performance
- **Baseline**: Extremely slow (only for small matrices)
- **NumPy**: Fast (BLAS-optimized)
- **Numba**: Comparable or faster than NumPy (depending on matrix size)

**Example Results** (for 1024×1024 matrices):
```
kernel    M      K      N     latency_ms    throughput_gflops
baseline  1024   1024   1024  (too slow)     -
numpy      1024   1024   1024  15.2           138.2
numba      1024   1024   1024  12.8           163.8
```

#### Softmax Performance
- **Baseline**: Slow for large inputs
- **NumPy**: Fast vectorized
- **Numba**: Often faster due to parallelization

**Example Results** (for batch_size=512, seq_len=2048):
```
kernel    batch_size  seq_len  latency_ms    throughput_mops
baseline  512         2048     (too slow)     -
numpy     512         2048     2.3            1823.5
numba     512         2048     1.1            3818.2
```

#### LayerNorm Performance
- **Baseline**: Slow for large inputs
- **NumPy**: Fast vectorized
- **Numba**: Competitive with NumPy

**Example Results** (for batch_size=256, hidden_dim=4096):
```
kernel    batch_size  hidden_dim  latency_ms    throughput_mops
baseline  256         4096        (too slow)     -
numpy     256         4096        1.2            6990.5
numba     256         4096        0.9            9320.7
```

*Note: Actual results depend on your CPU architecture and system configuration.*

## 🔍 Profiling Insights

### Baseline Profiling

Run profiling on baseline implementations:
```bash
python src/profiling/profile_baseline.py
```

**Expected Findings**:
- Most time spent in Python loop overhead
- Function call overhead dominates
- NumPy operations inside loops are slow

### Optimization Impact

**GEMM**:
- Baseline: ~99% time in Python loops
- Numba: ~95% time in compiled code, ~5% overhead

**Softmax**:
- Baseline: Multiple passes over data
- Numba: Fused operations, reduced memory traffic

**LayerNorm**:
- Baseline: Three separate passes
- Numba: Optimized passes, better cache behavior

## 🧮 Roofline Model Analysis

See `src/profiling/roofline_analysis.md` for detailed analysis.

### Summary

| Kernel | Arithmetic Intensity | Classification | Bottleneck |
|--------|---------------------|----------------|------------|
| GEMM (1024×1024) | ~170 FLOPs/byte | **Compute-bound** | CPU FLOPS |
| Softmax (512×2048) | ~0.33 FLOPs/byte | **Memory-bound** | Memory BW |
| LayerNorm (256×4096) | ~0.5 FLOPs/byte | **Memory-bound** | Memory BW |

### Key Insights

1. **GEMM** is compute-bound because:
   - High arithmetic intensity (grows with matrix size)
   - Data reuse (each element of A/B used multiple times)
   - Block tiling maximizes cache utilization
   - Optimizations target compute efficiency

2. **Softmax** is memory-bound because:
   - Low arithmetic intensity (< 1 FLOP/byte)
   - Limited data reuse
   - Expensive operations don't overcome memory limitation
   - Optimizations target memory bandwidth utilization

3. **LayerNorm** is memory-bound because:
   - Very low arithmetic intensity
   - Multiple passes over data
   - Limited computation per byte
   - Optimizations minimize memory traffic

## 🔗 End-to-End Inference Block

The `inference_block.py` chains kernels together:

```
Input → GEMM (Projection) → Softmax → LayerNorm → Output
```

### Amdahl's Law Analysis

If GEMM takes 70% of time with 10× speedup:
- Overall speedup = 1 / (0.3 + 0.7/10) = 3.7×

If all three kernels are optimized:
- Each contributes to overall speedup
- The slowest remaining component becomes the bottleneck

### Benchmarking

```bash
python src/bench/inference_block.py
```

**Expected Breakdown** (for typical transformer sizes):
- Projection (GEMM): 60-80% of total time
- Softmax: 10-20% of total time
- LayerNorm: 10-20% of total time

This demonstrates why GEMM optimization has the most impact on overall throughput.

## 📝 How to Reproduce Results

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Benchmarks**
   ```bash
   # All at once
   bash scripts/run_all_benchmarks.sh  # Linux/Mac
   scripts\run_all_benchmarks.bat      # Windows
   
   # Or individually
   python src/bench/bench_matmul.py
   python src/bench/bench_softmax.py
   python src/bench/bench_layernorm.py
   python src/bench/inference_block.py
   ```

3. **Check Results**
   - CSV files in `results/` directory
   - Contains latency, throughput, and percentiles

4. **Profile (Optional)**
   ```bash
   python src/profiling/profile_baseline.py
   ```

5. **View Roofline Analysis**
   - Read `src/profiling/roofline_analysis.md`

## 🎓 Key Learnings

1. **Cache Awareness**: Block tiling in GEMM shows massive performance gains
2. **Memory vs Compute**: Understanding arithmetic intensity is crucial
3. **Parallelization**: Not all kernels benefit equally from parallelization
4. **Numerical Stability**: Softmax max-subtraction is essential
5. **Amdahl's Law**: Optimizing the dominant kernel has the most impact

## 📚 References

- Numba Documentation: https://numba.readthedocs.io/
- Roofline Model: Williams, Waterman, Patterson (2009)
- Transformer Architecture: Vaswani et al. (2017)

## 📄 License

This project is provided as-is for educational and demonstration purposes.

---

**Built for AMD interview demonstration** - Showing fundamental performance understanding through CPU-first optimization.
