# 🚀 Inference Kernel Optimization Lab

**A hands-on CPU performance engineering project demonstrating fundamental optimization techniques for Transformer inference kernels.**

This project optimizes three critical kernels that power every Transformer model using **only Python, NumPy, and Numba** — no GPU, no CUDA, no PyTorch. Built to demonstrate deep understanding of cache behavior, memory access patterns, and compute optimization for AMD technical interviews.

---

## 🎯 What This Project Does

Transformers run on three core operations repeated millions of times:
1. **Matrix Multiplication (GEMM)** — Projecting embeddings through weight matrices
2. **Softmax** — Computing attention probabilities  
3. **LayerNorm** — Normalizing activations

Each kernel is implemented in **three versions**:
- 🐌 **Baseline**: Pure Python loops (intentionally slow, demonstrates the problem)
- ⚡ **NumPy**: Vectorized operations (what you'd write in production)
- 🔥 **Numba**: JIT-compiled with hand-tuned optimizations (cache blocking, parallelization, SIMD-friendly code)

**The Goal**: Understand *why* NumPy/BLAS is fast, and *how* to write kernel-level optimized code from scratch.

---

## 💡 Why This Matters (AMD Interview Perspective)

AMD reviewers want to see:
- ✅ **Cache awareness** (blocking, tiling, memory access patterns)
- ✅ **SIMD thinking** (vectorization, loop unrolling)
- ✅ **Memory-bound vs compute-bound** understanding (roofline model)
- ✅ **Systematic optimization** (benchmarking, profiling, analysis)

**Not** framework usage or GPU programming. This project demonstrates fundamental CPU optimization understanding.

---

## ⚡ Quick Start (3 Steps)

### Step 1: Setup Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Run Benchmarks

```powershell
# Run all benchmarks (recommended)
python src\bench\bench_matmul.py
python src\bench\bench_softmax.py
python src\bench\bench_layernorm.py

# Or use the automated script
scripts\run_all_benchmarks.bat
```

### Step 3: View Results

Results are saved as CSV files in `results/`:
- `matmul_results.csv` — GEMM performance data
- `softmax_results.csv` — Softmax performance data  
- `layernorm_results.csv` — LayerNorm performance data

View them in Excel, or generate plots:
```powershell
python src\bench\plot_results.py
```

---

## 📊 What You'll See

### Benchmark Output Example

```
======================================================================
GEMM Benchmark Suite - SINGLE-THREADED MODE
======================================================================
Mode: Single-threaded kernel efficiency comparison
  - NumPy BLAS: 1 thread (limited via env vars set before import)
  - Numba: 1 thread (set via set_num_threads)
  - Purpose: Fair kernel quality comparison (no threading effects)

Benchmarking GEMM: M=512, K=1024, N=512
  Testing baseline (pure Python)...
    Skipping baseline (problem too large)
  Testing NumPy (vectorized)...
  Testing Numba (JIT optimized)...
    Numba threads: 1
      block=16: median=46.234 ms, mean=47.123 ms, std=1.456 ms
      block=32: median=52.145 ms, mean=53.892 ms, std=2.123 ms
      block=64: median=58.234 ms, mean=59.876 ms, std=3.234 ms
    Optimal block size: 16

Results saved to: results/matmul_results.csv
```

### Expected Performance (Single-Threaded)

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

**Key Insight**: Numba won't beat NumPy/BLAS (that's expected!), but it demonstrates optimization understanding.

---

## 🧩 Project Structure

```
inference-kernel-optimization-lab/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── QUICKSTART.md               # Detailed quick start guide
│
├── src/
│   ├── kernels/                # Kernel implementations
│   │   ├── matmul_baseline.py      # Pure Python GEMM
│   │   ├── matmul_numpy.py         # NumPy GEMM (@ operator)
│   │   ├── matmul_numba.py         # Optimized Numba GEMM (tiled, parallel)
│   │   ├── softmax_baseline.py     # Pure Python softmax
│   │   ├── softmax_numpy.py        # NumPy softmax
│   │   ├── softmax_numba.py        # Parallel Numba softmax
│   │   ├── layernorm_baseline.py   # Pure Python LayerNorm
│   │   ├── layernorm_numpy.py      # NumPy LayerNorm
│   │   └── layernorm_numba.py      # Optimized Numba LayerNorm
│   │
│   ├── bench/                  # Benchmarking scripts
│   │   ├── bench_matmul.py         # GEMM benchmarks
│   │   ├── bench_softmax.py        # Softmax benchmarks
│   │   ├── bench_layernorm.py      # LayerNorm benchmarks
│   │   ├── inference_block.py      # End-to-end chain
│   │   └── plot_results.py         # Generate performance plots
│   │
│   └── profiling/              # Analysis tools
│       ├── profile_baseline.py     # Profile slow implementations
│       └── roofline_analysis.md    # Roofline model analysis
│
├── results/                    # Benchmark results (CSV files)
│   └── plots/                  # Generated performance plots
│
└── scripts/
    └── run_all_benchmarks.bat  # Run all benchmarks (Windows)
    └── run_all_benchmarks.sh   # Run all benchmarks (Linux/Mac)
```

---

## 🔍 Key Optimizations Demonstrated

### 1. GEMM (Matrix Multiply)

**The Problem**: Naive triple-loop matrix multiply is extremely slow (cache misses, no vectorization).

**The Solution**:
- ✅ **Block Tiling**: Divide matrices into cache-friendly tiles (32×32)
- ✅ **B Matrix Transpose**: Pre-transpose B so both matrices accessed row-major (cache-friendly)
- ✅ **Local Accumulators**: Keep tile results in L1 cache, write once
- ✅ **Parallelization**: Use `prange` for thread-level parallelism
- ✅ **4-Way Loop Unrolling**: Help Numba generate better SIMD code

**Result**: ~3-4 GFLOPs single-threaded (vs ~0.01 GFLOPs naive), still significantly slower than BLAS (~130-140 GFLOPs) but demonstrates optimization techniques.

### 2. Stable Softmax

**The Problem**: Standard softmax can overflow. Need numerical stability.

**The Solution**:
- ✅ **Max Subtraction**: Subtract row max before exp (prevents overflow)
- ✅ **Row-wise Parallelization**: Process rows independently in parallel
- ✅ **Fused Operations**: Minimize memory passes

**Result**: Fast, numerically stable, parallelized.

### 3. LayerNorm

**The Problem**: Multiple passes over data, temporary arrays.

**The Solution**:
- ✅ **Fused Operations**: Compute mean, variance, normalize in minimal passes
- ✅ **Contiguous Memory Access**: Row-major patterns for cache efficiency
- ✅ **Parallel Row Processing**: Independent rows computed in parallel

**Result**: Optimized memory-bound kernel.

---

## 📈 Benchmarking Methodology

### Single-Threaded Mode (Default)

**Purpose**: Fair kernel efficiency comparison without threading effects.

**Configuration**:
- NumPy BLAS: 1 thread (limited via environment variables)
- Numba: 1 thread (set via `set_num_threads(1)`)

**Why**: Focuses on kernel implementation quality (tiling, cache blocking) rather than parallelization.

**Run**:
```powershell
python src\bench\bench_matmul.py
```

### Robust Timing Methodology

- **Median-based**: Uses median (not mean) for primary latency metric — robust to outliers
- **Adequate iterations**: 
  - Small problems: 500-1000 runs (reduce variance)
  - Large problems: 5-10 runs (minimize overhead)
- **Block size sweep**: Tests multiple tile sizes (16, 32, 64), picks optimal
- **Warmup**: Sufficient warmup to stabilize CPU state and JIT compilation

---

## 🧮 Understanding the Results

### Roofline Model Analysis

See `HOW_IT_WORKS.md` for detailed roofline-style analysis.

**Key Classifications**:
- **GEMM**: **Compute-bound** (high arithmetic intensity, grows with matrix size)
- **Softmax**: **Memory-bound** (low arithmetic intensity, < 1 FLOP/byte)
- **LayerNorm**: **Memory-bound** (low arithmetic intensity, < 1 FLOP/byte)

### Why Numba Can't Match BLAS

NumPy's `A @ B` calls optimized BLAS libraries (MKL/OpenBLAS) that use:
- Hand-tuned assembly microkernels
- Explicit SIMD instructions (AVX/AVX2/AVX-512)
- Platform-specific optimizations
- Advanced prefetching

Numba relies on compiler auto-vectorization, which is less effective. A 30-40× performance gap is **expected and normal**.

**What Matters**: Demonstrating understanding of optimization techniques, not matching BLAS.

---

## 🎓 Learning Outcomes

By working through this project, you'll understand:

1. **Cache Behavior**
   - Why block tiling works
   - Memory access pattern optimization
   - Cache hierarchy awareness

2. **Memory vs Compute Bound**
   - Roofline model analysis
   - Arithmetic intensity calculation
   - Bottleneck identification

3. **Optimization Techniques**
   - Loop unrolling
   - Memory layout optimization (B-transpose)
   - Parallelization strategies

4. **Professional Benchmarking**
   - Fair comparisons (single-threaded mode)
   - Statistical rigor (median, percentiles)
   - Reproducible methodology

---

## 🛠 Advanced Usage

### Generate Performance Plots

```powershell
python src\bench\plot_results.py
```

Plots saved in `results/plots/`:
- Latency comparisons
- Throughput comparisons
- Time breakdowns

### Profile Baselines

```powershell
python src\profiling\profile_baseline.py
```

Shows where pure Python implementations spend time (spoiler: it's loops and function call overhead).

### Verbose Block Size Sweep

```powershell
python src\bench\bench_matmul.py --verbose-sweep
```

Shows detailed block size sweep results for all matrix sizes.

### Test Individual Kernels

```powershell
# Test correctness
python src\kernels\matmul_numba.py
python src\kernels\softmax_numba.py
python src\kernels\layernorm_numba.py
```

---

## 📋 Requirements

- **Python 3.8+**
- **NumPy** (>=1.24.0) — Vectorized operations
- **Numba** (>=0.58.0) — JIT compilation
- **pandas** (>=2.0.0) — Results analysis
- **matplotlib** (>=3.7.0) — Plotting
- **line_profiler** (>=4.1.0) — Profiling
- **py-cpuinfo** (>=9.0.0) — System info

Install all with:
```powershell
pip install -r requirements.txt
```

---

## 🔬 Technical Deep Dive

### Optimization Techniques Applied

#### Cache Blocking (GEMM)
```python
# Divide into 32×32 tiles
for tile_i in prange(num_tiles_i):
    for jj in range(0, N, block):
        tile_C = np.zeros((tile_height, tile_width))  # L1 cache
        for kk in range(0, K, block):
            # Compute tile contribution
            # Tile fits in cache, reused multiple times
```

#### B Matrix Transpose (GEMM)
```python
# Pre-transpose B once
BT = np.ascontiguousarray(B.T, dtype=np.float32)

# Now both A and BT accessed row-major (cache-friendly)
acc += A[i, k] * BT[j, k]  # Both row-major!
```

#### Loop Unrolling (GEMM)
```python
# 4-way unrolling for SIMD opportunities
while k + 3 < kkmax:
    acc += A[i, k] * BT[j, k]
    acc += A[i, k+1] * BT[j, k+1]
    acc += A[i, k+2] * BT[j, k+2]
    acc += A[i, k+3] * BT[j, k+3]
    k += 4
```

### Roofline Analysis

For detailed roofline model analysis, see `src/profiling/roofline_analysis.md`.

**Quick Summary**:
- GEMM: Compute-bound (high FLOPs/byte) → Optimize compute
- Softmax: Memory-bound (low FLOPs/byte) → Optimize memory access
- LayerNorm: Memory-bound (low FLOPs/byte) → Optimize memory access

---

## 📚 Documentation

- **QUICKSTART.md** — Detailed step-by-step guide
- **OPTIMIZATION_NOTES.md** — Optimization techniques explained
- **BENCHMARK_MODES.md** — Single-threaded vs multi-threaded modes
- **KERNEL_PERFORMANCE_ANALYSIS.md** — Performance characteristics
- **TIMING_METHODOLOGY_FIXES.md** — Benchmarking methodology

---

## 🎯 What This Demonstrates (For AMD Review)

### ✅ Systematic Optimization Approach
- Baseline → NumPy → Numba progression
- Clear understanding of each optimization's impact
- Block size tuning and parameter optimization

### ✅ Understanding of Fundamental Concepts
- Cache behavior and blocking
- Memory access patterns
- Compute vs memory bound kernels
- SIMD and vectorization

### ✅ Professional Engineering Practices
- Fair benchmarking methodology
- Statistical rigor (median, percentiles)
- Reproducible results
- Clear documentation

### ✅ Awareness of Limitations
- Understands why BLAS is faster
- Recognizes structural ceilings
- Can explain performance characteristics
- Knows what would be needed to match BLAS (hand-tuned SIMD)

---

## 🚦 Running Your First Benchmark

**The absolute fastest way to see it in action:**

```powershell
# 1. Setup (one time)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Run one benchmark
python src\bench\bench_matmul.py

# 3. Check results
# Open results\matmul_results.csv in Excel, or:
python src\bench\plot_results.py
```

**That's it!** You'll see:
- Baseline performance (slow)
- NumPy performance (fast, BLAS-backed)
- Numba performance (optimized, demonstrates techniques)
- Block size optimization in action

---

## 💻 System Requirements

- **OS**: Windows, Linux, or macOS
- **CPU**: Multi-core recommended (Numba uses parallelization)
- **RAM**: 4GB+ (for large matrix benchmarks)
- **Python**: 3.8 or higher

**Note**: This is CPU-only. No GPU required or used.

---

## 🐛 Troubleshooting

### Import Errors
```powershell
# Make sure you're in the project root directory
cd "path\to\inference-kernel-optimization-lab"
python src\bench\bench_matmul.py
```

### Numba Compilation Slow
- **Normal!** First run compiles JIT code (takes a few seconds)
- Subsequent runs are fast (compiled code cached)

### Performance Varies Between Runs
- **Expected**: CPU frequency scaling, thermal throttling, background processes
- **Solution**: Benchmark uses median (robust to outliers)
- Run when system is idle for most consistent results

### "No module named 'src'"
- Make sure you're running from the project root
- Check that `src/` directory exists

---

## 📊 Example Results Interpretation

### GEMM Results

```
kernel    M      K      N     latency_ms    throughput_gflops    block_size
baseline  64    128     64    82.006         0.013               -
numpy     64    128     64     0.036        28.862               -
numba     64    128     64     0.305         3.475               16
```

**Reading this**:
- Baseline is **6,400× slower** than NumPy (demonstrates the problem)
- NumPy is **9× faster** than Numba (BLAS is highly optimized)
- Numba shows optimization understanding (blocking, B-transpose)
- Block size 16 is optimal for this size

---

## 🎓 For Interview Discussions

When discussing this project, emphasize:

1. **Why optimizations work** (cache behavior, memory access)
2. **Systematic approach** (baseline → vectorized → optimized)
3. **Understanding of limits** (why BLAS is faster, what would be needed)
4. **Methodology** (fair benchmarking, statistical rigor)

**Key Talking Points**:
- "B-transpose optimization improved cache behavior by converting column-major to row-major access"
- "Block size selection depends on cache hierarchy and matrix dimensions"
- "Median-based timing provides robust measurements despite system jitter"
- "Single-threaded mode eliminates threading as a variable, focusing on kernel quality"

---

## 📖 Further Reading

- **Roofline Model**: Williams, Waterman, Patterson (2009) — "Roofline: An Insightful Visual Performance Model"
- **Numba Documentation**: https://numba.readthedocs.io/
- **BLAS Optimization**: GotoBLAS, OpenBLAS papers on cache blocking

---

## 🏆 Project Status

✅ **Complete and Production-Ready**

- All three kernels implemented (baseline, NumPy, Numba)
- Comprehensive benchmarking infrastructure
- Profiling and analysis tools
- Roofline model analysis
- End-to-end inference block
- Complete documentation

**Ready for technical review and interview discussion.**

---

## 📄 License

This project is provided as-is for educational and demonstration purposes.

---

**Built for AMD interview demonstration** — Showing fundamental performance understanding through CPU-first optimization. 🚀
