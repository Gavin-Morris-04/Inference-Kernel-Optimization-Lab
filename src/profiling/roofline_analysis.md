# Roofline Model Analysis

This document provides a roofline-style analysis of the three optimized kernels.

## Roofline Model Overview

The roofline model relates performance (operations per second) to arithmetic intensity (operations per byte). Key components:

- **Peak Compute**: Maximum FLOPS achievable by the processor
- **Peak Bandwidth**: Maximum memory bandwidth
- **Roofline**: min(Peak Compute, Peak Bandwidth × Arithmetic Intensity)

## GEMM (Matrix Multiply)

### Operation Count
For matrices A (M×K), B (K×N), output C (M×N):
- **FLOPs**: 2 × M × K × N (one multiply and one add per output element)
- **Memory Access**:
  - Read A: M × K × 4 bytes (float32)
  - Read B: K × N × 4 bytes (float32)
  - Write C: M × N × 4 bytes (float32)
  - Total: 4 × (M×K + K×N + M×N) bytes

### Arithmetic Intensity
```
AI = FLOPs / Bytes = (2 × M × K × N) / (4 × (M×K + K×N + M×N))
```

For square matrices where M = N = K = N_dim:
```
AI = (2 × N_dim³) / (4 × 3 × N_dim²) = N_dim / 6
```

For N_dim = 1024: AI ≈ 170.7 FLOPs/byte

### Classification
**Compute-bound** for large matrices.

Reasoning:
- High arithmetic intensity (grows linearly with matrix dimension)
- Each element of A and B is reused multiple times (K times for A, M times for B)
- With proper blocking/tiling, cache locality is excellent
- Modern CPUs can achieve high compute utilization with SIMD

### Optimization Strategies Applied
1. **Block Tiling (32×32 tiles)**: Maximizes cache reuse, reduces memory traffic
2. **Local Accumulators**: Keeps intermediate results in registers/L1 cache
3. **Parallelization**: Exploits multiple cores for outer loops
4. **FastMath**: Enables aggressive floating-point optimizations

### Expected Performance Characteristics
- Large matrices: Limited by compute capability (FLOPS)
- Small matrices: May be limited by overhead and memory bandwidth
- Numba JIT: Should approach ~50-80% of peak FLOPS on modern CPUs

---

## Stable Softmax

### Operation Count
For input x (batch_size × seq_len):
- **Operations per element**:
  - Find max: ~1 comparison per element
  - Compute exp(x - max): 1 operation
  - Sum: 1 accumulation
  - Normalize: 1 division
- **Total Ops**: ~4 × batch_size × seq_len

### Memory Access
- Read x: batch_size × seq_len × 4 bytes
- Write output: batch_size × seq_len × 4 bytes
- Temporary arrays (exp values): batch_size × seq_len × 4 bytes
- Total: ~3 × batch_size × seq_len × 4 bytes

### Arithmetic Intensity
```
AI = (4 × batch_size × seq_len) / (3 × batch_size × seq_len × 4) = 1/3 FLOPs/byte
```
**AI ≈ 0.33 FLOPs/byte**

### Classification
**Memory-bound** with compute components.

Reasoning:
- Very low arithmetic intensity (< 1 FLOP/byte)
- Each element is read once, written once (or twice with temporary)
- Limited data reuse
- Expensive transcendental function (exp) but still dominated by memory

### Optimization Strategies Applied
1. **Row-wise Parallelization**: Processes independent rows in parallel
2. **Fused Operations**: Minimizes passes over memory
3. **Numerical Stability**: Max subtraction reduces numerical errors
4. **Cache-Friendly Access**: Row-major access patterns

### Expected Performance Characteristics
- Limited by memory bandwidth
- Throughput should scale with memory BW, not compute
- Parallelization helps by utilizing memory channels across cores
- Numba JIT: Should achieve ~60-90% of peak memory bandwidth

---

## LayerNorm

### Operation Count
For input x (batch_size × hidden_dim):
- **Per element operations**:
  - Compute mean: 1 accumulation per element
  - Compute variance: 2 ops (subtract mean, square) per element
  - Normalize: 2 ops (subtract mean, divide by std) per element
  - Scale and shift: 2 ops (multiply by gamma, add beta) per element
- **Total Ops**: ~8 × batch_size × hidden_dim

### Memory Access
- Read x: batch_size × hidden_dim × 4 bytes
- Read gamma: hidden_dim × 4 bytes
- Read beta: hidden_dim × 4 bytes
- Write output: batch_size × hidden_dim × 4 bytes
- Temporary (mean, variance per row): batch_size × 2 × 4 bytes
- Total: ~(4 × batch_size × hidden_dim + 2 × hidden_dim + 2 × batch_size) × 4 bytes

For large hidden_dim:
```
≈ 4 × batch_size × hidden_dim × 4 bytes
```

### Arithmetic Intensity
```
AI = (8 × batch_size × hidden_dim) / (4 × batch_size × hidden_dim × 4) = 0.5 FLOPs/byte
```
**AI ≈ 0.5 FLOPs/byte**

### Classification
**Memory-bound**.

Reasoning:
- Low arithmetic intensity (< 1 FLOP/byte)
- Each element read once, written once
- Multiple passes over data (mean, variance, normalize)
- Reduction operations don't reuse data effectively

### Optimization Strategies Applied
1. **Fused Operations**: Minimizes memory passes where possible
2. **Row-wise Parallelization**: Processes independent rows in parallel
3. **Contiguous Memory Access**: Row-major access patterns
4. **Avoid Temporary Arrays**: Reuses memory locations

### Expected Performance Characteristics
- Strongly limited by memory bandwidth
- Throughput should scale with memory BW
- Multiple passes are necessary but should be cache-friendly
- Numba JIT: Should achieve ~70-90% of peak memory bandwidth

---

## Summary Table

| Kernel | FLOPs | Bytes | Arithmetic Intensity | Classification | Bottleneck |
|--------|-------|-------|---------------------|----------------|------------|
| GEMM (1024×1024) | 2.1B | 12MB | ~170 FLOPs/byte | Compute-bound | CPU FLOPS |
| Softmax (512×2048) | 4.2M | 12MB | ~0.33 FLOPs/byte | Memory-bound | Memory BW |
| LayerNorm (256×4096) | 8.4M | 4MB | ~0.5 FLOPs/byte | Memory-bound | Memory BW |

## Why Optimizations Worked

### GEMM Optimizations
- **Tiling**: Reduces cache misses by working on data that fits in L1/L2 cache
- **Local Accumulators**: Minimizes memory writes (write once per tile instead of K times)
- **Parallelization**: Utilizes multiple cores for independent tile computations
- Result: Approaches compute roofline (FLOPs-limited)

### Softmax Optimizations
- **Parallelization**: Utilizes multiple memory channels across cores
- **Fused Operations**: Reduces memory passes from 4+ to 2-3
- **Row-wise Processing**: Better cache locality than column-wise
- Result: Approaches memory bandwidth roofline

### LayerNorm Optimizations
- **Fused Operations**: Minimizes temporary arrays and memory passes
- **Parallelization**: Distributes memory load across cores
- **Contiguous Access**: Maximizes cache line utilization
- Result: Approaches memory bandwidth roofline

## Amdahl's Law Considerations

For an end-to-end inference block:
- If GEMM is 70% of time and gets 10× speedup → overall 3.7×
- If Softmax is 15% of time and gets 5× speedup → overall 1.6×
- If LayerNorm is 15% of time and gets 5× speedup → overall 1.6×
- Combined: Limited by the slowest remaining component

**Key Insight**: Optimizing all kernels is necessary, but the compute-bound GEMM optimization has the most impact on overall throughput.

