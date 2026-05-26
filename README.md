```markdown
# CoreForge

**CoreForge** is a high-performance C++ low-level optimization project focused on pushing compute-intensive kernels toward hardware limits. Under the strictly constrained global `-O2` compilation setting, the project deeply optimized 10 classic kernels through microarchitecture-aware analysis, memory-layout refactoring, and instruction-level tuning.

In the basic stage, CoreForge achieved a **1.475× geometric mean speedup**. In the Bonus stage, after compiler restrictions were lifted, the project introduced aggressive hardware-software co-optimization and ultimately reached a **12.100× extreme speedup**, while passing all correctness validations.

All performance testing and bottleneck analysis were conducted on the course server equipped with an **Intel Xeon Silver 4210R processor** based on the **Cascade Lake architecture**, featuring **40 cores**. The Linux `perf` tool was used to analyze cache behavior, instruction throughput, and performance bottlenecks.

---

## Highlights

- Optimized **10 compute-intensive C++ kernels** under strict `-O2` compilation constraints.
- Achieved **1.475× geometric mean speedup** in the basic stage.
- Achieved **12.100× extreme speedup** in the Bonus stage.
- Reached up to **165× peak acceleration** on matrix multiplication.
- Applied cache-aware memory optimization, SIMD vectorization, instruction-level parallelism, and multi-core parallelization.
- Passed all rigorous correctness validations.

---

## Optimization Overview

### 1. Memory Hierarchy Optimization

The core of high-performance optimization often lies in precise control over the memory hierarchy. In this project, several bottlenecks were resolved by reshaping data layouts and improving cache locality.

#### Filter Gradient

The original **SoA** layout was transformed into an **AoS** layout, allowing the RGB channels of the same pixel to be loaded together through a single L1 cache access.

This eliminated unnecessary cross-array memory accesses and significantly improved cache locality.

#### Graph Traversal

The original linked-list graph representation caused severe pointer chasing and poor cache performance. To address this, the graph structure was refactored into a compact one-dimensional **CSR** format.

This enabled continuous memory traversal, activated the hardware prefetcher, and combined with parallel reduction, improved graph traversal performance by up to **88×**.

#### Matrix Multiplication

For matrix multiplication, the memory access pattern was optimized through:

- `IKJ` loop reordering
- `64 × 64` cache blocking
- L1-cache-aware data reuse

This transformed inefficient column-major-like jumping accesses into continuous row-major accesses, allowing the core computation to remain inside the fast cache path.

---

## Arithmetic Throughput Optimization

Beyond feeding the CPU efficiently, CoreForge also improved arithmetic throughput through algorithmic simplification and instruction-level refactoring.

### Trace Replay

Rolling-hash computation introduced severe loop-carried dependencies. CoreForge applied **algebraic unrolling** to break the dependency chain and expose more instruction-level parallelism.

### Bitwise Kernels

For massive bitwise operations, the project adopted:

- 64-bit batch packing
- SWAR-style parallelism
- Boolean algebra simplification
- Branchless execution

This reduced complex conditional logic into pure bitwise operations and lowered the core instruction count to nearly one-eighth of the original version.

### Black-Scholes

For the Black-Scholes option pricing kernel, which involves expensive mathematical function calls, CoreForge used financial-mathematical simplification, pointer prefetching, and manual 4-way loop unrolling.

These optimizations reduced function-call overhead and improved instruction throughput.

---

## Extreme Optimization Stage

In the Bonus stage, CoreForge fully enabled hardware-level vectorization and thread-level parallelism.

### AVX-512 Vectorization

Hand-written AVX-512 instructions were introduced to exploit wide-vector computation. In matrix multiplication, an `8 × 16` register micro-kernel was combined with FMA instructions, pushing peak performance to an astonishing **165× speedup**.

### OpenMP Parallelization

OpenMP was used to schedule work across the 40-core Cascade Lake server. This significantly improved performance for compute-heavy kernels with sufficient parallel workload.

However, the project also revealed an important performance-engineering lesson: more parallelism does not always mean better performance.

For example, the simple ReLU activation kernel experienced nearly **50% performance regression** under high concurrency. Since ReLU is extremely memory-bound, waking up 40 cores to process tiny data blocks introduced thread scheduling overhead and memory bus contention that outweighed the benefits of parallel execution.

This confirmed that for lightweight memory-bound tasks, single-core scalar execution combined with SIMD can already approach the physical performance limit.

---

## Key Techniques

- L1 cache blocking
- AoS / SoA memory layout transformation
- CSR graph compression
- Cache-line-aware data alignment
- Loop reordering
- Algebraic loop unrolling
- Branchless design
- SWAR optimization
- Boolean algebra simplification
- Pointer prefetching
- Manual loop unrolling
- AVX-512 SIMD vectorization
- FMA fused multiply-add
- OpenMP multi-core scheduling
- Linux `perf` bottleneck analysis

---

## Performance Summary

| Stage | Optimization Scope | Result |
|---|---|---|
| Basic Stage | Global `-O2`, source-level optimization | **1.475× geometric mean speedup** |
| Bonus Stage | Hardware-software co-optimization | **12.100× extreme speedup** |
| Graph Traversal | CSR + parallel reduction | **88× speedup** |
| Matrix Multiplication | AVX-512 + `8 × 16` micro-kernel + FMA | **165× peak speedup** |

---

## Conclusion

CoreForge demonstrates that even under conservative compilation constraints, program performance is far from reaching its ceiling.

By treating C++ source code as a direct bridge to the underlying hardware, carefully managing cache-line boundaries, breaking pipeline dependencies, eliminating unnecessary branches, and respecting memory-bandwidth limits, developers can extract performance close to hardware limits through deliberate low-level engineering.

This project reflects the real value of hardware-aware software optimization: performance is not achieved by blindly adding computation power, but by understanding how every instruction, cache line, and memory access interacts with the processor.
```
