# CoreForge-High-Performance-C-Kernel-Optimizations
CoreForge optimizes 10 C++ compute-intensive kernels. Under strict `-O2` limits, we achieved a 1.475x speedup using cache blocking, AoS/SoA conversion, and loop unrolling. By further utilizing AVX-512, OpenMP, and software prefetching, we reached a 12.100x speedup. This showcases hardware-aware software engineering.
