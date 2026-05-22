#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <print>
#include <vector>

#include "bench.h"
#include "bitwise.h"
#include "blackscholes.h"
#include "filter_gradient.h"
#include "graph.h"
#include "grff.h"
#include "image_proc.h"
#include "matmul.h"
#include "relu.h"
#include "sparse_spmm.h"
#include "trace_replay.h"

#define GEOMETRIC_MEAN 1

int main() {
    std::cout << "Benchmark setup\n";
    std::cout << "\nRunning Benchmarks...\n";
    std::cout
        << "------------------------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(25) << "Benchmark" << std::setw(12)
              << "Status" << std::right << std::setw(18) << "Naive (ns)"
              << std::setw(18) << "Stu (ns)" << std::setw(14) << "vs Naive"
              << std::setw(16) << "vs Baseline" << '\n';
    std::cout
        << "------------------------------------------------------------------------------------------------\n";

    constexpr int k_best = 20;
    std::uint32_t seed = 12345u;

#if GEOMETRIC_MEAN
    std::vector<double> baseline_speedups;
    baseline_speedups.reserve(10);
    bool gm_enable = true;
#endif

    const auto measure_average_time = [&](lab_test_func func,
                                          void *ctx,
                                          const std::string &label) {
        std::chrono::nanoseconds avg_time{0};
        for (int i = 0; i < k_best; ++i) {
            flush_cache();
            const auto elapsed = measure_time([&] { func(ctx); });
            avg_time += elapsed;
            debug_log("DEBUG: {} {}-th measurement: {} ns\n",
                      label,
                      i,
                      static_cast<std::uint64_t>(elapsed.count()));
        }
        return avg_time / static_cast<std::uint64_t>(k_best);
    };

    const auto run_benchmark = [&](const bench_t &bench) {
        const auto naive_time = measure_average_time(
            bench.naiveFunc, bench.ref_args, bench.description + " naive");

        std::cout << std::left << std::setw(25) << bench.description;

        if (bench.tfunc == nullptr || bench.args == nullptr) {
            std::cout << "\033[1;33mTODO\033[0m" << std::right
                      << std::setw(18) << naive_time.count() << std::setw(18)
                      << "N/A" << std::setw(14) << "N/A" << std::setw(16)
                      << "N/A" << '\n';
            return;
        }

        const auto stu_time = measure_average_time(
            bench.tfunc, bench.args, bench.description + " stu");
        const bool correct =
            bench.checkFunc(bench.args, bench.ref_args, bench.naiveFunc);

        if (!correct) {
            std::cout << "\033[1;31mFAILED\033[0m" << std::right
                      << std::setw(18) << naive_time.count() << std::setw(18)
                      << stu_time.count() << std::setw(14) << "N/A"
                      << std::setw(16) << "N/A" << '\n';
            std::cout
                << "  Error: Results do not match naive implementation!\n";
#if GEOMETRIC_MEAN
            gm_enable = false;
#endif
            return;
        }

        const double naive_speedup = calculate_speedup(stu_time, naive_time);
        const double baseline_speedup = calculate_speedup(bench, stu_time);
        std::cout << "\033[1;32mPASSED\033[0m" << std::right << std::setw(18)
                  << naive_time.count() << std::setw(18) << stu_time.count()
                  << std::setw(13) << std::fixed << std::setprecision(3)
                  << naive_speedup << "x" << std::setw(15)
                  << baseline_speedup << "x";
        if (naive_speedup < bench.naive_speedup_lower_bound) {
            std::cout << " (SLOWER)";
        }
        std::cout << '\n';
        std::cout.unsetf(std::ios::floatfield);
        std::cout << std::setprecision(6);

#if GEOMETRIC_MEAN
        baseline_speedups.push_back(baseline_speedup);
#endif
    };

    {
        blackscholes_args black_args_ref;
        initialize_blackscholes(black_args_ref, 81920, seed);
        std::cout << "Black-Scholes options: "
                  << black_args_ref.spot_price.size() << '\n';

        blackscholes_args black_args_stu;
        initialize_blackscholes(black_args_stu, 81920, seed);
        run_benchmark({"Black-Scholes",
                       stu_BlkSchls_wrapper,
                       naive_BlkSchls_wrapper,
                       BlkSchls_check,
                       &black_args_stu,
                       &black_args_ref,
                       BASELINE_BLACKSCHOLES,
                       NAIVE_SPEEDUP_LOWER_BOUND_BLACKSCHOLES});
    }

    {
        sparse_spmm_args sparse_args_ref;
        initialize_spmm(sparse_args_ref, 512, 512, -1, {}, seed);
        std::cout << "Sparse A (CSR): " << sparse_args_ref.csr.rows << " x "
                  << sparse_args_ref.csr.cols
                  << ", nnz=" << sparse_args_ref.csr.values.size() << '\n';

        sparse_spmm_args sparse_args_stu;
        initialize_spmm(sparse_args_stu, 512, 512, -1, {}, seed);
        run_benchmark({"Sparse SpMM",
                       stu_sparse_spmm_wrapper,
                       naive_sparse_spmm_wrapper,
                       sparse_spmm_check,
                       &sparse_args_stu,
                       &sparse_args_ref,
                       BASELINE_SPARSE_SPMM,
                       NAIVE_SPEEDUP_LOWER_BOUND_SPARSE_SPMM});
    }

    {
        constexpr size_t relu_size = 1024000;
        relu_args relu_args_ref;
        initialize_relu(&relu_args_ref, relu_size, seed);
        std::println("ReLU: vector length={}", relu_size);

        relu_args relu_args_stu;
        initialize_relu(&relu_args_stu, relu_size, seed);
        run_benchmark({"ReLU",
                       stu_relu_wrapper,
                       naive_relu_wrapper,
                       relu_check,
                       &relu_args_stu,
                       &relu_args_ref,
                       BASELINE_RELU,
                       NAIVE_SPEEDUP_LOWER_BOUND_RELU});
    }

    {
        constexpr size_t bitwise_size = 1024000;
        bitwise_args bitwise_args_ref;
        initialize_bitwise(&bitwise_args_ref, bitwise_size, seed);
        std::println("Bitwise: vector length={}", bitwise_size);

        bitwise_args bitwise_args_stu;
        initialize_bitwise(&bitwise_args_stu, bitwise_size, seed);
        run_benchmark({"Bitwise",
                       stu_bitwise_wrapper,
                       naive_bitwise_wrapper,
                       bitwise_check,
                       &bitwise_args_stu,
                       &bitwise_args_ref,
                       BASELINE_BITWISE,
                       NAIVE_SPEEDUP_LOWER_BOUND_BITWISE});
    }

    {
        matmul_args matmul_args_ref;
        initialize_matmul(matmul_args_ref, 512, seed);
        std::cout << "MatMul: n=" << matmul_args_ref.n << '\n';

        matmul_args matmul_args_stu;
        initialize_matmul(matmul_args_stu, 512, seed);
        run_benchmark({"MatMul",
                       stu_matmul_wrapper,
                       naive_matmul_wrapper,
                       matmul_check,
                       &matmul_args_stu,
                       &matmul_args_ref,
                       BASELINE_MATMUL,
                       NAIVE_SPEEDUP_LOWER_BOUND_MATMUL});
    }

    {
        trace_replay_args trace_args_ref;
        initialize_trace_replay(trace_args_ref, 1 << 16, 1 << 20, seed);
        std::cout << "Trace Replay: records=" << trace_args_ref.records.size()
                  << ", trace_length=" << trace_args_ref.trace.size() << '\n';

        trace_replay_args trace_args_stu;
        initialize_trace_replay(trace_args_stu, 1 << 16, 1 << 20, seed);
        run_benchmark({"Trace Replay",
                       stu_trace_replay_wrapper,
                       naive_trace_replay_wrapper,
                       trace_replay_check,
                       &trace_args_stu,
                       &trace_args_ref,
                       BASELINE_TRACE_REPLAY,
                       NAIVE_SPEEDUP_LOWER_BOUND_TRACE_REPLAY});
    }

    {
        constexpr std::size_t graph_node_count = 1024000;
        constexpr int graph_avg_degree = 8;
        graph_args graph_args_ref;
        initialize_graph(&graph_args_ref,
                         graph_node_count,
                         graph_avg_degree,
                         seed);
        std::cout << "Graph: node_count=" << graph_node_count
                  << ", avg_degree=" << graph_avg_degree << '\n';

        graph_args graph_args_stu;
        initialize_graph(&graph_args_stu, graph_node_count, graph_avg_degree, seed);
        run_benchmark({"Graph",
                       stu_graph_wrapper,
                       naive_graph_wrapper,
                       graph_check,
                       &graph_args_stu,
                       &graph_args_ref,
                       BASELINE_GRAPH,
                       NAIVE_SPEEDUP_LOWER_BOUND_GRAPH});
    }

    {
        constexpr std::size_t grff_size = 1024000;
        grff_args grff_args_ref;
        initialize_grff(&grff_args_ref, grff_size, seed);
        std::cout << "GRFF: feature size=" << grff_args_ref.a_features.size()
                  << '\n';

        grff_args grff_args_stu;
        initialize_grff(&grff_args_stu, grff_size, seed);
        run_benchmark({"GRFF",
                       stu_grff_wrapper,
                       naive_grff_wrapper,
                       grff_check,
                       &grff_args_stu,
                       &grff_args_ref,
                       BASELINE_GRFF,
                       NAIVE_SPEEDUP_LOWER_BOUND_GRFF});
    }

    {
        constexpr std::size_t image_width = 1024;
        constexpr std::size_t image_height = 1000;
        image_proc_args image_args_ref;
        initialize_image_proc(&image_args_ref, image_width, image_height, seed);
        std::cout << "Image Proc: " << image_args_ref.width << " x "
                  << image_args_ref.height << '\n';

        image_proc_args image_args_stu;
        initialize_image_proc(&image_args_stu, image_width, image_height, seed);
        run_benchmark({"Image Proc",
                       stu_image_proc_wrapper,
                       naive_image_proc_wrapper,
                       image_proc_check,
                       &image_args_stu,
                       &image_args_ref,
                       BASELINE_IMAGE_PROC,
                       NAIVE_SPEEDUP_LOWER_BOUND_IMAGE_PROC});
    }

    {
        const std::size_t width = 1024;
        const std::size_t height = 1024;
        filter_gradient_args filter_gradient_args_ref;
        initialize_filter_gradient(&filter_gradient_args_ref,
                                   width,
                                   height,
                                   seed);
        std::cout << "Filter Gradient: " << height << " x " << width << '\n';

        filter_gradient_args filter_gradient_args_stu;
        initialize_filter_gradient(&filter_gradient_args_stu, width, height, seed);
        run_benchmark({"Filter Gradient",
                       stu_filter_gradient_wrapper,
                       naive_filter_gradient_wrapper,
                       filter_gradient_check,
                       &filter_gradient_args_stu,
                       &filter_gradient_args_ref,
                       BASELINE_FILTER_GRADIENT,
                       NAIVE_SPEEDUP_LOWER_BOUND_FILTER_GRADIENT});
    }

#if GEOMETRIC_MEAN
    if (gm_enable && !baseline_speedups.empty()) {
        const double geometric_mean_speedup =
            calculate_geometric_mean_speedup(baseline_speedups);
        std::println("\nGeometric mean speedup: {:.3f}x",
                     geometric_mean_speedup);
    } else {
        std::println("\nGeometric mean speedup: N/A");
    }
#endif

    return 0;
}