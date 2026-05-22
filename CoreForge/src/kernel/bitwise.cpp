#include "bitwise.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>

void initialize_bitwise(bitwise_args *args, const size_t size,
                                  const std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    constexpr std::int8_t LOWER_BOUND = std::numeric_limits<std::int8_t>::min();
    constexpr std::int8_t UPPER_BOUND = std::numeric_limits<std::int8_t>::max();

    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<int> dist(LOWER_BOUND, UPPER_BOUND);

    args->a.resize(size);
    args->b.resize(size);
    args->result.resize(size);

    for (std::size_t i = 0; i < size; ++i) {
        args->a[i] = static_cast<std::int8_t>(dist(gen));
        args->b[i] = static_cast<std::int8_t>(dist(gen));
        args->result[i] = 0;
    }
}


// The reference implementation of bitwise
// Student should not change this function
void naive_bitwise(std::span<std::int8_t> result,
                   std::span<const std::int8_t> a,
                   std::span<const std::int8_t> b) {
    constexpr std::uint8_t kMaskLo = 0x5Au;
    constexpr std::uint8_t kMaskHi = 0xC3u;

    const std::size_t n = std::min({result.size(), a.size(), b.size()});
    for (std::size_t i = 0; i < n; ++i) {
        const auto ua = static_cast<std::uint8_t>(a[i]);
        const auto ub = static_cast<std::uint8_t>(b[i]);

        const auto shared = static_cast<std::uint8_t>(ua & ub);
        const auto either = static_cast<std::uint8_t>(ua | ub);
        const auto diff = static_cast<std::uint8_t>(ua ^ ub);
        const auto mixed0 =
            static_cast<std::uint8_t>((diff & kMaskLo) | (~shared & ~kMaskLo));
        const auto mixed1 = static_cast<std::uint8_t>(
            ((either ^ kMaskHi) & (shared | ~kMaskHi)) ^ diff);

        result[i] = static_cast<std::int8_t>(mixed0 ^ mixed1);
    }
}

// Optimized bitwise function (Data Parallelism)
void stu_bitwise(std::span<std::int8_t> result, std::span<const std::int8_t> a,
                 std::span<const std::int8_t> b) {
    const std::size_t n = std::min({result.size(), a.size(), b.size()});
    
    // 核心数学优化：布尔代数化简万能掩码
    constexpr std::uint64_t K_MASK_1 = 0x2424242424242424ULL;
    constexpr std::uint64_t K_MASK_E = 0x1818181818181818ULL;
    constexpr std::uint64_t K_MASK_NE= 0x8181818181818181ULL;

    auto* __restrict__ res = reinterpret_cast<std::uint64_t*>(result.data());
    const auto* __restrict__ pa = reinterpret_cast<const std::uint64_t*>(a.data());
    const auto* __restrict__ pb = reinterpret_cast<const std::uint64_t*>(b.data());

    std::size_t chunk_count = n / 8;
    std::size_t i = 0;

    // 4 展开：每次处理 32 个 int8_t。由于指令极少，CPU 吞吐量将达到极限！
    for (; i + 3 < chunk_count; i += 4) {
        std::uint64_t e0 = pa[i]   | pb[i];
        std::uint64_t e1 = pa[i+1] | pb[i+1];
        std::uint64_t e2 = pa[i+2] | pb[i+2];
        std::uint64_t e3 = pa[i+3] | pb[i+3];

        res[i]   = K_MASK_1 | (e0 & K_MASK_E) | (~e0 & K_MASK_NE);
        res[i+1] = K_MASK_1 | (e1 & K_MASK_E) | (~e1 & K_MASK_NE);
        res[i+2] = K_MASK_1 | (e2 & K_MASK_E) | (~e2 & K_MASK_NE);
        res[i+3] = K_MASK_1 | (e3 & K_MASK_E) | (~e3 & K_MASK_NE);
    }

    // 剩余的 64 位块
    for (; i < chunk_count; ++i) {
        std::uint64_t e = pa[i] | pb[i];
        res[i] = K_MASK_1 | (e & K_MASK_E) | (~e & K_MASK_NE);
    }

    // 尾部处理 (处理剩下的不足 8 字节的单元素，使用 8 位掩码)
    constexpr std::uint8_t k_1 = 0x24;
    constexpr std::uint8_t k_E = 0x18;
    constexpr std::uint8_t k_NE = 0x81;
    for (std::size_t j = chunk_count * 8; j < n; ++j) {
        std::uint8_t e = static_cast<std::uint8_t>(a[j] | b[j]);
        result[j] = static_cast<std::int8_t>(k_1 | (e & k_E) | (~e & k_NE));
    }
}


void naive_bitwise_wrapper(void *ctx) {
    auto &args = *static_cast<bitwise_args *>(ctx);
    naive_bitwise(args.result, args.a, args.b);
}

void stu_bitwise_wrapper(void *ctx) {
    auto &args = *static_cast<bitwise_args *>(ctx);
    stu_bitwise(args.result, args.a, args.b);
}

bool bitwise_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    // Compute reference
    naive_func(ref_ctx);

    auto &stu_args = *static_cast<bitwise_args *>(stu_ctx);
    auto &ref_args = *static_cast<bitwise_args *>(ref_ctx);

    if (stu_args.result.size() != ref_args.result.size()) {
        debug_log("\tDEBUG: size mismatch: stu={} ref={}\n",
                  stu_args.result.size(),
                  ref_args.result.size());
        return false;
    }

    std::int32_t max_abs_diff = 0;
    size_t worst_i = 0;

    for (size_t i = 0; i < ref_args.result.size(); ++i) {
        const auto r = static_cast<std::int32_t>(ref_args.result[i]);
        const auto s = static_cast<std::int32_t>(stu_args.result[i]);

        if (r != s) {
            max_abs_diff = std::abs(r - s);
            worst_i = i;

            debug_log("\tDEBUG: fail at {}: ref={} stu={} abs_diff={}\n",
                      i,
                      r,
                      s,
                      max_abs_diff);
            return false;
        }
    }

    debug_log("\tDEBUG: bitwise_check passed. max_abs_diff={} at i={}\n",
              max_abs_diff,
              worst_i);
    return true;
}