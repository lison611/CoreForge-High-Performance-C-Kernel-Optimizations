#include "grff.h"
#include <algorithm>
#include <cmath>
#include <random>

void initialize_grff(grff_args *args, const size_t size, const std::uint_fast64_t seed) {
    if (!args) return;

    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    args->a_features.resize(size);
    args->b_features.resize(size);
    args->c_features.resize(size);
    args->f_output.resize(size);

    // 为我们新增的缓存分配空间
    args->buf_a_prime.resize(size);
    args->buf_b_prime_base.resize(size);

    for (size_t i = 0; i < size; ++i) {
        args->a_features[i] = dist(gen);
        args->b_features[i] = dist(gen);
        args->c_features[i] = dist(gen);
    }
}

// -------------------------------------------------------------------------
// Naive Implementation
// -------------------------------------------------------------------------
void naive_grff(grff_args& args) {
    size_t n = args.a_features.size();
    std::vector<float> G(n), A_prime(n), Smooth_A(n), B_prime(n), C_prime(n), H(n), E(n);

    for (size_t i = 0; i < n; ++i) G[i] = 0.5f * ((args.a_features[i] * args.b_features[i]) / (1.0f + std::abs(args.a_features[i] * args.b_features[i])) + 1.0f);
    for (size_t i = 0; i < n; ++i) A_prime[i] = args.a_features[i] + G[i];
    
    float sum_a = 0.0f;
    for (size_t i = 0; i < n; ++i) sum_a += A_prime[i];
    float avg_a = sum_a / static_cast<float>(n);

    Smooth_A[0] = A_prime[0];
    for (size_t i = 1; i < n; ++i) Smooth_A[i] = (A_prime[i] + A_prime[i-1]) * 0.5f; 
    for (size_t i = 0; i < n; ++i) B_prime[i] = args.b_features[i] * (1.0f - G[i]) * avg_a;
    for (size_t i = 0; i < n; ++i) C_prime[i] = args.c_features[i] + (Smooth_A[i] / (1.0f + std::abs(Smooth_A[i])));
    for (size_t i = 0; i < n; ++i) H[i] = Smooth_A[i] * C_prime[i];
    for (size_t i = 0; i < n; ++i) E[i] = (H[i] + B_prime[i]) / (1.0f + std::abs(Smooth_A[i]));
    
    for (size_t i = 0; i < n; ++i) {
        float result = C_prime[i] - E[i];
        args.f_output[i] = std::max(result, 0.0f);
    }
}

void stu_grff(grff_args& args) {
    const size_t n = args.a_features.size();
    
    // 1. 指针防别名（极其关键！告诉编译器这几块内存绝对不重叠，放心做并行加载）
    const float* __restrict__ a = args.a_features.data();
    const float* __restrict__ b = args.b_features.data();
    const float* __restrict__ c = args.c_features.data();
    
    float* __restrict__ a_prime = args.buf_a_prime.data();
    float* __restrict__ b_prime_base = args.buf_b_prime_base.data();
    float* __restrict__ out = args.f_output.data();

    // =================================================================
    // Pass 1: 计算 A_prime 和 B_prime_base (无数据依赖，易于自动向量化)
    // =================================================================
    #pragma GCC ivdep
    #pragma GCC unroll 8
    for (size_t i = 0; i < n; ++i) {
        float val_a = a[i];
        float val_b = b[i];
        float ab = val_a * val_b;
        
        // std::abs 被翻译成无分支的位掩码指令
        float g = 0.5f * ((ab / (1.0f + std::abs(ab))) + 1.0f);
        
        a_prime[i] = val_a + g;
        b_prime_base[i] = val_b * (1.0f - g);
    }

    // =================================================================
    // Pass 2: 严格串行累加 (保证 100% 匹配 Naive 精度的核心护城河)
    // =================================================================
    float sum_a = 0.0f;
    // 这里的求和顺序和 Naive 版本一模一样，杜绝了由于结合律导致的浮点数精度漂移！
    for (size_t i = 0; i < n; ++i) {
        sum_a += a_prime[i];
    }
    const float avg_a = sum_a / static_cast<float>(n);

    // =================================================================
    // Pass 3: 输出特征映射 (消除循环间依赖，释放指令级并行/向量化性能)
    // =================================================================
    if (n > 0) {
        // 首元素特判 (对应 i = 0，单独摘出来是为了让后续循环没有分支判断)
        float smooth_a0 = a_prime[0];
        float denom0 = 1.0f + std::abs(smooth_a0);
        float b_p0 = b_prime_base[0] * avg_a;
        float c_p0 = c[0] + (smooth_a0 / denom0);
        float h0 = smooth_a0 * c_p0;
        float e0 = (h0 + b_p0) / denom0;
        out[0] = std::max(c_p0 - e0, 0.0f);
        
        // 剩余元素 (i 从 1 开始)
        // a_prime[i] 和 a_prime[i-1] 是连续内存读取，现代 CPU 处理不对齐加载极快
        #pragma GCC ivdep
        #pragma GCC unroll 8
        for (size_t i = 1; i < n; ++i) {
            float smooth_a = (a_prime[i] + a_prime[i - 1]) * 0.5f;
            
            float denom = 1.0f + std::abs(smooth_a);
            float b_p = b_prime_base[i] * avg_a;
            float c_p = c[i] + (smooth_a / denom);
            
            float h = smooth_a * c_p;
            float e = (h + b_p) / denom;
            
            out[i] = std::max(c_p - e, 0.0f); // std::max 在底层会映射为无分支操作
        }
    }
}
// -------------------------------------------------------------------------
// Wrappers and Checker
// -------------------------------------------------------------------------
void naive_grff_wrapper(void *ctx) {
    auto &args = *static_cast<grff_args *>(ctx);
    naive_grff(args);
}

void stu_grff_wrapper(void *ctx) {
    auto &args = *static_cast<grff_args *>(ctx);
    stu_grff(args);
}

bool grff_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);
    auto &stu_args = *static_cast<grff_args *>(stu_ctx);
    auto &ref_args = *static_cast<grff_args *>(ref_ctx);
    const auto eps = ref_args.epsilon;
    const double atol = 1e-6;

    if (stu_args.f_output.size() != ref_args.f_output.size()) return false;

    for (size_t i = 0; i < ref_args.f_output.size(); ++i) {
        double r = static_cast<double>(ref_args.f_output[i]);
        double s = static_cast<double>(stu_args.f_output[i]);
        double err = std::abs(s - r);

        if (err > (atol + eps * std::abs(r))) {
            debug_log("DEBUG: GRFF fail at %zu: ref=%f stu=%f\n", i, r, s);
            return false;
        }
    }
    return true;
}