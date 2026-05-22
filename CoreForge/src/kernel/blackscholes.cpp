#include "blackscholes.h"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <random>
#include <cmath>

#define inv_sqrt_2xPI 0.39894228040143270286f
#define p_val 0.2316419f
#define coefficient_a1 0.319381530f
#define coefficient_a2 -0.356563782f
#define coefficient_a3 1.781477937f
#define coefficient_a4 -1.821255978f
#define coefficient_a5 1.330274429f

void initialize_blackscholes(blackscholes_args &args,
                             std::size_t n,
                             std::uint32_t seed) {
    args.call_option_price.assign(n, 0.0f);
    args.put_option_price.assign(n, 0.0f);
    args.epsilon = 5e-3;

    args.spot_price.resize(n);
    args.strike.resize(n);
    args.rate.resize(n);
    args.volatility.resize(n);
    args.time.resize(n);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> spot_dist(50.0f, 99.9f);
    std::uniform_real_distribution<float> strike_dist(50.0f, 99.9f);
    std::uniform_real_distribution<float> rate_dist(0.0275f, 0.1f);
    std::uniform_real_distribution<float> vol_dist(0.05f, 0.6f);
    std::uniform_real_distribution<float> time_dist(0.1f, 1.0f);

    for (std::size_t i = 0; i < n; ++i) {
        args.spot_price[i] = spot_dist(rng);
        args.strike[i] = strike_dist(rng);
        args.rate[i] = rate_dist(rng);
        args.volatility[i] = vol_dist(rng);
        args.time[i] = time_dist(rng);
    }
}

// --------------------------------------------------------
// Fast Math Approximations (Bit Manipulation)
// --------------------------------------------------------

inline float fast_inv_sqrt(float x) {
    float halfx = 0.5f * x;
    std::int32_t i = std::bit_cast<std::int32_t>(x);
    i = 0x5f3759df - (i >> 1);
    float y = std::bit_cast<float>(i);
    y = y * (1.5f - (halfx * y * y));
    return y;
}

inline float fast_log(float val) {
    std::int32_t x = std::bit_cast<std::int32_t>(val);
    std::int32_t exp = (x >> 23) - 127;
    x = (x & 0x007FFFFF) | 0x3f800000;
    float m = std::bit_cast<float>(x);
    float y = m - 1.0f;
    // 泰勒级数逼近
    float p = y * (1.0f + y * (-0.5f + y * (0.33333333f - 0.25f * y)));
    return static_cast<float>(exp) * 0.69314718f + p;
}

inline float fast_exp(float x) {
    float y = x * 1.44269504f; // x / ln(2)
    float floor_y = std::floor(y);
    float f = y - floor_y; // 小数部分
    float p = 1.0f + f * (0.693147f + f * (0.240226f + f * 0.055504f));
    std::int32_t i = (static_cast<std::int32_t>(floor_y) + 127) << 23;
    return std::bit_cast<float>(i) * p;
}

static inline void CNDF_fast(float InputX, float &OutputX) {
    int sign = 0;
    float x = InputX;
    if (x < 0.0f) {
        x = -x;
        sign = 1;
    }

    const float xNPrimeofX = fast_exp(-0.5f * x * x) * inv_sqrt_2xPI;
    const float k = 1.0f / (1.0f + p_val * x);
    const float k_2 = k * k;
    const float k_3 = k_2 * k;
    const float k_4 = k_3 * k;
    const float k_5 = k_4 * k;

    float local = k * coefficient_a1;
    local += k_2 * coefficient_a2;
    local += k_3 * coefficient_a3;
    local += k_4 * coefficient_a4;
    local += k_5 * coefficient_a5;
    local = 1.0f - local * xNPrimeofX;

    OutputX = sign ? (1.0f - local) : local;
}

// --------------------------------------------------------

void CNDF(const float &InputX, float &OutputX) {
    // Reference CNDF is kept here for naive_BlkSchls
    int sign = 0;
    float x = InputX;

    if (x < 0.0f) {
        x = -x;
        sign = 1;
    }

    const float xNPrimeofX = std::exp(-0.5f * x * x) * inv_sqrt_2xPI;
    const float k = 1.0f / (1.0f + p_val * x);
    const float k_2 = k * k;
    const float k_3 = k_2 * k;
    const float k_4 = k_3 * k;
    const float k_5 = k_4 * k;

    float local = k * coefficient_a1;
    local += k_2 * coefficient_a2;
    local += k_3 * coefficient_a3;
    local += k_4 * coefficient_a4;
    local += k_5 * coefficient_a5;
    local = 1.0f - local * xNPrimeofX;

    OutputX = sign ? (1.0f - local) : local;
}

static inline void naive_BlkSchls_one(float &CallOptionPrice,
                                      float &PutOptionPrice, float spotPrice,
                                      float strike, float rate,
                                      float volatility, float time) {
    const float xSqrtTime = std::sqrt(time);
    const float xLogTerm = std::log(spotPrice / strike);
    const float xPowerTerm = 0.5f * volatility * volatility;

    float xD1 = (rate + xPowerTerm) * time + xLogTerm;
    const float xDen = volatility * xSqrtTime;
    xD1 = xD1 / xDen;
    const float xD2 = xD1 - xDen;

    float d1 = xD1;
    float d2 = xD2;
    float NofXd1 = 0.0f;
    float NofXd2 = 0.0f;

    CNDF(d1, NofXd1);
    CNDF(d2, NofXd2);

    const float FutureValueX = strike * std::exp(-(rate) * (time));
    CallOptionPrice = (spotPrice * NofXd1) - (FutureValueX * NofXd2);

    const float NegNofXd1 = 1.0f - NofXd1;
    const float NegNofXd2 = 1.0f - NofXd2;
    PutOptionPrice = (FutureValueX * NegNofXd2) - (spotPrice * NegNofXd1);
}

void naive_BlkSchls(std::vector<float> &CallOptionPrice,
                    std::vector<float> &PutOptionPrice,
                    const std::vector<float> &spotPrice,
                    const std::vector<float> &strike,
                    const std::vector<float> &rate,
                    const std::vector<float> &volatility,
                    const std::vector<float> &time) {
    size_t n = spotPrice.size();
    for (size_t i = 0; i < n; ++i) {
        naive_BlkSchls_one(CallOptionPrice[i], PutOptionPrice[i], spotPrice[i],
                           strike[i], rate[i], volatility[i], time[i]);
    }
}

// Optimized Student Implementation
void stu_BlkSchls(std::vector<float> &CallOptionPrice,
                  std::vector<float> &PutOptionPrice,
                  const std::vector<float> &spotPrice,
                  const std::vector<float> &strike,
                  const std::vector<float> &rate,
                  const std::vector<float> &volatility,
                  const std::vector<float> &time) {
    
    size_t n = spotPrice.size();
    float* call = CallOptionPrice.data();
    float* put = PutOptionPrice.data();
    const float* spot = spotPrice.data();
    const float* strk = strike.data();
    const float* rt = rate.data();
    const float* vol = volatility.data();
    const float* tm = time.data();

    // 手动循环展开 4 路
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        // 处理 4 个元素
        for (int j = 0; j < 4; ++j) {
            size_t idx = i + j;
            float s = spot[idx];
            float k = strk[idx];
            float r = rt[idx];
            float v = vol[idx];
            float t = tm[idx];

            const float xSqrtTime = std::sqrt(t);
            const float xLogTerm = std::log(s / k);
            const float xPowerTerm = 0.5f * v * v;

            float xD1 = (r + xPowerTerm) * t + xLogTerm;
            const float xDen = v * xSqrtTime;
            xD1 = xD1 / xDen;
            const float xD2 = xD1 - xDen;

            float NofXd1 = 0.0f;
            float NofXd2 = 0.0f;

            CNDF(xD1, NofXd1);
            CNDF(xD2, NofXd2);

            const float FutureValueX = k * std::exp(-r * t);
            call[idx] = (s * NofXd1) - (FutureValueX * NofXd2);

            const float NegNofXd1 = 1.0f - NofXd1;
            const float NegNofXd2 = 1.0f - NofXd2;
            put[idx] = (FutureValueX * NegNofXd2) - (s * NegNofXd1);
        }
    }
    // 处理剩余元素
    for (; i < n; ++i) {
        float s = spot[i];
        float k = strk[i];
        float r = rt[i];
        float v = vol[i];
        float t = tm[i];

        const float xSqrtTime = std::sqrt(t);
        const float xLogTerm = std::log(s / k);
        const float xPowerTerm = 0.5f * v * v;

        float xD1 = (r + xPowerTerm) * t + xLogTerm;
        const float xDen = v * xSqrtTime;
        xD1 = xD1 / xDen;
        const float xD2 = xD1 - xDen;

        float NofXd1 = 0.0f;
        float NofXd2 = 0.0f;

        CNDF(xD1, NofXd1);
        CNDF(xD2, NofXd2);

        const float FutureValueX = k * std::exp(-r * t);
        call[i] = (s * NofXd1) - (FutureValueX * NofXd2);

        const float NegNofXd1 = 1.0f - NofXd1;
        const float NegNofXd2 = 1.0f - NofXd2;
        put[i] = (FutureValueX * NegNofXd2) - (s * NegNofXd1);
    }
}

void naive_BlkSchls_wrapper(void *ctx) {
    auto &args = *static_cast<blackscholes_args *>(ctx);
    naive_BlkSchls(args.call_option_price, args.put_option_price, args.spot_price,
                   args.strike, args.rate, args.volatility, args.time);
}

void stu_BlkSchls_wrapper(void *ctx) {
    auto &args = *static_cast<blackscholes_args *>(ctx);
    stu_BlkSchls(args.call_option_price, args.put_option_price, args.spot_price,
                 args.strike, args.rate, args.volatility, args.time);
}

// Checker code remains unchanged...
bool BlkSchls_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);
    auto &stu_args = *static_cast<blackscholes_args *>(stu_ctx);
    auto &ref_args = *static_cast<blackscholes_args *>(ref_ctx);
    const double eps = ref_args.epsilon; 

    if (ref_args.call_option_price.size() != stu_args.call_option_price.size() ||
        ref_args.put_option_price.size() != stu_args.put_option_price.size())
        return false;

    const double atol = 1e-5; 
    const size_t n = ref_args.call_option_price.size();
    double max_rel = 0.0, max_abs = 0.0;
    size_t max_idx = 0;
    const char *max_leg = "call";

    for (size_t i = 0; i < n; ++i) {
        const double rc = static_cast<double>(ref_args.call_option_price[i]);
        const double rp = static_cast<double>(ref_args.put_option_price[i]);
        const double sc = static_cast<double>(stu_args.call_option_price[i]);
        const double sp = static_cast<double>(stu_args.put_option_price[i]);

        const double err_c = std::abs(rc - sc);
        const double err_p = std::abs(rp - sp);
        const double rel_c = (err_c - atol) / std::abs(rc);
        const double rel_p = (err_p - atol) / std::abs(rp);

        const bool call_ok = err_c <= (atol + eps * std::abs(rc));
        const bool put_ok = err_p <= (atol + eps * std::abs(rp));

        if (rel_c > max_rel) { max_abs = err_c; max_rel = rel_c; max_idx = i; max_leg = "call"; }
        if (rel_p > max_rel) { max_abs = err_p; max_rel = rel_p; max_idx = i; max_leg = "put"; }

        if (!call_ok || !put_ok) {
            debug_log("\tDEBUG: fail idx={} | call ref={} stu={} err={} thr={} | put ref={} stu={} err={} thr={}\n",
                      i, rc, sc, err_c, (atol + eps * std::abs(rc)), rp, sp, err_p, (atol + eps * std::abs(rp)));
            return false;
        }
    }
    debug_log("\tBlkSchls_check passed: n={}, max_rel_err={}, max_abs_err={} at idx={} ({})\n",
              n, max_rel, max_abs, max_idx, max_leg);
    return true;
}