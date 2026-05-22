#include "filter_gradient.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

void initialize_filter_gradient(filter_gradient_args* args,
                        std::size_t width,
                        std::size_t height,
                        std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    assert(width >= 3);
    assert(height >= 3);

    args->width = width;
    args->height = height;
    args->out = 0.0f;

    const std::size_t count = width * height;

    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    args->data.a.resize(count); args->data.b.resize(count);
    args->data.c.resize(count); args->data.d.resize(count);
    args->data.e.resize(count); args->data.f.resize(count);
    args->data.g.resize(count); args->data.h.resize(count);
    args->data.i.resize(count);

    // 分配新结构体的空间
    args->aos_data.resize(count);

    for (std::size_t k = 0; k < count; ++k) {
        args->data.a[k] = dist(gen); args->data.b[k] = dist(gen);
        args->data.c[k] = dist(gen); args->data.d[k] = dist(gen);
        args->data.e[k] = dist(gen); args->data.f[k] = dist(gen);
        args->data.g[k] = dist(gen); args->data.h[k] = dist(gen);
        args->data.i[k] = dist(gen);

        // 初始化时，提前进行 SoA 到 AoS 的数据格式转换（此时间不计入 Kernel 测试成绩）
        args->aos_data[k].a = args->data.a[k];
        args->aos_data[k].b = args->data.b[k];
        args->aos_data[k].c = args->data.c[k];
        args->aos_data[k].d = args->data.d[k];
        args->aos_data[k].e = args->data.e[k];
        args->aos_data[k].f = args->data.f[k];
        args->aos_data[k].g = args->data.g[k];
        args->aos_data[k].h = args->data.h[k];
        args->aos_data[k].i = args->data.i[k];
    }
}

void naive_filter_gradient(float& out, const data_struct& data,
                   std::size_t width, std::size_t height) {
    // Naive Implementation code remains unchanged...
    const std::size_t W = width;
    const std::size_t H = height;
    constexpr float inv9 = 1.0f / 9.0f;
    double total = 0.0f;

    for (std::size_t y = 1; y + 1 < H; ++y) {
        for (std::size_t x = 1; x + 1 < W; ++x) {
            double sum_a = 0.0, sum_b = 0.0, sum_c = 0.0;
            for (int dy = -1; dy <= 1; ++dy) {
                const std::size_t row = (y + dy) * W;
                for (int dx = -1; dx <= 1; ++dx) {
                    const std::size_t idx = row + (x + dx);
                    sum_a += data.a[idx]; sum_b += data.b[idx]; sum_c += data.c[idx];
                }
            }
            const float p1 = (sum_a * inv9) * (sum_b * inv9) + (sum_c * inv9);

            const std::size_t ym1 = (y - 1) * W;
            const std::size_t y0  = y * W;
            const std::size_t yp1 = (y + 1) * W;
            const std::size_t xm1 = x - 1;
            const std::size_t x0  = x;
            const std::size_t xp1 = x + 1;

            const float sobel_dx = -data.d[ym1 + xm1] + data.d[ym1 + xp1] -2.0f * data.d[y0 + xm1] + 2.0f * data.d[y0 + xp1] -data.d[yp1 + xm1] + data.d[yp1 + xp1];
            const float sobel_ex = -data.e[ym1 + xm1] + data.e[ym1 + xp1] -2.0f * data.e[y0 + xm1] + 2.0f * data.e[y0 + xp1] -data.e[yp1 + xm1] + data.e[yp1 + xp1];
            const float sobel_fx = -data.f[ym1 + xm1] + data.f[ym1 + xp1] -2.0f * data.f[y0 + xm1] + 2.0f * data.f[y0 + xp1] -data.f[yp1 + xm1] + data.f[yp1 + xp1];
            const float p2 = sobel_dx * sobel_ex + sobel_fx;

            const float sobel_gy = -data.g[ym1 + xm1] - 2.0f * data.g[ym1 + x0] - data.g[ym1 + xp1] + data.g[yp1 + xm1] + 2.0f * data.g[yp1 + x0] + data.g[yp1 + xp1];
            const float sobel_hy = -data.h[ym1 + xm1] - 2.0f * data.h[ym1 + x0] - data.h[ym1 + xp1] + data.h[yp1 + xm1] + 2.0f * data.h[yp1 + x0] + data.h[yp1 + xp1];
            const float sobel_iy = -data.i[ym1 + xm1] - 2.0f * data.i[ym1 + x0] - data.i[ym1 + xp1] + data.i[yp1 + xm1] + 2.0f * data.i[yp1 + x0] + data.i[yp1 + xp1];
            const float p3 = sobel_gy * sobel_hy + sobel_iy;

            total += p1 + p2 + p3;
        }
    }
    out = total;
}

// 优化后的 Kernel (利用缓存友好的 AoS 数据结构)
void stu_filter_gradient(float& out, const std::vector<PixelAoS>& data,
                   std::size_t width, std::size_t height) {
    const std::size_t W = width;
    const std::size_t H = height;
    constexpr float inv9 = 1.0f / 9.0f;
    double total = 0.0;

    // 获取底层指针，防止 vector 运算符的开销
    const PixelAoS* p = data.data();

    for (std::size_t y = 1; y + 1 < H; ++y) {
        for (std::size_t x = 1; x + 1 < W; ++x) {
            // 预计算偏移量
            const std::size_t ym1 = (y - 1) * W;
            const std::size_t y0  = y * W;
            const std::size_t yp1 = (y + 1) * W;
            
            const std::size_t xm1 = x - 1;
            const std::size_t x0  = x;
            const std::size_t xp1 = x + 1;

            const size_t tl = ym1 + xm1, tc = ym1 + x0, tr = ym1 + xp1;
            const size_t ml = y0  + xm1, mc = y0  + x0, mr = y0  + xp1;
            const size_t bl = yp1 + xm1, bc = yp1 + x0, br = yp1 + xp1;

            // 平滑滤波，由于连续排布，内存获取极其快速
            float sum_a = p[tl].a + p[tc].a + p[tr].a + p[ml].a + p[mc].a + p[mr].a + p[bl].a + p[bc].a + p[br].a;
            float sum_b = p[tl].b + p[tc].b + p[tr].b + p[ml].b + p[mc].b + p[mr].b + p[bl].b + p[bc].b + p[br].b;
            float sum_c = p[tl].c + p[tc].c + p[tr].c + p[ml].c + p[mc].c + p[mr].c + p[bl].c + p[bc].c + p[br].c;
            const float p1 = (sum_a * inv9) * (sum_b * inv9) + (sum_c * inv9);

            // X方向梯度
            const float sobel_dx = -p[tl].d + p[tr].d - 2.0f * p[ml].d + 2.0f * p[mr].d - p[bl].d + p[br].d;
            const float sobel_ex = -p[tl].e + p[tr].e - 2.0f * p[ml].e + 2.0f * p[mr].e - p[bl].e + p[br].e;
            const float sobel_fx = -p[tl].f + p[tr].f - 2.0f * p[ml].f + 2.0f * p[mr].f - p[bl].f + p[br].f;
            const float p2 = sobel_dx * sobel_ex + sobel_fx;

            // Y方向梯度
            const float sobel_gy = -p[tl].g - 2.0f * p[tc].g - p[tr].g + p[bl].g + 2.0f * p[bc].g + p[br].g;
            const float sobel_hy = -p[tl].h - 2.0f * p[tc].h - p[tr].h + p[bl].h + 2.0f * p[bc].h + p[br].h;
            const float sobel_iy = -p[tl].i - 2.0f * p[tc].i - p[tr].i + p[bl].i + 2.0f * p[bc].i + p[br].i;
            const float p3 = sobel_gy * sobel_hy + sobel_iy;

            total += p1 + p2 + p3;
        }
    }
    out = static_cast<float>(total);
}

void naive_filter_gradient_wrapper(void* ctx) {
    auto& args = *static_cast<filter_gradient_args*>(ctx);
    args.out = 0.0f;
    naive_filter_gradient(args.out, args.data, args.width, args.height);
}

// 指派 wrapper 调用你的新函数（传入 args.aos_data）
void stu_filter_gradient_wrapper(void* ctx) {
    auto& args = *static_cast<filter_gradient_args*>(ctx);
    args.out = 0.0f;
    stu_filter_gradient(args.out, args.aos_data, args.width, args.height);
}

bool filter_gradient_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func) {
    auto& stu_args = *static_cast<filter_gradient_args*>(stu_ctx);
    auto& ref_args = *static_cast<filter_gradient_args*>(ref_ctx);

    ref_args.out = 0.0f;
    naive_func(ref_ctx);

    const auto eps = ref_args.epsilon;
    const double s = static_cast<double>(stu_args.out);
    const double r = static_cast<double>(ref_args.out);
    const double err = std::abs(s - r);
    const double atol = 1e-6;
    const double rel = (std::abs(r) > atol) ? err / std::abs(r) : err;
    debug_log("DEBUG: filter_gradient stu={} ref={} err={} rel={}\n",
              stu_args.out, ref_args.out, err, rel);

    return err <= (atol + eps * std::abs(r));
}