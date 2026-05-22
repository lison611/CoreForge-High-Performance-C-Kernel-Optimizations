#include "trace_replay.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace {

static inline uint64_t trace_replay_cost(const RequestRecord& record) {
    uint64_t cost = 0;
    cost += record.base_cost;
    cost += 2ull * record.retry_penalty;
    cost += record.miss_penalty;
    cost += record.bytes >> 4;
    return cost;
}

} // namespace

void initialize_trace_replay(trace_replay_args& args,
                             size_t record_count,
                             size_t trace_count,
                             uint32_t seed) {
    if (record_count == 0) {
        throw std::invalid_argument(
            "initialize_trace_replay: records must be non-empty.");
    }
    if (trace_count == 0) {
        throw std::invalid_argument(
            "initialize_trace_replay: trace must be non-empty.");
    }

    args.out = 0;
    args.records.resize(record_count);
    args.trace.resize(trace_count);

    uint32_t current = seed;

    for (size_t i = 0; i < args.records.size(); ++i) {
        current = current * 1664525u + 1013904223u;
        args.records[i].base_cost = current % 100;

        current = current * 1664525u + 1013904223u;
        args.records[i].retry_penalty = current % 50;

        current = current * 1664525u + 1013904223u;
        args.records[i].miss_penalty = current % 200;

        current = current * 1664525u + 1013904223u;
        args.records[i].bytes = current % 4096;
    }

    const uint32_t window_mask = 0x3FFF;
    const uint32_t stride = 7;
    const size_t segment_len = 1 << 14;

    for (size_t i = 0; i < args.trace.size(); ++i) {
        const uint32_t segment_idx = static_cast<uint32_t>(i / segment_len);
        const uint32_t base =
            (segment_idx * segment_len) % static_cast<uint32_t>(args.records.size());

        const uint32_t local =
            static_cast<uint32_t>(i % segment_len) & window_mask;
        args.trace[i] = base + ((local * stride) & window_mask);
    }
}

void naive_trace_replay(uint64_t& out,
                        const std::vector<RequestRecord>& records,
                        const std::vector<uint32_t>& trace) {
    uint64_t total = 0;
    const uint64_t order_mix = 1315423911ull;

    for (size_t i = 0; i < trace.size(); ++i) {
        total = total * order_mix + trace_replay_cost(records[trace[i]]);
    }

    out = total;
}

// 优化后的 Kernel：缓存打包 + 循环展开消除依赖
void stu_trace_replay(uint64_t& out,
                      const std::vector<RequestRecord>& records,
                      const std::vector<uint32_t>& trace) {
    
    // 1. 数据打包：提前将包含大量 Padding 的大结构体压缩为紧凑的 uint64_t 数组
    // 65536 个记录仅占用 512KB 内存，可以完全容纳在 L2 缓存中，消除主存访问延迟
    const size_t num_records = records.size();
    std::vector<uint64_t> packed_costs(num_records);
    uint64_t* cost_data = packed_costs.data();
    const RequestRecord* rec_data = records.data();

    for (size_t i = 0; i < num_records; ++i) {
        cost_data[i] = static_cast<uint64_t>(rec_data[i].base_cost) +
                       2ull * rec_data[i].retry_penalty +
                       rec_data[i].miss_penalty +
                       (rec_data[i].bytes >> 4);
    }

    // 2. 指令级并行：通过代数展开打破流水线上的 Loop-Carried Dependency
    uint64_t total = 0;
    const uint64_t C = 1315423911ull;
    const uint64_t C2 = C * C;
    const uint64_t C3 = C2 * C;
    const uint64_t C4 = C2 * C2;

    const size_t num_traces = trace.size();
    const uint32_t* trace_data = trace.data();
    size_t i = 0;

    // 每次处理 4 个元素
    for (; i + 3 < num_traces; i += 4) {
        uint64_t cost0 = cost_data[trace_data[i]];
        uint64_t cost1 = cost_data[trace_data[i + 1]];
        uint64_t cost2 = cost_data[trace_data[i + 2]];
        uint64_t cost3 = cost_data[trace_data[i + 3]];

        uint64_t block_contrib = cost0 * C3 + cost1 * C2 + cost2 * C + cost3;
        total = total * C4 + block_contrib;
    }

    // 尾部处理
    for (; i < num_traces; ++i) {
        total = total * C + cost_data[trace_data[i]];
    }

    out = total;
}

void naive_trace_replay_wrapper(void* ctx) {
    auto& args = *static_cast<trace_replay_args*>(ctx);
    naive_trace_replay(args.out, args.records, args.trace);
}

void stu_trace_replay_wrapper(void* ctx) {
    auto& args = *static_cast<trace_replay_args*>(ctx);
    stu_trace_replay(args.out, args.records, args.trace);
}

bool trace_replay_check(void* stu_ctx,
                        void* ref_ctx,
                        lab_test_func naive_func) {
    naive_func(ref_ctx);

    auto& stu_args = *static_cast<trace_replay_args*>(stu_ctx);
    auto& ref_args = *static_cast<trace_replay_args*>(ref_ctx);

    if (stu_args.out != ref_args.out) {
        debug_log("\tDEBUG: trace_replay fail: ref={} stu={}\n",
                  ref_args.out,
                  stu_args.out);
        return false;
    }

    debug_log("\tDEBUG: trace_replay_check passed. out={}\n", ref_args.out);
    return true;
}