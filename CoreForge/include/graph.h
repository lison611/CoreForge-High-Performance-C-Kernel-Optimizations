#ifndef GRAPH_H
#define GRAPH_H

#include "bench.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

const std::chrono::nanoseconds BASELINE_GRAPH{5000000};
inline constexpr double NAIVE_SPEEDUP_LOWER_BOUND_GRAPH{2.50};

struct Edge {
    int to;
    Edge* next;
};

struct Node {
    Edge* edges;
};

struct Graph {
    int n;
    Node* nodes;
};

struct CSRGraph {
    int n;
    const int* row_ptr;
    const int* col_idx;
};

struct graph_args {
    Graph graph;
    std::vector<Node> nodes;
    std::vector<Edge> edge_storage;
    std::vector<int> csr_row_ptr;
    std::vector<int> csr_col_idx;
    CSRGraph csr_graph;
    std::uint64_t out;
    double epsilon;
    // TODO: You may want to add new params at the end...

    explicit graph_args(double epsilon_in = 1e-6)
        : graph{0, nullptr}, csr_graph{0, nullptr, nullptr}, out{0}, epsilon{epsilon_in} {}
};

void naive_graph(std::uint64_t& out, const Graph& graph);
// TODO: You may need to add a function to convert data structure (not 
// included in time measurement), then implement your version in 
// stu_graph, whch is called by stu_graph_wrapper.
void stu_graph(std::uint64_t& out, const Graph& graph);

void naive_graph_wrapper(void* ctx);
void stu_graph_wrapper(void* ctx);

void initialize_graph(graph_args* args,
                       std::size_t node_count,
                       int avg_degree,
                       std::uint_fast64_t seed);

bool graph_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func);

#endif
