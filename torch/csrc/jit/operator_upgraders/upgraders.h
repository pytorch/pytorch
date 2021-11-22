#pragma once
#include <atomic>
#include <iostream>
#include <string>
#include <unordered_map>

namespace torch {
namespace jit {

#define IS_IN_POPULATION_PHASE ([] { \
    static std::atomic<bool> first_time(true); \
    return first_time.exchange(false); } ())

static std::unordered_map<std::string, std::string> upgraders_graph;

void populate_upgraders_map(std::unordered_map<std::string, std::string> content) {
    // make sure we populate the map only once
    if (!IS_IN_POPULATION_PHASE) return;

    for (const auto& entry: content) {
        upgraders_graph.insert(entry);
    }
}

int get_upgraders_map_size() {
    return upgraders_graph.size();
}

} // namespace jit
} // namespace torch
