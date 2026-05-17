#pragma once

#include <c10/macros/Macros.h>

#include <string>
#include <vector>

/**
 * This file provides a network flow implementation.
 * https://en.wikipedia.org/wiki/Flow_network
 *
 * It aims to mirror some of the behavior of networkx, which is/was used by
 * functorch partitioners for splitting the graph into a forward and backward
 * graph.
 */

namespace c10 {

enum class C10_API_ENUM MinCutStatus {
  SUCCESS = 0,
  UNBOUNDED = 1,
  OVERFLOW_INF = 2,
  INVALID = 3,
};

struct MinCutResult {
  MinCutStatus status;
  int64_t max_flow;
  std::vector<std::string> reachable;
  std::vector<std::string> unreachable;
};

// Modeled after networkx implementation
class C10_API NetworkFlowGraph {
 public:
  // selected such that INF + INF is < INT64_MAX
  constexpr static int64_t INF = (1LL << 62) - 1;

  struct Edge {
    std::string source, dest;
    int64_t capacity;
  };

  MinCutStatus add_edge(
      const std::string& source,
      const std::string& dest,
      int64_t capacity = 1);

  MinCutResult minimum_cut(const std::string& s, const std::string& t) const;

  std::vector<Edge> edges;
};

} // namespace c10
