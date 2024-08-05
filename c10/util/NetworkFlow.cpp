#include <c10/util/NetworkFlow.h>

#include <c10/util/Exception.h>

#include <iostream>
#include <optional>
#include <queue>
#include <unordered_map>
#include <vector>

namespace c10 {

namespace {

struct DinicFlowGraph {
  // [Note: Dinic graph format]
  // The graph is represented as an adjacency list:
  //   for a vertex u, adj[u] lists all the outgoing edges from u.
  //   adj[u][i] is the index of the i-th outgoing edge from u.
  //   To get information on the i-th outgoing edge from u, use
  //   edges[adj[i][i]].
  // The edges are directed and are paired with a reverse edge.
  //   For example, an edge u->v is paired with a v->u edge.
  //   The index of the reverse edge of e is stored as e.other_idx.
  // Capacities and flows: each edge has a capacity and a flow
  //   associated with it. When flow is added to an edge, it removes
  //   capacity from the reverse edge.
  struct Edge {
    size_t u, v;
    int64_t capacity;
    int64_t flow;
    size_t other_idx; // reverse edge

    int64_t residual_capacity() const {
      return capacity - flow;
    }
  };

  std::vector<Edge> edges;
  std::vector<std::vector<size_t>> adj; // adjacency list
  std::vector<std::string> vertex_names;
  std::unordered_map<std::string, size_t> mapping;
  size_t graph_size;

  void add_flow(Edge& e, int64_t more) {
    e.flow += more;
    edges[e.other_idx].flow -= more;
  }

  const Edge& reverse_edge(const Edge& e) const {
    return edges[e.other_idx];
  }

  DinicFlowGraph(const NetworkFlowGraph& g) {
    size_t vertex_count = 0;

    auto get_idx = [&vertex_count, this](const std::string& name) {
      if (!mapping.count(name)) {
        TORCH_CHECK(vertex_count == vertex_names.size());
        vertex_names.push_back(name);
        size_t idx = vertex_count;
        vertex_count++;
        mapping[name] = idx;
        return idx;
      }
      return mapping[name];
    };

    for (const auto& [source, dest, capacity] : g.edges) {
      auto u = get_idx(source);
      auto v = get_idx(dest);
      auto fwd_idx = edges.size();
      auto bwd_idx = edges.size() + 1;
      edges.push_back({u, v, capacity, 0, bwd_idx});
      edges.push_back({v, u, 0, 0, fwd_idx});
    }

    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    graph_size = mapping.size();
    adj.resize(graph_size);

    for (size_t i = 0; i < edges.size(); ++i) {
      adj[edges[i].u].push_back(i);
    }
  }

  std::vector<std::vector<size_t>> residual_level_graph(size_t s) const {
    // The residual graph is the graph including only edges
    //   where edge.residual_capacity() is nonzero, i.e.
    //   edge.capacity > edge.flow.
    // The residual level graph is constructed by:
    //   1. doing a BFS on the residual graph, assigning levels
    //      to each vertex.
    //   2. only include edges u->v where level[v] == leve[u] + 1
    std::queue<size_t> q;
    // let level[u] = 0 if it has not been visited yet.
    std::vector<size_t> level(graph_size, 0);
    // TODO(davidberard98) we can create this once and reuse it
    std::vector<std::vector<size_t>> output_adjacency(graph_size);
    level[s] = 1;
    q.push(s);
    while (!q.empty()) {
      size_t u = q.front();
      q.pop();
      for (const auto& edge_idx : adj[u]) {
        const auto& e = edges[edge_idx];
        if (e.residual_capacity()) {
          if (level[e.v] == 0) {
            level[e.v] = level[e.u] + 1;
            q.push(e.v);
          }
          if (level[e.v] == level[e.u] + 1) {
            output_adjacency[e.u].push_back(edge_idx);
          }
        }
      }
    }

    return output_adjacency;
  }

  std::pair<MinCutStatus, int64_t> augment_iteration(size_t s, size_t t) {
    // Perform one iteration of augmenting the flow.
    // 1. Create the level graph
    // 2. DFS to find augmenting paths
    // 3. If encountering edges that don't lead to augmenting paths,
    //    trim them from the level graph.
    // 4. Repeat 2-3 until we can't find any augmenting paths.
    std::vector<std::vector<size_t>> level_adj = residual_level_graph(s);

    // TODO(davidberard98): implement this DFS with a stack
    std::function<int64_t(size_t, size_t, int64_t)> dfs;
    dfs = [&level_adj, &dfs, this](
              size_t u, size_t t, int64_t cur_cap) -> int64_t {
      if (u == t) {
        return cur_cap;
      }
      while (!level_adj[u].empty()) {
        // Iterate over the outgoing edges from u.
        // If take an edge and find that we can't augment using this edge,
        //   then delete it from our level graph.
        // If we take an edge and it does find an augmenting path, then
        //   take the augmenting path and exit early
        auto edge_idx = level_adj[u].back();
        auto& e = edges[edge_idx];
        auto taken_cap = dfs(e.v, t, std::min(cur_cap, e.residual_capacity()));
        if (taken_cap) {
          add_flow(e, taken_cap);
          if (!e.residual_capacity()) {
            // this edge has no remaining residual capacity, remove it.
            level_adj[u].pop_back();
          }
          return taken_cap;
        } else {
          // we can't get any capacity from this edge, remove it.
          level_adj[u].pop_back();
        }
      }
      return 0;
    };

    int64_t additional_flow = 0;
    while (int64_t f = dfs(s, t, NetworkFlowGraph::INF)) {
      if (f == NetworkFlowGraph::INF) {
        return {MinCutStatus::UNBOUNDED, 0};
      }
      additional_flow += f;
      if (additional_flow >= NetworkFlowGraph::INF) {
        return {MinCutStatus::OVERFLOW_INF, 0};
      }
    }

    return {MinCutStatus::SUCCESS, additional_flow};
  }

  std::pair<MinCutStatus, int64_t> compute_max_flow(size_t s, size_t t) {
    int64_t total_flow = 0;
    while (true) {
      auto [status, additional_flow] = augment_iteration(s, t);
      if (status != MinCutStatus::SUCCESS) {
        return {status, 0};
      }
      if (additional_flow == 0) {
        break;
      }
      total_flow += additional_flow;
      if (total_flow >= NetworkFlowGraph::INF) {
        return {MinCutStatus::OVERFLOW_INF, 0};
      }
    }
    return {MinCutStatus::SUCCESS, total_flow};
  }

  std::vector<bool> reverse_bfs_reachable(size_t t) const {
    // Find all vertices that are reachable from t in the reverse
    //   residual graph.
    std::vector<bool> seen(graph_size, false);
    seen[t] = true;
    std::queue<size_t> q;
    q.push(t);
    while (!q.empty()) {
      auto x = q.front();
      q.pop();
      for (auto& edge_idx : adj[x]) {
        // the edge that goes u -> v where v == x
        const auto& e = reverse_edge(edges[edge_idx]);
        if (!e.residual_capacity()) {
          continue;
        }

        if (!seen[e.u]) {
          seen[e.u] = true;
          q.push(e.u);
        }
      }
    }
    return seen;
  }

  std::pair<std::vector<size_t>, std::vector<size_t>> partition(
      size_t s,
      size_t t) {
    // Note: the partitioning returns "reachable" / "unreachable",
    //   but specifically, for "unreachable", it returns "all vertices
    //   that are reachable from t in the reverse residual graph"
    //   and for "reachable" it returns all other nodes. This mirrors
    //   the behavior of networkx.
    auto can_reach_t = reverse_bfs_reachable(t);
    std::vector<size_t> reachable, unreachable;
    for (size_t i = 0; i < graph_size; ++i) {
      if (can_reach_t[i]) {
        unreachable.push_back(i);
      } else {
        reachable.push_back(i);
      }
    }
    return std::pair<std::vector<size_t>, std::vector<size_t>>(
        std::move(reachable), std::move(unreachable));
  }

  MinCutResult minimum_cut(const std::string& s, const std::string& t) {
    if (mapping.find(s) == mapping.end() || mapping.find(t) == mapping.end()) {
      return {
          MinCutStatus::INVALID, // status
          0, // max_flow
          {}, // reachable
          {}, // unreachable
      };
    }
    auto s_int = mapping[s];
    auto t_int = mapping[t];
    auto [status, max_flow] = compute_max_flow(s_int, t_int);
    if (status != MinCutStatus::SUCCESS) {
      return {
          status, // status
          0, // max_flow
          {}, // reachable
          {}, // unreachable
      };
    }

    auto [reachable_idxs, unreachable_idxs] = partition(s_int, t_int);
    std::vector<std::string> reachable, unreachable;

    auto idxs_to_names = [&](std::vector<size_t>& src,
                             std::vector<std::string>& dest) {
      dest.reserve(src.size());
      for (auto idx : src) {
        dest.push_back(vertex_names[idx]);
      }
    };

    idxs_to_names(reachable_idxs, reachable);
    idxs_to_names(unreachable_idxs, unreachable);

    return {
        MinCutStatus::SUCCESS,
        max_flow,
        reachable,
        unreachable,
    };
  }
};

} // namespace

MinCutStatus NetworkFlowGraph::add_edge(
    const std::string& source,
    const std::string& dest,
    int64_t capacity) {
  edges.push_back({source, dest, capacity});
  return MinCutStatus::SUCCESS;
}

MinCutResult NetworkFlowGraph::minimum_cut(
    const std::string& s,
    const std::string& t) const {
  auto flow_graph = DinicFlowGraph(*this);

  return flow_graph.minimum_cut(s, t);
}

} // namespace c10
