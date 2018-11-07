#include "torch/csrc/jit/passes/create_autodiff_subgraphs.h"

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/autodiff.h"
#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/jit/dynamic_dag.h"

#include <cstddef>
#include <limits>

namespace torch { namespace jit {

struct Graph;

namespace {

// Move nodes that exist in graph g into a 'group_node_kind' node.
// All inputs shared by the nodes become inputs to the new node.
// Outputs from 'nodes' are redirected to outputs of the new node,
// and the original nodes are removed.
// prereq: it is topologically valid to place the new node
// right before nodes[0] (i.e. it will not create cycles and all uses of
// new node will be after this position).
// prereq: nodes are in topological order
Node* mergeNodes(Block * block, Symbol group_node_kind, ArrayRef<Node*> nodes) {
  JIT_ASSERT(nodes.size() > 0);
  std::unordered_map<Value*, Value*> value_map;
  Graph * graph = block->owningGraph();

  auto new_graph = std::make_shared<Graph>();
  Node * group_node = graph->create(group_node_kind, 0);
  group_node->g_(attr::Subgraph, new_graph);

  auto getOrCreateInput = [&](Value * v) {
    if(value_map.count(v) > 0) {
      return value_map[v];
    }
    if (auto value = toIValue(v)) {
      Value * nv = new_graph->insertConstant(*value);
      value_map[v] = nv;
      return nv;
    }
    Value * nv = new_graph->addInput()->setType(v->type());
    group_node->addInput(v);
    value_map[v] = nv;
    return nv;
  };
  std::unordered_set<Node*> group_set(nodes.begin(), nodes.end());
  for(auto n : nodes) {
    auto nn = new_graph->appendNode(new_graph->createClone(n, getOrCreateInput));
    for(size_t i = 0; i < nn->outputs().size(); ++i) {
      auto old_output = n->outputs()[i];
      auto new_output = nn->outputs()[i];
      value_map[old_output] = new_output;
      std::vector<Use> to_replace;
      for(auto u : old_output->uses()) {
        // Uses within the set do not need to be made outputs
        if(group_set.count(u.user) > 0)
          continue;
        // Other uses do, but we
        // cannot replace them here or we invalid the uses list iterator
        to_replace.push_back(u);
      }
      if(to_replace.size() > 0) {
        new_graph->registerOutput(new_output);
        Value * external_output = group_node->addOutput()->setType(old_output->type());
        for(auto u : to_replace) {
          u.user->replaceInput(u.offset, external_output);
        }
      }
    }
  }
  group_node->insertBefore(nodes[0]);
  // delete backward, so that nodes are use-free before deletion
  for(size_t i = nodes.size(); i > 0; --i) {
    nodes[i - 1]->destroy();
  }
  JIT_ASSERT(isDifferentiable(*new_graph));
  return group_node;
}

bool shouldConsiderForMerge(detail::Vertex<Node*>* v) {
  if (v->data.size() >= 2) {
    return true;
  }
  JIT_ASSERT(v->data.size() == 1);
  auto * node = *v->data.begin();
  if (node->kind() == prim::Constant) {
    return false;
  }
  return isDifferentiable(node);
}

static detail::DynamicDAG<Node*> make_dependency_graph(Block * block) {
  detail::DynamicDAG<Node*> dag;
  std::unordered_map<Node*,detail::Vertex<Node*>*> node_to_vertex;
  // NB: the block's param and return nodes are not in the dependency graph.
  for (Node * node : block->nodes()) {
    node_to_vertex[node] = dag.newVertex(node);
  }
  for (auto * node : block->nodes()) {
    for (auto * v : node->outputs()) {
      for (auto & use : v->uses()) {
        // [Determine data dependencies]
        // Consider the following code:
        //     y = f(x)
        //     if k:
        //        w += y
        //     z = g(y)
        // This produces a dependency graph with 3 vertices:
        // (0: f)   (1: if k ...)   (2: g)
        // We need to peek into the if Node* to determine its data dependencies
        // (the body depends on the output of f, so Vertex 1 depends on Vertex 0).
        // For each Use of y, we find an owning node of y that is a part of the
        // dependency graph (in this case, the Vertex containing the if Node*)
        // and then record the dependency.
        auto * owning_node = use.user;
        if (owning_node == block->return_node()) {
          // The return node is not in the dag. Carry on.
          continue;
        }
        while (true) {
          auto search = node_to_vertex.find(owning_node);
          if (search == node_to_vertex.end()) {
            owning_node = owning_node->owningBlock()->owningNode();
            JIT_ASSERT(owning_node != nullptr);
            continue;
          }
          // NB: DynamicDAG is a simple graph (no multi-edges).
          // addEdge is a no-op if the edge already exists.
          dag.addEdge(node_to_vertex[node], search->second);
          break;
        }
      }
    }
  }
  return dag;
}

static void find_differentiable_groups(
    detail::DynamicDAG<Node*>& dep_graph,
    size_t distance_threshold=256,
    size_t producer_edge_threshold=16) {
  // A Vertex contains a Node* or a differentiable group of Node*.
  // Perform graph contraction on dep_graph: contract two vertices(x, y) if
  // the following conditions hold:
  // - x, y can be merged to form a differentiable group
  // - the contraction would not invalidate the dag (it creates no cycles).
  //
  // This performs a greedy algorithm. This greedy algorithm considers
  // dep_graph vertices in reverse topological order by reverse iterating through
  // ord indices. For a certain ord, we attempt to merge the vertex at that ord
  // with each of its parents. If the vertex at the ord cannot be merged with any
  // of its parents, then we move on to a smaller ord and repeat.
  //
  // Each contractEdge call is effectively constant because we limit the size
  // of the affected region (via the distance_threshold) and the fan in/fan out
  // via producer_edge_threshold.
  // In addition, each sort of in_edges is bounded by producer_edge threshold.
  // This makes the complexity of find_differential_groups effectively O(V + E).

  // Iterate in reverse topological order
  int64_t ord = dep_graph.max_size() - 1;
  for (int64_t ord = dep_graph.max_size() - 1; ord >= 0; --ord) {
    if (!dep_graph.at(ord)) continue;

    auto* consumer = dep_graph.at(ord).value();
    if (!shouldConsiderForMerge(consumer)) continue;

    // To bound the complexity of the sort. Makes the algorithm less optimal.
    if (consumer->in_edges().size() > producer_edge_threshold) continue;

    // Iterate through consumer->in_edges() in reverse topological order.
    // sort is performed once per ord in dep_graph and once per contraction.
    // There can be at most dep_graph.max_size() contractions, so
    // we do at most 2 * dep_graph.max_size() sorts.
    consumer->in_edges().sort();

    for (auto it = consumer->in_edges().rbegin(); it != consumer->in_edges().rend(); ++it) {
      auto * producer = *it;
      // The distance threshold makes this algorithm "not optimal": it will miss
      // some possible contraction opportunities, but it hopefully lets us:
      // 1) preserve locality of tensors. We don't want to keep them alive for too long.
      // 2) Help bound the computation complexity for contractEdge
      if (consumer->ord - producer->ord > distance_threshold) continue;
      if (!shouldConsiderForMerge(producer)) continue;

      // If the edge contraction is successful, dep_graph.at(ord) may have changed
      // as well as consumer->in_edges() so we break out of this loop
      if (dep_graph.contractEdge(producer, consumer)) {
        // Stay at the current ord until we are done considering the vertex
        // at this ord for contraction
        ++ord;
        break;
      }
    }
  }
}

static void reorder_according_to_dag(Block * block, const detail::DynamicDAG<Node*>& dep_graph) {
  for (size_t ord = 0; ord < dep_graph.max_size(); ++ord) {
    const auto& vertex = dep_graph.at(ord);
    if (!vertex.has_value()) continue;

    auto& nodes = vertex.value()->data;
    for (Node* node : nodes) {
      // Move all nodes according to the topological order in dep_graph. A lot
      // of the moves are unnecessary but this is a quick & easy solution.
      node->moveBefore(block->return_node());
    }
  }
}

static void merge_differentiable_groups(
    Block * block,
    const detail::DynamicDAG<Node*>& dep_graph,
    size_t size_threshold,
    std::vector<Node*>& diff_graphs) {
  for (size_t ord = 0; ord < dep_graph.max_size(); ++ord) {
    const auto& vertex = dep_graph.at(ord);
    if (!vertex) continue;
    if (!shouldConsiderForMerge(vertex.value())) continue;

    auto& nodes = vertex.value()->data;
    if (nodes.size() < size_threshold) continue;

    diff_graphs.push_back(mergeNodes(block, prim::DifferentiableGraph, nodes));
  }
}

void CreateAutodiffSubgraphsPK(
    Block * block,
    size_t size_threshold,
    std::vector<Node*>& diff_graphs) {
  for (auto * node : block->nodes()) {
    // Find subgraphs to run this on recursively.
    if (isDifferentiable(node)) continue;
    for (auto * sub_block : node->blocks()) {
      CreateAutodiffSubgraphsPK(sub_block, size_threshold, diff_graphs);
    }
  }

  auto dep_graph = make_dependency_graph(block);
  find_differentiable_groups(dep_graph);
  reorder_according_to_dag(block, dep_graph);
  merge_differentiable_groups(block, dep_graph, size_threshold, diff_graphs);
}

} // anonymous namespace

std::vector<Node*> CreateAutodiffSubgraphs(Graph & graph, size_t threshold) {
  std::vector<Node*> diff_nodes;
  CreateAutodiffSubgraphsPK(graph.block(), threshold, diff_nodes);
  return diff_nodes;
}

}}
