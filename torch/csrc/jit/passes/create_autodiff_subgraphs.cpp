#include <cstddef>
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/autodiff.h"

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
void mergeNodes(Block * block, Symbol group_node_kind, ArrayRef<Node*> nodes) {
  JIT_ASSERT(nodes.size() > 0);
  std::unordered_map<Value*, Value*> value_map;
  Graph * graph = block->owningGraph();

  auto new_graph = std::make_shared<Graph>();
  Node * group_node = graph->create(group_node_kind, 0);
  group_node->g_(kSubgraph, new_graph);

  auto getOrCreateInput = [&](Value * v) {
    if(value_map.count(v) > 0) {
      return value_map[v];
    }
    Value * nv = new_graph->addInput()->setType(v->type());
    group_node->addInput(v);
    value_map[v] = nv;
    return nv;
  };
  std::unordered_set<Node*> group_set;
  for(auto n : nodes) {
    group_set.insert(n);
  }
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
}

}

void CreateAutodiffSubgraphs(Block * block, size_t threshold) {
  // This implementation is not optimal, but it is simple.
  // It just scans through the list in order looking for runs of
  // differentiable ops, and then grouping them together when
  // it hits the first non-differentiable op.
  // It cannot handle things like:
  // a = f(x, y)
  // b = black_box(a)
  // c = g(a)
  // where you could group {f, g} together if the nodes were in a different
  // topological order

  // a better strategy would be to try to treat this like a fusion problem
  // and group maximal groups

  std::vector<Node*> groupable;
  for(Node * node : block->nodes()) { // Note: nodes() iterator stays valid since it is
                            // always pointing _after_ the nodes that mergeNodes
                            // mutates.
    if(isDifferentiable(node)) {
      groupable.push_back(node);
    } else {
      if(groupable.size() >= threshold) {
        mergeNodes(block, kGraphExecutor, groupable);
      }
      groupable.clear();
      for (Block * sub_block : node->blocks()) {
        CreateAutodiffSubgraphs(sub_block, threshold);
      }
    }
  }
  if(groupable.size() >= threshold) {
    mergeNodes(block, kGraphExecutor, groupable);
  }
}

void CreateAutodiffSubgraphs(Graph & graph, size_t threshold) {
  CreateAutodiffSubgraphs(graph.block(), threshold);
}


}}
