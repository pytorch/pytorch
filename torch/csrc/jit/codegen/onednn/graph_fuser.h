#pragma once

#include <torch/csrc/jit/codegen/onednn/graph_helper.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

struct WorkBlock : public std::pair<Node*, Node*> {
  using pair::pair;

  Node* begin() {
    return this->first;
  }
  Node* end() {
    return this->second;
  }
};

class GraphRewriter {
 public:
  GraphRewriter(Block* block, std::shared_ptr<Graph> graph, AliasDb& aliasDb)
      : block_(block),
        graph_(std::move(graph)),
        aliasDb_(aliasDb),
        llgaHelper_(graph_) {}

  void cleanupSubgraphs();
  void buildupSubgraphs();

 private:
  Block* block_;
  std::shared_ptr<Graph> graph_;
  AliasDb& aliasDb_;
  LlgaGraphHelper llgaHelper_;
  std::vector<WorkBlock> buildWorkBlocks();
  std::pair<graph_node_list::iterator, bool> scanNode(
      Node* consumer,
      graph_node_list::iterator workblock_begin);
  std::optional<Node*> tryMerge(Node* consumer, Node* producer);
};

// This pass creates the subgraphs for oneDNN Graph Fusion Nodes.
// Its code-structure has been vastly inspired from
// torch/csrc/jit/passes/create_autodiff_subgraphs.cpp
void CreateLlgaSubgraphs(std::shared_ptr<Graph>& graph);

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
