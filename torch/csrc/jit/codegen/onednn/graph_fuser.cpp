#include <torch/csrc/jit/codegen/onednn/graph_fuser.h>
#include <torch/csrc/jit/codegen/onednn/graph_helper.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

void GraphRewriter::run() {
  // We maintain alias db correctness in-place while building up the LLGA
  // subgraphs, however it is difficult to preserve correctness when
  // un-inlining autodiff subgraphs. We first recursively construct all
  // subgraphs and then recursively cleanup & unmerge the small subgraphs
  buildupSubgraphs();
  cleanupSubgraphs();
  // Run CSE globally onceto eliminate duplicates that may have occurred
  // while inlining subgraphs.
  EliminateCommonSubexpression(graph_);
  EliminateDeadCode(graph_);
}

void CreateLlgaSubgraphs(std::shared_ptr<Graph>& graph) {
  AliasDb db(graph);
  GraphRewriter(graph->block(), graph, db).run();
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
