#include <torch/csrc/jit/codegen/onednn/graph_fuser.h>
#include <torch/csrc/jit/codegen/onednn/graph_helper.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

namespace torch::jit::fuser::onednn {

void CreateLlgaSubgraphs(std::shared_ptr<Graph>& graph) {
  AliasDb db(graph);
  GraphRewriter graphRewriter(graph->block(), graph, db);
  // We maintain alias db correctness in-place while building up the LLGA
  // subgraphs, however it is difficult to preserve correctness when
  // un-inlining autodiff subgraphs. We first recursively construct all
  // subgraphs and then recursively cleanup & unmerge the small subgraphs
  graphRewriter.buildupSubgraphs();
  graphRewriter.cleanupSubgraphs();
  // Run CSE globally onceto eliminate duplicates that may have occurred
  // while inlining subgraphs.
  EliminateCommonSubexpression(graph);
  EliminateDeadCode(graph);
}

} // namespace torch::jit::fuser::onednn
