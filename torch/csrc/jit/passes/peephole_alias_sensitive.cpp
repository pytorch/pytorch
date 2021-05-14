#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/peephole_alias_sensitive.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/utils/memory.h>

namespace torch {
namespace jit {

// This pass only does optimizations which requires Alias Analysis
// It is seprated out from Peephole Pass so that Peephole does not have
// maintain alias db correctness throughout the pass.
// In the future `runAliasingSensitivePeepholeTransformations`
// in peephole.cpp can be incorporated and keep the alias-db
// correct throughout transformations so we only need to build it once
struct PeepholeOptimizeAliasSensitiveImpl {
  PeepholeOptimizeAliasSensitiveImpl(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)),
        aliasDb_(torch::make_unique<AliasDb>(graph_)) {}

  bool run() {
    return runBlock(graph_->block());
  }

 private:
  void replaceWithIValue(Value* v, IValue val) {
    WithInsertPoint guard(v->node());
    v->replaceAllUsesWith(v->owningGraph()->insertConstant(val));
  }

  bool runBlock(Block* block) {
    bool changed = false;
    for (Node* node : block->nodes()) {
      for (Block* b : node->blocks()) {
        changed |= runBlock(b);
      }

      // dim(conv(x)) extremely common and prevents Conv->BN fusion
      if (node->kind() == aten::conv1d || node->kind() == aten::conv2d ||
          node->kind() == aten::conv3d) {
        auto dim_uses = c10::filter(node->output()->uses(), [](const Use& use) {
          return use.user->kind() == aten::dim;
        });
        if (dim_uses.size() == 0) {
          continue;
        }
        auto kind = node->kind();
        int64_t output_size =
            kind == aten::conv1d ? 3 : (kind == aten::conv2d ? 4 : 5);
        // this is to handle potential resize_ calls, however unlikely
        // if we add more checks related to resize_ in the graph,
        // factor this out like collectResizeSet in shape_analysis
        if (!aliasDb_->hasWriters(node->output())) {
          for (const Use& dim_use : dim_uses) {
            replaceWithIValue(dim_use.user->output(), output_size);
          }
          changed = true;
        } else {
          for (const Use& dim_use : dim_uses) {
            if (aliasDb_->moveAfterTopologicallyValid(node, dim_use.user)) {
              replaceWithIValue(dim_use.user->output(), output_size);
              changed = true;
            }
          }
        }
        continue;
      }
    }
    return changed;
  }

  std::shared_ptr<Graph> graph_;
  std::unique_ptr<AliasDb> aliasDb_;
};

bool PeepholeOptimizeAliasSensitive(const std::shared_ptr<Graph>& graph) {
  PeepholeOptimizeAliasSensitiveImpl opt(graph);
  return opt.run();
}

} // namespace jit
} // namespace torch
