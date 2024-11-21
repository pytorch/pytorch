#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/peephole_alias_sensitive.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <unordered_set>

namespace torch::jit {

// This pass only does optimizations which requires Alias Analysis
// It is separated out from Peephole Pass so that Peephole does not have
// maintain alias db correctness throughout the pass.
struct PeepholeOptimizeAliasSensitiveImpl {
  PeepholeOptimizeAliasSensitiveImpl(
      std::shared_ptr<Graph> graph,
      bool shape_peepholes)
      : graph_(std::move(graph)),
        aliasDb_(std::make_unique<AliasDb>(graph_)),
        shape_peepholes_(shape_peepholes) {}

  bool run() {
    return runBlock(graph_->block());
  }

 private:
  void replaceWithIValue(Value* v, const IValue& val) {
    WithInsertPoint guard(v->node());
    v->replaceAllUsesWith(v->owningGraph()->insertConstant(val));
  }

  bool isFloatingPoint(TensorType& t) {
    auto input_dtype = t.scalarType();
    return (
        shape_peepholes_ && input_dtype && at::isFloatingType(*input_dtype));
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
        if (dim_uses.empty()) {
          continue;
        }
        auto kind = node->kind();
        int64_t output_size =
            kind == aten::conv1d ? 3 : (kind == aten::conv2d ? 4 : 5);
        // This is to handle potential resize_ calls, however unlikely.
        // If we add more checks related to resize_ in the graph,
        // factor this out like collectResizeSet in shape_analysis.
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
      } else if (
          node->matches(
              "aten::add(Tensor self, Scalar other, Scalar alpha) -> Tensor",
              /*const_inputs=*/{attr::alpha, attr::other}) ||
          node->matches(
              "aten::sub(Tensor self, Scalar other, Scalar alpha) -> Tensor",
              /*const_inputs=*/{attr::alpha, attr::other})) {
        // x + 0 == x - 0 == x
        // if either scalar input is a float, than removing this operator could
        // remove type promotion and affect semantics
        if (!isFloatingPoint(node->input(0)->type()->expectRef<TensorType>())) {
          auto inps = node->inputs();
          if (!inps.at(1)->type()->isSubtypeOf(IntType::get()) ||
              !inps.at(2)->type()->isSubtypeOf(IntType::get())) {
            continue;
          }
        }

        if (node->get<at::Scalar>(attr::alpha)->toDouble() == 1 &&
            node->get<at::Scalar>(attr::other)->toDouble() == 0) {
          if (tryToReplaceOutputWithInput(node->input(0), node->output())) {
            GRAPH_UPDATE(
                getHeader(node),
                " (x + 0 == x - 0 == x) is replaced with ",
                node->input(0)->debugName());
            node->output()->replaceAllUsesWith(node->input(0));
            changed = true;
          }
        }
      } else if (
          node->matches(
              "aten::mul(Tensor self, Scalar other) -> Tensor",
              /*const_inputs=*/attr::other) ||
          node->matches(
              "aten::div(Tensor self, Scalar other) -> Tensor",
              /*const_inputs=*/attr::other)) {
        // x * 1 == x / 1 == x
        // is the node is a division or other isn't an integer, than removing
        // this operator could remove type promotion and affect semantics
        if (!isFloatingPoint(node->input(0)->type()->expectRef<TensorType>())) {
          if (node->kind() == aten::div ||
              !node->input(1)->type()->isSubtypeOf(IntType::get())) {
            continue;
          }
        }

        if (node->get<at::Scalar>(attr::other)->toDouble() == 1) {
          if (tryToReplaceOutputWithInput(node->input(0), node->output())) {
            GRAPH_UPDATE(
                getHeader(node),
                " (x * 1 == x / 1 == x) is replaced with ",
                node->input(0)->debugName());

            changed = true;
          }
        }
      }
    }
    return changed;
  }

  bool tryToReplaceOutputWithInput(Value* input, Value* output) {
    if (!aliasDb_->safeToChangeAliasingRelationship(input, output)) {
      return false;
    }
    // whenever we replace an output with an input, all of the aliasing
    // properties of the output are now present on the input.
    // For example, if the output aliases a graph output, the input will now
    // as well.
    // in order to avoid re-instantiating an alias db on each change, which
    // would be O(n^2), or inplace modifying it, which would involve
    // invalidating all of the memory dag caches, we just keep a set of values
    // which are "stale" (aliasing properties not up to date), and avoid doing
    // further optimizations on values which alias them
    if (aliasDb_->mayAlias({input, output}, stale_alias_values_)) {
      return false;
    }
    output->replaceAllUsesWith(input);
    stale_alias_values_.insert(input);
    stale_alias_values_.insert(output);
    return true;
  }

  ValueSet stale_alias_values_;
  std::shared_ptr<Graph> graph_;
  std::unique_ptr<AliasDb> aliasDb_;
  bool shape_peepholes_;
};

bool PeepholeOptimizeAliasSensitive(
    const std::shared_ptr<Graph>& graph,
    bool shape_peepholes) {
  PeepholeOptimizeAliasSensitiveImpl opt(graph, shape_peepholes);
  return opt.run();
}

} // namespace torch::jit
