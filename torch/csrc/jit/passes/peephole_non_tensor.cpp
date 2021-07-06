#include <torch/csrc/jit/passes/peephole.h>

#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>

namespace torch {
namespace jit {

struct PeepholeOptimizeNonTensorImpl {
  // NOLINTNEXTLINE(modernize-pass-by-value)
  PeepholeOptimizeNonTensorImpl(const std::shared_ptr<Graph>& graph)
      : graph_(graph) {}

  bool run() {
    return optimizeBlock(graph_->block());
  }

  bool optimizeBlock(Block* block) {
    bool changed = false;
    for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
      auto* node = *it;

      for (Block* sub_block : node->blocks()) {
        changed |= optimizeBlock(sub_block);
      }

      if (node->kind() != prim::Constant) {
        WithInsertPoint guard(node);
        // Any Value whose type is None should be replaced with a Constant
        // This can occur if a module has an optional attribute, and it is
        // initialized as None.
        for (Value* output : node->outputs()) {
          if (output->type()->cast<NoneType>()) {
            output->replaceAllUsesWith(graph_->insertConstant(IValue()));
            changed = true;
          }
        }
      }
      // XXX: remember that if you want to simplify an expression by combining
      // multiple nodes into a different one, then you need to check that they
      // all belong to the given block
      // TODO: this doesn't work with Scalar-Tensor ops! We should
      // canonicalize those
      if (node->kind() == prim::If) {
        IfView n(node);
        // this handles redundant short circuits like "x and True" or "x or
        // False"
        for (size_t i = 0; i < n.outputs().size(); ++i) {
          if (n.outputs().at(i)->type() != BoolType::get()) {
            continue;
          }
          bool true_val =
              constant_as<bool>(n.thenOutputs().at(i)).value_or(false);
          bool false_val =
              constant_as<bool>(n.elseOutputs().at(i)).value_or(true);
          // if an if node's output equals its condition replace output with
          // condition
          if (true_val && !false_val) {
            GRAPH_UPDATE(
                "Replacing ",
                n.outputs().at(i)->debugName(),
                " (True or False) with ",
                n.cond()->debugName());
            n.outputs().at(i)->replaceAllUsesWith(n.cond());
            changed = true;
          }
        }
      } else if (
          node->kind() == aten::__is__ || node->kind() == aten::__isnot__) {
        // if we are comparing a None value with a value that can't be None
        // replace the output with true if node is __isnot__ or false if node is
        // __is__
        AT_ASSERT(node->inputs().size() == 2);
        for (size_t check_none_index : {0, 1}) {
          bool input_must_be_none =
              node->inputs().at(check_none_index)->mustBeNone();
          bool other_must_not_be_none =
              node->inputs().at(1 - check_none_index)->mustNotBeNone();
          if (input_must_be_none && other_must_not_be_none) {
            WithInsertPoint guard(node);
            auto output = node->owningGraph()->insertConstant(
                node->kind() == aten::__isnot__);
            GRAPH_UPDATE(
                "Folding ", getHeader(node), " to ", output->debugName());
            node->output()->replaceAllUsesWith(output);
            changed = true;
          }
        }
      } else if (
          node->kind() == prim::unchecked_unwrap_optional ||
          node->kind() == aten::_unwrap_optional) {
        // we are unwrapping an input that can't be None, remove the unwrap
        auto input = node->input();
        if (input->mustNotBeNone()) {
          GRAPH_UPDATE(
              "Unwrapping ",
              getHeader(node),
              " as ",
              node->input(),
              " can't be optional");
          node->output()->replaceAllUsesWith(node->input());
          changed = true;
        }
      } else if (node->kind() == prim::unchecked_cast) {
        // unchecked_cast is not generated for tensor properties, so we are not
        // losing anything by calling unshapedType here
        auto input_type = unshapedType(node->input()->type());
        auto output_type = unshapedType(node->output()->type());
        if (input_type->isSubtypeOf(output_type)) {
          GRAPH_UPDATE(
              "Removing ",
              getHeader(node),
              " as input type subtypes output type");
          node->output()->replaceAllUsesWith(node->input());
          changed = true;
        }
      } else if (node->kind() == aten::ne || node->kind() == aten::eq) {
        if (node->inputs().size() != 2 ||
            node->inputs().at(0) != node->inputs().at(1)) {
          continue;
        }
        auto inp_kind = node->inputs().at(0)->type()->kind();
        // only handling common immutable types here because other types like
        // Tensor or list of Tensor might throw on aten::eq
        switch (inp_kind) {
          case TypeKind::BoolType:
          case TypeKind::IntType:
          case TypeKind::FloatType: {
            WithInsertPoint guard(node);
            node->output()->replaceAllUsesWith(
                graph_->insertConstant(node->kind() == aten::eq));
            changed = true;
          }
          default:
            break;
        }
      }
    }
    return changed;
  }

 private:
  std::shared_ptr<Graph> graph_;
};

bool PeepholeOptimizeNonTensor(const std::shared_ptr<Graph>& graph) {
  PeepholeOptimizeNonTensorImpl peephole(graph);
  bool changed = peephole.run();
  GRAPH_DUMP("After PeepholeOptimize: ", graph);
  return changed;
}

} // namespace jit
} // namespace torch
