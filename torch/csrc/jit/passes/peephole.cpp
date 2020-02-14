#include <torch/csrc/jit/passes/peephole.h>
#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/graph_executor.h>
#include <torch/csrc/jit/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/alias_analysis.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/utils/memory.h>

namespace torch {
namespace jit {

// Conservatively compare two optionals. If both are undefined, assume
// they aren't equal
template <typename T>
static bool mustBeEqual(const c10::optional<T>& a, const c10::optional<T>& b) {
  return a == b && a.has_value();
}

struct PeepholeOptimizeImpl {
  PeepholeOptimizeImpl(
      const std::shared_ptr<Graph>& graph,
      bool addmm_fusion_enabled)
      : aliasDb_(nullptr),
        graph_(graph),
        changed_(true),
        addmm_fusion_enabled_(addmm_fusion_enabled) {
    run(graph->block());
  }

  // The intent for this optimization pass is to catch all of the small, easy to
  // catch peephole optimizations you might be interested in doing.
  //
  // Right now, it does:
  //    - Eliminate no-op 'expand' nodes
  //    - Simply x.t().t() to x
  //
  // TODO: Decide what kind of fixed point strategy we will have
  //
  // The parameter `addmm_fusion_enabled` exists because, as it is today, fusing
  // add + mm has no benefit within PyTorch running ATen ops. However, we rely
  // on seeing the fused version of addmm for ONNX export, since after ONNX
  // translation we would see redundant Gemm ops with sub-optimal inputs. This
  // flag is exposed so that ONNX export can pass `true` to get the fused
  // behavior, but normal JIT peephole optimization is left alone.
  void run(Block* block) {
    for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
      auto* node = *it;

      for (Block* sub_block : node->blocks()) {
        run(sub_block);
      }

      if (node->kind() != prim::Constant) {
        WithInsertPoint guard(node);
        // Any Value whose type is None should be replaced with a Constant
        // This can occur if a module has an optional attribute, and it is
        // initialized as None.
        for (Value* output : node->outputs()) {
          if (output->type()->cast<NoneType>()) {
            output->replaceAllUsesWith(graph_->insertConstant(IValue()));
          }
        }
      }

      // XXX: remember that if you want to simplify an expression by combining
      // multiple nodes into a different one, then you need to check that they
      // all belong to the given block
      if (node->matches(
              "aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
              /*const_inputs=*/attr::alpha)) {
        // z + x.mm(y) == z.addmm(x, y) == x.mm(y) + z
        // This optimization has been disabled at the moment, because it's not
        // helpful at all until we will be able to represent torch.addmm(a, b,
        // c, out=a). That's because addmm dispatches internally to gemm, which
        // computes:
        //   C = beta * C + alpha * A @ B
        // but aten::addmm(a, b, c, 1, 1) is really:
        //   D = beta * C + alpha * A @ B
        // and because it works out of place on C, we're only trading off an
        // explicit add for a copy inside the addmm function. Note that it
        // doesn't even result in fewer reads, because mm won't even load C
        // (because beta
        // == 0 for it).
        if (addmm_fusion_enabled_ &&
            node->get<at::Scalar>(attr::alpha).value().toDouble() == 1.) {
          // Look for mm from both sides of the add
          for (size_t mm_side = 0; mm_side < 2; mm_side++) {
            // Add will accept tensors of mismatched scalar types, as long as
            // one of them is a scalar. Addmm will throw in that case, so we can
            // only perform this fusion if we're sure that it is correct, and
            // for that we need the add_mat_type. An alternative would be to
            // insert a type_as conditional on the tensor shape being a scalar,
            // but that might add overhead, and make analysis harder.
            auto add_mat_type =
                node->input(1 - mm_side)->type()->expect<TensorType>();
            // if we don't have the rank, we can't tell if the bias is a scalar
            if (!add_mat_type->sizes().size()) {
              continue;
            }

            if (node->input(mm_side)->node()->matches(
                    "aten::mm(Tensor self, Tensor mat2) -> Tensor")) {
              WithInsertPoint guard(node);

              auto* graph = node->owningGraph();
              auto* mm_node = node->input(mm_side)->node();
              auto* add_mat = node->input(1 - mm_side);
              auto* mat1 = mm_node->input(0);
              auto* mat2 = mm_node->input(1);

              // Attempts to find a matrix with a defined scalar type to type as
              auto* type_as_mat = mat1;
              if (!type_as_mat->type()->expect<TensorType>()->scalarType()) {
                type_as_mat = mat2;
              }
              auto mat_scalar_type =
                  type_as_mat->type()->expect<TensorType>()->scalarType();

              // we can't use type_as if we don't know the target type (mm), the
              // bias needs to be coerced to
              if (!mat_scalar_type) {
                continue;
              }

              // We insert the type_as if we're sure that the added element is a
              // scalar, and we either don't know what is the type of the
              // scalar, or know the type, and know that it's
              // mismatched.
              if (add_mat_type->sizes().size() &&
                  *add_mat_type->sizes().size() == 0 &&
                  !mustBeEqual(add_mat_type->scalarType(), mat_scalar_type)) {
                auto* type_as_node =
                    graph->insertNode(graph->create(aten::type_as, 1));
                type_as_node->addInput(add_mat);
                type_as_node->addInput(type_as_mat);
                add_mat = type_as_node->output();
                if (add_mat_type->isComplete()) {
                  auto new_type = add_mat_type->withScalarType(mat_scalar_type)
                                      ->contiguous();
                  add_mat->setType(new_type);
                }
              }

              auto* cOne = graph->insertConstant(1);
              auto* addmm_node =
                  graph->insertNode(graph->create(aten::addmm, 1));
              addmm_node->addInput(add_mat);
              addmm_node->addInput(mat1);
              addmm_node->addInput(mat2);
              addmm_node->addInput(cOne);
              addmm_node->addInput(cOne);
              auto* addmm_value = addmm_node->output();

              // Copy shape information from output node
              addmm_value->copyMetadata(node->output());
              GRAPH_UPDATE(
                  "Fusing ",
                  mm_node->input(0)->debugName(),
                  ", ",
                  mm_node->input(1)->debugName(),
                  " and ",
                  node->input(1 - mm_side)->debugName(),
                  " into ",
                  addmm_value->debugName());
              node->output()->replaceAllUsesWith(addmm_value);
              changed_ = true;
              continue;
            }
          }
        }
        // TODO: this doesn't work with Scalar-Tensor ops! We should
        // canonicalize those
      } else if (
          node->matches(
              "aten::_grad_sum_to_size(Tensor(a) self, int[]? size) -> Tensor(a)")) {
        if (node->input(1)->mustBeNone()) {
          GRAPH_UPDATE(
              *node,
              " (x._grad_sum_to_size(x, None) == x) is replaced with ",
              node->input(0)->debugName());
          node->output()->replaceAllUsesWith(node->input(0));
          changed_ = true;
        } else {
          auto uses = node->output()->uses();
          for (Use u : uses) {
            if (u.user->matches(
                    "aten::_grad_sum_to_size(Tensor(a) self, int[]? size) -> Tensor(a)") &&
                u.user->input(1)->type()->isSubtypeOf(ListType::ofInts())) {
              GRAPH_UPDATE(
                  *node,
                  " (x._grad_sum_to_size(y)._grad_sum_to_size(z) == x._grad_sum_to_size(z)) is replaced with ",
                  node->inputs().at(0)->debugName());
              u.user->replaceInput(0, node->inputs().at(0));
              changed_ = true;
            }
          }
        }
      } else if (
          node->matches(
              "aten::expand(Tensor self, int[] size, *, bool implicit) -> Tensor",
              /*const_inputs=*/attr::size)) {
        // x.expand(x.size()) == x
        if (auto input_type =
                node->namedInput(attr::self)->type()->cast<TensorType>()) {
          auto expanded_sizes = node->get<c10::List<int64_t>>(attr::size);
          auto input_type_sizes = input_type->sizes().concrete_sizes();
          if (expanded_sizes.has_value() && input_type_sizes &&
              expanded_sizes->vec() == *input_type_sizes) {
            GRAPH_UPDATE(
                *node,
                " (x.expand(x.size()) == x) is replaced with ",
                node->namedInput(attr::self)->debugName());
            node->output()->replaceAllUsesWith(node->namedInput(attr::self));
            changed_ = true;
          }
        }
      } else if (node->matches("aten::t(Tensor self) -> Tensor")) {
        // x.t().t() == x
        Node* input_node = node->input()->node();
        if (input_node->matches("aten::t(Tensor self) -> Tensor")) {
          GRAPH_UPDATE(
              *node,
              " (x.t().t() == x) is replaced with ",
              input_node->input()->debugName());
          node->output()->replaceAllUsesWith(input_node->input());
          changed_ = true;
        }
      } else if (node->matches(
                     "aten::type_as(Tensor self, Tensor other) -> Tensor")) {
        // x.type_as(y) == x iff x.type() == y.type()
        auto self_type = node->input(0)->type()->expect<TensorType>();
        auto other_type = node->input(1)->type()->expect<TensorType>();
        if (mustBeEqual(self_type->scalarType(), other_type->scalarType()) &&
            mustBeEqual(self_type->device(), other_type->device())) {
          GRAPH_UPDATE(
              *node,
              " (x.type_as(y) == x) is replaced with ",
              node->input(0)->debugName());
          node->output()->replaceAllUsesWith(node->input(0));
          changed_ = true;
        }
      } else if (
          node->kind() == aten::Float || node->kind() == aten::Int ||
          node->kind() == aten::FloatImplicit || node->kind() == aten::IntImplicit ||
          node->kind() == aten::ScalarImplicit) {
        Node* input_node = node->input()->node();
        if (input_node->kind() == prim::NumToTensor) {
          GRAPH_UPDATE(
              *node,
              " (x.NumToTensor().TensorToNum() == x.NumToTensor()) is replaced with ",
              node->input()->debugName());
          node->output()->replaceAllUsesWith(input_node->input());
          changed_ = true;
        }
      } else if (node->matches("aten::size(Tensor self) -> int[]") ||
                 node->kind() == prim::shape) {
        if (auto ptt = node->input()->type()->cast<TensorType>()) {
          if (auto sizes = ptt->sizes().concrete_sizes()) {
            WithInsertPoint guard(node);
            IValue ival(sizes);
            auto const_sizes_val = node->owningGraph()->insertConstant(ival);
            node->output()->replaceAllUsesWith(const_sizes_val);
            changed_ = true;
          }
        }
      } else if (node->kind() == prim::If) {
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
            changed_ = true;
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
            GRAPH_UPDATE("Folding ", *node, " to ", output->debugName());
            node->output()->replaceAllUsesWith(output);
            changed_ = true;
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
              *node,
              " as ",
              node->input(),
              " can't be optional");
          node->output()->replaceAllUsesWith(node->input());
          changed_ = true;
        }
      } else if (node->kind() == prim::unchecked_cast) {
        // unchecked_cast is not generated for tensor properties, so we are not
        // losing anything by calling unshapedType here
        auto input_type = unshapedType(node->input()->type());
        auto output_type = unshapedType(node->output()->type());
        if (input_type->isSubtypeOf(output_type)) {
          GRAPH_UPDATE(
              "Removing ", *node, " as input type subtypes output type");
          node->output()->replaceAllUsesWith(node->input());
        }
      } else if (node->matches("prim::dtype(Tensor a) -> int")) {
        auto ptt = node->input()->type()->expect<TensorType>();
        if (ptt->scalarType()) {
          WithInsertPoint guard(node);
          auto output = node->owningGraph()->insertConstant(
              static_cast<int64_t>(*ptt->scalarType()));
          GRAPH_UPDATE(
              "Replacing ",
              *node,
              " with a type constant ",
              output->debugName());
          node->output()->replaceAllUsesWith(output);
          changed_ = true;
        }
      } else if (node->matches("prim::device(Tensor a) -> Device")) {
        auto ptt = node->input()->type()->expect<TensorType>();
        if (ptt->device()) {
          WithInsertPoint guard(node);
          auto output = node->owningGraph()->insertConstant(*ptt->device());
          GRAPH_UPDATE(
              "Replacing ",
              *node,
              " with a device constant ",
              output->debugName());
          node->output()->replaceAllUsesWith(output);
          changed_ = true;
        }
      } else if (node->matches("aten::dim(Tensor self) -> int")) {
        auto ptt = node->input()->type()->expect<TensorType>();
        if (auto dim = ptt->sizes().size()) {
          WithInsertPoint guard(node);
          auto output =
              node->owningGraph()->insertConstant(static_cast<int64_t>(*dim));
          GRAPH_UPDATE(
              "Replacing ",
              *node,
              " with a \"dim\" constant ",
              output->debugName());
          node->output()->replaceAllUsesWith(output);
          changed_ = true;
        }
      } else if (node->matches("prim::is_cuda(Tensor a) -> bool")) {
        auto ptt = node->input()->type()->expect<TensorType>();
        if (ptt->device()) {
          WithInsertPoint guard(node);
          auto output =
              node->owningGraph()->insertConstant((*ptt->device()).is_cuda());
          GRAPH_UPDATE(
              "Replacing ",
              *node,
              " with a is_cuda constant ",
              output->debugName());
          node->output()->replaceAllUsesWith(output);
          changed_ = true;
        }
      }

      // these transformations rely on AA for correctness
      // see `runAliasingSensitivePeepholeTransformations` for more details
      runAliasingSensitivePeepholeTransformations(node);
    }
  }

  bool safeToChangeAliasingRelationship(Node* node) {
    if (changed_) {
      aliasDb_ = torch::make_unique<AliasDb>(graph_);
      changed_ = false;
    }

    return aliasDb_->safeToChangeAliasingRelationship(
        node->inputs(), node->outputs());
  }

  // if either the inputs or outputs of an op alias graph's inputs or
  // outputs, the transformations below are invalid
  // An example:
  //
  // def test_write(x):
  //     s = 0
  //     s += x
  //     s += x
  //     return s
  //
  void runAliasingSensitivePeepholeTransformations(Node* node) {
    if (node->matches(
            "aten::add(Tensor self, Scalar other, Scalar alpha) -> Tensor",
            /*const_inputs=*/{attr::alpha, attr::other}) ||
        node->matches(
            "aten::sub(Tensor self, Scalar other, Scalar alpha) -> Tensor",
            /*const_inputs=*/{attr::alpha, attr::other})) {
      // x + 0 == x - 0 == x
      if (node->get<at::Scalar>(attr::alpha)->toDouble() == 1 &&
          node->get<at::Scalar>(attr::other)->toDouble() == 0) {
        if (!safeToChangeAliasingRelationship(node)) {
          return;
        }
        GRAPH_UPDATE(
            *node,
            " (x + 0 == x - 0 == x) is replaced with ",
            node->input(0)->debugName());
        node->output()->replaceAllUsesWith(node->input(0));
        changed_ = true;
      }
    } else if (
        node->matches(
            "aten::mul(Tensor self, Scalar other) -> Tensor",
            /*const_inputs=*/attr::other) ||
        node->matches(
            "aten::div(Tensor self, Scalar other) -> Tensor",
            /*const_inputs=*/attr::other)) {
      // x * 1 == x / 1 == x
      if (node->get<at::Scalar>(attr::other)->toDouble() == 1) {
        if (!safeToChangeAliasingRelationship(node)) {
          return;
        }
        GRAPH_UPDATE(
            *node,
            " (x * 1 == x / 1 == x) is replaced with ",
            node->input(0)->debugName());
        node->output()->replaceAllUsesWith(node->input(0));
        changed_ = true;
      }
    }
  }

 private:
  std::unique_ptr<AliasDb> aliasDb_ = nullptr;
  std::shared_ptr<Graph> graph_;
  bool changed_;
  bool addmm_fusion_enabled_;
};

void PeepholeOptimize(
    const std::shared_ptr<Graph>& graph,
    bool addmm_fusion_enabled) {
  PeepholeOptimizeImpl peephole(graph, addmm_fusion_enabled);
  GRAPH_DUMP("After PeepholeOptimize: ", graph);
  // Eliminate dead code created by any peephole passes we've just done
  EliminateDeadCode(graph->block());
}

} // namespace jit
} // namespace torch
