#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/symbolic_variable.h>

namespace torch {
namespace jit {

// Conservatively compare two optionals. If both are undefined, assume
// they aren't equal
template <typename T>
static bool mustBeEqual(const c10::optional<T>& a, const c10::optional<T>& b) {
  return a == b && a.has_value();
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
// add + mm has no benefit within PyTorch running ATen ops. However, we rely on
// seeing the fused version of addmm for ONNX export, since after ONNX
// translation we would see redundant Gemm ops with sub-optimal inputs. This
// flag is exposed so that ONNX export can pass `true` to get the fused
// behavior, but normal JIT peephole optimization is left alone.
void PeepholeOptimizeImpl(Block* block, bool addmm_fusion_enabled) {
  for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
    auto* node = *it;

    for (Block* sub_block : node->blocks()) {
      PeepholeOptimizeImpl(sub_block, addmm_fusion_enabled);
    }

    // XXX: remember that if you want to simplify an expression by combining
    // multiple nodes into a different one, then you need to check that they all
    // belong to the given block
    if (node->matches(
            "aten::expand(Tensor self, int[] size, *, bool implicit) -> Tensor",
            /*const_inputs=*/attr::size)) {
      // x.expand(x.size()) == x
      if (auto input_type = node->namedInput(attr::self)
                                ->type()
                                ->cast<ProfiledTensorType>()) {
        auto expanded_sizes = node->get<c10::List<int64_t>>(attr::size);
        auto input_type_sizes = input_type->sizes().concrete_sizes();
        if (expanded_sizes.has_value() && input_type_sizes &&
            c10::impl::toVector(*expanded_sizes) == *input_type_sizes) {
          GRAPH_UPDATE(
              *node,
              " (x.expand(x.size()) == x) is replaced with ",
              node->namedInput(attr::self)->debugName());
          node->output()->replaceAllUsesWith(node->namedInput(attr::self));
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
      }
    } else if (node->matches(
                   "aten::type_as(Tensor self, Tensor other) -> Tensor")) {
      // x.type_as(y) == x iff x.type() == y.type()
      auto self_type = ProfiledTensorType::create(node->input(0)->type());
      auto other_type = ProfiledTensorType::create(node->input(1)->type());
      if (mustBeEqual(self_type->scalarType(), other_type->scalarType()) &&
          mustBeEqual(self_type->device(), other_type->device())) {
        GRAPH_UPDATE(
            *node,
            " (x.type_as(y) == x) is replaced with ",
            node->input(0)->debugName());
        node->output()->replaceAllUsesWith(node->input(0));
      }
    } else if (
        node->matches(
            "aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
            /*const_inputs=*/attr::alpha)) {
      // z + x.mm(y) == z.addmm(x, y) == x.mm(y) + z
      // This optimization has been disabled at the moment, because it's not
      // helpful at all until we will be able to represent torch.addmm(a, b, c,
      // out=a). That's because addmm dispatches internally to gemm, which
      // computes:
      //   C = beta * C + alpha * A @ B
      // but aten::addmm(a, b, c, 1, 1) is really:
      //   D = beta * C + alpha * A @ B
      // and because it works out of place on C, we're only trading off an
      // explicit add for a copy inside the addmm function. Note that it doesn't
      // even result in fewer reads, because mm won't even load C (because beta
      // == 0 for it).
      if (addmm_fusion_enabled &&
          node->get<at::Scalar>(attr::alpha).value().toDouble() == 1.) {
        // Look for mm from both sides of the add
        for (size_t mm_side = 0; mm_side < 2; mm_side++) {
          // Add will accept tensors of mismatched scalar types, as long as one
          // of them is a scalar. Addmm will throw in that case, so we can only
          // perform this fusion if we're sure that it is correct, and for that
          // we need the add_mat_type. An alternative would be to insert a
          // type_as conditional on the tensor shape being a scalar, but that
          // might add overhead, and make analysis harder.
          auto add_mat_type =
              ProfiledTensorType::create(node->input(1 - mm_side)->type());
          // if we don't have the rank, we can't tell if the bias is a scalar
          if (!add_mat_type->sizes().size()) {
            continue;
          }

          if (node->input(mm_side)->node()->matches(
                  "aten::mm(Tensor self, Tensor mat2) -> Tensor")) {
            WithInsertPoint guard(node);

            auto mm_node = node->input(mm_side)->node();
            SymbolicVariable add_mat(node->input(1 - mm_side));
            SymbolicVariable mat1(mm_node->input(0));
            SymbolicVariable mat2(mm_node->input(1));

            auto mat1_type = ProfiledTensorType::create(mat1.value()->type());
            auto mat_scalar_type = mat1_type->scalarType();
            if (!mat_scalar_type) {
              auto mat2_type = ProfiledTensorType::create(mat2.value()->type());
              mat_scalar_type = mat2_type->scalarType();
            }

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
              add_mat = add_mat.type_as(mat1);
            }

            SymbolicVariable addmm_value = add_mat.addmm(mat1, mat2);

            // Copy shape information from output node
            ((Value*)addmm_value)->copyMetadata(node->output());
            GRAPH_UPDATE(
                "Fusing ",
                mm_node->input(0)->debugName(),
                ", ",
                mm_node->input(1)->debugName(),
                " and ",
                node->input(1 - mm_side)->debugName(),
                " into ",
                addmm_value.value()->debugName());
            node->output()->replaceAllUsesWith(addmm_value);
          }
        }
      }
      // TODO: this doesn't work with Scalar-Tensor ops! We should canonicalize
      // those
    } else if (
        node->matches(
            "aten::mul(Tensor self, Scalar other) -> Tensor",
            /*const_inputs=*/attr::other) ||
        node->matches(
            "aten::div(Tensor self, Scalar other) -> Tensor",
            /*const_inputs=*/attr::other)) {
      // x * 1 == x / 1 == x
      if (node->get<at::Scalar>(attr::other)->toDouble() == 1) {
        GRAPH_UPDATE(
            *node,
            " (x * 1 == x / 1 == x) is replaced with ",
            node->input(0)->debugName());
        node->output()->replaceAllUsesWith(node->input(0));
      }
    } else if (
        node->matches(
            "aten::add(Tensor self, Scalar other, Scalar alpha) -> Tensor",
            /*const_inputs=*/{attr::alpha, attr::other}) ||
        node->matches(
            "aten::sub(Tensor self, Scalar other, Scalar alpha) -> Tensor",
            /*const_inputs=*/{attr::alpha, attr::other})) {
      // x + 0 == x - 0 == x
      if (node->get<at::Scalar>(attr::alpha)->toDouble() == 1 &&
          node->get<at::Scalar>(attr::other)->toDouble() == 0) {
        GRAPH_UPDATE(
            *node,
            " (x + 0 == x - 0 == x) is replaced with ",
            node->input(0)->debugName());
        node->output()->replaceAllUsesWith(node->input(0));
      }
    } else if (
        node->kind() == aten::Float || node->kind() == aten::Int ||
        node->kind() == prim::ImplicitTensorToNum) {
      Node* input_node = node->input()->node();
      if (input_node->kind() == prim::NumToTensor) {
        GRAPH_UPDATE(
            *node,
            " (x.NumToTensor().ImplicitTensorToNum() == x.NumToTensor()) is replaced with ",
            node->input()->debugName());
        node->output()->replaceAllUsesWith(input_node->input());
      }
    } else if (
        node->matches(
            "aten::_grad_sum_to_size(Tensor(a) self, int[]? size) -> Tensor(a)")) {
      if (node->input(1)->mustBeNone()) {
        GRAPH_UPDATE(
            *node,
            " (x._grad_sum_to_size(x, None) == x) is replaced with ",
            node->input(0)->debugName());
        node->output()->replaceAllUsesWith(node->input(0));
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
          }
        }
      }
    } else if (node->kind() == prim::If) {
      IfView n(node);
      // this handles redundant short circuits like "x and True" or "x or False"
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
        }
      }
    } else if (
        node->kind() == prim::unchecked_unwrap_optional ||
        node->kind() == aten::_unwrap_optional) {
      // we are unwrapping an input that can't be None, remove the unwrap
      auto input = node->input();
      if (input->mustNotBeNone()) {
        GRAPH_UPDATE(
            "Unwrapping ", *node, " as ", node->input(), " can't be optional");
        node->output()->replaceAllUsesWith(node->input());
      }
    } else if (node->matches("prim::dtype(Tensor a) -> int")) {
      auto ptt = ProfiledTensorType::create(node->input()->type());
      if (ptt->scalarType()) {
        WithInsertPoint guard(node);
        auto output = node->owningGraph()->insertConstant(
            static_cast<int64_t>(*ptt->scalarType()));
        GRAPH_UPDATE(
            "Replacing ", *node, " with a type constant ", output->debugName());
        node->output()->replaceAllUsesWith(output);
      }
    } else if (node->matches("prim::device(Tensor a) -> Device")) {
      auto ptt = ProfiledTensorType::create(node->input()->type());
      if (ptt->device()) {
        WithInsertPoint guard(node);
        auto output = node->owningGraph()->insertConstant(*ptt->device());
        GRAPH_UPDATE(
            "Replacing ",
            *node,
            " with a device constant ",
            output->debugName());
        node->output()->replaceAllUsesWith(output);
      }
    } else if (node->matches("aten::dim(Tensor self) -> int")) {
      auto ptt = ProfiledTensorType::create(node->input()->type());
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
      }
    } else if (node->matches("prim::is_cuda(Tensor a) -> bool")) {
      auto ptt = ProfiledTensorType::create(node->input()->type());
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
      }
    }
  }
}

void PeepholeOptimize(Block* block, bool addmm_fusion_enabled) {
  PeepholeOptimizeImpl(block, addmm_fusion_enabled);
  GRAPH_DUMP("After PeepholeOptimize: ", block->owningGraph());
  // Eliminate dead code created by any peephole passes we've just done
  EliminateDeadCode(block);
}

void PeepholeOptimize(
    const std::shared_ptr<Graph>& graph,
    bool addmm_fusion_enabled) {
  PeepholeOptimize(graph->block(), addmm_fusion_enabled);
}
} // namespace jit
} // namespace torch
