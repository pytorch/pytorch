#include <torch/csrc/jit/passes/peephole.h>

#include <torch/csrc/jit/symbolic_variable.h>

#include <torch/csrc/jit/passes/dead_code_elimination.h>

namespace torch {
namespace jit {

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
                                ->cast<CompleteTensorType>()) {
        auto expanded_sizes = node->get<std::vector<int64_t>>(attr::size);
        if (expanded_sizes == input_type->sizes()) {
          node->output()->replaceAllUsesWith(node->namedInput(attr::self));
        }
      }
    } else if (node->matches("aten::t(Tensor self) -> Tensor")) {
      // x.t().t() == x
      Node* input_node = node->input()->node();
      if (input_node->matches("aten::t(Tensor self) -> Tensor")) {
        node->output()->replaceAllUsesWith(input_node->input());
      }
    } else if (node->matches(
                   "aten::type_as(Tensor self, Tensor other) -> Tensor")) {
      // x.type_as(y) == x iff x.type() == y.type()
      auto self_type = node->input(0)->type()->cast<DimensionedTensorType>();
      auto other_type = node->input(1)->type()->cast<DimensionedTensorType>();
      if (self_type && other_type &&
          self_type->scalarType() == other_type->scalarType() &&
          self_type->device() == other_type->device()) {
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
              node->input(1 - mm_side)->type()->cast<DimensionedTensorType>();
          if (!add_mat_type)
            continue;

          if (node->input(mm_side)->node()->matches(
                  "aten::mm(Tensor self, Tensor mat2) -> Tensor")) {
            WithInsertPoint guard(node);

            auto mm_node = node->input(mm_side)->node();
            SymbolicVariable add_mat(node->input(1 - mm_side));
            SymbolicVariable mat1(mm_node->input(0));
            SymbolicVariable mat2(mm_node->input(1));

            auto mat_type = mat1.value()->type()->cast<DimensionedTensorType>();
            if (!mat_type) {
              mat_type = mat2.value()->type()->cast<DimensionedTensorType>();
            }
            // We insert the type_as if we're sure that the added element is a
            // scalar, and we either don't know what is the type of the
            // multiplied matrices, or know the type, and know that it's
            // mismatched.
            if (add_mat_type->dim() == 0 &&
                (!mat_type ||
                 add_mat_type->scalarType() != mat_type->scalarType())) {
              add_mat = add_mat.type_as(mat1);
            }

            SymbolicVariable addmm_value = add_mat.addmm(mat1, mat2);

            // Copy shape information from output node
            ((Value*)addmm_value)->copyMetadata(node->output());
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
        node->output()->replaceAllUsesWith(node->input(0));
      }
    } else if (
        node->kind() == prim::Float || node->kind() == prim::Int ||
        node->kind() == prim::ImplicitTensorToNum) {
      Node* input_node = node->input()->node();
      if (input_node->kind() == prim::NumToTensor) {
        node->output()->replaceAllUsesWith(input_node->input());
      }
    } else if (
        node->matches(
            "aten::_grad_sum_to_size(Tensor(a) self, int[] size) -> Tensor(a)")) {
      auto uses = node->output()->uses();
      for (Use u : uses) {
        if (u.user->matches(
                "aten::_grad_sum_to_size(Tensor(a) self, int[] size) -> Tensor(a)")) {
          u.user->replaceInput(0, node->inputs().at(0));
        }
      }
    }
  }
}

void PeepholeOptimize(Block* block, bool addmm_fusion_enabled) {
  PeepholeOptimizeImpl(block, addmm_fusion_enabled);
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
