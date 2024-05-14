#include <torch/csrc/jit/passes/peephole.h>

#include <ATen/core/jit_type.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/concat_opt.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/peephole_alias_sensitive.h>
#include <torch/csrc/jit/passes/peephole_dict_idioms.h>
#include <torch/csrc/jit/passes/peephole_list_idioms.h>
#include <torch/csrc/jit/passes/peephole_non_tensor.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

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
      // NOLINTNEXTLINE(modernize-pass-by-value)
      const std::shared_ptr<Graph>& graph,
      bool disable_shape_peepholes)
      : graph_(graph), shape_peepholes_(!disable_shape_peepholes) {}

  bool run() {
    bool changed = optimizeBlock(graph_->block());
    changed |= PeepholeOptimizeListIdioms(graph_);
    changed |= PeepholeOptimizeDictIdioms(graph_);
    changed |= PeepholeOptimizeAliasSensitive(graph_, shape_peepholes_);
    changed |= PeepholeOptimizeNonTensor(graph_);
    changed |= CombineConcats(graph_);
    return changed;
  }

  // The intent for this optimization pass is to catch all of the small, easy to
  // catch peephole optimizations you might be interested in doing.
  //
  // TODO: Decide what kind of fixed point strategy we will have
  bool optimizeBlock(Block* block) {
    bool changed = false;
    for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
      auto* node = *it;

      for (Block* sub_block : node->blocks()) {
        changed |= optimizeBlock(sub_block);
      }

      // XXX: remember that if you want to simplify an expression by combining
      // multiple nodes into a different one, then you need to check that they
      // all belong to the given block
      // TODO: this doesn't work with Scalar-Tensor ops! We should
      // canonicalize those
      if (node->matches(
              "aten::_grad_sum_to_size(Tensor(a) self, int[]? size) -> Tensor(a)")) {
        // Eliminate no-op _grad_sum_to_size.
        // TODO: this doesn't work with Scalar-Tensor ops! We should
        // canonicalize those
        if (node->input(1)->mustBeNone()) {
          GRAPH_UPDATE(
              getHeader(node),
              " (x._grad_sum_to_size(x, None) == x) is replaced with ",
              node->input(0)->debugName());
          node->output()->replaceAllUsesWith(node->input(0));
          changed = true;
        } else {
          auto uses = node->output()->uses();
          for (Use u : uses) {
            if (u.user->matches(
                    "aten::_grad_sum_to_size(Tensor(a) self, int[]? size) -> Tensor(a)") &&
                u.user->input(1)->type()->isSubtypeOf(*ListType::ofInts())) {
              GRAPH_UPDATE(
                  getHeader(node),
                  " (x._grad_sum_to_size(y)._grad_sum_to_size(z) == x._grad_sum_to_size(z)) is replaced with ",
                  node->inputs().at(0)->debugName());
              u.user->replaceInput(0, node->inputs().at(0));
              changed = true;
            }
          }
        }
      } else if (
          node->matches(
              "aten::expand(Tensor self, int[] size, *, bool implicit) -> Tensor",
              /*const_inputs=*/attr::size)) {
        // x.expand(x.size()) == x
        auto input_type =
            node->namedInput(attr::self)->type()->cast<TensorType>();
        if (input_type && shape_peepholes_) {
          auto expanded_sizes = node->get<c10::List<int64_t>>(attr::size);
          auto input_type_sizes = input_type->sizes().concrete_sizes();
          if (expanded_sizes.has_value() && input_type_sizes &&
              expanded_sizes->vec() == *input_type_sizes) {
            GRAPH_UPDATE(
                getHeader(node),
                " (x.expand(x.size()) == x) is replaced with ",
                node->namedInput(attr::self)->debugName());
            node->output()->replaceAllUsesWith(node->namedInput(attr::self));
            changed = true;
          }
        }
      } else if (node->matches("aten::t(Tensor self) -> Tensor")) {
        // x.t().t() == x
        Node* input_node = node->input()->node();
        if (input_node->matches("aten::t(Tensor self) -> Tensor")) {
          GRAPH_UPDATE(
              getHeader(node),
              " (x.t().t() == x) is replaced with ",
              input_node->input()->debugName());
          node->output()->replaceAllUsesWith(input_node->input());
          changed = true;
        }
      } else if (
          node->matches("aten::type_as(Tensor self, Tensor other) -> Tensor") &&
          shape_peepholes_) {
        // x.type_as(y) == x iff x.type() == y.type()
        auto self_type = node->input(0)->type()->expect<TensorType>();
        auto other_type = node->input(1)->type()->expect<TensorType>();
        if (mustBeEqual(self_type->scalarType(), other_type->scalarType()) &&
            mustBeEqual(self_type->device(), other_type->device())) {
          GRAPH_UPDATE(
              getHeader(node),
              " (x.type_as(y) == x) is replaced with ",
              node->input(0)->debugName());
          node->output()->replaceAllUsesWith(node->input(0));
          changed = true;
        }
      } else if (
          node->kind() == aten::Float || node->kind() == aten::Int ||
          node->kind() == aten::FloatImplicit ||
          node->kind() == aten::IntImplicit ||
          node->kind() == aten::ScalarImplicit) {
        Node* input_node = node->input()->node();
        if (input_node->kind() == prim::NumToTensor) {
          GRAPH_UPDATE(
              getHeader(node),
              " (x.NumToTensor() == x) is replaced with ",
              node->input()->debugName());
          node->output()->replaceAllUsesWith(input_node->input());
          changed = true;
        }
      } else if (
          node->matches("aten::size(Tensor self) -> int[]") &&
          shape_peepholes_) {
        if (auto ptt = node->input()->type()->cast<TensorType>()) {
          if (auto sizes = ptt->sizes().concrete_sizes()) {
            GRAPH_UPDATE(
                getHeader(node),
                " (x.size()) is replaced with ",
                node->input()->debugName());
            WithInsertPoint guard(node);
            IValue ival(sizes);
            auto const_sizes_val = node->owningGraph()->insertConstant(ival);
            node->output()->replaceAllUsesWith(const_sizes_val);
            changed = true;
          }
        }
      } else if (
          node->matches("aten::len.t(t[] a) -> int") &&
          node->input()->node()->matches("aten::size(Tensor self) -> int[]") &&
          shape_peepholes_) {
        auto ptt = node->input()->node()->input()->type()->expect<TensorType>();
        // only handle one use case for now to avoid modifying mutated lists
        // TODO: canonicalize as aten::dim ?
        if (ptt->sizes().size() && node->input()->uses().size() == 1) {
          WithInsertPoint guard(node);
          auto output = node->owningGraph()->insertConstant(
              static_cast<int64_t>(*ptt->sizes().size()));
          GRAPH_UPDATE(
              "Replacing ",
              getHeader(node),
              " with a \"dim\" constant ",
              output->debugName());
          node->output()->replaceAllUsesWith(output);
          changed = true;
        }
      } else if (
          node->matches("aten::size(Tensor self, int dim) -> int") &&
          shape_peepholes_) {
        if (auto ptt = node->inputs().at(0)->type()->cast<TensorType>()) {
          if (auto maybe_ndim = ptt->sizes().size()) {
            auto ndim = *maybe_ndim;
            auto maybe_index = toIValue(node->inputs().at(1));
            if (!maybe_index) {
              continue;
            }
            int64_t index = maybe_index->toInt();
            int64_t norm_index = index < 0 ? ndim + index : index;
            if (norm_index >= 0 && norm_index < static_cast<int64_t>(ndim) &&
                ptt->sizes()[norm_index]) {
              WithInsertPoint guard(node);
              IValue ival(*ptt->sizes()[norm_index]);
              auto const_sizes_val = node->owningGraph()->insertConstant(ival);
              node->output()->replaceAllUsesWith(const_sizes_val);
              GRAPH_UPDATE(
                  getHeader(node),
                  " (x.size(dim)) is replaced with constant ",
                  const_sizes_val->debugName());
              changed = true;
            }
          }
        }
      } else if (
          node->matches("aten::is_floating_point(Tensor self) -> bool") &&
          shape_peepholes_) {
        auto ptt = node->inputs().at(0)->type()->cast<TensorType>();
        if (auto maybe_dtype = ptt->scalarType()) {
          c10::ScalarType dtype = *maybe_dtype;
          WithInsertPoint guard(node);
          IValue ival(at::isFloatingType(dtype));
          auto new_constant = node->owningGraph()->insertConstant(ival);
          node->output()->replaceAllUsesWith(new_constant);
          GRAPH_UPDATE(
              getHeader(node),
              " (x.is_floating_point()) is replaced with ",
              new_constant->debugName());
          changed = true;
        }
      } else if (
          node->matches("aten::is_complex(Tensor self) -> bool") &&
          shape_peepholes_) {
        auto ptt = node->inputs().at(0)->type()->cast<TensorType>();
        if (auto maybe_dtype = ptt->scalarType()) {
          c10::ScalarType dtype = *maybe_dtype;
          WithInsertPoint guard(node);
          IValue ival(at::isComplexType(dtype));
          auto new_constant = node->owningGraph()->insertConstant(ival);
          node->output()->replaceAllUsesWith(new_constant);
          GRAPH_UPDATE(
              getHeader(node),
              " (x.is_complex()) is replaced with ",
              new_constant->debugName());
          changed = true;
        }
      } else if (
          node->matches("prim::dtype(Tensor a) -> int") && shape_peepholes_) {
        auto ptt = node->input()->type()->expect<TensorType>();
        if (ptt->scalarType()) {
          WithInsertPoint guard(node);
          auto output = node->owningGraph()->insertConstant(
              static_cast<int64_t>(*ptt->scalarType()));
          GRAPH_UPDATE(
              "Replacing ",
              getHeader(node),
              " with a type constant ",
              output->debugName());
          node->output()->replaceAllUsesWith(output);
          changed = true;
        }
      } else if (
          node->matches("prim::device(Tensor a) -> Device") &&
          shape_peepholes_) {
        auto ptt = node->input()->type()->expect<TensorType>();
        if (ptt->device()) {
          WithInsertPoint guard(node);
          auto output = node->owningGraph()->insertConstant(*ptt->device());
          GRAPH_UPDATE(
              "Replacing ",
              getHeader(node),
              " with a device constant ",
              output->debugName());
          node->output()->replaceAllUsesWith(output);
          changed = true;
        }
      } else if (
          node->matches("aten::device(str type, int index) -> Device") &&
          shape_peepholes_) {
        auto string_type = node->inputs().at(0)->type()->expect<StringType>();
        if (string_type) {
          WithInsertPoint guard(node);
          std::string type_str = node->inputs().at(0)->node()->s(attr::value);
          auto maybe_index = toIValue(node->inputs().at(1));
          int64_t index = 0;
          if (maybe_index) {
            index = maybe_index->toInt();
          }
          auto device = c10::Device(type_str + ":" + std::to_string(index));
          auto output = node->owningGraph()->insertConstant(device);
          GRAPH_UPDATE(
              "Replacing ",
              getHeader(node),
              " with a device constant ",
              output->debugName());
          node->output()->replaceAllUsesWith(output);
          changed = true;
        }
      } else if (
          node->matches("aten::dim(Tensor self) -> int") && shape_peepholes_) {
        auto ptt = node->input()->type()->expect<TensorType>();
        if (auto dim = ptt->sizes().size()) {
          WithInsertPoint guard(node);
          auto output =
              node->owningGraph()->insertConstant(static_cast<int64_t>(*dim));
          GRAPH_UPDATE(
              "Replacing ",
              getHeader(node),
              " with a \"dim\" constant ",
              output->debugName());
          node->output()->replaceAllUsesWith(output);
          changed = true;
        }
      } else if (
          node->matches("prim::is_cuda(Tensor a) -> bool") &&
          shape_peepholes_) {
        auto ptt = node->input()->type()->expect<TensorType>();
        if (ptt->device()) {
          WithInsertPoint guard(node);
          auto output =
              node->owningGraph()->insertConstant((*ptt->device()).is_cuda());
          GRAPH_UPDATE(
              "Replacing ",
              getHeader(node),
              " with a is_cuda constant ",
              output->debugName());
          node->output()->replaceAllUsesWith(output);
          changed = true;
        }
      }
    }
    return changed;
  }

 private:
  std::shared_ptr<Graph> graph_;
  bool shape_peepholes_;
};

static bool FuseAddMM(Block* block) {
  bool changed = false;
  for (Node* node : block->nodes()) {
    // XXX: remember that if you want to simplify an expression by combining
    // multiple nodes into a different one, then you need to check that they
    // all belong to the given block
    if (node->matches(
            "aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
            /*const_inputs=*/attr::alpha)) {
      // z + x.mm(y) == z.addmm(x, y) == x.mm(y) + z
      if (node->get<at::Scalar>(attr::alpha).value().toDouble() == 1.) {
        // Look for mm from both sides of the add
        for (const auto mm_side : c10::irange(2)) {
          // Add will accept tensors of mismatched scalar types, as long as
          // one of them is a scalar, but addmm will throw in that case, so we
          // can only perform this fusion if we're sure that it is correct,
          // and for that we need the add_mat_type. An alternative would be to
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
            if (!type_as_mat->type()->expectRef<TensorType>().scalarType()) {
              type_as_mat = mat2;
            }
            auto mat_scalar_type =
                type_as_mat->type()->expectRef<TensorType>().scalarType();

            // we can't use type_as if we don't know the target type (mm), the
            // bias needs to be coerced to
            if (!mat_scalar_type) {
              continue;
            }

            // We insert the type_as if we're sure that the added element is a
            // scalar, and we either don't know the type of the scalar, or
            // know that it's mismatched.
            if (add_mat_type->sizes().size() &&
                *add_mat_type->sizes().size() == 0 &&
                !mustBeEqual(add_mat_type->scalarType(), mat_scalar_type)) {
              auto* type_as_node =
                  graph->insertNode(graph->create(aten::type_as, 1));
              type_as_node->addInput(add_mat);
              type_as_node->addInput(type_as_mat);
              add_mat = type_as_node->output();
              if (add_mat_type->isComplete()) {
                auto new_type =
                    add_mat_type->withScalarType(mat_scalar_type)->contiguous();
                add_mat->setType(new_type);
              }
            }

            auto* cOne = graph->insertConstant(1);
            auto* addmm_node = graph->insertNode(graph->create(aten::addmm, 1));
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
            changed = true;
            continue;
          }
        }
      }
    }
    for (Block* b : node->blocks()) {
      changed |= FuseAddMM(b);
    }
  }
  return changed;
}

// FuseAddMM is a separate pass from peephole optimize because it is currently
// used for exporting to ONNX.
// Today, fusing add + MM has no benefit within PyTorch running ATen
// ops. However, we rely on seeing the fused version of AddMM for ONNX export,
// since otherwise after ONNX translation we would see redundant Gemm ops with
// sub-optimal inputs.
// It won't be helpful for ATen until we're able to represent
//   torch.addmm(a, b, c, out=a).
// That's because addmm dispatches internally to gemm, which computes:
//   C = beta * C + alpha * A @ B
// but aten::addmm(a, b, c, 1, 1) is really:
//   D = beta * C + alpha * A @ B
// and because it works out of place on C, we're only trading off an
// explicit add for a copy inside the addmm function. Note that it
// doesn't even result in fewer reads, because mm won't even load C
// (because beta == 0 for it).
bool FuseAddMM(const std::shared_ptr<Graph>& graph) {
  bool changed = FuseAddMM(graph->block());
  GRAPH_DUMP("After FuseAddMM: ", graph);
  return changed;
}

bool PeepholeOptimize(
    const std::shared_ptr<Graph>& graph,
    bool addmm_fusion_enabled) {
  PeepholeOptimizeImpl peephole(graph, addmm_fusion_enabled);
  bool changed = peephole.run();
  GRAPH_DUMP("After PeepholeOptimize: ", graph);
  // Eliminate dead code created by any peephole passes we've just done
  if (changed) {
    EliminateDeadCode(graph->block());
  }
  return changed;
}

} // namespace jit
} // namespace torch
