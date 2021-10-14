#include <torch/csrc/jit/passes/onnx/preprocess_for_onnx.h>

#include <c10/util/irange.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/helper.h>

namespace torch {
namespace jit {

namespace onnx {
using namespace ::c10::onnx;
}

namespace {

at::optional<Node*> FindFusibleListUnpack(Node* n) {
  // 1. number of outputs is restricted to 1.
  // 2. output is only used by prim::ListUnpack.
  if (n->outputs().size() != 1) {
    return at::nullopt;
  }
  if (n->output()->uses().size() != 1) {
    return at::nullopt;
  }
  auto listUnpackNode = n->output()->uses()[0].user;
  if (listUnpackNode->kind() != prim::ListUnpack) {
    return at::nullopt;
  }
  return listUnpackNode;
}

// Fuse node + ListUnpack
// Node such as split/unbind produces tensor[] of static size,
// that is later unpacked by ListUnpack.
// This pass fuses the two nodes, and adds an additional input "_outputs" such
// that the symbolic function is aware of the number of outputs.
//
// Example IR
//  split.Tensor(Tensor(a) self, int split_size, int dim=0) -> Tensor(a)[]
//  split_with_sizes(Tensor self, int[] split_sizes, int dim=0) -> Tensor[]
//
// graph(%input : Float(5, 4, 3, strides=[12, 3, 1])):
//   %13 : int[] = prim::Constant[value=[2, 1, 2]]()
//   %7 : int = prim::Constant[value=0]()
//   %8 : Tensor[] = aten::split_with_sizes(%input, %13, %7)
//   %9 : Float(2, 4, 3, strides=[12, 3, 1]), %10 : Float(1, 4, 3, strides=[12,
//   3, 1]), %11 : Float(2, 4, 3, strides=[12, 3, 1]) = prim::ListUnpack(%8)
//   return (%9, %10, %11)
//
// After fusion
// graph(%input : Float(5, 4, 3, strides=[12, 3, 1])):
//   %13 : int[] = prim::Constant[value=[2, 1, 2]]()
//   %7 : int = prim::Constant[value=0]()
//   %8 : int = prim::Constant[value=3]()  # Adding addtional input of value 3
//      representing the number of outputs.
//   %14 : Float(2, 4, 3, strides=[12, 3, 1]), %15 : Float(1, 4, 3, strides=[12,
//      3, 1]), %16 : Float(2, 4, 3, strides=[12, 3, 1] =
//      aten::split_with_sizes(%input, %13, %7, %8) return (%14, %15, %16)
void FuseWithListUnpack(Node* n) {
  auto found_listUnpack = FindFusibleListUnpack(n);
  if (!found_listUnpack) {
    return;
  }

  auto listUnpack_node = found_listUnpack.value();

  TORCH_INTERNAL_ASSERT(n->outputs().size() == 1);
  // 1. Add internal input "_outputs" to node, so that later symbolic function
  //    conversion is aware of the number of outputs.
  // 2. Add the exact number of outputs to n, copy metadata and replace uses of
  //    listUnpack outputs.
  n->i_(
      Symbol::fromQualString("attr::_outputs"),
      static_cast<int64_t>(listUnpack_node->outputs().size()));

  for (size_t i = 0; i < listUnpack_node->outputs().size(); ++i) {
    auto new_output = n->addOutput();
    new_output->copyMetadata(listUnpack_node->output(i));
  }
  listUnpack_node->removeAllInputs();
  // remove original output, which is input to listUnpack node.
  n->eraseOutput(0);
  listUnpack_node->replaceAllUsesWith(n);
}

static void FuseWithListUnpack(Block* b) {
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    for (auto* child_block : it->blocks()) {
      FuseWithListUnpack(child_block);
    }

    auto n_kind = it->kind();
    switch (n_kind) {
      case aten::split:
      case aten::split_with_sizes:
      case aten::unsafe_split:
      case aten::unsafe_split_with_sizes:
      case aten::unbind:
      case aten::unsafe_chunk:
      case aten::where:
      case aten::nonzero_numpy:
        FuseWithListUnpack(*it);
        break;
      default:
        break;
    }
  }
}

// Replace aten::add with onnx::Concat
// when inputs to the add node are two int lists
//
// before the pass:
// graph(%x.1 : Float(2, 3, 4, strides=[12, 4, 1], requires_grad=0, device=cpu),
//  %y.1 : Float(1, 2, 3, strides=[6, 3, 1], requires_grad=0, device=cpu)):
//  %2 : None = prim::Constant()
//  %3 : int[] = aten::size(%x.1)
//  %l1.1 : int[] = aten::list(%3
//  %5 : int[] = aten::size(%y.1)
//  %l2.1 : int[] = aten::list(%5)
//  %7 : int[] = aten::add(%l1.1, %l2.1)
//  %8 : Tensor = aten::new_zeros(%x.1, %7, %2, %2, %2, %2)
//  return (%8)
//
// after the pass:
// graph(%x.1 : Float(2, 3, 4, strides=[12, 4, 1], requires_grad=0, device=cpu),
//  %y.1 : Float(1, 2, 3, strides=[6, 3, 1], requires_grad=0, device=cpu)):
//  %2 : None = prim::Constant()
//  %3 : int[] = aten::size(%x.1)
//  %l1.1 : int[] = aten::list(%3)
//  %5 : int[] = aten::size(%y.1)
//  %l2.1 : int[] = aten::list(%5)
//  %9 : Tensor = onnx::Concat[axis=0](%l1.1, %l2.1)
//  %8 : Tensor = aten::new_zeros(%x.1, %9, %2, %2, %2, %2)
//  return (%8)
static void ReplaceAddWithConcat(Block* b) {
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    for (auto* child_block : it->blocks()) {
      ReplaceAddWithConcat(child_block);
    }
    if (it->kind() == aten::add) {
      if (!it->input(0)->type()->cast<ListType>() ||
          !it->input(1)->type()->cast<ListType>()) {
        continue;
      }

      TypePtr elem =
          it->input(0)->type()->castRaw<ListType>()->getElementType();
      if (elem->cast<IntType>()) {
        Node* concat_node = b->owningGraph()->create(onnx::Concat, 1);
        concat_node->i_(attr::axis, 0);
        concat_node->insertBefore(*it);
        concat_node->addInput(it->input(0));
        concat_node->addInput(it->input(1));
        concat_node->outputs()[0]->setType(
            TensorType::fromNumberType(std::move(elem)));
        it->replaceAllUsesWith(concat_node);
        it->removeAllInputs();
        it.destroyCurrent();
      }
    }
  }
}

// This pass also covers the case when the input to ListUnpack
// is int[] comming from some other op than ListConstruct (like Slice or Shape)
//
// before the pass
// graph(%x.1 : Float(2, 3, strides=[3, 1], requires_grad=0, device=cpu)):
//   %1 : None = prim::Constant()
//   %2 : int[] = aten::size(%x.1)
//   %a.1 : int, %b.1 : int = prim::ListUnpack(%2)
//   %5 : int[] = prim::ListConstruct(%a.1, %b.1)
//   %6 : Tensor = aten::new_zeros(%x.1, %5, %1, %1, %1, %1)
//
// after the pass:
// graph(%x.1 : Float(2, 3, strides=[3, 1], requires_grad=0, device=cpu)):
//   %1 : None = prim::Constant()
//   %2 : int[] = aten::size(%x.1)
//   %7 : Tensor = onnx::Constant[value={0}]()
//   %8 : Tensor = onnx::Gather(%2, %7)
//   %9 : Tensor = onnx::Constant[value={1}]()
//   %10 : Tensor = onnx::Gather(%2, %9)
//   %a.1 : int, %b.1 : int = prim::ListUnpack(%2)
//   %5 : int[] = prim::ListConstruct(%8, %10)
//   %6 : Tensor = aten::new_zeros(%x.1, %5, %1, %1, %1, %1)
static void fuseListAndListUnpack(Block* b) {
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    for (auto* child_block : it->blocks()) {
      fuseListAndListUnpack(child_block);
    }
    if (it->kind() == prim::ListUnpack) {
      for (const auto i : c10::irange(it->outputs().size())) {
        auto output = it->outputs().at(i);
        if (it->inputs().size() == 1 &&
            it->input()->node()->kind() != prim::ListConstruct &&
            it->input()->type()->cast<ListType>() &&
            it->input()
                ->type()
                ->castRaw<ListType>()
                ->getElementType()
                ->cast<IntType>()) {
          Node* gather_indices = b->owningGraph()->create(onnx::Constant, 1);
          gather_indices->insertBefore(*it);
          gather_indices->t_(
              attr::value, at::scalar_to_tensor(at::Scalar(int(i))));
          Node* gather_node = b->owningGraph()->create(onnx::Gather, 1);
          gather_node->insertBefore(*it);
          gather_node->addInput(it->input());
          gather_node->addInput(gather_indices->output());
          output->replaceAllUsesWith(gather_node->output());
        }
      }
    }
  }
}

} // namespace

void PreprocessForONNX(std::shared_ptr<Graph>& graph) {
  FuseWithListUnpack(graph->block());
  GRAPH_DUMP("After FuseWithListUnpack: ", graph);
  ReplaceAddWithConcat(graph->block());
  GRAPH_DUMP("After ReplaceAddWithConcat: ", graph);
  fuseListAndListUnpack(graph->block());
  GRAPH_DUMP("After fuseListAndListUnpack: ", graph);
}

} // namespace jit
} // namespace torch
