#include <torch/csrc/lazy/core/tensor_list.h>

namespace torch {
namespace lazy {

TensorList::TensorList(OpList values)
  : Node(
      /*op=*/tensor_list_opkind,
      /*operands=*/values,
      /*shapes=*/std::vector<Shape>(),
      /*num_outputs=*/1,
      /*hash_seed=*/OperandHashes(values, /*seed=*/kHashSeed, enableDynamicShape())) {}

const Shape& TensorList::shape(size_t output_index) const {
  TORCH_CHECK(false, "NotImplementedError");
}

}  // namespace lazy
}  // namespace torch
