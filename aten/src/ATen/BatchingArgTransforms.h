#pragma once

#include <ATen/BatchedTensorImpl.h>

namespace at {

// This file contains abstractions used for transforming *logical* vmap arguments
// into *physical* arguments. (Keep reading for definitions of these terms).

// NOTE: [Logical vs physical args]
// Consider the following vmap.
//   vmap(vmap(func, in_dims=(2,)), in_dims=(0,))(torch.ones(2, 3, 4))
// This would produce a BatchedTensor wrapping a Tensor of size [2, 3, 4],
// with batch dims 0 and 2:
//   BatchedTensor(ones(2, 3, 4), bdims=[(lvl=1,dim=0),(lvl=2,dim=2)])
//
// We say the *logical* view of the tensor has size [3] -- tensors inside
// `func` appear to have size [3].
// However, the *physical* underlying tensor (the one passed to vmap) has size
// [2, 3, 4].
//
// This notion of logical vs physical also extends to non-tensor arguments.
// Consider the previous tensor; let's assume the user called
// `torch.sum(tensor, dim=0)` inside of `func`. Then the logical
// dimension they are reducing over is dim 0 but the physical dim is dim 1
// (the first non-batch dimension)

// Forward declared; see NOTE: [What is a PhysicalView?]
struct PhysicalView;

// NOTE: [What is an ArgTransform?]
// An *ArgTransform* converts logical views of tensors to physical views.
//
// Batching rules use ArgTransforms to convert logical arguments to
// physical arguments, then call one or more at:: operator that handles the
// physical arguments, and then converts the physical result back to a logical
// argument.

// ArgTransform for operators that take tensors with multiple batch dims.
// Given one or more logical views on Tensors, `logicalToPhysical` 
// permutes all of the batch dims to the front of the tensor, aligns
// and expands the batch dims to match each other (according to their `level`),
// and returns a PhysicalView on the tensor(s).
struct TORCH_API MultiBatchArgTransform {
  static PhysicalView logicalToPhysical(const Tensor& logical_tensor);
  static std::vector<PhysicalView> logicalToPhysical(TensorList logical_tensors);
};

// NOTE: [What is a PhysicalView?]
// PhysicalView represents a physical view on a Tensor.
//
// One can use it to further convert logical dimension indices, logical shapes,
// and more to their physical variants, or convert a new (physical) tensor into
// a logical BatchedTensor. (TODO(rzou): these are not yet implemented).
struct TORCH_API PhysicalView {
  PhysicalView(Tensor&& tensor)
      : tensor_(tensor) {
    TORCH_INTERNAL_ASSERT(!isBatched(tensor));
  }

  Tensor& tensor() { return tensor_; }

 private:
  Tensor tensor_;
};


} // namespace at
