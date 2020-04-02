#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/tensor_meta.h>

/*
 * This file currently contains items associated with tensors, tensor domains,
 * tensor views and transforms associated with them (split, merge, reorder,
 * compute_at).
 *
 * Tensor is our link to the tensors described and used in the JIT. We create
 * our own wrapper version as a stepping stone into our IR structure, this
 * allows us to link our concept of tensors with that of the JIT.
 *
 * IterDomain for now is an annotated size. The size is a range for us to
 * iterate over (number of elements, not including stride). The annotations are
 * associated with if there's a parallelization mechanism associated with the
 * iter domain, and if we need to reduce over it.
 *
 * TensorDomain holds a vector (could be changed to an array) of IterDomains. It
 * holds an IterDomain for every logical axis in its associated tensor.
 * TensorDomain does not directly hold the Tensor it is associated.
 * TensorDomain's primary responsibility is to hold the history of
 * transformations that were used to generate it. This is done through the
 * normal interaction of Expr/Val in Fusion. i.e. if we want to know the
 * previous operation generating a particular TensorDomain we can simply call
 * FusionGuard::getCurFusion()->origin(a_tensor_domain) which should give us an
 * operation in the list [split, merge, reorder] or similar operations that take
 * in a TensorDomain, applies a transformation and outputs a tensor domain.
 *
 * TensorView is the glue between TensorDomain and Tensor. TensorView is
 * intended to be used directly in mathematical operations. TensorView is
 * directly used in the "what" is being computed. TensorView holds a reference
 * to the Tensor it's a view of, as well as the TensorDomain of that particular
 * view. TensorView provides the history of the what is being computed and that
 * history can be accessed, similar to the mechanism TensorDomain uses, through
 * normal Expr/Val interactions in Fusion. i.e.
 * FusionGuard::getCurFusion()->origin(a_tensor_view) which should give us an
 * operation that takes in a TensorView, other inputs (other TensorViews, or
 * Scalars) applies a mathematical operation and outputs a TensorView (and other
 * outputs?).
 *
 * The reason we need TensorView and TensorDomain is that we need to have a
 * record of both what is being computed and how it is being computed. For
 * Example we may have the operation: TV3[I, J, K] = TV2[I, J, K] + TV1[I, J, K]
 * The mathematical operationss here are on the tensor views TV1, TV2, and TV3.
 * This operation is a pointwise operation. To compute this pointwise operation
 * we iterate over the 3D TensorDomain [I, J, K], where K is the fastest
 * changing dimension.
 *
 * For now the functions split, merge, reorder, and compute_at are also in this
 * file and its associated .cpp file. However, they may be moved later.
 *
 */

namespace torch {
namespace jit {
namespace fuser {

struct TransformReplay;
struct TensorView;

TORCH_CUDA_API TensorView* split_(TensorView*, int axis, int factor);
TORCH_CUDA_API TensorView* merge_(TensorView*, int axis);
TORCH_CUDA_API TensorView* reorder_(
    TensorView*,
    const std::unordered_map<int, int>&);

} // namespace fuser
} // namespace jit
} // namespace torch
