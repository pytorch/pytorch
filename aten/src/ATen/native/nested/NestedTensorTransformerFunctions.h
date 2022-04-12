/**
 * Transformer-specific NestedTensor utility functions.
 *
 * Not co-located with NestedTensor core code yet because they only
 * support specific cases needed in transformers.
 */
#pragma once

#include <vector>

#include <c10/macros/Macros.h>
#include <c10/util/Optional.h>

namespace c10 {
class Scalar;
} // namespace c10

namespace at {
class Tensor;
namespace native {
struct NestedTensorImpl;

// Requires that self is a contiguous NestedTensor, other is not a
// NestedTensor, self.dim() == 3, and other.dim() == 2. Also, self
// must have a consistent last dimension across its included Tensors
// and that dimension must match other.size(0).
Tensor NestedTensor_matmul(const Tensor& self, const Tensor& other);

// Requires that mat1 is a contiguous NestedTensor, self & mat2 are
// not NestedTensors, mat1.dim() == 3, mat2.dim() == 2, and that mat1
// has a consistent last dimension across its included Tensors that
// matches mat2.size(0).
Tensor NestedTensor_times_Tensor_plus_Tensor_addmm(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const c10::Scalar& beta,
    const c10::Scalar& alpha,
    c10::optional<bool> use_gelu = c10::nullopt);

Tensor NestedTensor_add_NestedTensor_in_place(
    const Tensor& self,
    const Tensor& other);

TORCH_API Tensor NestedTensor_batch_offsets_from_size_tensor(
    const Tensor& sizes,
    int64_t extra_elements);

Tensor NestedTensor_from_padded_tensor_cpu(
    const Tensor& padded,
    const NestedTensorImpl& nt);

Tensor NestedTensor_to_mask(const Tensor& nt, c10::optional<int64_t> mask_dim);


template <typename T>
void remove_padding_kernelLauncher(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size);

template <typename T>
void remove_padding_transform0213_kernelLauncher(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size);
} // namespace native
} // namespace at
