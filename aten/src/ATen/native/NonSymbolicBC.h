#pragma once
#include <ATen/core/Tensor.h>
#include <c10/util/irange.h>
#include <ATen/core/IListRef.h>

namespace at::native {
// This file contains non-symbolic signatures for ops that we have sym-intified the signature of.
// However, in certain cases (such as static runtime), we call the native versions of the ops directly.
// In those cases, we will duplicate the signature here with non-symbolic ints, and also duplicate the C++ implementation.
TORCH_API at::Tensor reshape(const at::Tensor& self, at::IntArrayRef proposed_shape);
TORCH_API at::Tensor narrow(const at::Tensor& self, int64_t dim, int64_t start, int64_t length);
TORCH_API at::Tensor _sparse_coo_tensor_unsafe(const at::Tensor & indices, const at::Tensor & values, at::IntArrayRef size, c10::optional<at::ScalarType> dtype=c10::nullopt, c10::optional<at::Layout> layout=c10::nullopt, c10::optional<at::Device> device=c10::nullopt, c10::optional<bool> pin_memory=c10::nullopt);
TORCH_API at::Tensor nll_loss(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor>& weight_opt, int64_t reduction, int64_t ignore_index);
TORCH_API at::Tensor nll_loss2d(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor>& weight_opt, int64_t reduction, int64_t ignore_index);
// The below ops don't get a duplicated C++ implementation.
// They are backward ops, which make them very unlikely to be called directly
// by external code (at::native::trace_backward).
// They get their own declaration for BC purposes however.
TORCH_API at::Tensor _embedding_bag_backward(const at::Tensor & grad, const at::Tensor & indices, const at::Tensor & offsets, const at::Tensor & offset2bag, const at::Tensor & bag_size, const at::Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor> & per_sample_weights, int64_t padding_idx=-1);
TORCH_API at::Tensor _embedding_bag_sparse_backward(const at::Tensor & grad, const at::Tensor & indices, const at::Tensor & offsets, const at::Tensor & offset2bag, const at::Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const c10::optional<at::Tensor> & per_sample_weights, int64_t padding_idx=-1);
TORCH_API at::Tensor value_selecting_reduction_backward(const at::Tensor & grad, int64_t dim, const at::Tensor & indices, at::IntArrayRef sizes, bool keepdim);
TORCH_API at::Tensor trace_backward(const at::Tensor & grad, at::IntArrayRef sizes);
TORCH_API at::Tensor index_select_backward(const at::Tensor & grad, at::IntArrayRef self_sizes, int64_t dim, const at::Tensor & index);
TORCH_API at::Tensor select(const at::Tensor& self, int64_t dim, int64_t index);
TORCH_API std::vector<Tensor> tensor_split(const Tensor& self, IntArrayRef indices, int64_t dim);
} // namespace at::native
