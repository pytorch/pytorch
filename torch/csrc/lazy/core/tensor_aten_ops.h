#pragma once

#include <torch/csrc/lazy/core/tensor.h>

namespace torch {
namespace lazy {

//////////////////////////////////////////////////////////////////////////////
// ATEN operators follows here, listed in alphabetical order.
//////////////////////////////////////////////////////////////////////////////
// Takes a slice from the input as R1 at the specified offset and reshapes it
// into the provided size.
TORCH_API torch::lazy::LazyTensorPtr as_strided(const torch::lazy::LazyTensorPtr& input, std::vector<int64_t> size,
                      std::vector<int64_t> stride,
                      c10::optional<int64_t> storage_offset);

// In-place version of the method above.
TORCH_API void as_strided_(torch::lazy::LazyTensorPtr& input, std::vector<int64_t> size,
                 std::vector<int64_t> stride,
                 c10::optional<int64_t> storage_offset);

TORCH_API torch::lazy::LazyTensorPtr expand(const torch::lazy::LazyTensorPtr& input,
                  std::vector<int64_t> size);

// Fills the input with the given value.
TORCH_API void fill_(torch::lazy::LazyTensorPtr& input, const at::Scalar& value);

// Returns a new tensor that is a narrowed view of the input in the given
// dimension.
TORCH_API torch::lazy::LazyTensorPtr narrow(const torch::lazy::LazyTensorPtr& input, int64_t dim, int64_t start,
                  int64_t length);

TORCH_API std::tuple<torch::lazy::LazyTensorPtr, torch::lazy::LazyTensorPtr, torch::lazy::LazyTensorPtr> native_batch_norm(
    const torch::lazy::LazyTensorPtr& input, const torch::lazy::LazyTensorPtr& weight, const torch::lazy::LazyTensorPtr& bias,
    torch::lazy::LazyTensorPtr& running_mean, torch::lazy::LazyTensorPtr& running_var, bool training,
    double momentum, double eps);

TORCH_API std::tuple<torch::lazy::LazyTensorPtr, torch::lazy::LazyTensorPtr, torch::lazy::LazyTensorPtr> native_batch_norm_backward(
    const torch::lazy::LazyTensorPtr& grad_out, const torch::lazy::LazyTensorPtr& input,
    const torch::lazy::LazyTensorPtr& weight, const torch::lazy::LazyTensorPtr& running_mean,
    const torch::lazy::LazyTensorPtr& running_var, const torch::lazy::LazyTensorPtr& save_mean,
    const torch::lazy::LazyTensorPtr& save_invstd, bool training, double eps,
    c10::ArrayRef<bool> output_mask);

// Permute the dimensions of this tensor according to the given permutation.
TORCH_API torch::lazy::LazyTensorPtr permute(const torch::lazy::LazyTensorPtr& input, c10::ArrayRef<int64_t> dims);

// Repeats the input tensor along each dimension by the given number of
// repeats.
TORCH_API torch::lazy::LazyTensorPtr repeat(const torch::lazy::LazyTensorPtr& input, std::vector<int64_t> repeats);

TORCH_API void copy_(torch::lazy::LazyTensorPtr& input, torch::lazy::LazyTensorPtr& src);

TORCH_API torch::lazy::LazyTensorPtr select(const torch::lazy::LazyTensorPtr& input, int64_t dim, int64_t index);

TORCH_API torch::lazy::LazyTensorPtr slice(const torch::lazy::LazyTensorPtr& input, int64_t dim, int64_t start,
                 int64_t end, int64_t step);

// Squeeze out all trivial (size 1) dimensions.
TORCH_API torch::lazy::LazyTensorPtr squeeze(const torch::lazy::LazyTensorPtr& input);

// Squeeze out the specified dimension index, if trivial (size 1). Returns
// unchanged input otherwise.
TORCH_API torch::lazy::LazyTensorPtr squeeze(const torch::lazy::LazyTensorPtr& input, int64_t dim);

// In-place versions of the methods above.
TORCH_API void squeeze_(torch::lazy::LazyTensorPtr& input);
TORCH_API void squeeze_(torch::lazy::LazyTensorPtr& input, int64_t dim);


TORCH_API std::tuple<torch::lazy::LazyTensorPtr, torch::lazy::LazyTensorPtr, torch::lazy::LazyTensorPtr> svd(
    const torch::lazy::LazyTensorPtr& input,
    bool some, bool compute_uv);

// Swap given dimensions of the input.
TORCH_API torch::lazy::LazyTensorPtr transpose(const torch::lazy::LazyTensorPtr& input, int64_t dim0, int64_t dim1);

// In-place version of the method above.
TORCH_API void transpose_(torch::lazy::LazyTensorPtr& input, int64_t dim0, int64_t dim1);

// Insert a dimension of size one at the specified position.
TORCH_API torch::lazy::LazyTensorPtr unsqueeze(const torch::lazy::LazyTensorPtr& input, int64_t dim);

// In-place version of the method above.
TORCH_API void unsqueeze_(torch::lazy::LazyTensorPtr& input, int64_t dim);

// Like reshape, but it returns a view into the original tensor.
TORCH_API torch::lazy::LazyTensorPtr view(const torch::lazy::LazyTensorPtr& input, c10::ArrayRef<int64_t> output_size);
}  // namespace lazy
}  // namespace torch
