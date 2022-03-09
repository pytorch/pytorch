#pragma once

#include <torch/csrc/lazy/core/tensor.h>

namespace torch_lazy_tensors {
namespace lazy_tensor_aten_ops {

//////////////////////////////////////////////////////////////////////////////
// ATEN operators follows here, listed in alphabetical order.
//////////////////////////////////////////////////////////////////////////////
// Takes a slice from the input as R1 at the specified offset and reshapes it
// into the provided size.
torch::lazy::LazyTensorPtr as_strided(const torch::lazy::LazyTensorPtr& input, std::vector<int64_t> size,
                      std::vector<int64_t> stride,
                      c10::optional<int64_t> storage_offset);

// In-place version of the method above.
void as_strided_(torch::lazy::LazyTensorPtr& input, std::vector<int64_t> size,
                 std::vector<int64_t> stride,
                 c10::optional<int64_t> storage_offset);

torch::lazy::LazyTensorPtr bernoulli(const torch::lazy::LazyTensorPtr& input, double probability);
torch::lazy::LazyTensorPtr bernoulli(const torch::lazy::LazyTensorPtr& input);
void bernoulli_(torch::lazy::LazyTensorPtr& input, double probability);
void bernoulli_(torch::lazy::LazyTensorPtr& input, const torch::lazy::LazyTensorPtr& probability);

torch::lazy::LazyTensorPtr expand(const torch::lazy::LazyTensorPtr& input,
                  std::vector<int64_t> size);

// Fills the input with the given value.
void fill_(torch::lazy::LazyTensorPtr& input, const at::Scalar& value);

// Returns a new tensor that is a narrowed view of the input in the given
// dimension.
torch::lazy::LazyTensorPtr narrow(const torch::lazy::LazyTensorPtr& input, int64_t dim, int64_t start,
                  int64_t length);

std::tuple<torch::lazy::LazyTensorPtr, torch::lazy::LazyTensorPtr, torch::lazy::LazyTensorPtr> ts_native_batch_norm(
    const torch::lazy::LazyTensorPtr& input, const torch::lazy::LazyTensorPtr& weight, const torch::lazy::LazyTensorPtr& bias,
    torch::lazy::LazyTensorPtr& running_mean, torch::lazy::LazyTensorPtr& running_var, bool training,
    double momentum, double eps);

std::tuple<torch::lazy::LazyTensorPtr, torch::lazy::LazyTensorPtr, torch::lazy::LazyTensorPtr> ts_native_batch_norm_backward(
    const torch::lazy::LazyTensorPtr& grad_out, const torch::lazy::LazyTensorPtr& input,
    const torch::lazy::LazyTensorPtr& weight, const torch::lazy::LazyTensorPtr& running_mean,
    const torch::lazy::LazyTensorPtr& running_var, const torch::lazy::LazyTensorPtr& save_mean,
    const torch::lazy::LazyTensorPtr& save_invstd, bool training, double eps,
    c10::ArrayRef<bool> output_mask);

// Permute the dimensions of this tensor according to the given permutation.
torch::lazy::LazyTensorPtr permute(const torch::lazy::LazyTensorPtr& input, c10::ArrayRef<int64_t> dims);

// Repeats the input tensor along each dimension by the given number of
// repeats.
torch::lazy::LazyTensorPtr repeat(const torch::lazy::LazyTensorPtr& input, std::vector<int64_t> repeats);

void copy_(torch::lazy::LazyTensorPtr& input, torch::lazy::LazyTensorPtr& src);

torch::lazy::LazyTensorPtr select(const torch::lazy::LazyTensorPtr& input, int64_t dim, int64_t index);

torch::lazy::LazyTensorPtr slice(const torch::lazy::LazyTensorPtr& input, int64_t dim, int64_t start,
                 int64_t end, int64_t step);

// Squeeze out all trivial (size 1) dimensions.
torch::lazy::LazyTensorPtr squeeze(const torch::lazy::LazyTensorPtr& input);

// Squeeze out the specified dimension index, if trivial (size 1). Returns
// unchanged input otherwise.
torch::lazy::LazyTensorPtr squeeze(const torch::lazy::LazyTensorPtr& input, int64_t dim);

// In-place versions of the methods above.
void squeeze_(torch::lazy::LazyTensorPtr& input);
void squeeze_(torch::lazy::LazyTensorPtr& input, int64_t dim);


std::tuple<torch::lazy::LazyTensorPtr, torch::lazy::LazyTensorPtr, torch::lazy::LazyTensorPtr> svd(
    const torch::lazy::LazyTensorPtr& input,
    bool some, bool compute_uv);

// Swap given dimensions of the input.
torch::lazy::LazyTensorPtr transpose(const torch::lazy::LazyTensorPtr& input, int64_t dim0, int64_t dim1);

// In-place version of the method above.
void transpose_(torch::lazy::LazyTensorPtr& input, int64_t dim0, int64_t dim1);

// Insert a dimension of size one at the specified position.
torch::lazy::LazyTensorPtr unsqueeze(const torch::lazy::LazyTensorPtr& input, int64_t dim);

// In-place version of the method above.
void unsqueeze_(torch::lazy::LazyTensorPtr& input, int64_t dim);

// Like reshape, but it returns a view into the original tensor.
torch::lazy::LazyTensorPtr view(const torch::lazy::LazyTensorPtr& input, c10::ArrayRef<int64_t> output_size);

}  // namespace lazy_tensor_aten_ops
}  // namespace torch_lazy_tensors
