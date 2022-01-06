#pragma once

#include <torch/csrc/lazy/core/tensor.h>

namespace torch_lazy_tensors {
namespace lazy_tensor_aten_ops {

//////////////////////////////////////////////////////////////////////////////
// ATEN operators follows here, listed in alphabetical order.
//////////////////////////////////////////////////////////////////////////////
// Takes a slice from the input as R1 at the specified offset and reshapes it
// into the provided size.
torch::lazy::LazyTensor as_strided(const torch::lazy::LazyTensor& input, std::vector<int64_t> size,
                      std::vector<int64_t> stride,
                      c10::optional<int64_t> storage_offset);

// In-place version of the method above.
void as_strided_(torch::lazy::LazyTensor& input, std::vector<int64_t> size,
                 std::vector<int64_t> stride,
                 c10::optional<int64_t> storage_offset);

torch::lazy::LazyTensor bernoulli(const torch::lazy::LazyTensor& input, double probability);
torch::lazy::LazyTensor bernoulli(const torch::lazy::LazyTensor& input);
void bernoulli_(torch::lazy::LazyTensor& input, double probability);
void bernoulli_(torch::lazy::LazyTensor& input, const torch::lazy::LazyTensor& probability);

torch::lazy::LazyTensor expand(const torch::lazy::LazyTensor& input,
                  std::vector<int64_t> size);

// Fills the input with the given value.
void fill_(torch::lazy::LazyTensor& input, const at::Scalar& value);

// Returns a new tensor that is a narrowed view of the input in the given
// dimension.
torch::lazy::LazyTensor narrow(const torch::lazy::LazyTensor& input, int64_t dim, int64_t start,
                  int64_t length);

std::tuple<torch::lazy::LazyTensor, torch::lazy::LazyTensor, torch::lazy::LazyTensor> ts_native_batch_norm(
    const torch::lazy::LazyTensor& input, const torch::lazy::LazyTensor& weight, const torch::lazy::LazyTensor& bias,
    torch::lazy::LazyTensor& running_mean, torch::lazy::LazyTensor& running_var, bool training,
    double momentum, double eps);

std::tuple<torch::lazy::LazyTensor, torch::lazy::LazyTensor, torch::lazy::LazyTensor> ts_native_batch_norm_backward(
    const torch::lazy::LazyTensor& grad_out, const torch::lazy::LazyTensor& input,
    const torch::lazy::LazyTensor& weight, const torch::lazy::LazyTensor& running_mean,
    const torch::lazy::LazyTensor& running_var, const torch::lazy::LazyTensor& save_mean,
    const torch::lazy::LazyTensor& save_invstd, bool training, double eps,
    c10::ArrayRef<bool> output_mask);

std::pair<torch::lazy::LazyTensor, torch::lazy::LazyTensor> nms(const torch::lazy::LazyTensor& boxes,
                                      const torch::lazy::LazyTensor& scores,
                                      const torch::lazy::LazyTensor& score_threshold,
                                      const torch::lazy::LazyTensor& iou_threshold,
                                      int64_t output_size);

// Permute the dimensions of this tensor according to the given permutation.
torch::lazy::LazyTensor permute(const torch::lazy::LazyTensor& input, c10::ArrayRef<int64_t> dims);

// Repeats the input tensor along each dimension by the given number of
// repeats.
torch::lazy::LazyTensor repeat(const torch::lazy::LazyTensor& input, std::vector<int64_t> repeats);

torch::lazy::LazyTensor rsub(const torch::lazy::LazyTensor& input, const at::Scalar& other,
                const at::Scalar& alpha);

void copy_(torch::lazy::LazyTensor& input, torch::lazy::LazyTensor& src);

torch::lazy::LazyTensor select(const torch::lazy::LazyTensor& input, int64_t dim, int64_t index);

torch::lazy::LazyTensor slice(const torch::lazy::LazyTensor& input, int64_t dim, int64_t start,
                 int64_t end, int64_t step);

// Squeeze out all trivial (size 1) dimensions.
torch::lazy::LazyTensor squeeze(const torch::lazy::LazyTensor& input);

// Squeeze out the specified dimension index, if trivial (size 1). Returns
// unchanged input otherwise.
torch::lazy::LazyTensor squeeze(const torch::lazy::LazyTensor& input, int64_t dim);

// In-place versions of the methods above.
void squeeze_(torch::lazy::LazyTensor& input);
void squeeze_(torch::lazy::LazyTensor& input, int64_t dim);

torch::lazy::LazyTensor stack(c10::ArrayRef<torch::lazy::LazyTensor> tensors, int64_t dim);

torch::lazy::LazyTensor sub(const torch::lazy::LazyTensor& input, const torch::lazy::LazyTensor& other,
               const at::Scalar& alpha);
torch::lazy::LazyTensor sub(const torch::lazy::LazyTensor& input, const at::Scalar& other,
               const at::Scalar& alpha);

std::tuple<torch::lazy::LazyTensor, torch::lazy::LazyTensor, torch::lazy::LazyTensor> svd(const torch::lazy::LazyTensor& input,
                                                   bool some, bool compute_uv);

// Swap given dimensions of the input.
torch::lazy::LazyTensor transpose(const torch::lazy::LazyTensor& input, int64_t dim0, int64_t dim1);

// In-place version of the method above.
void transpose_(torch::lazy::LazyTensor& input, int64_t dim0, int64_t dim1);

// Insert a dimension of size one at the specified position.
torch::lazy::LazyTensor unsqueeze(const torch::lazy::LazyTensor& input, int64_t dim);

// In-place version of the method above.
void unsqueeze_(torch::lazy::LazyTensor& input, int64_t dim);

// Like reshape, but it returns a view into the original tensor.
torch::lazy::LazyTensor view(const torch::lazy::LazyTensor& input, c10::ArrayRef<int64_t> output_size);

}  // namespace lazy_tensor_aten_ops
}  // namespace torch_lazy_tensors
