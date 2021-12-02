#pragma once

#include "lazy_tensor_core/csrc/tensor.h"

namespace torch_lazy_tensors {
namespace lazy_tensor_aten_ops {

//////////////////////////////////////////////////////////////////////////////
// ATEN operators follows here, listed in alphabetical order.
//////////////////////////////////////////////////////////////////////////////
// Takes a slice from the input as R1 at the specified offset and reshapes it
// into the provided size.
LazyTensor as_strided(const LazyTensor& input, std::vector<int64_t> size,
                      std::vector<int64_t> stride,
                      c10::optional<int64_t> storage_offset);

// In-place version of the method above.
void as_strided_(LazyTensor& input, std::vector<int64_t> size,
                 std::vector<int64_t> stride,
                 c10::optional<int64_t> storage_offset);

LazyTensor bernoulli(const LazyTensor& input, double probability);
LazyTensor bernoulli(const LazyTensor& input);
void bernoulli_(LazyTensor& input, double probability);
void bernoulli_(LazyTensor& input, const LazyTensor& probability);

// Pad with the given value and size specified by the given list of low and
// high paddings.
LazyTensor constant_pad_nd(const LazyTensor& input, c10::ArrayRef<int64_t> pad,
                           const at::Scalar& value);

LazyTensor convolution_overrideable(
    const LazyTensor& input, const LazyTensor& weight, const LazyTensor& bias,
    std::vector<int64_t> stride, std::vector<int64_t> padding,
    std::vector<int64_t> dilation, bool transposed,
    std::vector<int64_t> output_padding, int64_t groups);

LazyTensor convolution_overrideable(
    const LazyTensor& input, const LazyTensor& weight,
    std::vector<int64_t> stride, std::vector<int64_t> padding,
    std::vector<int64_t> dilation, bool transposed,
    std::vector<int64_t> output_padding, int64_t groups);

std::tuple<LazyTensor, LazyTensor, LazyTensor>
convolution_backward_overrideable(
    const LazyTensor& out_backprop, const LazyTensor& input,
    const LazyTensor& weight, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    bool transposed, std::vector<int64_t> output_padding, int64_t groups,
    std::array<bool, 3> output_mask);

LazyTensor expand(const LazyTensor& input,
                  std::vector<int64_t> size);

// Fills the input with the given value.
void fill_(LazyTensor& input, const at::Scalar& value);

LazyTensor mul(const LazyTensor& input, const LazyTensor& other);
LazyTensor mul(const LazyTensor& input, const at::Scalar& other);

// Returns a new tensor that is a narrowed view of the input in the given
// dimension.
LazyTensor narrow(const LazyTensor& input, int64_t dim, int64_t start,
                  int64_t length);

std::tuple<LazyTensor, LazyTensor, LazyTensor> ts_native_batch_norm(
    const LazyTensor& input, const LazyTensor& weight, const LazyTensor& bias,
    LazyTensor& running_mean, LazyTensor& running_var, bool training,
    double momentum, double eps);

std::tuple<LazyTensor, LazyTensor, LazyTensor> ts_native_batch_norm_backward(
    const LazyTensor& grad_out, const LazyTensor& input,
    const LazyTensor& weight, const LazyTensor& running_mean,
    const LazyTensor& running_var, const LazyTensor& save_mean,
    const LazyTensor& save_invstd, bool training, double eps,
    c10::ArrayRef<bool> output_mask);

std::pair<LazyTensor, LazyTensor> nms(const LazyTensor& boxes,
                                      const LazyTensor& scores,
                                      const LazyTensor& score_threshold,
                                      const LazyTensor& iou_threshold,
                                      int64_t output_size);

// Permute the dimensions of this tensor according to the given permutation.
LazyTensor permute(const LazyTensor& input, c10::ArrayRef<int64_t> dims);

// Repeats the input tensor along each dimension by the given number of
// repeats.
LazyTensor repeat(const LazyTensor& input, std::vector<int64_t> repeats);

LazyTensor rsub(const LazyTensor& input, const at::Scalar& other,
                const at::Scalar& alpha);

void copy_(LazyTensor& input, LazyTensor& src);

LazyTensor select(const LazyTensor& input, int64_t dim, int64_t index);

LazyTensor slice(const LazyTensor& input, int64_t dim, int64_t start,
                 int64_t end, int64_t step);

// Squeeze out all trivial (size 1) dimensions.
LazyTensor squeeze(const LazyTensor& input);

// Squeeze out the specified dimension index, if trivial (size 1). Returns
// unchanged input otherwise.
LazyTensor squeeze(const LazyTensor& input, int64_t dim);

// In-place versions of the methods above.
void squeeze_(LazyTensor& input);
void squeeze_(LazyTensor& input, int64_t dim);

LazyTensor stack(c10::ArrayRef<LazyTensor> tensors, int64_t dim);

LazyTensor sub(const LazyTensor& input, const LazyTensor& other,
               const at::Scalar& alpha);
LazyTensor sub(const LazyTensor& input, const at::Scalar& other,
               const at::Scalar& alpha);

std::tuple<LazyTensor, LazyTensor, LazyTensor> svd(const LazyTensor& input,
                                                   bool some, bool compute_uv);

LazyTensor tanh_backward(const LazyTensor& grad_output,
                         const LazyTensor& output);

// Swap given dimensions of the input.
LazyTensor transpose(const LazyTensor& input, int64_t dim0, int64_t dim1);

// In-place version of the method above.
void transpose_(LazyTensor& input, int64_t dim0, int64_t dim1);

// Insert a dimension of size one at the specified position.
LazyTensor unsqueeze(const LazyTensor& input, int64_t dim);

// In-place version of the method above.
void unsqueeze_(LazyTensor& input, int64_t dim);

// Like reshape, but it returns a view into the original tensor.
LazyTensor view(const LazyTensor& input, c10::ArrayRef<int64_t> output_size);

}  // namespace lazy_tensor_aten_ops
}  // namespace torch_lazy_tensors
