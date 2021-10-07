#pragma once

#include "lazy_tensor_core/csrc/tensor.h"

namespace torch_lazy_tensors {
namespace tensor_aten_ops {

// TODO: Remove methods from here as we support codegen for more ops

//////////////////////////////////////////////////////////////////////////////
// ATEN operators follows here, listed in alphabetical order.
//////////////////////////////////////////////////////////////////////////////
void __ilshift__(LazyTensor& input, const at::Scalar& other);
void __ilshift__(LazyTensor& input, const LazyTensor& other);

void __irshift__(LazyTensor& input, const at::Scalar& other);
void __irshift__(LazyTensor& input, const LazyTensor& other);

LazyTensor __lshift__(
    const LazyTensor& input, const at::Scalar& other,
    c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
LazyTensor __lshift__(
    const LazyTensor& input, const LazyTensor& other,
    c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

LazyTensor __rshift__(
    const LazyTensor& input, const at::Scalar& other,
    c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
LazyTensor __rshift__(
    const LazyTensor& input, const LazyTensor& other,
    c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

LazyTensor adaptive_avg_pool3d(const LazyTensor& input,
                               std::vector<lazy_tensors::int64> output_size);

LazyTensor adaptive_avg_pool3d_backward(const LazyTensor& grad_output,
                                        const LazyTensor& input);

LazyTensor _adaptive_avg_pool2d(const LazyTensor& input,
                                std::vector<lazy_tensors::int64> output_size);

LazyTensor _adaptive_avg_pool2d_backward(const LazyTensor& grad_output,
                                         const LazyTensor& input);

void _amp_foreach_non_finite_check_and_unscale_(std::vector<LazyTensor> self,
                                                LazyTensor& found_inf,
                                                const LazyTensor& inv_scale);

void _amp_update_scale_(LazyTensor& current_scale, LazyTensor& growth_tracker,
                        const LazyTensor& found_inf, double scale_growth_factor,
                        double scale_backoff_factor, int growth_interval);

LazyTensor abs(const LazyTensor& input);

LazyTensor acos(const LazyTensor& input);

LazyTensor acosh(const LazyTensor& input);

LazyTensor add(
    const LazyTensor& input, const LazyTensor& other, const at::Scalar& alpha,
    c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
LazyTensor add(
    const LazyTensor& input, const at::Scalar& other, const at::Scalar& alpha,
    c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

void addcdiv_(LazyTensor& input, const at::Scalar& value,
              const LazyTensor& tensor1, const LazyTensor& tensor2);

LazyTensor addmm(const LazyTensor& input, const LazyTensor& weight,
                 const LazyTensor& bias);

LazyTensor all(const LazyTensor& input,
               std::vector<lazy_tensors::int64> dimensions,
               bool keep_reduced_dimensions);

LazyTensor amax(const LazyTensor& input,
                std::vector<lazy_tensors::int64> dimensions,
                bool keep_reduced_dimensions);

LazyTensor amin(const LazyTensor& input,
                std::vector<lazy_tensors::int64> dimensions,
                bool keep_reduced_dimensions);

LazyTensor any(const LazyTensor& input,
               std::vector<lazy_tensors::int64> dimensions,
               bool keep_reduced_dimensions);

void arange_out(LazyTensor& out, const at::Scalar& start, const at::Scalar& end,
                const at::Scalar& step, at::ScalarType scalar_type);

LazyTensor argmax(const LazyTensor& input, lazy_tensors::int64 dim,
                  bool keepdim);
LazyTensor argmax(const LazyTensor& input);

LazyTensor argmin(const LazyTensor& input, lazy_tensors::int64 dim,
                  bool keepdim);
LazyTensor argmin(const LazyTensor& input);

// Takes a slice from the input as R1 at the specified offset and reshapes it
// into the provided size.
LazyTensor as_strided(const LazyTensor& input,
                      std::vector<lazy_tensors::int64> size,
                      std::vector<lazy_tensors::int64> stride,
                      c10::optional<lazy_tensors::int64> storage_offset);

// In-place version of the method above.
void as_strided_(LazyTensor& input, std::vector<lazy_tensors::int64> size,
                 std::vector<lazy_tensors::int64> stride,
                 c10::optional<lazy_tensors::int64> storage_offset);

LazyTensor asin(const LazyTensor& input);

LazyTensor asinh(const LazyTensor& input);

LazyTensor atan(const LazyTensor& input);

LazyTensor atanh(const LazyTensor& input);

LazyTensor atan2(
    const LazyTensor& input, const LazyTensor& other,
    c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

LazyTensor avg_pool_nd(const LazyTensor& input,
                       lazy_tensors::int64 spatial_dim_count,
                       std::vector<lazy_tensors::int64> kernel_size,
                       std::vector<lazy_tensors::int64> stride,
                       std::vector<lazy_tensors::int64> padding, bool ceil_mode,
                       bool count_include_pad);

LazyTensor avg_pool_nd_backward(const LazyTensor& out_backprop,
                                const LazyTensor& input,
                                lazy_tensors::int64 spatial_dim_count,
                                std::vector<lazy_tensors::int64> kernel_size,
                                std::vector<lazy_tensors::int64> stride,
                                std::vector<lazy_tensors::int64> padding,
                                bool ceil_mode, bool count_include_pad);

LazyTensor baddbmm(const LazyTensor& input, const LazyTensor& batch1,
                   const LazyTensor& batch2, const at::Scalar& beta,
                   const at::Scalar& alpha);

LazyTensor bernoulli(const LazyTensor& input, double probability);
LazyTensor bernoulli(const LazyTensor& input);
void bernoulli_(LazyTensor& input, double probability);
void bernoulli_(LazyTensor& input, const LazyTensor& probability);

LazyTensor binary_cross_entropy(const LazyTensor& input,
                                const LazyTensor& target,
                                const LazyTensor& weight,
                                lazy_tensors::int64 reduction);

LazyTensor binary_cross_entropy_backward(const LazyTensor& grad_output,
                                         const LazyTensor& input,
                                         const LazyTensor& target,
                                         const LazyTensor& weight,
                                         lazy_tensors::int64 reduction);

void logical_and_out(LazyTensor& out, const LazyTensor& input,
                     const LazyTensor& other);

LazyTensor bitwise_and(const LazyTensor& input, const at::Scalar& other);

LazyTensor bitwise_and(const LazyTensor& input, const LazyTensor& other);

void bitwise_not_out(LazyTensor& out, const LazyTensor& input);

void bitwise_or_out(LazyTensor& out, const LazyTensor& input,
                    const at::Scalar& other);

void bitwise_or_out(LazyTensor& out, const LazyTensor& input,
                    const LazyTensor& other);

void bitwise_xor_out(LazyTensor& out, const LazyTensor& input,
                     const at::Scalar& other);

void bitwise_xor_out(LazyTensor& out, const LazyTensor& input,
                     const LazyTensor& other);

// Batch matrix multiplication. Both tensors must be 3D, the batch size must
// match and the remaining two dimensions must be compatible for matrix
// multiplication.
LazyTensor bmm(const LazyTensor& batch1, const LazyTensor& batch2);

// Broadcasts the given tensors according to broadcasting semantics.
std::vector<LazyTensor> broadcast_tensors(
    lazy_tensors::Span<const LazyTensor> tensors);

LazyTensor cat(lazy_tensors::Span<const LazyTensor> tensors,
               lazy_tensors::int64 dim);

LazyTensor ceil(const LazyTensor& input);

LazyTensor cholesky(const LazyTensor& input, bool upper);

LazyTensor clamp(const LazyTensor& input, const c10::optional<at::Scalar>& min,
                 const c10::optional<at::Scalar>& max);
LazyTensor clamp(const LazyTensor& input, const c10::optional<at::Tensor>& min,
                 const c10::optional<at::Tensor>& max);
void clamp_out(LazyTensor& out, const LazyTensor& input,
               const c10::optional<at::Tensor>& min,
               const c10::optional<at::Tensor>& max);

LazyTensor clone(const LazyTensor& input);

// Pad with the given value and size specified by the given list of low and
// high paddings.
LazyTensor constant_pad_nd(const LazyTensor& input,
                           lazy_tensors::Span<const lazy_tensors::int64> pad,
                           const at::Scalar& value);

LazyTensor convolution_overrideable(
    const LazyTensor& input, const LazyTensor& weight, const LazyTensor& bias,
    std::vector<lazy_tensors::int64> stride,
    std::vector<lazy_tensors::int64> padding,
    std::vector<lazy_tensors::int64> dilation, bool transposed,
    std::vector<lazy_tensors::int64> output_padding,
    lazy_tensors::int64 groups);

std::tuple<LazyTensor, LazyTensor, LazyTensor>
convolution_backward_overrideable(
    const LazyTensor& out_backprop, const LazyTensor& input,
    const LazyTensor& weight, std::vector<lazy_tensors::int64> stride,
    std::vector<lazy_tensors::int64> padding,
    std::vector<lazy_tensors::int64> dilation, bool transposed,
    std::vector<lazy_tensors::int64> output_padding, lazy_tensors::int64 groups,
    std::array<bool, 3> output_mask);

LazyTensor convolution_overrideable(
    const LazyTensor& input, const LazyTensor& weight,
    std::vector<lazy_tensors::int64> stride,
    std::vector<lazy_tensors::int64> padding,
    std::vector<lazy_tensors::int64> dilation, bool transposed,
    std::vector<lazy_tensors::int64> output_padding,
    lazy_tensors::int64 groups);

LazyTensor cosh(const LazyTensor& input);

// Returns the cross product of the two input tensors in the given dimension.
// If the dimension is not given, it defaults to the first dimension found
// with the size 3.
LazyTensor cross(const LazyTensor& input, const LazyTensor& other,
                 c10::optional<lazy_tensors::int64> dim);

// Returns the cumulative product of elements of input in the given dimension.
LazyTensor cumprod(const LazyTensor& input, lazy_tensors::int64 dim,
                   c10::optional<at::ScalarType> dtype);

// Returns the cumulative sum of elements of input in the given dimension.
LazyTensor cumsum(const LazyTensor& input, lazy_tensors::int64 dim,
                  c10::optional<at::ScalarType> dtype);

// If the input is a matrix (2-D tensor), returns a 1-D tensor with the
// diagonal elements of the input. If the input is a vector (1-D tensor),
// returns a 2-D square tensor with the elements of input as the diagonal.
LazyTensor diag(const LazyTensor& input, lazy_tensors::int64 offset);

// Returns the diagonal of a matrix (2-D tensor) or batch of matrices. The
// matrix dimensions are specified by dim1 and dim2, the diagonal by offset.
LazyTensor diagonal(const LazyTensor& input, lazy_tensors::int64 offset,
                    lazy_tensors::int64 dim1, lazy_tensors::int64 dim2);

LazyTensor div(
    const LazyTensor& input, const LazyTensor& other,
    const c10::optional<c10::string_view>& rounding_mode = c10::nullopt,
    c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
LazyTensor div(const LazyTensor& input, const at::Scalar& other);

// A generalized contraction between tensors of arbitrary dimension defined by
// the given equation and applied to the input tensors.
LazyTensor einsum(const std::string& equation,
                  lazy_tensors::Span<const LazyTensor> tensors);

LazyTensor elu(const LazyTensor& input, const at::Scalar& alpha,
               const at::Scalar& scale, const at::Scalar& input_scale);
void elu_(LazyTensor& input, const at::Scalar& alpha, const at::Scalar& scale,
          const at::Scalar& input_scale);
LazyTensor elu_backward(const LazyTensor& grad_output, const at::Scalar& alpha,
                        const at::Scalar& scale, const at::Scalar& input_scale,
                        const LazyTensor& output);

LazyTensor embedding_dense_backward(const LazyTensor& grad_output,
                                    const LazyTensor& indices,
                                    lazy_tensors::int64 num_weights,
                                    lazy_tensors::int64 padding_idx,
                                    bool scale_grad_by_freq);

LazyTensor ts_embedding_dense_backward(const LazyTensor& grad_output,
                                       const LazyTensor& indices,
                                       lazy_tensors::int64 num_weights,
                                       lazy_tensors::int64 padding_idx,
                                       bool scale_grad_by_freq);

LazyTensor eq(const LazyTensor& input, const at::Scalar& other);

LazyTensor eq(const LazyTensor& input, const LazyTensor& other);

LazyTensor erf(const LazyTensor& input);

LazyTensor erfc(const LazyTensor& input);

LazyTensor erfinv(const LazyTensor& input);

LazyTensor exp(const LazyTensor& input);

LazyTensor expand(const LazyTensor& input,
                  std::vector<lazy_tensors::int64> size);

LazyTensor expm1(const LazyTensor& input);

void exponential_(LazyTensor& input, double lambd);

// Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.
LazyTensor eye(lazy_tensors::int64 lines, lazy_tensors::int64 cols,
               const Device& device, at::ScalarType element_type);

void eye_out(LazyTensor& out, lazy_tensors::int64 lines,
             lazy_tensors::int64 cols);

// Fills the input with the given value.
void fill_(LazyTensor& input, const at::Scalar& value);

// Flips (reverses) the values in the dimensions of the input tensor.
LazyTensor flip(const LazyTensor& input,
                lazy_tensors::Span<const lazy_tensors::int64> dims);

LazyTensor floor(const LazyTensor& input);

LazyTensor fmod(
    const LazyTensor& input, const LazyTensor& other,
    c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
LazyTensor fmod(
    const LazyTensor& input, const at::Scalar& other,
    c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

LazyTensor frac(const LazyTensor& input);

LazyTensor full(lazy_tensors::Span<const lazy_tensors::int64> size,
                const at::Scalar& fill_value, const Device& device,
                at::ScalarType scalar_type);
LazyTensor full_like(const LazyTensor& input, const at::Scalar& fill_value,
                     const Device& device,
                     c10::optional<at::ScalarType> scalar_type);

LazyTensor gather(const LazyTensor& input, lazy_tensors::int64 dim,
                  const LazyTensor& index);

LazyTensor ge(const LazyTensor& input, const at::Scalar& other);

LazyTensor ge(const LazyTensor& input, const LazyTensor& other);

LazyTensor gelu(const LazyTensor& input);
LazyTensor gelu_backward(const LazyTensor& grad, const LazyTensor& input);

LazyTensor ger(const LazyTensor& input, const LazyTensor& vec2);

LazyTensor gt(const LazyTensor& input, const at::Scalar& other);

LazyTensor gt(const LazyTensor& input, const LazyTensor& other);

// Gather slices from input into a result with shape specified by indices. The
// shape of the indices are first made consistent using broadcast semantics.
// For input of shape d1 x d2 x ... x dn and p indices of shape i1 x i2 x ...
// x ik, the output shape is d1 x ... x d(start_dim) x i1 x ... x ik x
// d(start_dim+p+1) x ... x dn.
LazyTensor index(const LazyTensor& input,
                 lazy_tensors::Span<const LazyTensor> indices,
                 lazy_tensors::int64 start_dim);

LazyTensor index_add(const LazyTensor& input, lazy_tensors::int64 dim,
                     const LazyTensor& index, const LazyTensor& source);

void index_add_(LazyTensor& input, lazy_tensors::int64 dim,
                const LazyTensor& index, const LazyTensor& source);

LazyTensor index_copy(const LazyTensor& input, lazy_tensors::int64 dim,
                      const LazyTensor& index, const LazyTensor& source);

void index_copy_(LazyTensor& input, lazy_tensors::int64 dim,
                 const LazyTensor& index, const LazyTensor& source);

// Fills the elements of the base tensor with the given value in the given
// dimension, at positions given by the index. The index must be a rank-1
// tensor.
LazyTensor index_fill(const LazyTensor& input, lazy_tensors::int64 dim,
                      const LazyTensor& index, const at::Scalar& value);

// Same as above, but the value is wrapped as a rank-0 tensor.
LazyTensor index_fill(const LazyTensor& input, lazy_tensors::int64 dim,
                      const LazyTensor& index, const LazyTensor& value);

void index_fill_(LazyTensor& input, lazy_tensors::int64 dim,
                 const LazyTensor& index, const LazyTensor& value);

void index_fill_(LazyTensor& input, lazy_tensors::int64 dim,
                 const LazyTensor& index, const at::Scalar& value);

// Puts values into the input tensor using the given indices (a tuple of
// tensors) and returns the result.
LazyTensor index_put(
    const LazyTensor& input, lazy_tensors::Span<const LazyTensor> indices,
    lazy_tensors::int64 start_dim, const LazyTensor& values, bool accumulate,
    lazy_tensors::Span<const lazy_tensors::int64> result_permutation);

void index_put_(
    LazyTensor& input, const LazyTensor& canonical_base,
    lazy_tensors::Span<const LazyTensor> indices, lazy_tensors::int64 start_dim,
    const LazyTensor& values, bool accumulate,
    lazy_tensors::Span<const lazy_tensors::int64> result_permutation);

LazyTensor index_select(const LazyTensor& input, lazy_tensors::int64 dim,
                        const LazyTensor& index);

LazyTensor inverse(const LazyTensor& input);

LazyTensor isnan(const LazyTensor& input);

LazyTensor kl_div_backward(const LazyTensor& grad_output,
                           const LazyTensor& input, const LazyTensor& target,
                           lazy_tensors::int64 reduction, bool log_target);

std::tuple<LazyTensor, LazyTensor> kthvalue(const LazyTensor& input,
                                            lazy_tensors::int64 k,
                                            lazy_tensors::int64 dim,
                                            bool keepdim);

LazyTensor l1_loss(const LazyTensor& input, const LazyTensor& target,
                   lazy_tensors::int64 reduction);

LazyTensor l1_loss_backward(const LazyTensor& grad_output,
                            const LazyTensor& input, const LazyTensor& target,
                            lazy_tensors::int64 reduction);

LazyTensor le(const LazyTensor& input, const at::Scalar& other);

LazyTensor le(const LazyTensor& input, const LazyTensor& other);

LazyTensor hardshrink(const LazyTensor& input, const at::Scalar& lambda);
LazyTensor hardshrink_backward(const LazyTensor& grad_out,
                               const LazyTensor& input,
                               const at::Scalar& lambda);

LazyTensor hardsigmoid(const LazyTensor& input);

LazyTensor hardsigmoid_backward(const LazyTensor& grad_output,
                                const LazyTensor& input);

LazyTensor hardtanh_backward(const LazyTensor& grad_output,
                             const LazyTensor& input, const at::Scalar& min_val,
                             const at::Scalar& max_val);

LazyTensor leaky_relu(const LazyTensor& input, double negative_slope);
LazyTensor leaky_relu_backward(const LazyTensor& grad_output,
                               const LazyTensor& input, double negative_slope,
                               bool self_is_result);
LazyTensor lerp(const LazyTensor& input, const LazyTensor& end,
                const LazyTensor& weight);
LazyTensor lerp(const LazyTensor& input, const LazyTensor& end,
                const at::Scalar& weight);

LazyTensor log(const LazyTensor& input);

LazyTensor log_base(const LazyTensor& input, ir::OpKind op, double base);

LazyTensor log_sigmoid(const LazyTensor& input);
std::tuple<LazyTensor, LazyTensor> log_sigmoid_forward(const LazyTensor& input);
LazyTensor log_sigmoid_backward(const LazyTensor& grad_output,
                                const LazyTensor& input,
                                const LazyTensor& buffer);

LazyTensor log_softmax_backward(const LazyTensor& grad_output,
                                const LazyTensor& output,
                                lazy_tensors::int64 dim);

LazyTensor ts_log_softmax_backward(const LazyTensor& grad_output,
                                   const LazyTensor& output,
                                   lazy_tensors::int64 dim,
                                   const LazyTensor& self);

LazyTensor log1p(const LazyTensor& input);
void log1p_(LazyTensor& input);

LazyTensor logdet(const LazyTensor& input);

LazyTensor logsumexp(const LazyTensor& input,
                     std::vector<lazy_tensors::int64> dimensions,
                     bool keep_reduced_dimensions);

LazyTensor lt(const LazyTensor& input, const at::Scalar& other);

LazyTensor lt(const LazyTensor& input, const LazyTensor& other);

// In-place version of the method above.
void masked_fill_(LazyTensor& input, const LazyTensor& mask,
                  const at::Scalar& value);

void masked_scatter_(LazyTensor& input, const LazyTensor& mask,
                     const LazyTensor& source);

LazyTensor masked_select(const LazyTensor& input, const LazyTensor& mask);

LazyTensor matmul(const LazyTensor& input, const LazyTensor& other);

LazyTensor max(
    const LazyTensor& input, const LazyTensor& other,
    c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

LazyTensor max(const LazyTensor& input);

std::tuple<LazyTensor, LazyTensor> max(const LazyTensor& input,
                                       lazy_tensors::int64 dim, bool keepdim);

void max_out(LazyTensor& max, LazyTensor& max_values, const LazyTensor& input,
             lazy_tensors::int64 dim, bool keepdim);

std::tuple<LazyTensor, LazyTensor> max_pool_nd(
    const LazyTensor& input, lazy_tensors::int64 spatial_dim_count,
    std::vector<lazy_tensors::int64> kernel_size,
    std::vector<lazy_tensors::int64> stride,
    std::vector<lazy_tensors::int64> padding, bool ceil_mode);

LazyTensor max_pool_nd_backward(const LazyTensor& out_backprop,
                                const LazyTensor& input,
                                lazy_tensors::int64 spatial_dim_count,
                                std::vector<lazy_tensors::int64> kernel_size,
                                std::vector<lazy_tensors::int64> stride,
                                std::vector<lazy_tensors::int64> padding,
                                bool ceil_mode);

LazyTensor max_unpool(const LazyTensor& input, const LazyTensor& indices,
                      std::vector<lazy_tensors::int64> output_size);

LazyTensor max_unpool_backward(const LazyTensor& grad_output,
                               const LazyTensor& input,
                               const LazyTensor& indices,
                               std::vector<lazy_tensors::int64> output_size);

LazyTensor min(
    const LazyTensor& input, const LazyTensor& other,
    c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

LazyTensor min(const LazyTensor& input);

std::tuple<LazyTensor, LazyTensor> min(const LazyTensor& input,
                                       lazy_tensors::int64 dim, bool keepdim);

void min_out(LazyTensor& min, LazyTensor& min_indices, const LazyTensor& input,
             lazy_tensors::int64 dim, bool keepdim);

LazyTensor mse_loss(const LazyTensor& input, const LazyTensor& target,
                    lazy_tensors::int64 reduction);

LazyTensor mse_loss_backward(const LazyTensor& grad_output,
                             const LazyTensor& input, const LazyTensor& target,
                             lazy_tensors::int64 reduction);

LazyTensor mul(
    const LazyTensor& input, const LazyTensor& other,
    c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
LazyTensor mul(
    const LazyTensor& input, const at::Scalar& other,
    c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

// Returns a new tensor that is a narrowed view of the input in the given
// dimension.
LazyTensor narrow(const LazyTensor& input, lazy_tensors::int64 dim,
                  lazy_tensors::int64 start, lazy_tensors::int64 length);

// Like batch_norm, but returns additional save_mean and save_invstd used by
// the backward pass.
std::tuple<LazyTensor, LazyTensor, LazyTensor> native_batch_norm(
    const LazyTensor& input, const LazyTensor& weight, const LazyTensor& bias,
    LazyTensor& running_mean, LazyTensor& running_var, bool training,
    double momentum, double eps);

std::tuple<LazyTensor, LazyTensor, LazyTensor> ts_native_batch_norm(
    const LazyTensor& input, const LazyTensor& weight, const LazyTensor& bias,
    LazyTensor& running_mean, LazyTensor& running_var, bool training,
    double momentum, double eps);

// Returns the input, weight and bias gradients.
std::tuple<LazyTensor, LazyTensor, LazyTensor> native_batch_norm_backward(
    const LazyTensor& grad_out, const LazyTensor& input,
    const LazyTensor& weight, const LazyTensor& save_mean,
    const LazyTensor& save_invstd, bool training, double eps);

std::tuple<LazyTensor, LazyTensor, LazyTensor> ts_native_batch_norm_backward(
    const LazyTensor& grad_out, const LazyTensor& input,
    const LazyTensor& weight, const LazyTensor& running_mean,
    const LazyTensor& running_var, const LazyTensor& save_mean,
    const LazyTensor& save_invstd, bool training, double eps,
    lazy_tensors::Span<const bool> output_mask);

LazyTensor ne(const LazyTensor& input, const at::Scalar& other);

LazyTensor ne(const LazyTensor& input, const LazyTensor& other);

LazyTensor neg(const LazyTensor& input);

std::tuple<LazyTensor, LazyTensor> nll_loss_forward(
    const LazyTensor& input, const LazyTensor& target, const LazyTensor& weight,
    lazy_tensors::int64 reduction, int ignore_index);

LazyTensor nll_loss2d(const LazyTensor& input, const LazyTensor& target,
                      const LazyTensor& weight, lazy_tensors::int64 reduction,
                      int ignore_index);

LazyTensor nll_loss2d_backward(const LazyTensor& grad_output,
                               const LazyTensor& input,
                               const LazyTensor& target,
                               const LazyTensor& weight,
                               lazy_tensors::int64 reduction, int ignore_index,
                               const LazyTensor& total_weight);

LazyTensor nll_loss_backward(const LazyTensor& grad_output,
                             const LazyTensor& input, const LazyTensor& target,
                             const LazyTensor& weight,
                             lazy_tensors::int64 reduction, int ignore_index,
                             const LazyTensor& total_weight);

std::pair<LazyTensor, LazyTensor> nms(const LazyTensor& boxes,
                                      const LazyTensor& scores,
                                      const LazyTensor& score_threshold,
                                      const LazyTensor& iou_threshold,
                                      lazy_tensors::int64 output_size);

LazyTensor nonzero(const LazyTensor& input);

LazyTensor norm(const LazyTensor& input, const c10::optional<at::Scalar>& p,
                c10::optional<at::ScalarType> dtype, at::IntArrayRef dim,
                bool keepdim);

LazyTensor normal(double mean, const LazyTensor& std);

LazyTensor normal(const LazyTensor& mean, double std);

LazyTensor normal(const LazyTensor& mean, const LazyTensor& std);

void normal_(LazyTensor& input, double mean, double std);

LazyTensor not_supported(std::string description, lazy_tensors::Shape shape,
                         const Device& device);

// Permute the dimensions of this tensor according to the given permutation.
LazyTensor permute(const LazyTensor& input,
                   lazy_tensors::Span<const lazy_tensors::int64> dims);

LazyTensor pow(const LazyTensor& input, const at::Scalar& exponent);
LazyTensor pow(const LazyTensor& input, const LazyTensor& exponent);
LazyTensor pow(const at::Scalar& input, const LazyTensor& exponent);

LazyTensor prod(const LazyTensor& input,
                std::vector<lazy_tensors::int64> dimensions,
                bool keep_reduced_dimensions,
                c10::optional<at::ScalarType> dtype);

void put_(LazyTensor& input, const LazyTensor& index, const LazyTensor& source,
          bool accumulate);

std::tuple<LazyTensor, LazyTensor> qr(const LazyTensor& input, bool some);

void random_(LazyTensor& input);

LazyTensor randperm(lazy_tensors::int64 n, const Device& device,
                    at::ScalarType scalar_type);

LazyTensor reciprocal(const LazyTensor& input);

LazyTensor reflection_pad2d(const LazyTensor& input,
                            std::vector<lazy_tensors::int64> padding);

LazyTensor reflection_pad2d_backward(const LazyTensor& grad_output,
                                     const LazyTensor& input,
                                     std::vector<lazy_tensors::int64> padding);

LazyTensor relu(const LazyTensor& input);
void relu_(LazyTensor& input);

LazyTensor remainder(const LazyTensor& input, const LazyTensor& other);
LazyTensor remainder(const LazyTensor& input, const at::Scalar& other);

// Repeats the input tensor along each dimension by the given number of
// repeats.
LazyTensor repeat(const LazyTensor& input,
                  std::vector<lazy_tensors::int64> repeats);

LazyTensor replication_pad1d(const LazyTensor& input,
                             std::vector<lazy_tensors::int64> padding);
LazyTensor replication_pad1d_backward(const LazyTensor& grad_output,
                                      const LazyTensor& input,
                                      std::vector<lazy_tensors::int64> padding);

LazyTensor replication_pad2d(const LazyTensor& input,
                             std::vector<lazy_tensors::int64> padding);
LazyTensor replication_pad2d_backward(const LazyTensor& grad_output,
                                      const LazyTensor& input,
                                      std::vector<lazy_tensors::int64> padding);

void resize_(LazyTensor& input, std::vector<lazy_tensors::int64> size);

LazyTensor round(const LazyTensor& input);

LazyTensor rrelu_with_noise(const LazyTensor& input, LazyTensor& noise,
                            const at::Scalar& lower, const at::Scalar& upper,
                            bool training);

LazyTensor rrelu_with_noise_backward(const LazyTensor& grad_output,
                                     const LazyTensor& input,
                                     const LazyTensor& noise,
                                     const at::Scalar& lower,
                                     const at::Scalar& upper, bool training);

LazyTensor rsqrt(const LazyTensor& input);

LazyTensor rsub(
    const LazyTensor& input, const LazyTensor& other, const at::Scalar& alpha,
    c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
LazyTensor rsub(
    const LazyTensor& input, const at::Scalar& other, const at::Scalar& alpha,
    c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

void copy_(LazyTensor& input, LazyTensor& src);

void scatter_out(LazyTensor& out, const LazyTensor& input,
                 lazy_tensors::int64 dim, const LazyTensor& index,
                 const LazyTensor& src);
void scatter_out(LazyTensor& out, const LazyTensor& input,
                 lazy_tensors::int64 dim, const LazyTensor& index,
                 const at::Scalar& value);

void scatter_add_(LazyTensor& input, lazy_tensors::int64 dim,
                  const LazyTensor& index, const LazyTensor& src);
void scatter_add_out(LazyTensor& out, const LazyTensor& input,
                     lazy_tensors::int64 dim, const LazyTensor& index,
                     const LazyTensor& src);
void scatter_add_out(LazyTensor& out, const LazyTensor& input,
                     lazy_tensors::int64 dim, const LazyTensor& index,
                     const at::Scalar& value);

LazyTensor select(const LazyTensor& input, lazy_tensors::int64 dim,
                  lazy_tensors::int64 index);

void silu_out(LazyTensor& input, LazyTensor& out);
LazyTensor sigmoid(const LazyTensor& input);
LazyTensor sigmoid_backward(const LazyTensor& grad_output,
                            const LazyTensor& output);

LazyTensor sign(const LazyTensor& input);

LazyTensor sin(const LazyTensor& input);

LazyTensor sinh(const LazyTensor& input);

LazyTensor slice(const LazyTensor& input, lazy_tensors::int64 dim,
                 lazy_tensors::int64 start, lazy_tensors::int64 end,
                 lazy_tensors::int64 step);

// Computes a loss that uses a squared term if the absolute element-wise error
// falls below 1 and an L1 term otherwise.
LazyTensor smooth_l1_loss(const LazyTensor& input, const LazyTensor& target,
                          lazy_tensors::int64 reduction, double beta);

// Returns the gradient of the input of a smooth_l1_loss operation.
LazyTensor smooth_l1_loss_backward(const LazyTensor& grad_output,
                                   const LazyTensor& input,
                                   const LazyTensor& target,
                                   lazy_tensors::int64 reduction, double beta);

LazyTensor softmax(const LazyTensor& input, lazy_tensors::int64 dim,
                   c10::optional<at::ScalarType> dtype);
LazyTensor softmax_backward(const LazyTensor& grad_output,
                            const LazyTensor& output, lazy_tensors::int64 dim);

LazyTensor softplus(const LazyTensor& input, const at::Scalar& beta,
                    const at::Scalar& threshold);
LazyTensor softplus_backward(const LazyTensor& grad_output,
                             const LazyTensor& input, const at::Scalar& beta,
                             const at::Scalar& threshold,
                             const LazyTensor& output);

LazyTensor softshrink(const LazyTensor& input, const at::Scalar& lambda);
LazyTensor softshrink_backward(const LazyTensor& grad_out,
                               const LazyTensor& input,
                               const at::Scalar& lambda);

std::vector<LazyTensor> split(const LazyTensor& input,
                              lazy_tensors::int64 split_size,
                              lazy_tensors::int64 dim);

std::vector<LazyTensor> split_with_sizes(
    const LazyTensor& input, std::vector<lazy_tensors::int64> split_size,
    lazy_tensors::int64 dim);

LazyTensor sqrt(const LazyTensor& input);

// Squeeze out all trivial (size 1) dimensions.
LazyTensor squeeze(const LazyTensor& input);

// Squeeze out the specified dimension index, if trivial (size 1). Returns
// unchanged input otherwise.
LazyTensor squeeze(const LazyTensor& input, lazy_tensors::int64 dim);

// In-place versions of the methods above.
void squeeze_(LazyTensor& input);
void squeeze_(LazyTensor& input, lazy_tensors::int64 dim);

LazyTensor stack(lazy_tensors::Span<const LazyTensor> tensors,
                 lazy_tensors::int64 dim);

LazyTensor std(const LazyTensor& input,
               std::vector<lazy_tensors::int64> dimensions,
               bool keep_reduced_dimensions, lazy_tensors::int64 correction);

std::tuple<LazyTensor, LazyTensor> std_mean(
    const LazyTensor& input, std::vector<lazy_tensors::int64> dimensions,
    lazy_tensors::int64 correction, bool keep_reduced_dimensions);

LazyTensor sub(
    const LazyTensor& input, const LazyTensor& other, const at::Scalar& alpha,
    c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
LazyTensor sub(
    const LazyTensor& input, const at::Scalar& other, const at::Scalar& alpha,
    c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

LazyTensor sum(const LazyTensor& input,
               std::vector<lazy_tensors::int64> dimensions,
               bool keep_reduced_dimensions,
               c10::optional<at::ScalarType> dtype);

std::tuple<LazyTensor, LazyTensor, LazyTensor> svd(const LazyTensor& input,
                                                   bool some, bool compute_uv);

std::tuple<LazyTensor, LazyTensor> symeig(const LazyTensor& input,
                                          bool eigenvectors, bool upper);

LazyTensor take(const LazyTensor& input, const LazyTensor& index);

LazyTensor tan(const LazyTensor& input);

LazyTensor tanh(const LazyTensor& input);
LazyTensor tanh_backward(const LazyTensor& grad_output,
                         const LazyTensor& output);

LazyTensor threshold(const LazyTensor& input, float threshold, float value);

LazyTensor threshold_backward(const LazyTensor& grad_output,
                              const LazyTensor& input, float threshold);

LazyTensor to(LazyTensor& input, c10::optional<Device> device,
              c10::optional<at::ScalarType> scalar_type);

std::tuple<LazyTensor, LazyTensor> topk(const LazyTensor& input,
                                        lazy_tensors::int64 k,
                                        lazy_tensors::int64 dim, bool largest,
                                        bool sorted);

// Returns the sum of the elements of the diagonal of the input 2-D matrix.
LazyTensor trace(const LazyTensor& input);

// Swap given dimensions of the input.
LazyTensor transpose(const LazyTensor& input, lazy_tensors::int64 dim0,
                     lazy_tensors::int64 dim1);

// In-place version of the method above.
void transpose_(LazyTensor& input, lazy_tensors::int64 dim0,
                lazy_tensors::int64 dim1);

std::tuple<LazyTensor, LazyTensor> triangular_solve(const LazyTensor& rhs,
                                                    const LazyTensor& lhs,
                                                    bool left_side, bool upper,
                                                    bool transpose,
                                                    bool unitriangular);

// Returns the lower triangular part of a matrix (2-D tensor) or batch of
// matrices input, the other elements of the result tensor out are set to 0.
LazyTensor tril(const LazyTensor& input, lazy_tensors::int64 diagonal);

// In-place version of the method above.
void tril_(LazyTensor& input, lazy_tensors::int64 diagonal);

// Returns the upper triangular part of a matrix (2-D tensor) or batch of
// matrices input, the other elements of the result tensor out are set to 0.
LazyTensor triu(const LazyTensor& input, lazy_tensors::int64 diagonal);

// In-place version of the method above.
void triu_(LazyTensor& input, lazy_tensors::int64 diagonal);

LazyTensor trunc(const LazyTensor& input);

LazyTensor ts_softmax_backward(const LazyTensor& grad_output,
                               const LazyTensor& output,
                               lazy_tensors::int64 dim, const LazyTensor& self);

// Returns a tuple of all slices along a given dimension with that dimension
// removed.
std::vector<LazyTensor> unbind(const LazyTensor& input,
                               lazy_tensors::int64 dim);

void uniform_(LazyTensor& input, double from, double to);

// Insert a dimension of size one at the specified position.
LazyTensor unsqueeze(const LazyTensor& input, lazy_tensors::int64 dim);

// In-place version of the method above.
void unsqueeze_(LazyTensor& input, lazy_tensors::int64 dim);

LazyTensor upsample_bilinear2d(const LazyTensor& input,
                               std::vector<lazy_tensors::int64> output_size,
                               bool align_corners);

LazyTensor upsample_bilinear2d_backward(
    const LazyTensor& grad_output, std::vector<lazy_tensors::int64> output_size,
    std::vector<lazy_tensors::int64> input_size, bool align_corners);

LazyTensor upsample_nearest2d(const LazyTensor& input,
                              std::vector<lazy_tensors::int64> output_size);

LazyTensor upsample_nearest2d_backward(
    const LazyTensor& grad_output, std::vector<lazy_tensors::int64> output_size,
    std::vector<lazy_tensors::int64> input_size);

LazyTensor var(const LazyTensor& input,
               std::vector<lazy_tensors::int64> dimensions,
               lazy_tensors::int64 correction, bool keep_reduced_dimensions);

std::tuple<LazyTensor, LazyTensor> var_mean(
    const LazyTensor& input, std::vector<lazy_tensors::int64> dimensions,
    lazy_tensors::int64 correction, bool keep_reduced_dimensions);

// Like reshape, but it returns a view into the original tensor.
LazyTensor view(const LazyTensor& input,
                lazy_tensors::Span<const lazy_tensors::int64> output_size);

void zero_(LazyTensor& input);

LazyTensor where(const LazyTensor& condition, const LazyTensor& input,
                 const LazyTensor& other);

}  // namespace tensor_aten_ops
}  // namespace torch_lazy_tensors
