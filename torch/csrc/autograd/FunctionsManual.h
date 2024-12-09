#pragma once

// NB: Must be at the top of file to avoid including the deprecated "math.h".
// https://stackoverflow.com/questions/6563810/m-pi-works-with-math-h-but-not-with-cmath-in-visual-studio
#ifdef _MSC_VER
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#endif

#include <ATen/ATen.h>
#include <torch/csrc/autograd/generated/Functions.h>

namespace torch::autograd::generated::details {

extern const char* kCudnnDoubleBackwardMsg;

// A simple way to imperatively compute index ranges for slots
// that have been flattened
struct TORCH_API IndexRangeGenerator {
  IndexRange range(size_t range_size) {
    i += range_size;
    return {i - range_size, i};
  }
  size_t size() {
    return i;
  }

 private:
  size_t i = 0;
};

TORCH_API Tensor toNonOptFwGrad(const std::optional<Tensor>& t);
TORCH_API Tensor toNonOptPrimal(const std::optional<Tensor>& t);
TORCH_API Tensor toNonOptTensor(const std::optional<Tensor>& t);

TORCH_API inline std::optional<Tensor> wrap_opt_if(
    const Tensor& t,
    const bool cond) {
  using OptTensor = std::optional<Tensor>;
  return cond ? OptTensor(t) : static_cast<OptTensor>(std::nullopt);
}

TORCH_API Tensor
apply_loss_reduction(const Tensor& unreduced, int64_t reduction);
TORCH_API bool any_variable_defined(const variable_list& variables);
TORCH_API void copy_range(
    variable_list& out,
    IndexRange range,
    const at::Tensor& t);
TORCH_API void copy_range(
    variable_list& out,
    IndexRange range,
    at::ArrayRef<at::Tensor> t);
TORCH_API at::Tensor copysign_tensor_self_backward(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& result);
TORCH_API at::Tensor not_implemented(const char* name, const char* reason = "");
TORCH_API std::vector<Tensor> not_implemented_list(
    const char* name,
    const char* reason = "");
at::Tensor handle_r_to_c(ScalarType self_st, Tensor gradient_result);
at::Tensor maybe_multiply(const at::Tensor& t, const at::Scalar& s);
int64_t _safe_size(IntArrayRef sizes, IntArrayRef dim);
Tensor restore_reduced_dims(
    const Tensor& output,
    IntArrayRef dims,
    bool keepdim);
Tensor scale_grad_by_count(
    const Tensor& grad,
    const Tensor& mask,
    IntArrayRef dims);
at::Tensor norm_backward(
    const at::Tensor& grad,
    const at::Tensor& self,
    const std::optional<at::Scalar>& p_,
    const at::Tensor& norm);
at::Tensor norm_backward(
    at::Tensor grad,
    const at::Tensor& self,
    const std::optional<at::Scalar>& p_,
    at::Tensor norm,
    at::IntArrayRef dim,
    bool keepdim);
Tensor norm_jvp(
    const Tensor& self_p,
    const Tensor& self_t,
    const std::optional<Scalar>& p_,
    Tensor norm,
    IntArrayRef dim,
    bool keepdim);
Tensor norm_jvp(
    const Tensor& grad,
    const Tensor& self,
    const std::optional<Scalar>& p_,
    Tensor norm);
Tensor _nested_from_padded_backward(
    const Tensor& grad,
    const Tensor& input,
    const bool do_transform_0213);
std::tuple<Tensor, Tensor, Tensor> linear_double_backward(
    const variable_list& grads,
    const Tensor& self,
    const Tensor& grad_output,
    const Tensor& weight);
Tensor linalg_vector_norm_jvp(
    const Tensor& self_p,
    const Tensor& self_t,
    const Scalar& scalar_ord,
    Tensor norm,
    const at::OptionalIntArrayRef& opt_dim,
    bool keepdim);
at::Tensor linalg_vector_norm_backward(
    at::Tensor grad,
    const at::Tensor& self,
    const at::Scalar& ord,
    at::Tensor norm,
    const at::OptionalIntArrayRef& opt_dim,
    bool keepdim);
at::Tensor pow_backward(
    at::Tensor grad,
    const at::Tensor& self,
    const at::Scalar& exponent_);
at::Tensor pow_backward_self(
    const at::Tensor& grad,
    const at::Tensor& self,
    const at::Tensor& exponent);
at::Tensor pow_backward_exponent(
    const at::Tensor& grad,
    const at::Tensor& self,
    const at::Tensor& exponent,
    const at::Tensor& result);
at::Tensor pow_backward_exponent(
    const at::Tensor& grad,
    const at::Scalar& base,
    const at::Tensor& exponent,
    const at::Tensor& result);
at::Tensor angle_backward(const at::Tensor& grad, const at::Tensor& self);
template <typename T>
at::Tensor mul_tensor_backward(const Tensor& grad, T other, ScalarType self_st);
template <typename T>
at::Tensor div_tensor_self_backward(
    const Tensor& grad,
    T other,
    ScalarType self_st,
    const std::optional<std::string_view>& rounding_mode = std::nullopt);
at::Tensor div_tensor_other_backward(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& other,
    const std::optional<std::string_view>& rounding_mode = std::nullopt);
at::Tensor mvlgamma_backward(
    const at::Tensor& grad,
    const at::Tensor& self,
    int64_t p);
at::Tensor permute_backwards(const at::Tensor& grad, at::IntArrayRef fwd_dims);
at::Tensor rad2deg_backward(const at::Tensor& grad);
at::Tensor deg2rad_backward(const at::Tensor& grad);
at::Tensor unsqueeze_multiple(
    const at::Tensor& t,
    at::OptionalIntArrayRef opt_dim,
    size_t n_dims);
at::Tensor sum_backward(
    const at::Tensor& grad,
    at::SymIntArrayRef sizes,
    at::OptionalIntArrayRef opt_dims,
    bool keepdim);
at::Tensor sum_backward(
    const at::Tensor& grad,
    c10::SymIntArrayRef sizes,
    c10::IntArrayRef dims,
    bool keepdim);
at::Tensor nansum_backward(
    const at::Tensor& grad,
    const at::Tensor& self,
    at::OptionalIntArrayRef dims,
    bool keepdim);
std::vector<int64_t> reverse_list(const at::IntArrayRef list);
std::vector<c10::SymInt> reverse_list_symint(const c10::SymIntArrayRef list);
at::Tensor reverse_dim(const at::Tensor& t, int64_t dim);
at::Tensor prod_safe_zeros_backward(
    const at::Tensor& grad,
    const at::Tensor& inp,
    int64_t dim);
at::Tensor prod_backward(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& result);
at::Tensor prod_backward(
    at::Tensor grad,
    const at::Tensor& input,
    at::Tensor result,
    int64_t dim,
    bool keepdim);
at::Tensor solve_jvp(
    const Tensor& X,
    const Tensor& A,
    const Tensor& dA,
    const Tensor& dB);
at::Tensor solve_backward_self(
    const at::Tensor& grad,
    const at::Tensor& self,
    const at::Tensor& A);
at::Tensor solve_backward_A(
    const at::Tensor& grad,
    const at::Tensor& self,
    const at::Tensor& A,
    const at::Tensor& solution);
at::Tensor cumsum_backward(const at::Tensor& grad, int64_t dim);
at::Tensor logsumexp_backward(
    at::Tensor grad,
    const at::Tensor& self,
    at::Tensor result,
    at::IntArrayRef dim,
    bool keepdim);
at::Tensor logsumexp_jvp(
    const at::Tensor& self_p,
    const at::Tensor& self_t,
    IntArrayRef dim,
    bool keepdim);
at::Tensor safe_logsumexp_jvp(
    const at::Tensor& self_p,
    const at::Tensor& self_t,
    IntArrayRef dim,
    bool keepdim);
at::Tensor logcumsumexp_backward(
    at::Tensor grad,
    const at::Tensor& self,
    const at::Tensor& result,
    int64_t dim);
at::Tensor logcumsumexp_jvp(
    const at::Tensor& self_p,
    const at::Tensor& self_t,
    int64_t dim);
at::Tensor unbind_backward(const variable_list& grads, int64_t dim);
at::Tensor unbind_backward_nested(
    const variable_list& grads,
    const Tensor& nt_sizes,
    int64_t dim,
    const at::TensorOptions& options);
at::Tensor unbind_backward_nested_jagged(
    const variable_list& grads,
    const Tensor& self,
    int64_t dim);
at::Tensor unsqueeze_to(const at::Tensor& self, c10::SymIntArrayRef sym_sizes);
at::Tensor unsqueeze_to(
    const at::Tensor& self,
    int64_t dim,
    c10::SymIntArrayRef sym_sizes);
at::Tensor unsqueeze_to(
    const at::Tensor& self,
    IntArrayRef dim,
    c10::SymIntArrayRef sym_sizes);
std::vector<at::Tensor> cat_tensors_backward(
    const at::Tensor& grad,
    const std::vector<std::vector<c10::SymInt>>& sizes,
    const std::vector<ScalarType>& dtypes,
    int64_t dim);
std::vector<at::Tensor> stack_tensors_backward(
    const at::Tensor& grad,
    int64_t dim,
    const std::vector<ScalarType>& dtypes);
std::vector<at::Tensor> block_diag_backward(
    const at::Tensor& grad,
    const std::vector<std::vector<int64_t>>& sizes,
    const std::vector<ScalarType>& dtypes);
at::Tensor clamp_backward(
    const at::Tensor& grad,
    const at::Tensor& self,
    const std::optional<at::Scalar>& min,
    const std::optional<at::Scalar>& max);
at::Tensor clamp_backward(
    const at::Tensor& grad,
    const at::Tensor& self,
    const at::Tensor& min,
    const at::Tensor& max);
std::tuple<at::Tensor, at::Tensor> clamp_backward_min_max(
    const at::Tensor& grad,
    const at::Tensor& self,
    const at::Tensor& min,
    const at::Tensor& max,
    const std::array<bool, 2>&);
at::Tensor clamp_jvp(
    const Tensor& self_p,
    const Tensor& self_t,
    const Tensor& min_p,
    const Tensor& min_t,
    const Tensor& max_p,
    const Tensor& max_t);
at::SymIntArrayRef strides_or_error(
    const Tensor& input,
    std::string_view const& input_name);
at::Tensor mm_mat1_backward(
    const Tensor& grad,
    const Tensor& mat2,
    at::SymIntArrayRef mat1_sizes,
    at::SymIntArrayRef mat1_strides,
    c10::Layout mat1_layout,
    const Scalar& alpha);
at::Tensor mm_mat2_backward(
    const at::Tensor& grad,
    const at::Tensor& mat1,
    at::SymIntArrayRef sizes,
    at::SymIntArrayRef strides,
    c10::Layout layout,
    const at::Scalar& alpha);
at::Tensor mm_mat1_sparse_backward(
    const at::Tensor& grad,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& alpha);
std::tuple<Tensor, Tensor, Tensor> sparse_sampled_addmm_backward(
    const Tensor& grad,
    const Tensor& self,
    const std::optional<Tensor>& mat1,
    const std::optional<Tensor>& mat2,
    const Scalar& alpha,
    const Scalar& beta,
    const std::array<bool, 3>& grad_input_mask);
at::Tensor sparse_mask_backward(
    const at::Tensor& grad,
    const at::Tensor& mask,
    c10::Layout self_layout);
at::Tensor sparse_sparse_matmul_backward(
    const at::Tensor& grad,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    int64_t grad_order);
at::Tensor renorm_backward(
    const at::Tensor& grad,
    const at::Tensor& self,
    const at::Scalar& p,
    int64_t dim,
    const at::Scalar& maxnorm);
at::Tensor renorm_jvp(
    const at::Tensor& self_p,
    const at::Tensor& self_t,
    const at::Scalar& p,
    int64_t dim,
    const at::Scalar& maxnorm);
at::Tensor repeat_backward(
    at::Tensor grad,
    at::SymIntArrayRef repeats,
    at::SymIntArrayRef input_shape);
at::Tensor _fused_dropout_backward(
    const at::Tensor& grad,
    const at::Tensor& mask,
    double p1m);
at::Tensor infinitely_differentiable_native_dropout_backward(
    const at::Tensor& grad,
    const at::Tensor& mask,
    double scale);
at::Tensor native_dropout_double_backward(
    const at::Tensor& ggI,
    const at::Tensor& grad,
    const at::Tensor& mask,
    double scale);
at::Tensor evenly_distribute_backward(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& value);
Tensor sgn_backward(const Tensor& x, const Tensor& gx, const Tensor& sgn);
Tensor masked_fill_backward(const Tensor& grad, const Tensor& mask);
at::Tensor var_backward(
    at::Tensor grad,
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const std::optional<c10::Scalar>& correction,
    bool keepdim);
at::Tensor var_jvp(
    const at::Tensor& self_t,
    const at::Tensor& self_p,
    const at::Tensor& result,
    at::OptionalIntArrayRef dim_opt,
    const std::optional<c10::Scalar>& correction,
    bool keepdim);
at::Tensor std_backward(
    const at::Tensor& result,
    const at::Tensor& grad,
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const std::optional<c10::Scalar>& correction,
    bool keepdim);
Tensor mean_backward(
    const Tensor& grad,
    c10::SymIntArrayRef shape,
    at::OptionalIntArrayRef opt_dim,
    c10::SymInt numel,
    bool keepdim);
Tensor var_mean_backward(
    const Tensor& gvar,
    const Tensor& gmean,
    const Tensor& self,
    at::OptionalIntArrayRef dim_opt,
    const std::optional<c10::Scalar>& correction,
    bool keepdim);
Tensor std_mean_backward(
    const Tensor& gstd,
    const Tensor& gmean,
    const Tensor& self,
    const Tensor& std,
    at::OptionalIntArrayRef dim_opt,
    const std::optional<c10::Scalar>& correction,
    bool keepdim);
at::Tensor cholesky_backward(
    const at::Tensor& grad,
    bool upper,
    const at::Tensor& L);
at::Tensor cholesky_jvp(
    const at::Tensor& input_tangent,
    const at::Tensor& L,
    bool upper);
at::Tensor cholesky_inverse_backward(
    const at::Tensor& grad,
    const at::Tensor& L,
    bool upper,
    const at::Tensor& inverse);
at::Tensor cholesky_inverse_jvp(
    const at::Tensor& F,
    const at::Tensor& dF,
    const at::Tensor& X,
    bool upper);
Tensor pinv_jvp(const Tensor& A, const Tensor& pinvA, const Tensor& dA);
Tensor pinv_backward(const Tensor& grad, const Tensor& pinvA, const Tensor& A);
Tensor chunk_backward_nested(
    const std::vector<torch::autograd::Variable>& grads,
    const Tensor& self,
    int64_t chunks,
    int64_t dim);
at::Tensor split_with_sizes_backward(
    const std::vector<torch::autograd::Variable>& grads,
    c10::SymIntArrayRef split_sizes,
    int64_t dim,
    c10::SymIntArrayRef sizes,
    const at::TensorOptions& options);
at::Tensor _nested_split_with_sizes_backward(
    const std::vector<torch::autograd::Variable>& grads,
    c10::SymIntArrayRef split_sizes,
    int64_t dim,
    const Tensor& nt_sizes,
    const at::TensorOptions& options);
at::Tensor split_backward(
    const std::vector<torch::autograd::Variable>& grads,
    const c10::SymInt& split_size,
    int64_t dim,
    c10::SymIntArrayRef sizes,
    const at::TensorOptions& options);
at::Tensor max_pool_double_backward(
    const at::Tensor& grad,
    const at::Tensor& indices,
    int dim);
at::Tensor error_for_max_pool2d_double_backward();
at::Tensor glu_double_backward(
    const at::Tensor& grad,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t dim);
at::Tensor glu_double_backward_grad_output(
    const at::Tensor& grad,
    const at::Tensor& input,
    int64_t dim);
at::Tensor infinitely_differentiable_silu_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input);
at::Tensor infinitely_differentiable_mish_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input);
Tensor infinitely_differentiable_logit_backward(
    const Tensor& grad,
    const Tensor& self,
    std::optional<double> eps);
Tensor binary_cross_entropy_target_backward(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight,
    int64_t reduction);
Tensor binary_cross_entropy_double_backward_target(
    const Tensor& grad,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight,
    int64_t reduction);
Tensor binary_cross_entropy_with_logits_backward(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& target,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& pos_weight_opt,
    int64_t reduction);
at::Tensor binary_cross_entropy_with_logits_target_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& pos_weight,
    int64_t reduction);
at::Tensor log_sigmoid_double_backward(
    const at::Tensor& grad,
    const at::Tensor& input);
at::Tensor softmax_double_backward(
    const at::Tensor& grad,
    const at::Tensor& grad_output,
    int dim,
    const at::Tensor& output);
at::Tensor binary_cross_entropy_double_backward(
    const at::Tensor& grad_output,
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& target,
    const std::optional<at::Tensor>& weight,
    int64_t reduction);
at::Tensor binary_cross_entropy_double_backward_grad_output(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& target,
    const std::optional<at::Tensor>& weight,
    int64_t reduction);
at::Tensor smooth_l1_loss_double_backward(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& target,
    int64_t reduction,
    double beta);
at::Tensor huber_loss_double_backward(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& target,
    int64_t reduction,
    double delta);
at::Tensor huber_loss_double_backward_grad_output(
    const at::Tensor& grad,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& target,
    int64_t reduction,
    double delta);
at::Tensor mse_loss_double_backward(
    const at::Tensor& grad,
    const at::Tensor& input,
    int64_t reduction);
at::Tensor soft_margin_loss_double_backward(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& target,
    int64_t reduction);
at::Tensor soft_margin_loss_double_backward_grad_output(
    const at::Tensor& grad,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& target,
    int64_t reduction);
at::Tensor softplus_double_backward(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Scalar& beta,
    const at::Scalar& threshold);
std::tuple<at::Tensor, at::Tensor> slogdet_jvp(
    const at::Tensor& LU,
    const at::Tensor& pivots,
    const at::Tensor& dA,
    const at::Tensor& sign,
    const bool use_A_T);
at::Tensor slogdet_backward(
    const at::Tensor& grad_sign,
    const at::Tensor& grad_logabsdet,
    const at::Tensor& A,
    const at::Tensor& signdet,
    const at::Tensor& LU,
    const at::Tensor& pivots);
at::Tensor log1p_backward(const at::Tensor& grad, const at::Tensor& self);
at::Tensor sinc_backward(const at::Tensor& grad, const at::Tensor& self);
at::Tensor sparse_constructor_values_backward(
    const at::Tensor& sparse_grad_out,
    const at::Tensor& indices);
at::Tensor embedding_dense_double_backward_symint(
    const at::Tensor& grad,
    const at::Tensor& indices,
    const c10::SymInt& padding_idx);
at::Tensor index_backward(
    at::Tensor zeros_like_self,
    const torch::List<std::optional<Tensor>>& indices,
    const at::Tensor& grad);
at::Tensor _cudnn_ctc_loss_backward(
    const at::Tensor& grad_out,
    const at::Tensor& loss,
    const at::Tensor& raw_grad,
    bool zero_infinity);
at::Tensor elu_double_backward(
    const Tensor& grad,
    const Tensor& grad_output,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale,
    bool is_result,
    const Tensor& self_or_result);

Tensor svd_backward(
    const Tensor& gU,
    const Tensor& gS,
    const Tensor& gVh,
    const Tensor& U,
    const Tensor& S,
    const Tensor& Vh);

std::tuple<Tensor, Tensor, Tensor> linalg_svd_jvp(
    const Tensor& dA,
    const Tensor& U,
    const Tensor& S,
    const Tensor& Vh,
    const bool full_matrices);
Tensor slice_backward_wrapper(
    const at::Tensor& grad,
    const c10::SymIntArrayRef& input_sizes,
    int64_t dim,
    std::optional<c10::SymInt> start,
    std::optional<c10::SymInt> end,
    c10::SymInt step);
std::tuple<Tensor, Tensor> linalg_eig_jvp(
    const Tensor& dA,
    const Tensor& L,
    const Tensor& V,
    const bool is_hermitian);
Tensor linalg_eig_backward(
    const Tensor& gL,
    const Tensor& gV,
    const Tensor& L,
    const Tensor& V,
    const bool is_hermitian,
    const bool symeig_eigenvectors = true);
Tensor linalg_lstsq_jvp(
    const Tensor& A,
    const Tensor& B,
    const Tensor& dA,
    const Tensor& dB);
std::tuple<Tensor, Tensor> triangular_solve_backward(
    const Tensor& grad_x,
    const Tensor& grad_m,
    const Tensor& b,
    const Tensor& a,
    const Tensor& x,
    const bool upper,
    const bool transpose,
    const bool unitriangular,
    std::array<bool, 2> output_mask);
Tensor triangular_solve_jvp(
    const Tensor& X,
    const Tensor& A,
    const Tensor& dA,
    const Tensor& dB,
    const bool upper,
    const bool transpose,
    const bool unitriangular);
Tensor linalg_solve_triangular_forward_AD(
    const Tensor& A_t,
    const Tensor& B_t,
    const Tensor& A,
    const Tensor& X,
    const bool upper,
    const bool left,
    const bool unitriangular);
std::tuple<Tensor, Tensor> linalg_solve_triangular_backward(
    const Tensor& grad,
    const Tensor& A,
    const Tensor& X,
    const bool upper,
    const bool left,
    const bool unitriangular,
    std::array<bool, 2> output_mask);
std::tuple<Tensor, Tensor, Tensor> _trilinear_backward(
    const Tensor& grad_out,
    const std::optional<Tensor>& i1,
    const std::optional<Tensor>& i2,
    const std::optional<Tensor>& i3,
    IntArrayRef expand1,
    IntArrayRef expand2,
    IntArrayRef expand3,
    IntArrayRef sumdim,
    std::array<bool, 3> grad_mask);
std::tuple<Tensor, Tensor> linalg_qr_jvp(
    const Tensor& dA,
    const Tensor& Q,
    const Tensor& R,
    const std::string_view mode);
Tensor linalg_qr_backward(
    const Tensor& gQ,
    const Tensor& gR,
    const Tensor& Q,
    const Tensor& R,
    const std::string_view mode);
Tensor linalg_matrix_exp_differential(
    const Tensor& self,
    const Tensor& grad,
    bool adjoint);
std::tuple<Tensor, Tensor, Tensor> batchnorm_double_backward(
    const Tensor& input,
    const std::optional<Tensor>& gamma,
    const Tensor& ggI,
    const Tensor& ggG,
    const Tensor& ggB,
    const Tensor& gO,
    const std::optional<Tensor>& running_mean,
    const std::optional<Tensor>& running_var,
    bool training,
    double eps,
    const std::optional<Tensor>& save_mean,
    const std::optional<Tensor>& save_invstd,
    std::array<bool, 3> output_mask);
std::tuple<Tensor, Tensor> _euclidean_dist_backward(
    const Tensor& grad,
    const Tensor& x1,
    const Tensor& x2,
    const Tensor& res);
Tensor fft_backward(
    const Tensor& self,
    const Tensor& grad,
    int64_t signal_ndim,
    bool complex_input,
    bool complex_output,
    bool inverse,
    IntArrayRef checked_signal_sizes,
    int64_t normalization,
    bool onesided,
    IntArrayRef output_sizes);
Tensor fft_r2c_backward(
    const Tensor& grad,
    at::IntArrayRef dim,
    int64_t normalization,
    bool onesided,
    const c10::SymInt& last_dim_size);
Tensor fft_c2r_backward(
    const Tensor& grad,
    IntArrayRef dim,
    int64_t normalization);
Tensor constant_pad_nd_backward(const Tensor& grad, c10::SymIntArrayRef pad);
std::tuple<Tensor, Tensor> cholesky_solve_backward(
    const Tensor& grad_x,
    const Tensor& self,
    const Tensor& input2,
    const Tensor& result,
    const bool upper,
    std::array<bool, 2> output_mask);
Tensor cholesky_solve_jvp(
    const Tensor& X,
    const Tensor& U,
    const Tensor& dU,
    const Tensor& dB,
    const bool upper);
std::tuple<Tensor, Tensor, Tensor>
infinitely_differentiable_native_group_norm_backward(
    const Tensor& dY,
    const Tensor& dmean,
    const Tensor& drstd,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const std::optional<Tensor>& gamma,
    c10::SymInt N,
    const c10::SymInt& C,
    c10::SymInt HxW,
    int64_t group,
    double eps,
    std::array<bool, 3> grad_input_mask);
Tensor gelu_double_backward(
    const Tensor& ggI,
    const Tensor& gO,
    const Tensor& input,
    std::string_view approximate);
Tensor as_strided_backward(
    Tensor grad,
    const TensorGeometry& input_geometry,
    c10::SymIntArrayRef sizes,
    c10::SymIntArrayRef strides,
    const std::optional<c10::SymInt>& storage_offset_);
Tensor as_strided_scatter_backward(
    const Tensor& grad,
    const TensorGeometry& input_geometry,
    const TensorGeometry& src_geometry,
    c10::SymIntArrayRef sizes,
    c10::SymIntArrayRef strides,
    std::optional<c10::SymInt> storage_offset);
std::tuple<Tensor, Tensor> atan2_backward(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& other,
    std::array<bool, 2> output_mask);
Tensor amaxamin_jvp(
    const Tensor& x,
    const Tensor& dx,
    const Tensor& result,
    IntArrayRef dim,
    bool keepdim);
std::tuple<Tensor, Tensor, Tensor> layer_norm_double_backward(
    const Tensor& input,
    const std::optional<Tensor>& gamma,
    const Tensor& ggI,
    const Tensor& ggG,
    const Tensor& ggB,
    const Tensor& gO,
    const Tensor& save_mean,
    const Tensor& save_invstd,
    c10::SymIntArrayRef normalized_shape,
    std::array<bool, 3> output_mask);

std::tuple<Tensor, Tensor> householder_product_backward(
    const Tensor& grad,
    const Tensor& result,
    const Tensor& input,
    const Tensor& tau,
    const bool flip_order = false);
Tensor householder_product_jvp(
    const Tensor& dV,
    const Tensor& dtau,
    const Tensor& prod,
    const Tensor& V,
    const Tensor& tau);
std::tuple<Tensor, Tensor, Tensor> ormqr_backward(
    const Tensor& grad,
    const Tensor& result,
    const Tensor& self,
    const Tensor& tau,
    const Tensor& other,
    bool left,
    bool transpose,
    std::array<bool, 3> grad_output_mask);
std::tuple<Tensor, Tensor> polar_backward(
    const Tensor& grad,
    const Tensor& result);
Tensor i1_backward(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& result);
Tensor i1e_backward(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& result);
Tensor linalg_lu_solve_LU(
    const Tensor& grad,
    const Tensor& LU,
    const Tensor& pivots,
    const Tensor& X,
    const bool left,
    const bool adjoint);
Tensor linalg_lu_solve_jvp(
    const Tensor& X,
    const Tensor& LU,
    const Tensor& pivots,
    const Tensor& dLU,
    const Tensor& dB,
    const bool left,
    const bool adjoint);
std::tuple<Tensor, Tensor> linalg_solve_backward(
    const Tensor& gX,
    const Tensor& X,
    const Tensor& A,
    const Tensor& LU,
    const Tensor& pivots,
    const bool left,
    const bool B_requires_grad);
Tensor linalg_solve_jvp(
    const Tensor& dA,
    const Tensor& dB,
    const Tensor& X,
    const Tensor& LU,
    const Tensor& pivots,
    const bool left,
    const bool use_A_T);
Tensor lu_unpack_backward(
    const Tensor& L_grad,
    const Tensor& U_grad,
    const c10::SymInt& m,
    const c10::SymInt& n);

Tensor linalg_det_backward(
    const Tensor& grad,
    const Tensor& det,
    const Tensor& A,
    const Tensor& LU,
    const Tensor& pivots);
Tensor linalg_det_jvp(
    const Tensor& dA,
    const Tensor& det,
    const Tensor& LU,
    const Tensor& pivots,
    const bool use_A_T);
std::tuple<Tensor, Tensor> linalg_lstsq_backward(
    const Tensor& grad,
    const Tensor& A,
    const Tensor& B_,
    const std::array<bool, 2>& grad_input_mask);
Tensor linalg_lu_backward(
    const Tensor& L_grad,
    const Tensor& U_grad,
    const Tensor& P,
    const Tensor& L,
    const Tensor& U,
    const bool pivot);

std::tuple<Tensor, Tensor> linalg_lu_jvp(
    const Tensor& dA,
    const Tensor& P,
    const Tensor& L,
    const Tensor& U,
    const bool pivot);

Tensor lu_factor_ex_backward(
    const Tensor& grad,
    const Tensor& LU,
    const Tensor& pivs,
    const bool pivot);
Tensor lu_factor_ex_jvp(
    const Tensor& dX,
    const Tensor& LU,
    const Tensor& pivs,
    const bool pivot);

Tensor batch_norm_jvp(
    const Tensor& input_p,
    const Tensor& input_t,
    const Tensor& weight_p,
    const Tensor& weight_t,
    const Tensor& bias_p,
    const Tensor& bias_t,
    const std::optional<Tensor>& running_mean,
    const std::optional<Tensor>& running_var,
    const Tensor& saved_mean,
    const Tensor& saved_invstd,
    bool train,
    double eps);

Tensor layer_norm_jvp(
    const Tensor& input_p,
    const Tensor& input_t,
    const Tensor& weight_p,
    const Tensor& weight_t,
    const Tensor& bias_p,
    const Tensor& bias_t,
    const Tensor& saved_mean,
    const Tensor& saved_invstd,
    c10::SymIntArrayRef normalized_shape);

Tensor group_norm_jvp(
    const Tensor& input_p,
    const Tensor& input_t,
    const Tensor& weight_p,
    const Tensor& weight_t,
    const Tensor& bias_p,
    const Tensor& bias_t,
    const Tensor& saved_mean,
    const Tensor& saved_invstd,
    int64_t groups);
Tensor group_norm_mean_jvp(
    const Tensor& input_t,
    const Tensor& mean_p,
    int64_t groups);
Tensor group_norm_invstd_jvp(
    const Tensor& input_p,
    const Tensor& input_t,
    const Tensor& mean_p,
    const Tensor& invstd_p,
    int64_t groups);

Tensor convolution_jvp(
    const Tensor& input_p,
    const Tensor& input_t,
    const Tensor& weight_p,
    const Tensor& weight_t,
    const Tensor& bias_p,
    const Tensor& bias_t,
    at::SymIntArrayRef stride,
    at::SymIntArrayRef padding,
    at::SymIntArrayRef dilation,
    bool transposed,
    at::SymIntArrayRef output_padding,
    const c10::SymInt& groups);

Tensor _convolution_jvp(
    const Tensor& input_p,
    const Tensor& input_t,
    const Tensor& weight_p,
    const Tensor& weight_t,
    const Tensor& bias_p,
    const Tensor& bias_t,
    at::SymIntArrayRef stride,
    at::SymIntArrayRef padding,
    at::SymIntArrayRef dilation,
    bool transposed,
    at::SymIntArrayRef output_padding,
    const c10::SymInt& groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32);

Tensor convolution_backward_jvp_grad_bias(
    const Tensor& grad_out_t,
    const Tensor& grad_bias);

Tensor cat_jvp(const at::ITensorListRef& tensors, int64_t dim);
Tensor block_diag_jvp(at::TensorList tensors);
Tensor stack_jvp(at::TensorList tensors, int64_t dim);
Tensor cumprod_jvp(
    const Tensor& self_t,
    const Tensor& self_p,
    const Tensor& result,
    int dim);
Tensor gather_with_keepdimed_indices(
    const Tensor& input,
    int64_t dim,
    const Tensor& indices,
    bool keepdim);
Tensor evenly_read_jvp(
    const Tensor& fw_grad,
    const Tensor& input,
    const Tensor& value);
Tensor warn_backwards(const Tensor& grad_output);

std::tuple<Tensor, Tensor> _cudnn_convolution_backward(
    const at::Tensor& self,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    at::SymIntArrayRef padding,
    at::SymIntArrayRef output_padding,
    at::SymIntArrayRef stride,
    at::SymIntArrayRef dilation,
    bool transposed,
    c10::SymInt groups,
    ::std::array<bool, 2> output_mask);

Tensor scatter_reduce_jvp(
    const Tensor& self_p,
    const Tensor& self_t,
    int dim,
    const Tensor& index,
    const Tensor& src_p,
    const Tensor& src_t,
    std::string_view reduce,
    bool include_self,
    const Tensor& result);

std::tuple<Tensor, Tensor> scatter_reduce_backward(
    const Tensor& grad,
    const Tensor& self,
    int dim,
    const Tensor& index,
    const Tensor& src,
    std::string_view reduce,
    bool include_self,
    const Tensor& result);

Tensor _to_copy_backward(
    const Tensor& grad,
    const c10::TensorOptions& self_options);

std::tuple<Tensor, Tensor> index_reduce_backward(
    const Tensor& grad,
    const Tensor& self,
    int dim,
    const Tensor& index,
    const Tensor& source,
    std::string_view reduce,
    bool include_self,
    const Tensor& result);

Tensor take_backward(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& indices);

Tensor to_sparse_backward(
    const Tensor& grad,
    const c10::Layout self_layout,
    const c10::OptionalArrayRef<c10::SymInt>& self_blocksize);

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>
mkldnn_rnn_layer_differentiable_backward(
    const Tensor& input,
    const Tensor& weight0,
    const Tensor& weight1,
    const Tensor& weight2,
    const Tensor& weight3,
    const Tensor& hx_,
    const Tensor& cx_tmp,
    const Tensor& output,
    const Tensor& hy_,
    const Tensor& cy_,
    const std::optional<Tensor>& grad_output_r_opt,
    const std::optional<Tensor>& grad_hy_r_opt,
    const std::optional<Tensor>& grad_cy_r_opt,
    bool reverse,
    int64_t mode,
    int64_t hidden_size,
    int64_t num_layers,
    bool has_biases,
    bool train,
    bool bidirectional,
    at::IntArrayRef batch_sizes,
    bool batch_first,
    const at::Tensor& workspace);

Tensor values_backward(const Tensor& grad, const Tensor& self);

} // namespace torch::autograd::generated::details
