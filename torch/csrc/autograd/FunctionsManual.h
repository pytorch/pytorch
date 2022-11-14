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

namespace torch {
namespace autograd {
namespace generated {
namespace details {

extern const char* kCudnnDoubleBackwardMsg;

// A simple way to imperatively compute index ranges for slots
// that have been flattened
struct IndexRangeGenerator {
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

Tensor toNonOptFwGrad(const c10::optional<Tensor>& t);
Tensor toNonOptPrimal(const c10::optional<Tensor>& t);
Tensor toNonOptTensor(const c10::optional<Tensor>& t);

Tensor apply_loss_reduction(const Tensor& unreduced, int64_t reduction);
bool any_variable_defined(const variable_list& variables);
void copy_range(variable_list& out, IndexRange range, const at::Tensor& t);
void copy_range(
    variable_list& out,
    IndexRange range,
    at::ArrayRef<at::Tensor> t);
at::Tensor copysign_tensor_self_backward(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& result);
at::Tensor not_implemented(const char* name, const char* reason = "");
std::vector<Tensor> not_implemented_list(
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
    const optional<at::Scalar>& p_,
    const at::Tensor& norm);
at::Tensor norm_backward(
    at::Tensor grad,
    const at::Tensor& self,
    const optional<at::Scalar>& p_,
    at::Tensor norm,
    at::IntArrayRef dim,
    bool keepdim);
Tensor norm_jvp(
    const Tensor& self_p,
    const Tensor& self_t,
    const optional<Scalar>& p_,
    Tensor norm,
    IntArrayRef dim,
    bool keepdim);
Tensor norm_jvp(
    const Tensor& grad,
    const Tensor& self,
    const optional<Scalar>& p_,
    Tensor norm);
Tensor _nested_from_padded_backward(
    const Tensor& grad,
    const Tensor& input,
    const bool do_transform_0213);
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
    at::Tensor grad,
    const at::Tensor& self,
    const at::Tensor& exponent);
at::Tensor pow_backward_exponent(
    at::Tensor grad,
    const at::Tensor& self,
    const at::Tensor& exponent,
    at::Tensor result);
at::Tensor pow_backward_exponent(
    at::Tensor grad,
    const at::Scalar& base,
    const at::Tensor& exponent,
    at::Tensor result);
at::Tensor angle_backward(at::Tensor grad, const at::Tensor& self);
at::Tensor mul_tensor_backward(Tensor grad, Tensor other, ScalarType self_st);
at::Tensor div_tensor_self_backward(
    Tensor grad,
    Tensor other,
    ScalarType self_st);
at::Tensor div_tensor_other_backward(Tensor grad, Tensor self, Tensor other);
at::Tensor div_tensor_self_backward(
    Tensor grad,
    Tensor other,
    ScalarType self_st,
    const c10::optional<c10::string_view>& rounding_mode);
at::Tensor div_tensor_other_backward(
    Tensor grad,
    Tensor self,
    Tensor other,
    const c10::optional<c10::string_view>& rounding_mode);
at::Tensor mvlgamma_backward(
    at::Tensor grad,
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
at::Tensor logcumsumexp_backward(
    at::Tensor grad,
    const at::Tensor& self,
    at::Tensor result,
    int64_t dim);
at::Tensor unbind_backward(const variable_list& grads, int64_t dim);
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
    const optional<at::Scalar>& min,
    const optional<at::Scalar>& max);
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
    c10::string_view const& input_name);
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
at::Tensor repeat_backward(
    at::Tensor grad,
    at::SymIntArrayRef repeats,
    at::SymIntArrayRef input_shape);
at::Tensor _fused_dropout_backward(
    at::Tensor grad,
    at::Tensor mask,
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
    at::Tensor grad,
    const at::Tensor& input,
    const at::Tensor& value);
Tensor sgn_backward(const Tensor& x, const Tensor& gx, const Tensor& sgn);
Tensor masked_fill_backward(const Tensor& grad, const Tensor& mask);
at::Tensor var_backward(
    at::Tensor grad,
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    c10::optional<int64_t> correction,
    bool keepdim);
at::Tensor var_jvp(
    const at::Tensor& self_t,
    const at::Tensor& self_p,
    const at::Tensor& result,
    at::OptionalIntArrayRef dim_opt,
    c10::optional<int64_t> correction_opt,
    bool keepdim);
at::Tensor std_backward(
    const at::Tensor& result,
    const at::Tensor& grad,
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    c10::optional<int64_t> correction,
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
    c10::optional<int64_t> correction_opt,
    bool keepdim);
Tensor std_mean_backward(
    const Tensor& gstd,
    const Tensor& gmean,
    const Tensor& self,
    const Tensor& std,
    at::OptionalIntArrayRef dim_opt,
    c10::optional<int64_t> correction_opt,
    bool keepdim);
at::Tensor masked_scatter_backward(
    const at::Tensor& grad,
    const at::Tensor& mask,
    c10::SymIntArrayRef sizes);
at::Tensor cholesky_backward(
    const at::Tensor& grad,
    bool upper,
    const at::Tensor& L);
at::Tensor cholesky_jvp(
    const at::Tensor& input_tangent,
    const at::Tensor& L,
    bool upper);
at::Tensor cholesky_inverse_backward(
    at::Tensor grad,
    at::Tensor L,
    bool upper,
    at::Tensor inverse);
at::Tensor cholesky_inverse_jvp(
    const at::Tensor& F,
    const at::Tensor& dF,
    const at::Tensor& X,
    bool upper);
Tensor pinv_jvp(const Tensor& A, const Tensor& pinvA, const Tensor& dA);
Tensor pinv_backward(const Tensor& grad, const Tensor& pinvA, const Tensor& A);
at::Tensor split_with_sizes_backward(
    const std::vector<torch::autograd::Variable>& grads,
    c10::SymIntArrayRef split_sizes,
    int64_t dim,
    c10::SymIntArrayRef sizes,
    const at::TensorOptions& options);
at::Tensor split_backward(
    const std::vector<torch::autograd::Variable>& grads,
    c10::SymInt split_size,
    int64_t dim,
    c10::SymIntArrayRef sizes,
    const at::TensorOptions& options);
at::Tensor max_pool_double_backward(
    const at::Tensor& grad,
    const at::Tensor& indices,
    int dim);
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
    c10::optional<double> eps);
Tensor binary_cross_entropy_target_backward(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight,
    int64_t reduction);
Tensor binary_cross_entropy_double_backward_target(
    const Tensor& grad,
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight,
    int64_t reduction);
Tensor binary_cross_entropy_with_logits_backward(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& target,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& pos_weight_opt,
    int64_t reduction);
at::Tensor binary_cross_entropy_with_logits_target_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& pos_weight,
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
    const c10::optional<at::Tensor>& weight,
    int64_t reduction);
at::Tensor binary_cross_entropy_double_backward_grad_output(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
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
at::Tensor embedding_dense_double_backward(
    const at::Tensor& grad,
    const at::Tensor& indices,
    int64_t padding_idx);
at::Tensor index_backward(
    at::Tensor zeros_like_self,
    const torch::List<c10::optional<Tensor>>& indices,
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
    c10::optional<c10::SymInt> start,
    c10::optional<c10::SymInt> end,
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
    const Tensor& i1,
    const Tensor& i2,
    const Tensor& i3,
    IntArrayRef expand1,
    IntArrayRef expand2,
    IntArrayRef expand3,
    IntArrayRef sumdim,
    std::array<bool, 3> grad_mask);
std::tuple<Tensor, Tensor> linalg_qr_jvp(
    const Tensor& dA,
    const Tensor& Q,
    const Tensor& R,
    const c10::string_view mode);
Tensor linalg_qr_backward(
    const Tensor& gQ,
    const Tensor& gR,
    const Tensor& Q,
    const Tensor& R,
    const c10::string_view mode);
Tensor linalg_matrix_exp_differential(
    const Tensor& self,
    const Tensor& grad,
    bool adjoint);
std::tuple<Tensor, Tensor, Tensor> batchnorm_double_backward(
    const Tensor& input,
    const c10::optional<Tensor>& gamma,
    const Tensor& ggI,
    const Tensor& ggG,
    const Tensor& ggB,
    const Tensor& gO,
    const c10::optional<Tensor>& running_mean,
    const c10::optional<Tensor>& running_var,
    bool training,
    double eps,
    const c10::optional<Tensor>& save_mean,
    const c10::optional<Tensor>& save_invstd,
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
    c10::SymInt last_dim_size);
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
    const bool upper);
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
    const c10::optional<Tensor>& gamma,
    c10::SymInt N,
    c10::SymInt C,
    c10::SymInt HxW,
    int64_t group,
    double eps,
    std::array<bool, 3> grad_input_mask);
Tensor prelu_jvp(
    const Tensor& x,
    const Tensor& dx,
    const Tensor& w,
    const Tensor& dw);
std::tuple<Tensor, Tensor, Tensor> prelu_double_backward(
    const Tensor& grad_grad_input,
    const Tensor& grad_grad_weight,
    const Tensor& grad_out,
    const Tensor& input_,
    const Tensor& weight_);
Tensor prelu_backward_self_jvp(
    const Tensor& x,
    const Tensor& w,
    const Tensor& dw,
    const Tensor& g,
    const Tensor& dg);
Tensor prelu_backward_weight_jvp(
    const Tensor& w,
    const Tensor& x,
    const Tensor& dx,
    const Tensor& g,
    const Tensor& dg);
Tensor gelu_double_backward(
    const Tensor& ggI,
    const Tensor& gO,
    const Tensor& input,
    c10::string_view approximate);
Tensor as_strided_backward(
    Tensor grad,
    TensorGeometry input_geometry,
    c10::SymIntArrayRef sizes,
    c10::SymIntArrayRef strides,
    optional<c10::SymInt> storage_offset_);
Tensor as_strided_scatter_backward(
    Tensor grad,
    TensorGeometry input_geometry,
    TensorGeometry src_geometry,
    c10::SymIntArrayRef sizes,
    c10::SymIntArrayRef strides,
    optional<c10::SymInt> storage_offset);
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
    const c10::optional<Tensor>& gamma,
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
    const c10::SymInt m,
    const c10::SymInt n);

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
    const Tensor& B,
    const c10::optional<double> rcond,
    const c10::optional<c10::string_view> driver,
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
    const c10::optional<Tensor>& running_mean,
    const c10::optional<Tensor>& running_var,
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
    IntArrayRef stride,
    at::SymIntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    at::SymIntArrayRef output_padding,
    int64_t groups);

Tensor _convolution_jvp(
    const Tensor& input_p,
    const Tensor& input_t,
    const Tensor& weight_p,
    const Tensor& weight_t,
    const Tensor& bias_p,
    const Tensor& bias_t,
    IntArrayRef stride,
    at::SymIntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    at::SymIntArrayRef output_padding,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool cudnn_enabled,
    bool allow_tf32);

Tensor convolution_backward_jvp_grad_bias(
    const Tensor& grad_out_t,
    const Tensor& grad_bias);

Tensor cat_jvp(at::ITensorListRef tensors, int64_t dim);
Tensor block_diag_jvp(at::TensorList tensors);
Tensor stack_jvp(at::TensorList tensors, int64_t dim);
Tensor cumprod_jvp(Tensor self_t, Tensor self_p, Tensor result, int dim);
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
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    bool transposed,
    int64_t groups,
    ::std::array<bool, 2> output_mask);

Tensor scatter_reduce_jvp(
    const Tensor& self_p,
    const Tensor& self_t,
    int dim,
    const Tensor& index,
    const Tensor& src_p,
    const Tensor& src_t,
    c10::string_view reduce,
    bool include_self,
    const Tensor& result);

std::tuple<Tensor, Tensor> scatter_reduce_backward(
    const Tensor& grad,
    const Tensor& self,
    int dim,
    const Tensor& index,
    const Tensor& src,
    c10::string_view reduce,
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
    c10::string_view reduce,
    bool include_self,
    const Tensor& result);

Tensor take_backward(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& indices);

} // namespace details
} // namespace generated
} // namespace autograd
} // namespace torch
