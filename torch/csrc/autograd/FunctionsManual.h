#pragma once

// NB: Must be at the top of file to avoid including the deprecated "math.h".
// https://stackoverflow.com/questions/6563810/m-pi-works-with-math-h-but-not-with-cmath-in-visual-studio
#ifdef _MSC_VER
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#endif

#include <torch/csrc/autograd/generated/Functions.h>
#include <ATen/ATen.h>

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
  size_t size() { return i; }
  private:
    size_t i = 0;
};

Tensor toNonOptFwGrad(const c10::optional<Tensor>& t);
Tensor toNonOptPrimal(const c10::optional<Tensor>& t);
Tensor toNonOptTensor(const c10::optional<Tensor>& t);

Tensor apply_loss_reduction(const Tensor& unreduced, int64_t reduction);
bool any_variable_defined(const variable_list& variables);
void copy_range(variable_list& out, IndexRange range, const at::Tensor & t);
void copy_range(variable_list& out, IndexRange range, at::ArrayRef<at::Tensor> t);
at::Tensor copysign_tensor_self_backward(const Tensor & grad, const Tensor & self, const Tensor & result);
at::Tensor not_implemented(const char* name, const char* reason="");
std::vector<Tensor> not_implemented_list(const char* name, const char* reason="");
at::Tensor handle_r_to_c(ScalarType self_st, Tensor gradient_result);
at::Tensor maybe_multiply(const at::Tensor & t, const at::Scalar & s);
int64_t _safe_size(IntArrayRef sizes, IntArrayRef dim);
Tensor restore_reduced_dims(const Tensor &output, IntArrayRef dims, bool keepdim);
Tensor scale_grad_by_count(const Tensor &grad, const Tensor &mask, IntArrayRef dims);
at::Tensor norm_backward(const at::Tensor & grad, const at::Tensor & self, const optional<at::Scalar> & p_, const at::Tensor & norm);
at::Tensor norm_backward(at::Tensor grad, const at::Tensor & self, const optional<at::Scalar> & p_, at::Tensor norm, at::IntArrayRef dim, bool keepdim);
at::Tensor linalg_vector_norm_backward(at::Tensor grad, const at::Tensor & self, const at::Scalar & ord, at::Tensor norm, const c10::optional<at::IntArrayRef> & opt_dim, bool keepdim);
at::Tensor pow_backward(at::Tensor grad, const at::Tensor & self, const at::Scalar & exponent_);
at::Tensor pow_backward_self(at::Tensor grad, const at::Tensor & self, const at::Tensor & exponent);
at::Tensor pow_backward_exponent(at::Tensor grad, const at::Tensor& self, const at::Tensor& exponent, at::Tensor result);
at::Tensor pow_backward_exponent(at::Tensor grad, const at::Scalar & base, const at::Tensor& exponent, at::Tensor result);
at::Tensor angle_backward(at::Tensor grad, const at::Tensor& self);
at::Tensor mul_tensor_backward(Tensor grad, Tensor other, ScalarType self_st);
at::Tensor div_tensor_self_backward(Tensor grad, Tensor other, ScalarType self_st);
at::Tensor div_tensor_other_backward(Tensor grad, Tensor self, Tensor other);
at::Tensor div_tensor_self_backward(Tensor grad, Tensor other, ScalarType self_st, const c10::optional<c10::string_view>& rounding_mode);
at::Tensor div_tensor_other_backward(Tensor grad, Tensor self, Tensor other, const c10::optional<c10::string_view>& rounding_mode);
at::Tensor mvlgamma_backward(at::Tensor grad, const at::Tensor & self, int64_t p);
at::Tensor permute_backwards(const at::Tensor & grad, at::IntArrayRef fwd_dims);
at::Tensor rad2deg_backward(const at::Tensor& grad);
at::Tensor deg2rad_backward(const at::Tensor& grad);
at::Tensor unsqueeze_multiple(const at::Tensor & t, at::IntArrayRef dim, size_t n_dims);
at::Tensor sum_backward(const at::Tensor & grad, at::IntArrayRef sizes, at::IntArrayRef dims, bool keepdim);
at::Tensor nansum_backward(const at::Tensor & grad, const at::Tensor & self, at::IntArrayRef dims, bool keepdim);
std::vector<int64_t> reverse_list(const at::IntArrayRef list);
at::Tensor reverse_dim(const at::Tensor& t, int64_t dim);
at::Tensor prod_safe_zeros_backward(const at::Tensor &grad, const at::Tensor& inp, int64_t dim);
at::Tensor prod_backward(const at::Tensor& grad, const at::Tensor& input, const at::Tensor& result);
at::Tensor prod_backward(at::Tensor grad, const at::Tensor& input, at::Tensor result, int64_t dim, bool keepdim);
at::Tensor solve_jvp(const Tensor& X, const Tensor& A, const Tensor& dA, const Tensor& dB);
at::Tensor solve_backward_self(const at::Tensor & grad, const at::Tensor & self, const at::Tensor & A);
at::Tensor solve_backward_A(const at::Tensor & grad, const at::Tensor & self, const at::Tensor & A, const at::Tensor & solution);
at::Tensor cumsum_backward(const at::Tensor & grad, int64_t dim);
at::Tensor logsumexp_backward(at::Tensor grad, const at::Tensor & self, at::Tensor result, at::IntArrayRef dim, bool keepdim);
at::Tensor logcumsumexp_backward(at::Tensor grad, const at::Tensor & self, at::Tensor result, int64_t dim);
at::Tensor unbind_backward(const variable_list& grads, int64_t dim);
at::Tensor unsqueeze_to(const at::Tensor & self, at::IntArrayRef sizes);
at::Tensor unsqueeze_to(const at::Tensor & self, int64_t dim, at::IntArrayRef sizes);
std::vector<at::Tensor> cat_tensors_backward(const at::Tensor & grad, const std::vector<std::vector<int64_t>> &sizes, const std::vector<ScalarType> &dtypes, int64_t dim);
at::Tensor clamp_backward(const at::Tensor & grad, const at::Tensor &self, const optional<at::Scalar>& min, const optional<at::Scalar>& max);
at::Tensor clamp_backward(const at::Tensor & grad, const at::Tensor &self, const at::Tensor& min, const at::Tensor& max);
std::tuple<at::Tensor, at::Tensor> clamp_backward_min_max(const at::Tensor& grad, const at::Tensor& self, const at::Tensor& min, const at::Tensor& max, const std::array<bool, 2>&);
at::IntArrayRef strides_or_error(const Tensor & input, c10::string_view const & input_name);
at::Tensor mm_mat1_backward(const Tensor & grad, const Tensor & mat2, at::IntArrayRef mat1_sizes, at::IntArrayRef mat1_strides, const Scalar & alpha);
at::Tensor mm_mat2_backward(const at::Tensor & grad, const at::Tensor & mat1, at::IntArrayRef sizes, at::IntArrayRef strides, const at::Scalar & alpha);
at::Tensor _sparse_addmm_sparse_backward(const at::Tensor& grad, const at::Tensor& sparse_, const at::Tensor& dense, const at::Scalar& alpha);
at::Tensor sparse_sparse_matmul_backward(const at::Tensor& grad, const at::Tensor& mat1, const at::Tensor& mat2,int64_t grad_order);
at::Tensor renorm_backward(const at::Tensor & grad, const at::Tensor & self, const at::Scalar& p, int64_t dim, const at::Scalar& maxnorm);
at::Tensor repeat_backward(at::Tensor grad, at::IntArrayRef repeats, at::IntArrayRef input_shape);
at::Tensor _fused_dropout_backward(at::Tensor grad, at::Tensor mask, double p1m);
at::Tensor infinitely_differentiable_native_dropout_backward(const at::Tensor& grad, const at::Tensor& mask, double scale);
at::Tensor native_dropout_double_backward(const at::Tensor& ggI, const at::Tensor& grad, const at::Tensor& mask, double scale);
at::Tensor evenly_distribute_backward(at::Tensor grad, const at::Tensor & input, const at::Tensor & value);
at::Tensor sgn_backward(Tensor result, Tensor grad, Tensor self);
at::Tensor var_backward(at::Tensor grad, const at::Tensor& self, c10::optional<IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim);
at::Tensor var_jvp(const at::Tensor& self_t, const at::Tensor& self_p, const at::Tensor& result, c10::optional<IntArrayRef> dim_opt, c10::optional<int64_t> correction_opt, bool keepdim);
at::Tensor std_backward(const at::Tensor& result, const at::Tensor& grad, const at::Tensor& self, c10::optional<IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim);
at::Tensor mean_backward(at::Tensor grad, const at::IntArrayRef sizes, at::IntArrayRef dim, bool keepdim);
at::Tensor mean_backward(at::Tensor grad, const at::IntArrayRef sizes, int64_t numel);
at::Tensor var_std_mean_backward(const variable_list& grads, const at::Tensor& self, const at::Tensor& r1, const at::Tensor& r2, c10::optional<IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim, bool is_std);
at::Tensor masked_scatter_backward(const at::Tensor & grad, const at::Tensor & mask, at::IntArrayRef sizes);
at::Tensor cholesky_backward(at::Tensor grad, bool upper, at::Tensor L);
at::Tensor cholesky_jvp(const at::Tensor& input_tangent, const at::Tensor& L, bool upper);
at::Tensor cholesky_inverse_backward(at::Tensor grad, at::Tensor L, bool upper, at::Tensor inverse);
Tensor pinv_jvp(
  const Tensor& A,
  const Tensor& pinvA,
  const Tensor& dA
);
Tensor pinv_backward(
  const Tensor& grad,
  const Tensor& pinvA,
  const Tensor& A
);
at::Tensor split_with_sizes_backward(const std::vector<torch::autograd::Variable> &grads,
                                     IntArrayRef split_sizes, int64_t dim, IntArrayRef sizes, const at::TensorOptions &options);
at::Tensor split_backward(const std::vector<torch::autograd::Variable> &grads, int64_t split_size, int64_t dim, at::IntArrayRef sizes, const at::TensorOptions &options);
at::Tensor max_pool_double_backward(const at::Tensor & grad, const at::Tensor & indices, int dim);
at::Tensor glu_double_backward(const at::Tensor & grad, const at::Tensor & grad_output, const at::Tensor & input, int64_t dim);
at::Tensor glu_double_backward_grad_output(const at::Tensor & grad, const at::Tensor & input, int64_t dim);
at::Tensor infinitely_differentiable_silu_backward(const at::Tensor& grad_output, const at::Tensor& input);
at::Tensor infinitely_differentiable_mish_backward(const at::Tensor& grad_output, const at::Tensor& input);
Tensor infinitely_differentiable_logit_backward(const Tensor& grad, const Tensor& self, c10::optional<double> eps);
at::Tensor kl_div_double_backward_grad_output(const at::Tensor & grad, const at::Tensor & input, const at::Tensor & target, int64_t reduction, bool log_target);
Tensor binary_cross_entropy_target_backward(
  const Tensor& grad,
  const Tensor& self,
  const Tensor& target,
  const c10::optional<Tensor>& weight,
  int64_t reduction);
at::Tensor binary_cross_entropy_with_logits_target_backward(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& target, const c10::optional<at::Tensor>& weight, const c10::optional<at::Tensor>& pos_weight, int64_t reduction);
at::Tensor binary_cross_entropy_with_logits_jvp(const Tensor& input_t, const Tensor& target_t, const Tensor& input_p, const Tensor& target_p, const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& pos_weight_opt, int64_t reduction);
at::Tensor log_sigmoid_double_backward(const at::Tensor & grad, const at::Tensor & input);
at::Tensor softmax_double_backward(const at::Tensor & grad, const at::Tensor & grad_output, int dim, const at::Tensor & output);
at::Tensor binary_cross_entropy_double_backward(const at::Tensor & grad_output, const at::Tensor & grad, const at::Tensor & input, const at::Tensor & target, const c10::optional<at::Tensor>& weight, int64_t reduction);
at::Tensor binary_cross_entropy_double_backward_grad_output(const at::Tensor & grad, const at::Tensor & input, const at::Tensor & target, const c10::optional<at::Tensor>& weight, int64_t reduction);
at::Tensor l1_loss_double_backward(const at::Tensor & grad, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & target, int64_t reduction);
at::Tensor l1_loss_double_backward_grad_output(const at::Tensor & grad, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & target, int64_t reduction);
at::Tensor smooth_l1_loss_double_backward(const at::Tensor & grad, const at::Tensor & input, const at::Tensor & target, int64_t reduction, double beta);
at::Tensor smooth_l1_loss_double_backward_grad_output(const at::Tensor & grad, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & target, int64_t reduction, double beta);
at::Tensor huber_loss_double_backward(const at::Tensor & grad, const at::Tensor & input, const at::Tensor & target, int64_t reduction, double delta);
at::Tensor huber_loss_double_backward_grad_output(const at::Tensor & grad, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & target, int64_t reduction, double delta);
at::Tensor mse_loss_double_backward(const at::Tensor & grad, const at::Tensor & input, int64_t reduction);
at::Tensor mse_loss_double_backward_grad_output(const at::Tensor & grad, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & target, int64_t reduction);
at::Tensor soft_margin_loss_double_backward(const at::Tensor & grad, const at::Tensor & input, const at::Tensor & target, int64_t reduction);
at::Tensor soft_margin_loss_double_backward_grad_output(const at::Tensor & grad, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & target, int64_t reduction);
at::Tensor softplus_double_backward(const at::Tensor & grad, const at::Tensor & input, const at::Scalar& beta, const at::Scalar& threshold);
at::Tensor logdet_backward(const at::Tensor & grad, const at::Tensor& self, const at::Tensor& logdet);
at::Tensor slogdet_backward(const at::Tensor& grad_logabsdet, const at::Tensor& self, const at::Tensor& signdet, const at::Tensor& logabsdet);
at::Tensor log1p_backward(const at::Tensor& grad, const at::Tensor& self);
at::Tensor sinc_backward(const at::Tensor& grad, const at::Tensor& self);
at::Tensor sparse_constructor_values_backward(const at::Tensor& sparse_grad_out, const at::Tensor& indices);
at::Tensor embedding_dense_double_backward(const at::Tensor & grad, const at::Tensor & indices, int64_t padding_idx);
at::Tensor index_backward(at::Tensor zeros_like_self, const torch::List<c10::optional<Tensor>>& indices, const at::Tensor& grad);
at::Tensor _cudnn_ctc_loss_backward(const at::Tensor& grad_out, const at::Tensor& loss, const at::Tensor& raw_grad, bool zero_infinity);
at::Tensor elu_double_backward(const Tensor& grad, const Tensor& grad_output, const Scalar& alpha, const Scalar& scale, const Scalar& input_scale, bool is_result, const Tensor& self_or_result);

Tensor svd_backward(const Tensor& gU,
                    const Tensor& gS,
                    const Tensor& gVh,
                    const Tensor& U,
                    const Tensor& S,
                    const Tensor& Vh);

std::tuple<Tensor, Tensor, Tensor> linalg_svd_jvp(const Tensor& dA,
                                                  const Tensor& U,
                                                  const Tensor& S,
                                                  const Tensor& Vh,
                                                  const bool full_matrices);
Tensor slice_backward_wrapper(
    const at::Tensor& grad,
    const c10::IntArrayRef& input_sizes,
    int64_t dim,
    c10::optional<int64_t> start,
    c10::optional<int64_t> end,
    int64_t step);
std::tuple<Tensor, Tensor> linalg_eig_jvp(const Tensor& dA,
                                          const Tensor& L,
                                          const Tensor& V,
                                          const bool is_hermitian);
Tensor linalg_eig_backward(const Tensor& gL,
                           const Tensor& gV,
                           const Tensor& L,
                           const Tensor& V,
                           const bool is_hermitian,
                           const bool symeig_eigenvectors=true);
Tensor linalg_lstsq_jvp(
  const Tensor& A,
  const Tensor& B,
  const Tensor& dA,
  const Tensor& dB
);
std::tuple<Tensor, Tensor> triangular_solve_backward(
    const Tensor & grad_x, const Tensor & grad_m,
    const Tensor & b, const Tensor & a, const Tensor & x,
    const bool upper, const bool transpose, const bool unitriangular,
    std::array<bool, 2> output_mask);
Tensor triangular_solve_jvp(
  const Tensor& X, const Tensor& A,
  const Tensor& dA, const Tensor& dB,
  const bool upper,
  const bool transpose,
  const bool unitriangular
);
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
std::tuple<Tensor, Tensor, Tensor> _trilinear_backward(const Tensor& grad_out, const Tensor& i1, const Tensor& i2, const Tensor& i3,
                                                       IntArrayRef expand1, IntArrayRef expand2, IntArrayRef expand3,
                                                       IntArrayRef sumdim, std::array<bool, 3> grad_mask);
std::tuple<Tensor, Tensor> linalg_qr_jvp(
  const Tensor& dA,
  const Tensor& Q,
  const Tensor& R
);
Tensor linalg_qr_jvp_Q(
  const Tensor& dA,
  const Tensor& Q,
  const Tensor& R
);
Tensor linalg_qr_jvp_R(
  const Tensor& dA,
  const Tensor& Q,
  const Tensor& R
);
Tensor linalg_qr_backward(const std::vector<torch::autograd::Variable> &grads, const Tensor& self,
                          c10::string_view mode, const Tensor& Q, const Tensor& R);
Tensor eig_backward(const std::vector<torch::autograd::Variable> &grads, const Tensor& self,
                    bool eigenvectors, const Tensor& lambda, const Tensor& v);
Tensor linalg_matrix_exp_differential(const Tensor& self, const Tensor& grad, bool adjoint);
Tensor linalg_det_backward(const Tensor & grad, const Tensor& self, const Tensor& det);
std::tuple<Tensor, Tensor, Tensor> batchnorm_double_backward(
    const Tensor & input,
    const c10::optional<Tensor> & gamma,
    const Tensor & ggI,
    const Tensor & ggG,
    const Tensor & ggB,
    const Tensor & gO,
    const c10::optional<Tensor> & running_mean,
    const c10::optional<Tensor> & running_var,
    bool training,
    double eps,
    const c10::optional<Tensor> & save_mean,
    const c10::optional<Tensor> & save_invstd,
    std::array<bool,3> output_mask);
std::tuple<Tensor, Tensor> _euclidean_dist_backward(const Tensor & grad, const Tensor & x1, const Tensor & x2, const Tensor & res);
Tensor kl_div_target_backward(Tensor grad_output, Tensor self, Tensor target, int64_t reduction, bool log_target);
Tensor fft_backward(const Tensor& self, const Tensor& grad, int64_t signal_ndim,
                    bool complex_input, bool complex_output,
                    bool inverse, IntArrayRef checked_signal_sizes,
                    int64_t normalization, bool onesided,
                    IntArrayRef output_sizes);
Tensor fft_r2c_backward(const Tensor& grad, IntArrayRef dim, int64_t normalization,
                        bool onesided, int64_t last_dim_size);
Tensor fft_c2r_backward(const Tensor& grad, IntArrayRef dim, int64_t normalization);
Tensor constant_pad_nd_backward(const Tensor& grad, IntArrayRef pad);
std::tuple<Tensor, Tensor> cholesky_solve_backward(
    const Tensor& grad_x, const Tensor& self,
    const Tensor& input2, const Tensor& result, const bool upper);
Tensor cholesky_solve_jvp(
    const Tensor& X,
    const Tensor& U,
    const Tensor& dU,
    const Tensor& dB,
    const bool upper
);
std::tuple<Tensor, Tensor, Tensor>
infinitely_differentiable_native_group_norm_backward(
    const Tensor& dY,
    const Tensor& dmean,
    const Tensor& drstd,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const c10::optional<Tensor>& gamma,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    double eps,
    std::array<bool, 3> grad_input_mask);
std::tuple<Tensor, Tensor, Tensor> prelu_double_backward(
    const Tensor & grad_grad_input,
    const Tensor & grad_grad_weight,
    const Tensor & grad_out,
    const Tensor & input_,
    const Tensor & weight_);
Tensor as_strided_backward(Tensor grad, TensorGeometry input_geometry, IntArrayRef sizes, IntArrayRef strides, optional<int64_t> storage_offset_);
std::tuple<Tensor, Tensor> atan2_backward(const Tensor& grad, const Tensor& self, const Tensor& other, std::array<bool, 2> output_mask);
std::tuple<Tensor, Tensor, Tensor> layer_norm_double_backward(
    const Tensor & input,
    const c10::optional<Tensor> & gamma,
    const Tensor & ggI,
    const Tensor & ggG,
    const Tensor & ggB,
    const Tensor & gO,
    const Tensor & save_mean,
    const Tensor & save_invstd,
    IntArrayRef normalized_shape,
    std::array<bool,3> output_mask);

std::tuple<Tensor, Tensor> householder_product_backward(const Tensor& grad, const Tensor& result, const Tensor& input, const Tensor& tau);
Tensor householder_product_jvp(
    const Tensor& dV,
    const Tensor& dtau,
    const Tensor& prod,
    const Tensor& V,
    const Tensor& tau
);
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
std::tuple<Tensor, Tensor> lu_solve_backward(
  const Tensor& grad,
  const Tensor& X,
  const Tensor& LU_data,
  const Tensor& LU_pivots,
  const std::array<bool, 2>& grad_input_mask
);
Tensor lu_solve_jvp(
  const Tensor& X,
  const Tensor& LU_data,
  const Tensor& dLU_data,
  const Tensor& dB,
  const Tensor& LU_pivots
);
Tensor lu_unpack_backward(
  const variable_list& grads,
  const Tensor& LU_data,
  bool unpack_data
);

Tensor _det_lu_based_helper_backward(
  const Tensor& det_grad,
  const Tensor& det,
  const Tensor& self,
  const Tensor& lu,
  const Tensor& pivs
);

std::tuple<Tensor, Tensor> linalg_lstsq_backward(
  const Tensor& grad,
  const Tensor& A,
  const Tensor& B,
  const c10::optional<double> rcond,
  const c10::optional<c10::string_view> driver,
  const std::array<bool, 2>& grad_input_mask
);

Tensor lu_backward_base(
  const variable_list& grads,
  const Tensor& self,
  const Tensor& P,
  const Tensor& L,
  const Tensor& U
);
Tensor lu_factor_ex_backward(
  const Tensor& grad,
  const Tensor& self,
  const Tensor& LU,
  const Tensor& pivs
);
Tensor lu_factor_ex_jvp(
  const Tensor& dX,
  const Tensor& LU,
  const Tensor& pivs
);

Tensor batch_norm_jvp(
  const Tensor& input_p, const Tensor& input_t,
  const Tensor& weight_p, const Tensor& weight_t,
  const Tensor& bias_p, const Tensor& bias_t,
  const c10::optional<Tensor>& running_mean,
  const c10::optional<Tensor>& running_var,
  const Tensor& saved_mean, const Tensor& saved_invstd,
  bool train,
  double eps
);

Tensor layer_norm_jvp(
  const Tensor& input_p, const Tensor& input_t,
  const Tensor& weight_p, const Tensor& weight_t,
  const Tensor& bias_p, const Tensor& bias_t,
  const Tensor& saved_mean, const Tensor& saved_invstd,
  IntArrayRef normalized_shape
);

Tensor group_norm_jvp(
  const Tensor& input_p, const Tensor& input_t,
  const Tensor& weight_p, const Tensor& weight_t,
  const Tensor& bias_p, const Tensor& bias_t,
  const Tensor& saved_mean, const Tensor& saved_invstd,
  int64_t groups
);
Tensor group_norm_mean_jvp(
  const Tensor& input_t,
  const Tensor& mean_p,
  int64_t groups
);
Tensor group_norm_invstd_jvp(
  const Tensor& input_p, const Tensor& input_t,
  const Tensor& mean_p, const Tensor& invstd_p,
  int64_t groups
);

Tensor convolution_jvp(
  const Tensor& input_p, const Tensor& input_t,
  const Tensor& weight_p, const Tensor& weight_t,
  const Tensor& bias_p, const Tensor& bias_t,
  IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
  bool transposed, IntArrayRef output_padding, int64_t groups
);

Tensor _convolution_jvp(
  const Tensor& input_p, const Tensor& input_t,
  const Tensor& weight_p, const Tensor& weight_t,
  const Tensor& bias_p, const Tensor& bias_t,
  IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
  bool transposed, IntArrayRef output_padding, int64_t groups,
  bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32
);

Tensor convolution_backward_jvp_grad_bias(const Tensor& grad_out_t, const Tensor& grad_bias);

Tensor stack_jvp(at::TensorList tensors, int64_t dim);
Tensor cat_jvp(at::ITensorList tensors, int64_t dim);
Tensor cumprod_jvp(Tensor self_t, Tensor self_p, Tensor result, int dim);
Tensor gather_with_keepdimed_indices(const Tensor& input, int64_t dim, const Tensor& indices, bool keepdim);
Tensor evenly_read_jvp(const Tensor& fw_grad, const Tensor & input, const Tensor & value);
Tensor warn_backwards(const Tensor &grad_output);

std::tuple<Tensor, Tensor> _cudnn_convolution_backward(
    const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding,
    at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, bool transposed, int64_t groups,
    ::std::array<bool,2> output_mask);

Tensor scatter_reduce_backward(
  const Tensor& grad,
  const Tensor& input,
  int dim,
  const Tensor& index,
  c10::string_view reduce,
  const Tensor& result
);


} // namespace details
} // namespace generated
} // namespace autograd
} // namespace torch
