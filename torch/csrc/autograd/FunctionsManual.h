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

bool isFwGradDefined(const c10::optional<Tensor>& t);
Tensor toLegacyFwGrad(const c10::optional<Tensor>& t);
Tensor toLegacyPrimal(const c10::optional<Tensor>& t);

bool any_variable_defined(variable_list& variables);
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
at::Tensor pow_backward(at::Tensor grad, const at::Tensor & self, const at::Scalar & exponent_);
at::Tensor pow_backward_self(at::Tensor grad, const at::Tensor & self, const at::Tensor & exponent);
at::Tensor pow_backward_exponent(at::Tensor grad, const at::Tensor& self, const at::Tensor& exponent, at::Tensor result);
at::Tensor pow_backward_exponent(at::Tensor grad, const at::Scalar & base, const at::Tensor& exponent, at::Tensor result);
at::Tensor angle_backward(at::Tensor grad, const at::Tensor& self);
at::Tensor mul_tensor_backward(Tensor grad, Tensor other, ScalarType self_st);
at::Tensor div_tensor_self_backward(Tensor grad, Tensor other, ScalarType self_st);
at::Tensor div_tensor_other_backward(Tensor grad, Tensor self, Tensor other);
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
at::Tensor solve_backward_self(const at::Tensor & grad, const at::Tensor & self, const at::Tensor & A);
at::Tensor solve_backward_A(const at::Tensor & grad, const at::Tensor & self, const at::Tensor & A, const at::Tensor & solution);
at::Tensor cumsum_backward(const at::Tensor & x, int64_t dim);
at::Tensor logsumexp_backward(at::Tensor grad, const at::Tensor & self, at::Tensor result, at::IntArrayRef dim, bool keepdim);
at::Tensor logcumsumexp_backward(at::Tensor grad, const at::Tensor & self, at::Tensor result, int64_t dim);
at::Tensor unbind_backward(const variable_list& grads, int64_t dim);
at::Tensor unsqueeze_to(const at::Tensor & self, at::IntArrayRef sizes);
at::Tensor unsqueeze_to(const at::Tensor & self, int64_t dim, at::IntArrayRef sizes);
std::vector<at::Tensor> cat_tensors_backward(const at::Tensor & grad, const std::vector<std::vector<int64_t>> &sizes, int64_t dim);
at::Tensor clamp_backward(const at::Tensor & grad, const at::Tensor &self, const optional<at::Scalar> & min, const optional<at::Scalar> & max);
at::IntArrayRef strides_or_error(const Tensor & input, c10::string_view const & input_name);
at::Tensor mm_mat1_backward(const Tensor & grad, const Tensor & mat2, at::IntArrayRef mat1_sizes, at::IntArrayRef mat1_strides, const Scalar & alpha);
at::Tensor mm_mat2_backward(const at::Tensor & grad, const at::Tensor & mat1, at::IntArrayRef sizes, at::IntArrayRef strides, const at::Scalar & alpha);
at::Tensor _sparse_addmm_sparse_backward(const at::Tensor& grad, const at::Tensor& sparse_, const at::Tensor& dense, const at::Scalar& alpha);
at::Tensor sparse_sparse_matmul_backward(const at::Tensor& grad, const at::Tensor& mat1, const at::Tensor& mat2,int64_t grad_order);
at::Tensor renorm_backward(const at::Tensor & grad, const at::Tensor & self, at::Scalar p, int64_t dim, at::Scalar maxnorm);
at::Tensor repeat_backward(at::Tensor grad, at::IntArrayRef repeats, at::IntArrayRef input_shape);
at::Tensor _fused_dropout_backward(at::Tensor grad, at::Tensor mask, double p1m);
at::Tensor evenly_distribute_backward(at::Tensor grad, const at::Tensor & input, const at::Tensor & value);
at::Tensor sgn_backward(Tensor result, Tensor grad, Tensor self);
at::Tensor var_backward(at::Tensor grad, const at::Tensor& self, c10::optional<IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim);
at::Tensor std_backward(const at::Tensor& result, const at::Tensor& grad, const at::Tensor& self, c10::optional<IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim);
at::Tensor mean_backward(at::Tensor grad, const at::IntArrayRef sizes, at::IntArrayRef dim, bool keepdim);
at::Tensor mean_backward(at::Tensor grad, const at::IntArrayRef sizes, int64_t numel);
at::Tensor var_std_mean_backward(const variable_list& grads, const at::Tensor& self, const at::Tensor& r1, const at::Tensor& r2, c10::optional<IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim, bool is_std);
at::Tensor masked_scatter_backward(const at::Tensor & grad, const at::Tensor & mask, at::IntArrayRef sizes);
at::Tensor cholesky_backward(at::Tensor grad, bool upper, at::Tensor L);
at::Tensor cholesky_inverse_backward(at::Tensor grad, at::Tensor L, bool upper, at::Tensor inverse);
at::Tensor split_with_sizes_backward(const std::vector<torch::autograd::Variable> &grads,
                                     IntArrayRef split_sizes, int64_t dim, IntArrayRef sizes, const at::TensorOptions &options);
at::Tensor split_backward(const std::vector<torch::autograd::Variable> &grads, int64_t split_size, int64_t dim, at::IntArrayRef sizes, const at::TensorOptions &options);
at::Tensor max_pool_double_backward(const at::Tensor & grad, const at::Tensor & indices, int dim);
at::Tensor glu_double_backward(const at::Tensor & grad, const at::Tensor & grad_output, const at::Tensor & input, int64_t dim);
at::Tensor glu_double_backward_grad_output(const at::Tensor & grad, const at::Tensor & input, int64_t dim);
at::Tensor infinitely_differentiable_silu_backward(const at::Tensor& grad_output, const at::Tensor& input);
Tensor infinitely_differentiable_logit_backward(const Tensor& grad, const Tensor& self, c10::optional<double> eps);
at::Tensor kl_div_double_backward_grad_output(const at::Tensor & grad, const at::Tensor & input, const at::Tensor & target, int64_t reduction, bool log_target);
at::Tensor binary_cross_entropy_with_logits_target_backward(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& target, const c10::optional<at::Tensor>& weight, const c10::optional<at::Tensor>& pos_weight, int64_t reduction);
at::Tensor log_sigmoid_double_backward(const at::Tensor & grad, const at::Tensor & input);
at::Tensor softmax_double_backward(const at::Tensor & grad, const at::Tensor & grad_output, int dim, const at::Tensor & output);
at::Tensor log_softmax_double_backward(const at::Tensor & grad, const at::Tensor & grad_output, int dim, const at::Tensor & output);
at::Tensor binary_cross_entropy_double_backward(const at::Tensor & grad_output, const at::Tensor & grad, const at::Tensor & input, const at::Tensor & target, const c10::optional<at::Tensor>& weight, int64_t reduction);
at::Tensor binary_cross_entropy_double_backward_grad_output(const at::Tensor & grad, const at::Tensor & input, const at::Tensor & target, const c10::optional<at::Tensor>& weight, int64_t reduction);
at::Tensor l1_loss_double_backward(const at::Tensor & grad, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & target, int64_t reduction);
at::Tensor l1_loss_double_backward_grad_output(const at::Tensor & grad, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & target, int64_t reduction);
at::Tensor smooth_l1_loss_double_backward(const at::Tensor & grad, const at::Tensor & input, const at::Tensor & target, int64_t reduction, double beta);
at::Tensor smooth_l1_loss_double_backward_grad_output(const at::Tensor & grad, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & target, int64_t reduction, double beta);
at::Tensor mse_loss_double_backward(const at::Tensor & grad, const at::Tensor & input, int64_t reduction);
at::Tensor mse_loss_double_backward_grad_output(const at::Tensor & grad, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & target, int64_t reduction);
at::Tensor soft_margin_loss_double_backward(const at::Tensor & grad, const at::Tensor & input, const at::Tensor & target, int64_t reduction);
at::Tensor soft_margin_loss_double_backward_grad_output(const at::Tensor & grad, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & target, int64_t reduction);
at::Tensor softplus_double_backward(const at::Tensor & grad, const at::Tensor & input, at::Scalar beta, at::Scalar threshold);
at::Tensor logdet_backward(const at::Tensor & grad, const at::Tensor& self, const at::Tensor& logdet);
at::Tensor slogdet_backward(const at::Tensor& grad_logabsdet, const at::Tensor& self, const at::Tensor& signdet, const at::Tensor& logabsdet);
at::Tensor log1p_backward(const at::Tensor& grad, const at::Tensor& self);
at::Tensor sparse_constructor_values_backward(const at::Tensor& sparse_grad_out, const at::Tensor& indices, at::IntArrayRef values_shape);
at::Tensor embedding_dense_double_backward(const at::Tensor & grad, const at::Tensor & indices, int64_t padding_idx);
at::Tensor index_backward(at::Tensor zeros_like_self, const torch::List<c10::optional<Tensor>>& indices, const at::Tensor& grad);
at::Tensor _cudnn_ctc_loss_backward(const at::Tensor& grad_out, const at::Tensor& loss, const at::Tensor& raw_grad, bool zero_infinity);
at::Tensor elu_double_backward(const Tensor& grad, const Tensor& grad_output, Scalar alpha, Scalar scale, Scalar input_scale, bool is_result, const Tensor& self_or_result);

Tensor svd_backward(const std::vector<torch::autograd::Variable> &grads, const Tensor& self,
          bool some, bool compute_uv, const Tensor& raw_u, const Tensor& sigma, const Tensor& raw_v);
Tensor slice_backward_wrapper(
    const at::Tensor& grad,
    const c10::IntArrayRef& input_sizes,
    int64_t dim,
    c10::optional<int64_t> start,
    c10::optional<int64_t> end,
    int64_t step);
Tensor symeig_backward(const std::vector<torch::autograd::Variable> &grads, const Tensor& self,
                    bool eigenvectors, bool upper, const Tensor& lambda, const Tensor& v);
std::tuple<Tensor, Tensor> triangular_solve_backward(
    const Tensor & grad_x, const Tensor & grad_m,
    const Tensor & b, const Tensor & a, const Tensor & x,
    const bool upper, const bool transpose, const bool unitriangular,
    std::array<bool, 2> output_mask);
std::tuple<Tensor, Tensor, Tensor> _trilinear_backward(const Tensor& grad_out, const Tensor& i1, const Tensor& i2, const Tensor& i3,
                                                       IntArrayRef expand1, IntArrayRef expand2, IntArrayRef expand3,
                                                       IntArrayRef sumdim, int64_t unroll_dim, std::array<bool, 3> grad_mask);
Tensor linalg_qr_backward(const std::vector<torch::autograd::Variable> &grads, const Tensor& self,
                          std::string mode, const Tensor& Q, const Tensor& R);
Tensor eig_backward(const std::vector<torch::autograd::Variable> &grads, const Tensor& self,
                    bool eigenvectors, const Tensor& lambda, const Tensor& v);
Tensor det_backward(const Tensor & grad, const Tensor& self, const Tensor& det);
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
std::tuple<Tensor, Tensor, Tensor>
infinitely_differentiable_native_layer_norm_backward(
    const Tensor& dY,
    const Tensor& dmean,
    const Tensor& drstd,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const c10::optional<Tensor>& gamma,
    IntArrayRef normalized_shape,
    double eps,
    std::array<bool, 3> grad_input_mask);


} // namespace details
} // namespace generated
} // namespace autograd
} // namespace torch
