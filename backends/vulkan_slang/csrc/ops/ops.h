#pragma once

#include <torch/torch.h>

namespace torch_vulkan { namespace ops {

// Binary ops
at::Tensor vulkan_add(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha);
at::Tensor vulkan_sub(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha);
at::Tensor vulkan_mul(const at::Tensor& self, const at::Tensor& other);
at::Tensor vulkan_div(const at::Tensor& self, const at::Tensor& other);
at::Tensor vulkan_pow(const at::Tensor& self, const at::Tensor& exponent);
at::Tensor vulkan_pow_scalar(const at::Tensor& self, const at::Scalar& exponent);

// Binary ops (scalar)
at::Tensor vulkan_add_scalar(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha);
at::Tensor vulkan_sub_scalar(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha);
at::Tensor vulkan_rsub_scalar(const at::Tensor& self, const at::Scalar& scalar);
at::Tensor vulkan_mul_scalar(const at::Tensor& self, const at::Scalar& other);
at::Tensor vulkan_div_scalar(const at::Tensor& self, const at::Scalar& other);

// In-place binary ops (tensor)
at::Tensor& vulkan_add_(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha);
at::Tensor& vulkan_sub_(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha);
at::Tensor& vulkan_mul_(at::Tensor& self, const at::Tensor& other);
at::Tensor& vulkan_div_(at::Tensor& self, const at::Tensor& other);

// In-place binary ops (scalar)
at::Tensor& vulkan_add_scalar_(at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha);
at::Tensor& vulkan_sub_scalar_(at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha);
at::Tensor& vulkan_mul_scalar_(at::Tensor& self, const at::Scalar& other);
at::Tensor& vulkan_div_scalar_(at::Tensor& self, const at::Scalar& other);

// Unary ops
at::Tensor vulkan_neg(const at::Tensor& self);
at::Tensor vulkan_abs(const at::Tensor& self);
at::Tensor vulkan_exp(const at::Tensor& self);
at::Tensor vulkan_log(const at::Tensor& self);
at::Tensor vulkan_sqrt(const at::Tensor& self);
at::Tensor vulkan_rsqrt(const at::Tensor& self);
at::Tensor vulkan_ceil(const at::Tensor& self);
at::Tensor vulkan_floor(const at::Tensor& self);
at::Tensor vulkan_round(const at::Tensor& self);
at::Tensor vulkan_sign(const at::Tensor& self);

// Comparison ops (tensor)
at::Tensor vulkan_eq(const at::Tensor& self, const at::Tensor& other);
at::Tensor vulkan_ne(const at::Tensor& self, const at::Tensor& other);
at::Tensor vulkan_lt(const at::Tensor& self, const at::Tensor& other);
at::Tensor vulkan_gt(const at::Tensor& self, const at::Tensor& other);
at::Tensor vulkan_le(const at::Tensor& self, const at::Tensor& other);
at::Tensor vulkan_ge(const at::Tensor& self, const at::Tensor& other);
at::Tensor vulkan_where(const at::Tensor& condition, const at::Tensor& self, const at::Tensor& other);

// Comparison ops (scalar)
at::Tensor vulkan_eq_scalar(const at::Tensor& self, const at::Scalar& other);
at::Tensor vulkan_ne_scalar(const at::Tensor& self, const at::Scalar& other);
at::Tensor vulkan_lt_scalar(const at::Tensor& self, const at::Scalar& other);
at::Tensor vulkan_gt_scalar(const at::Tensor& self, const at::Scalar& other);
at::Tensor vulkan_le_scalar(const at::Tensor& self, const at::Scalar& other);
at::Tensor vulkan_ge_scalar(const at::Tensor& self, const at::Scalar& other);

// Activations
at::Tensor vulkan_relu(const at::Tensor& self);
at::Tensor& vulkan_relu_(at::Tensor& self);
at::Tensor vulkan_sigmoid(const at::Tensor& self);
at::Tensor vulkan_tanh(const at::Tensor& self);
at::Tensor vulkan_gelu(const at::Tensor& self, c10::string_view approximate);
at::Tensor vulkan_silu(const at::Tensor& self);
at::Tensor vulkan_leaky_relu(const at::Tensor& self, const at::Scalar& negative_slope);
at::Tensor vulkan_elu(const at::Tensor& self, const at::Scalar& alpha,
                      const at::Scalar& scale, const at::Scalar& input_scale);
at::Tensor vulkan_clamp(const at::Tensor& self,
                        const std::optional<at::Scalar>& min_val,
                        const std::optional<at::Scalar>& max_val);
at::Tensor vulkan_clamp_min(const at::Tensor& self, const at::Scalar& min_val);
at::Tensor& vulkan_clamp_min_(at::Tensor& self, const at::Scalar& min_val);
at::Tensor vulkan_clamp_min_tensor(const at::Tensor& self, const at::Tensor& min_val);
at::Tensor& vulkan_clamp_min_tensor_out(const at::Tensor& self, const at::Tensor& min_val, at::Tensor& out);
at::Tensor vulkan_clamp_max(const at::Tensor& self, const at::Scalar& max_val);
at::Tensor& vulkan_clamp_max_(at::Tensor& self, const at::Scalar& max_val);
at::Tensor& vulkan_clamp_min_out(const at::Tensor& self, const at::Scalar& min_val, at::Tensor& out);
at::Tensor& vulkan_clamp_max_out(const at::Tensor& self, const at::Scalar& max_val, at::Tensor& out);
at::Tensor vulkan_selu(const at::Tensor& self);
at::Tensor vulkan_prelu(const at::Tensor& self, const at::Tensor& weight);
at::Tensor vulkan_hardtanh(const at::Tensor& self, const at::Scalar& min_val, const at::Scalar& max_val);
at::Tensor& vulkan_hardtanh_(at::Tensor& self, const at::Scalar& min_val, const at::Scalar& max_val);
at::Tensor vulkan_hardswish(const at::Tensor& self);
at::Tensor& vulkan_hardswish_(at::Tensor& self);
at::Tensor vulkan_hardsigmoid(const at::Tensor& self);
at::Tensor& vulkan_hardsigmoid_(at::Tensor& self);
at::Tensor vulkan_softplus(const at::Tensor& self, const at::Scalar& beta, const at::Scalar& threshold);
at::Tensor vulkan_mish(const at::Tensor& self);
at::Tensor vulkan_mish_backward(const at::Tensor& grad_output, const at::Tensor& self);
at::Tensor vulkan_swiglu(const at::Tensor& gate, const at::Tensor& up);
std::tuple<at::Tensor, at::Tensor> vulkan_swiglu_backward(const at::Tensor& grad_output, const at::Tensor& gate, const at::Tensor& up);
at::Tensor vulkan_swiglu_autograd(const at::Tensor& gate, const at::Tensor& up);
// Scaled BMM: scale * (q @ k.T) in single dispatch — for attention
at::Tensor vulkan_scaled_bmm_autograd(const at::Tensor& q, const at::Tensor& k, double scale);

// Reductions
at::Tensor vulkan_sum(const at::Tensor& self, at::OptionalIntArrayRef dim,
                      bool keepdim, std::optional<at::ScalarType> dtype);
at::Tensor vulkan_mean(const at::Tensor& self, at::OptionalIntArrayRef dim,
                       bool keepdim, std::optional<at::ScalarType> dtype);
at::Tensor vulkan_amax(const at::Tensor& self, at::IntArrayRef dim, bool keepdim);
at::Tensor vulkan_amin(const at::Tensor& self, at::IntArrayRef dim, bool keepdim);
std::tuple<at::Tensor, at::Tensor> vulkan_max_dim(const at::Tensor& self, int64_t dim, bool keepdim);
std::tuple<at::Tensor, at::Tensor> vulkan_max_dim_out(const at::Tensor& self, int64_t dim, bool keepdim,
                                                       at::Tensor& values_out, at::Tensor& indices_out);
std::tuple<at::Tensor, at::Tensor> vulkan_min_dim(const at::Tensor& self, int64_t dim, bool keepdim);
std::tuple<at::Tensor, at::Tensor> vulkan_min_dim_out(const at::Tensor& self, int64_t dim, bool keepdim,
                                                       at::Tensor& values_out, at::Tensor& indices_out);
at::Tensor vulkan_prod(const at::Tensor& self, int64_t dim, bool keepdim,
                        std::optional<at::ScalarType> dtype);
at::Tensor vulkan_argmax(const at::Tensor& self, std::optional<int64_t> dim, bool keepdim);
at::Tensor vulkan_argmin(const at::Tensor& self, std::optional<int64_t> dim, bool keepdim);
at::Tensor vulkan_any(const at::Tensor& self);
at::Tensor vulkan_any_dim(const at::Tensor& self, int64_t dim, bool keepdim);
at::Tensor vulkan_all(const at::Tensor& self);
at::Tensor vulkan_all_dim(const at::Tensor& self, int64_t dim, bool keepdim);
at::Tensor vulkan_norm(const at::Tensor& self, const at::Scalar& ord,
                        at::OptionalIntArrayRef dim, bool keepdim,
                        std::optional<at::ScalarType> dtype);
at::Tensor vulkan_norm_ScalarOpt_dim(const at::Tensor& self,
                                      const std::optional<at::Scalar>& p,
                                      at::IntArrayRef dim, bool keepdim);

// BLAS
at::Tensor vulkan_mm(const at::Tensor& self, const at::Tensor& mat2);
// Matmul with explicit transpose flags — avoids GPU permute copy from .t()
at::Tensor vulkan_mm_ex(const at::Tensor& self, const at::Tensor& mat2,
                         bool transpose_a, bool transpose_b);
at::Tensor vulkan_addmm(const at::Tensor& bias, const at::Tensor& self, const at::Tensor& mat2,
                         const at::Scalar& beta, const at::Scalar& alpha);
at::Tensor vulkan_bmm(const at::Tensor& self, const at::Tensor& mat2);
// Batched matmul with explicit transpose flags — avoids GPU permute copy
at::Tensor vulkan_bmm_ex(const at::Tensor& self, const at::Tensor& mat2,
                          bool transpose_a, bool transpose_b, float scale = 1.0f);
// scale * (q @ k.T) in single dispatch — fused attention score computation
at::Tensor vulkan_scaled_bmm_forward(const at::Tensor& q, const at::Tensor& k, float scale);
at::Tensor vulkan_linear(const at::Tensor& input, const at::Tensor& weight,
                          const std::optional<at::Tensor>& bias_opt);
at::Tensor vulkan_scaled_mm(
    const at::Tensor& self, const at::Tensor& mat2,
    const at::Tensor& scale_a, const at::Tensor& scale_b,
    const std::optional<at::Tensor>& bias_opt,
    const std::optional<at::Tensor>& scale_result_opt,
    std::optional<at::ScalarType> out_dtype_opt,
    bool use_fast_accum);

// Softmax
at::Tensor vulkan_softmax(const at::Tensor& self, int64_t dim, std::optional<at::ScalarType> dtype);
at::Tensor vulkan_log_softmax(const at::Tensor& self, int64_t dim, std::optional<at::ScalarType> dtype);

// Normalization
std::tuple<at::Tensor, at::Tensor, at::Tensor> vulkan_layer_norm(
    const at::Tensor& input, at::IntArrayRef normalized_shape,
    const std::optional<at::Tensor>& weight_opt, const std::optional<at::Tensor>& bias_opt,
    double eps);
at::Tensor vulkan_batch_norm(
    const at::Tensor& input, const std::optional<at::Tensor>& weight_opt,
    const std::optional<at::Tensor>& bias_opt,
    const std::optional<at::Tensor>& running_mean_opt,
    const std::optional<at::Tensor>& running_var_opt,
    bool training, double momentum, double eps, bool cudnn_enabled);
std::tuple<at::Tensor, at::Tensor, at::Tensor> vulkan_group_norm(
    const at::Tensor& input, int64_t num_groups,
    const std::optional<at::Tensor>& weight_opt, const std::optional<at::Tensor>& bias_opt,
    double eps);

// Pooling
at::Tensor vulkan_max_pool2d(const at::Tensor& self, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode);
at::Tensor vulkan_avg_pool2d(const at::Tensor& self, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode,
    bool count_include_pad, std::optional<int64_t> divisor_override);
at::Tensor vulkan_adaptive_avg_pool2d(const at::Tensor& self, at::IntArrayRef output_size);
at::Tensor vulkan_adaptive_avg_pool2d_backward(
    const at::Tensor& grad_output, const at::Tensor& input);

// Autograd wrappers
at::Tensor vulkan_adaptive_avg_pool2d_autograd(const at::Tensor& self, at::IntArrayRef output_size);

// Indexing
at::Tensor vulkan_embedding(const at::Tensor& weight, const at::Tensor& indices,
    int64_t padding_idx, bool scale_grad_by_freq, bool sparse);
at::Tensor vulkan_index_select(const at::Tensor& self, int64_t dim, const at::Tensor& index);
at::Tensor& vulkan_masked_fill(at::Tensor& self, const at::Tensor& mask, const at::Scalar& value);
at::Tensor vulkan_masked_scatter(const at::Tensor& self, const at::Tensor& mask, const at::Tensor& source);
at::Tensor& vulkan_masked_scatter_(at::Tensor& self, const at::Tensor& mask, const at::Tensor& source);

// Convolution
at::Tensor vulkan_conv2d(const at::Tensor& input, const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_opt, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups);
at::Tensor vulkan_conv_transpose2d(
    const at::Tensor& input, const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_opt, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef output_padding,
    int64_t groups, at::IntArrayRef dilation);

// Shape ops
at::Tensor vulkan_view(const at::Tensor& self, at::IntArrayRef size);
at::Tensor vulkan_reshape(const at::Tensor& self, at::IntArrayRef shape);
at::Tensor vulkan_unsqueeze(const at::Tensor& self, int64_t dim);
at::Tensor vulkan_squeeze(const at::Tensor& self);
at::Tensor vulkan_squeeze_dim(const at::Tensor& self, int64_t dim);
at::Tensor vulkan_permute(const at::Tensor& self, at::IntArrayRef dims);
at::Tensor vulkan_transpose(const at::Tensor& self, int64_t dim0, int64_t dim1);
at::Tensor vulkan_t(const at::Tensor& self);
at::Tensor vulkan_expand(const at::Tensor& self, at::IntArrayRef size, bool implicit);
at::Tensor vulkan_cat(const at::ITensorListRef& tensors, int64_t dim);
at::Tensor vulkan_select(const at::Tensor& self, int64_t dim, int64_t index);
at::Tensor vulkan_slice(const at::Tensor& self, int64_t dim,
                        std::optional<int64_t> start, std::optional<int64_t> end,
                        int64_t step);
std::vector<at::Tensor> vulkan_split(const at::Tensor& self, int64_t split_size, int64_t dim);

// Fill (GPU shader version)
at::Tensor& vulkan_fill_scalar_gpu(at::Tensor& self, const at::Scalar& value);

// Clone
at::Tensor vulkan_clone(const at::Tensor& self, std::optional<at::MemoryFormat> memory_format);

// Tensor factories
at::Tensor vulkan_arange(const at::Scalar& start, const at::Scalar& end,
    const at::Scalar& step, std::optional<at::ScalarType> dtype_opt,
    std::optional<at::Layout> layout_opt, std::optional<at::Device> device_opt,
    std::optional<bool> pin_memory_opt);
at::Tensor vulkan_linspace(const at::Scalar& start, const at::Scalar& end,
    int64_t steps, std::optional<at::ScalarType> dtype_opt,
    std::optional<at::Layout> layout_opt, std::optional<at::Device> device_opt,
    std::optional<bool> pin_memory_opt);
at::Tensor vulkan_eye(int64_t n, std::optional<at::ScalarType> dtype_opt,
    std::optional<at::Layout> layout_opt, std::optional<at::Device> device_opt,
    std::optional<bool> pin_memory_opt);
at::Tensor vulkan_eye_m(int64_t n, int64_t m, std::optional<at::ScalarType> dtype_opt,
    std::optional<at::Layout> layout_opt, std::optional<at::Device> device_opt,
    std::optional<bool> pin_memory_opt);
at::Tensor vulkan_full(at::IntArrayRef size, const at::Scalar& fill_value,
    std::optional<at::ScalarType> dtype_opt, std::optional<at::Layout> layout_opt,
    std::optional<at::Device> device_opt, std::optional<bool> pin_memory_opt);
at::Tensor vulkan_scalar_tensor(const at::Scalar& s,
    std::optional<at::ScalarType> dtype_opt, std::optional<at::Layout> layout_opt,
    std::optional<at::Device> device_opt, std::optional<bool> pin_memory_opt);

// RNG
void vulkan_manual_seed(uint64_t seed);
at::Tensor& vulkan_uniform_(at::Tensor& self, double from, double to,
    std::optional<at::Generator> generator);
at::Tensor& vulkan_normal_(at::Tensor& self, double mean, double std,
    std::optional<at::Generator> generator);
std::tuple<at::Tensor, at::Tensor> vulkan_native_dropout(
    const at::Tensor& input, double p, std::optional<bool> train);
at::Tensor vulkan_native_dropout_backward(
    const at::Tensor& grad_output, const at::Tensor& mask, double scale);
at::Tensor& vulkan_bernoulli_(at::Tensor& self, double p,
    std::optional<at::Generator> generator);
at::Tensor& vulkan_bernoulli_p(at::Tensor& self, const at::Tensor& p,
    std::optional<at::Generator> generator);

// Optimizer ops
at::Tensor& vulkan_addcmul_(at::Tensor& self, const at::Tensor& tensor1,
    const at::Tensor& tensor2, const at::Scalar& value);
at::Tensor& vulkan_addcdiv_(at::Tensor& self, const at::Tensor& tensor1,
    const at::Tensor& tensor2, const at::Scalar& value);
at::Tensor& vulkan_lerp_(at::Tensor& self, const at::Tensor& end, const at::Scalar& weight);
at::Tensor& vulkan_clamp_(at::Tensor& self,
    const std::optional<at::Scalar>& min_val, const std::optional<at::Scalar>& max_val);

// Fused SGD step
void vulkan_sgd_step(
    at::Tensor& param, const at::Tensor& grad, at::Tensor& momentum_buf,
    float lr, float momentum, float dampening, float weight_decay,
    bool nesterov, bool has_momentum_buf);

// Batched SGD step (no momentum): up to 15 params per dispatch
void vulkan_sgd_batch_step(
    const std::vector<at::Tensor*>& params,
    const std::vector<const at::Tensor*>& grads,
    float lr, float weight_decay);

// Batched AdamW step: up to 7 params per dispatch
void vulkan_adamw_batch_step(
    const std::vector<at::Tensor*>& params,
    const std::vector<const at::Tensor*>& grads,
    const std::vector<at::Tensor*>& m_bufs,
    const std::vector<at::Tensor*>& v_bufs,
    float lr, float beta1, float beta2, float eps,
    float weight_decay, float bc1, float bc2);

// Fused AdamW step
void vulkan_adamw_step(
    at::Tensor& param, const at::Tensor& grad,
    at::Tensor& m, at::Tensor& v,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    int64_t step);

// Loss functions
std::tuple<at::Tensor, at::Tensor> vulkan_nll_loss_forward(
    const at::Tensor& self, const at::Tensor& target,
    const std::optional<at::Tensor>& weight, int64_t reduction, int64_t ignore_index);
at::Tensor vulkan_nll_loss_backward(
    const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& target,
    const std::optional<at::Tensor>& weight, int64_t reduction, int64_t ignore_index,
    const at::Tensor& total_weight);
at::Tensor vulkan_cross_entropy_loss(
    const at::Tensor& self, const at::Tensor& target,
    const std::optional<at::Tensor>& weight, int64_t reduction,
    int64_t ignore_index, double label_smoothing);

// Attention
at::Tensor vulkan_scaled_dot_product_attention(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask, double dropout_p,
    bool is_causal, std::optional<double> scale);
// Returns (output, attn_weights_3d) — attn_weights is [B*H, N, S] for backward reuse
std::tuple<at::Tensor, at::Tensor> vulkan_scaled_dot_product_attention_with_attn(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask, double dropout_p,
    bool is_causal, std::optional<double> scale);
at::Tensor vulkan_rope(const at::Tensor& input, double theta = 10000.0);

// Advanced ops
at::Tensor vulkan_cumsum(const at::Tensor& self, int64_t dim,
    std::optional<at::ScalarType> dtype);
std::tuple<at::Tensor, at::Tensor> vulkan_sort(const at::Tensor& self,
    int64_t dim, bool descending);
std::tuple<at::Tensor, at::Tensor> vulkan_topk(const at::Tensor& self,
    int64_t k, int64_t dim, bool largest, bool sorted);
at::Tensor vulkan_gather(const at::Tensor& self, int64_t dim,
    const at::Tensor& index, bool sparse_grad);
at::Tensor& vulkan_scatter_(at::Tensor& self, int64_t dim,
    const at::Tensor& index, const at::Tensor& src);
at::Tensor vulkan_upsample_nearest2d(
    const at::Tensor& self, at::IntArrayRef output_size,
    std::optional<double> scales_h, std::optional<double> scales_w);
at::Tensor vulkan_upsample_nearest2d_backward(
    const at::Tensor& grad_output, at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    std::optional<double> scales_h, std::optional<double> scales_w);
at::Tensor vulkan_upsample_bilinear2d(
    const at::Tensor& self, at::IntArrayRef output_size, bool align_corners,
    std::optional<double> scales_h, std::optional<double> scales_w);
at::Tensor vulkan_upsample_bilinear2d_backward(
    const at::Tensor& grad_output, at::IntArrayRef output_size,
    at::IntArrayRef input_size, bool align_corners,
    std::optional<double> scales_h, std::optional<double> scales_w);
at::Tensor vulkan_grid_sampler_2d(
    const at::Tensor& input, const at::Tensor& grid,
    int64_t interpolation_mode, int64_t padding_mode, bool align_corners);
at::Tensor& vulkan_index_put_(at::Tensor& self, const c10::List<std::optional<at::Tensor>>& indices,
                                const at::Tensor& values, bool accumulate);

// Foreach ops (fused optimizer support)
void vulkan_foreach_add_scalar_(at::TensorList self, const at::Scalar& scalar);
void vulkan_foreach_add_list_(at::TensorList self, at::TensorList other, const at::Scalar& alpha);
void vulkan_foreach_mul_scalar_(at::TensorList self, const at::Scalar& scalar);
void vulkan_foreach_addcmul_(at::TensorList self, at::TensorList tensor1,
                               at::TensorList tensor2, const at::Scalar& value);
void vulkan_foreach_addcdiv_(at::TensorList self, at::TensorList tensor1,
                               at::TensorList tensor2, const at::Scalar& value);
std::vector<at::Tensor> vulkan_foreach_sqrt(at::TensorList self);
std::vector<at::Tensor> vulkan_foreach_neg(at::TensorList self);
void vulkan_foreach_div_scalar_(at::TensorList self, const at::Scalar& scalar);
void vulkan_foreach_lerp_(at::TensorList self, at::TensorList end, const at::Scalar& weight);
std::vector<at::Tensor> vulkan_foreach_maximum(at::TensorList self, at::TensorList other);

// Backward helper ops (for PyTorch's built-in autograd decompositions)
at::Tensor vulkan_threshold_backward(const at::Tensor& grad_output, const at::Tensor& self, const at::Scalar& threshold);
at::Tensor vulkan_sigmoid_backward(const at::Tensor& grad_output, const at::Tensor& output);
at::Tensor vulkan_tanh_backward(const at::Tensor& grad_output, const at::Tensor& output);
at::Tensor vulkan_gelu_backward(const at::Tensor& grad_output, const at::Tensor& self, c10::string_view approximate);
at::Tensor vulkan_silu_backward(const at::Tensor& grad_output, const at::Tensor& self);
at::Tensor vulkan_leaky_relu_backward(const at::Tensor& grad_output, const at::Tensor& self, const at::Scalar& negative_slope, bool self_is_result);
at::Tensor vulkan_elu_backward(const at::Tensor& grad_output, const at::Scalar& alpha, const at::Scalar& scale, const at::Scalar& input_scale, bool is_result, const at::Tensor& self_or_result);
at::Tensor vulkan_softmax_backward_data(const at::Tensor& grad_output, const at::Tensor& output, int64_t dim, at::ScalarType input_dtype);
at::Tensor vulkan_hardtanh_backward(const at::Tensor& grad_output, const at::Tensor& self, const at::Scalar& min_val, const at::Scalar& max_val);
at::Tensor vulkan_hardswish_backward(const at::Tensor& grad_output, const at::Tensor& self);
at::Tensor vulkan_hardsigmoid_backward(const at::Tensor& grad_output, const at::Tensor& self);
at::Tensor vulkan_softplus_backward(const at::Tensor& grad_output, const at::Tensor& self, const at::Scalar& beta, const at::Scalar& threshold);
at::Tensor vulkan_log_softmax_backward_data(const at::Tensor& grad_output, const at::Tensor& output, int64_t dim, at::ScalarType input_dtype);
at::Tensor vulkan_avg_pool2d_backward(const at::Tensor& grad_output, const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, std::optional<int64_t> divisor_override);
std::tuple<at::Tensor, at::Tensor> vulkan_max_pool2d_with_indices(const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode);
at::Tensor vulkan_max_pool2d_with_indices_backward(const at::Tensor& grad_output, const at::Tensor& self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor& indices);
at::Tensor vulkan_embedding_dense_backward(const at::Tensor& grad_output, const at::Tensor& indices, c10::SymInt num_weights, c10::SymInt padding_idx, bool scale_grad_by_freq);
std::tuple<at::Tensor, at::Tensor, at::Tensor> vulkan_native_layer_norm_backward(const at::Tensor& grad_out, const at::Tensor& input, c10::SymIntArrayRef normalized_shape, const at::Tensor& mean, const at::Tensor& rstd, const std::optional<at::Tensor>& weight, const std::optional<at::Tensor>& bias, std::array<bool, 3> output_mask);
std::tuple<at::Tensor, at::Tensor, at::Tensor> vulkan_native_group_norm_backward(const at::Tensor& grad_out, const at::Tensor& input, const at::Tensor& mean, const at::Tensor& rstd, const std::optional<at::Tensor>& weight, c10::SymInt N, c10::SymInt C, c10::SymInt HxW, int64_t group, std::array<bool, 3> output_mask);
std::tuple<at::Tensor, at::Tensor, at::Tensor> vulkan_native_batch_norm_backward(const at::Tensor& grad_out, const at::Tensor& input, const std::optional<at::Tensor>& weight, const std::optional<at::Tensor>& running_mean, const std::optional<at::Tensor>& running_var, const std::optional<at::Tensor>& save_mean, const std::optional<at::Tensor>& save_invstd, bool train, double eps, std::array<bool, 3> output_mask);
std::tuple<at::Tensor, at::Tensor, at::Tensor> vulkan_linear_backward(const at::Tensor& self, const at::Tensor& grad_output, const at::Tensor& weight, std::array<bool, 3> output_mask);

// Phase 3: Model coverage ops
at::Tensor vulkan_triu(const at::Tensor& self, int64_t diagonal);
at::Tensor vulkan_tril(const at::Tensor& self, int64_t diagonal);
at::Tensor vulkan_constant_pad_nd(const at::Tensor& self, c10::IntArrayRef pad, const at::Scalar& value);
at::Tensor vulkan_index_tensor(const at::Tensor& self, const c10::List<std::optional<at::Tensor>>& indices);
at::Tensor vulkan_repeat(const at::Tensor& self, c10::IntArrayRef repeats);
at::Tensor vulkan_repeat_interleave_self_int(const at::Tensor& self, int64_t repeats,
    std::optional<int64_t> dim, std::optional<int64_t> output_size);
at::Tensor vulkan_stack(at::TensorList tensors, int64_t dim);
std::vector<at::Tensor> vulkan_chunk(const at::Tensor& self, int64_t chunks, int64_t dim);
at::Tensor vulkan_erf(const at::Tensor& self);
at::Tensor& vulkan_erf_(at::Tensor& self);
at::Tensor vulkan_narrow(const at::Tensor& self, int64_t dim, int64_t start, int64_t length);
at::Tensor vulkan_flip(const at::Tensor& self, at::IntArrayRef dims);
at::Tensor vulkan_roll(const at::Tensor& self, at::IntArrayRef shifts, at::IntArrayRef dims);
at::Tensor vulkan_unsafe_view(const at::Tensor& self, at::IntArrayRef size);
at::Tensor vulkan_contiguous(const at::Tensor& self, at::MemoryFormat memory_format);
at::Tensor vulkan_to_copy(const at::Tensor& self, std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout, std::optional<at::Device> device,
    std::optional<bool> pin_memory, bool non_blocking,
    std::optional<at::MemoryFormat> memory_format);
at::Tensor vulkan_as_strided(const at::Tensor& self, at::IntArrayRef size,
    at::IntArrayRef stride, std::optional<int64_t> storage_offset);
const at::Tensor& vulkan_resize_(const at::Tensor& self, at::IntArrayRef size,
    std::optional<at::MemoryFormat> memory_format);

// AMP ops
void vulkan_amp_non_finite_check_and_unscale_(at::TensorList scaled_grads,
    at::Tensor& found_inf, const at::Tensor& inv_scale);

at::Tensor& vulkan_amp_update_scale_(at::Tensor& current_scale, at::Tensor& growth_tracker,
    const at::Tensor& found_inf, double scale_growth_factor,
    double scale_backoff_factor, int64_t growth_interval);

// Additional unary ops
at::Tensor vulkan_reciprocal(const at::Tensor& self);
at::Tensor vulkan_sin(const at::Tensor& self);
at::Tensor vulkan_cos(const at::Tensor& self);
at::Tensor vulkan_tan(const at::Tensor& self);
at::Tensor vulkan_atan(const at::Tensor& self);
at::Tensor vulkan_log2(const at::Tensor& self);
at::Tensor vulkan_log10(const at::Tensor& self);
at::Tensor vulkan_log1p(const at::Tensor& self);
at::Tensor vulkan_logical_not(const at::Tensor& self);
at::Tensor vulkan_bitwise_not(const at::Tensor& self);
at::Tensor& vulkan_bitwise_and_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& out);
at::Tensor& vulkan_random_from(at::Tensor& self, int64_t from, std::optional<int64_t> to,
                                std::optional<at::Generator> generator);

// Check ops
at::Tensor vulkan_isnan(const at::Tensor& self);
at::Tensor vulkan_isinf(const at::Tensor& self);
at::Tensor vulkan_isfinite(const at::Tensor& self);

// Additional binary ops
at::Tensor vulkan_fmod(const at::Tensor& self, const at::Tensor& other);
at::Tensor vulkan_remainder(const at::Tensor& self, const at::Tensor& other);
at::Tensor vulkan_atan2(const at::Tensor& self, const at::Tensor& other);

// Additional reduction ops
at::Tensor vulkan_cumprod(const at::Tensor& self, int64_t dim,
    std::optional<at::ScalarType> dtype);

// Loss ops
at::Tensor vulkan_mse_loss(const at::Tensor& self, const at::Tensor& target, int64_t reduction);
at::Tensor vulkan_mse_loss_backward(const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, int64_t reduction);
at::Tensor vulkan_binary_cross_entropy(const at::Tensor& self, const at::Tensor& target,
    const std::optional<at::Tensor>& weight, int64_t reduction);
at::Tensor vulkan_binary_cross_entropy_backward(const at::Tensor& grad_output,
    const at::Tensor& self, const at::Tensor& target,
    const std::optional<at::Tensor>& weight, int64_t reduction);
at::Tensor vulkan_binary_cross_entropy_with_logits(const at::Tensor& self, const at::Tensor& target,
    const std::optional<at::Tensor>& weight, const std::optional<at::Tensor>& pos_weight,
    int64_t reduction);
at::Tensor vulkan_smooth_l1_loss(const at::Tensor& self, const at::Tensor& target,
    int64_t reduction, double beta);
at::Tensor vulkan_smooth_l1_loss_backward(const at::Tensor& grad_output,
    const at::Tensor& self, const at::Tensor& target, int64_t reduction, double beta);
at::Tensor vulkan_huber_loss(const at::Tensor& self, const at::Tensor& target,
    int64_t reduction, double delta);
at::Tensor vulkan_huber_loss_backward(const at::Tensor& grad_output,
    const at::Tensor& self, const at::Tensor& target, int64_t reduction, double delta);
at::Tensor vulkan_l1_loss(const at::Tensor& self, const at::Tensor& target,
    int64_t reduction);
at::Tensor vulkan_l1_loss_backward(const at::Tensor& grad_output,
    const at::Tensor& self, const at::Tensor& target, int64_t reduction);
at::Tensor vulkan_kl_div(const at::Tensor& self, const at::Tensor& target,
    int64_t reduction, bool log_target);
at::Tensor vulkan_kl_div_backward(const at::Tensor& grad_output,
    const at::Tensor& self, const at::Tensor& target,
    int64_t reduction, bool log_target);

// Autograd-aware wrappers (register at AutogradPrivateUse1)
at::Tensor vulkan_view_autograd(const at::Tensor& self, at::IntArrayRef size);
at::Tensor vulkan_reshape_autograd(const at::Tensor& self, at::IntArrayRef shape);
at::Tensor vulkan_permute_autograd(const at::Tensor& self, at::IntArrayRef dims);
at::Tensor vulkan_transpose_autograd(const at::Tensor& self, int64_t dim0, int64_t dim1);
at::Tensor vulkan_t_autograd(const at::Tensor& self);
at::Tensor vulkan_unsqueeze_autograd(const at::Tensor& self, int64_t dim);
at::Tensor vulkan_squeeze_autograd(const at::Tensor& self);
at::Tensor vulkan_squeeze_dim_autograd(const at::Tensor& self, int64_t dim);
at::Tensor vulkan_expand_autograd(const at::Tensor& self, at::IntArrayRef size, bool implicit);
at::Tensor vulkan_select_autograd(const at::Tensor& self, int64_t dim, int64_t index);
at::Tensor vulkan_slice_autograd(const at::Tensor& self, int64_t dim,
                                  std::optional<int64_t> start, std::optional<int64_t> end,
                                  int64_t step);
at::Tensor vulkan_relu_autograd(const at::Tensor& self);
at::Tensor vulkan_mm_autograd(const at::Tensor& self, const at::Tensor& mat2);
at::Tensor vulkan_addmm_autograd(const at::Tensor& bias, const at::Tensor& self,
                                   const at::Tensor& mat2,
                                   const at::Scalar& beta, const at::Scalar& alpha);
at::Tensor vulkan_linear_autograd(const at::Tensor& input, const at::Tensor& weight,
                                    const std::optional<at::Tensor>& bias_opt);
at::Tensor vulkan_bmm_autograd(const at::Tensor& self, const at::Tensor& mat2);
at::Tensor vulkan_sigmoid_autograd(const at::Tensor& self);
at::Tensor vulkan_tanh_autograd(const at::Tensor& self);

// Stage 3 Tier 3 — Additional autograd wrappers
at::Tensor vulkan_gelu_autograd(const at::Tensor& self, c10::string_view approximate);
at::Tensor vulkan_silu_autograd(const at::Tensor& self);
at::Tensor vulkan_leaky_relu_autograd(const at::Tensor& self, const at::Scalar& negative_slope);
at::Tensor vulkan_elu_autograd(const at::Tensor& self, const at::Scalar& alpha,
                                const at::Scalar& scale, const at::Scalar& input_scale);
at::Tensor vulkan_softmax_autograd(const at::Tensor& self, int64_t dim,
                                     std::optional<at::ScalarType> dtype);
at::Tensor vulkan_log_softmax_autograd(const at::Tensor& self, int64_t dim,
                                         std::optional<at::ScalarType> dtype);
at::Tensor vulkan_conv2d_autograd(const at::Tensor& input, const at::Tensor& weight,
                                    const std::optional<at::Tensor>& bias_opt,
                                    at::IntArrayRef stride, at::IntArrayRef padding,
                                    at::IntArrayRef dilation, int64_t groups);
at::Tensor vulkan_max_pool2d_autograd(const at::Tensor& self, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode);
at::Tensor vulkan_avg_pool2d_autograd(const at::Tensor& self, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode,
    bool count_include_pad, std::optional<int64_t> divisor_override);
at::Tensor vulkan_batch_norm_autograd(
    const at::Tensor& input, const std::optional<at::Tensor>& weight_opt,
    const std::optional<at::Tensor>& bias_opt,
    const std::optional<at::Tensor>& running_mean_opt,
    const std::optional<at::Tensor>& running_var_opt,
    bool training, double momentum, double eps, bool cudnn_enabled);
at::Tensor vulkan_embedding_autograd(const at::Tensor& weight, const at::Tensor& indices,
    int64_t padding_idx, bool scale_grad_by_freq, bool sparse);
std::tuple<at::Tensor, at::Tensor, at::Tensor> vulkan_layer_norm_autograd(
    const at::Tensor& input, at::IntArrayRef normalized_shape,
    const std::optional<at::Tensor>& weight_opt, const std::optional<at::Tensor>& bias_opt,
    double eps);
std::tuple<at::Tensor, at::Tensor, at::Tensor> vulkan_group_norm_autograd(
    const at::Tensor& input, int64_t num_groups,
    const std::optional<at::Tensor>& weight_opt, const std::optional<at::Tensor>& bias_opt,
    double eps);
at::Tensor vulkan_sdpa_autograd(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask, double dropout_p,
    bool is_causal, std::optional<double> scale);
at::Tensor vulkan_prelu_autograd(const at::Tensor& self, const at::Tensor& weight);
at::Tensor vulkan_selu_autograd(const at::Tensor& self);
at::Tensor vulkan_clamp_autograd(const at::Tensor& self,
                                   const std::optional<at::Scalar>& min_val,
                                   const std::optional<at::Scalar>& max_val);
at::Tensor vulkan_rope_autograd(const at::Tensor& input, double theta = 10000.0);

// RMSNorm (used by Qwen3, Llama, etc.)
std::tuple<at::Tensor, at::Tensor> vulkan_rms_norm(
    const at::Tensor& input, const at::Tensor& weight, double eps);
std::tuple<at::Tensor, at::Tensor> vulkan_rms_norm_backward(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& weight, const at::Tensor& rstd);
at::Tensor vulkan_rms_norm_autograd(
    const at::Tensor& input, const at::Tensor& weight, double eps);
// Fused Add + RMSNorm: h_new = residual + shortcut; out = weight * (h_new / rms(h_new))
// Returns (normed_output, h_new, rstd). Saves 1 dispatch vs separate add + rms_norm.
std::tuple<at::Tensor, at::Tensor, at::Tensor> vulkan_add_rms_norm(
    const at::Tensor& residual, const at::Tensor& shortcut,
    const at::Tensor& weight, double eps);
// Fused backward: returns (grad_h, grad_weight) where grad_h = grad_residual = grad_shortcut
std::tuple<at::Tensor, at::Tensor> vulkan_add_rms_norm_backward(
    const at::Tensor& grad_normed, const at::Tensor& grad_h_new,
    const at::Tensor& h_new, const at::Tensor& weight, const at::Tensor& rstd);
// Returns (normed_output, h_new) — both differentiable via shared autograd Function.
std::tuple<at::Tensor, at::Tensor> vulkan_add_rms_norm_apply(
    const at::Tensor& residual, const at::Tensor& shortcut,
    const at::Tensor& weight, double eps);

// RMSNormGated: out = weight * rms_norm(input) * silu(gate) — Qwen3.5 GatedDeltaNet
std::tuple<at::Tensor, at::Tensor> vulkan_rms_norm_gated(
    const at::Tensor& input, const at::Tensor& gate,
    const at::Tensor& weight, double eps);
std::tuple<at::Tensor, at::Tensor> vulkan_rms_norm_gated_backward(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& gate, const at::Tensor& weight, const at::Tensor& rstd);
at::Tensor vulkan_rms_norm_gated_autograd(
    const at::Tensor& input, const at::Tensor& gate,
    const at::Tensor& weight, double eps);

// Cached causal mask: [BH, N, S] additive mask (0=attend, -1e9=block).
at::Tensor get_causal_mask(int64_t BH, int64_t N, int64_t S,
                            const at::TensorOptions& opts);

// Flash Attention: fused QK^T + softmax + @V in one dispatch per (b,h,q).
// Eliminates intermediate [B*H, N, S] attention weight matrix.
// Returns (output [B,H,N,D], lse [B,H,N]) — lse = log-sum-exp for backward.
std::tuple<at::Tensor, at::Tensor> vulkan_flash_attention_forward(
    const at::Tensor& Q, const at::Tensor& K, const at::Tensor& V,
    float scale, bool is_causal, bool q_seq_major = false);
// Backward: returns (grad_Q, grad_K, grad_V)
std::tuple<at::Tensor, at::Tensor, at::Tensor> vulkan_flash_attention_backward(
    const at::Tensor& grad_out, const at::Tensor& Q,
    const at::Tensor& K, const at::Tensor& V,
    const at::Tensor& out, const at::Tensor& lse,
    float scale, bool is_causal);
// Autograd wrapper (returns output only; lse saved internally)
at::Tensor vulkan_flash_attention_autograd(
    const at::Tensor& Q, const at::Tensor& K, const at::Tensor& V,
    double scale, bool is_causal, bool q_seq_major = false);

}} // namespace torch_vulkan::ops
