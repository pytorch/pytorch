#include <ATen/autocast_mode.h>

#include <exception>
#include <mutex>
#include <ATen/CachedTensorUtils.h>
#include <c10/util/flat_hash_map.h>

namespace at {
namespace autocast {

bool is_enabled() {
  return !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::AutocastCUDA);
}

void set_enabled(bool new_enabled) {
  c10::impl::tls_set_dispatch_key_excluded(DispatchKey::AutocastCUDA, !new_enabled);
}

bool is_cpu_enabled() {
  return !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::AutocastCPU);
}

void set_cpu_enabled(bool new_enabled) {
  c10::impl::tls_set_dispatch_key_excluded(DispatchKey::AutocastCPU, !new_enabled);
}

bool is_xpu_enabled() {
  return !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::AutocastXPU);
}

void set_xpu_enabled(bool new_enabled) {
  c10::impl::tls_set_dispatch_key_excluded(DispatchKey::AutocastXPU, !new_enabled);
}

bool is_ipu_enabled() {
  return !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::AutocastIPU);
}

void set_ipu_enabled(bool new_enabled) {
  c10::impl::tls_set_dispatch_key_excluded(DispatchKey::AutocastIPU, !new_enabled);
}

bool is_hpu_enabled() {
  return !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::AutocastHPU);
}

void set_hpu_enabled(bool new_enabled) {
  c10::impl::tls_set_dispatch_key_excluded(DispatchKey::AutocastHPU, !new_enabled);
}

bool is_xla_enabled() {
  return !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::AutocastXLA);
}

void set_xla_enabled(bool new_enabled) {
  c10::impl::tls_set_dispatch_key_excluded(DispatchKey::AutocastXLA, !new_enabled);
}

bool is_privateuseone_enabled() {
  return !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::AutocastPrivateUse1);
}

void set_privateuseone_enabled(bool new_enabled) {
  c10::impl::tls_set_dispatch_key_excluded(DispatchKey::AutocastPrivateUse1, !new_enabled);
}

namespace {
// Imitate Apex and cache some of the casts to streamline parameter reuse.
// Our heuristic is to cache lower_precision_fp casts of fp32 model weights (see cached_cast below).
//
// After discussion with @ezyang, the cache uses the following structure:
// The key is the fp32 source tensor's TensorImpl*, a proxy for a Tensor uuid that's
// unchanged across shallow copies.
// The value is a tuple with a weakref to the source tensor's TensorImpl as the first
// element and the casted tensor as the second element.
//
// The weakref keeps the source's TensorImpl from being deleted.  We need to because we're
// using the source TensorImpl* as the key.  If it were deleted, another random Tensor could
// be allocated whose TensorImpl* happened to have the same value.  This TensorImpl* would
// then mistakenly hit in cache:  a rare, intermittent, unpredictable bug.
//
// I'm not using the weak_intrusive_ptr as the key because it's more difficult to compare
// directly against incoming TensorImpl*s.
using weakref_type = c10::weak_intrusive_ptr<TensorImpl, UndefinedTensorImpl>;
using val_type = std::tuple<weakref_type, Tensor>;
ska::flat_hash_map<TensorImpl*, val_type> cached_casts;
std::mutex cached_casts_mutex;


// nesting tracks the nesting depth of the Python-side context manager.
// When the autocast context manager exits to a nesting level that's outside
// any instance of autocast (which should occur at the end of each forward pass)
// it calls clear_cache() to ensure cached Tensors don't leak outside the autocasting region.
thread_local int nesting = 0;

// autocast_cpu_dtype is the lower_precision_fp used by AutocastCPU.
thread_local at::ScalarType autocast_cpu_dtype = at::kBFloat16;

// autocast_xpu_dtype is the lower_precision_fp used by AutocastXPU.
thread_local at::ScalarType autocast_xpu_dtype = at::kBFloat16;

// autocast_ipu_dtype is the lower_precision_fp used by AutocastIPU.
thread_local at::ScalarType autocast_ipu_dtype = at::kHalf;

// autocast_hpu_dtype is the lower_precision_fp used by AutocastHPU.
thread_local at::ScalarType autocast_hpu_dtype = at::kBFloat16;

// autocast_xla_dtype is the lower_precision_fp used by AutocastXLA.
thread_local at::ScalarType autocast_xla_dtype = at::kBFloat16;

// should we enabled the cache inside autocast.
thread_local bool cache_enabled = true;

// autocast_gpu_dtype is the lower_precision_fp used by AutocastGPU.
thread_local at::ScalarType autocast_gpu_dtype = at::kHalf;

// autocast_privateuseone_dtype is the lower_precision_fp used by AutocastPrivateUse1.
thread_local at::ScalarType autocast_privateuseone_dtype = at::kHalf;
}

void clear_cache() {
  const std::lock_guard<std::mutex> lock(cached_casts_mutex);
  cached_casts.clear();
}

int increment_nesting() {
  return ++nesting;
}

int decrement_nesting() {
  return --nesting;
}

at::ScalarType get_autocast_gpu_dtype() {
  return autocast_gpu_dtype;
}

at::ScalarType get_autocast_cpu_dtype() {
  return autocast_cpu_dtype;
}

at::ScalarType get_autocast_xpu_dtype() {
  return autocast_xpu_dtype;
}

at::ScalarType get_autocast_ipu_dtype() {
  return autocast_ipu_dtype;
}

at::ScalarType get_autocast_hpu_dtype() {
  return autocast_hpu_dtype;
}

at::ScalarType get_autocast_xla_dtype() {
  return autocast_xla_dtype;
}

at::ScalarType get_autocast_privateuseone_dtype() {
  return autocast_privateuseone_dtype;
}

void set_autocast_cpu_dtype(at::ScalarType dtype) {
  TORCH_CHECK(
      dtype == at::kBFloat16,
      "Currently, AutocastCPU only support Bfloat16 as the autocast_cpu_dtype");
  autocast_cpu_dtype = dtype;
}

void set_autocast_gpu_dtype(at::ScalarType dtype) {
  autocast_gpu_dtype = dtype;
}

void set_autocast_xpu_dtype(at::ScalarType dtype) {
  autocast_xpu_dtype = dtype;
}

void set_autocast_ipu_dtype(at::ScalarType dtype) {
  autocast_ipu_dtype = dtype;
}

void set_autocast_hpu_dtype(at::ScalarType dtype) {
  autocast_hpu_dtype = dtype;
}

void set_autocast_xla_dtype(at::ScalarType dtype) {
  autocast_xla_dtype = dtype;
}

void set_autocast_privateuseone_dtype(at::ScalarType dtype) {
  autocast_privateuseone_dtype = dtype;
}

bool is_autocast_cache_enabled() {
  return cache_enabled;
}

void set_autocast_cache_enabled(bool enabled) {
  cache_enabled = enabled;
}

// Overload to catch Tensor args
// TODO (possible optimization):
// Move cast_cache to an inline function in a header with cached_casts declared as
// extern thread_local in the header.
Tensor cached_cast(at::ScalarType to_type, const Tensor& arg, DeviceType device_type) {
  if (is_eligible(arg, device_type) && (arg.scalar_type() != to_type)) {
    // Heuristic:  Do what Apex does, and cache lower_precision_fp casts of fp32 model weights (leaves).
    // See cached_casts declaration above for detailed strategy.
    bool can_try_cache = (to_type == get_lower_precision_fp_from_device_type(device_type) &&
                         arg.scalar_type() == at::kFloat && arg.requires_grad() &&
                         arg.is_leaf() && !arg.is_view() && cache_enabled &&
                         !at::caching::is_cached_tensor(arg));

    if (can_try_cache) {
      const std::lock_guard<std::mutex> lock(cached_casts_mutex);
      auto it = cached_casts.find(arg.unsafeGetTensorImpl());
      if (it != cached_casts.end()) {
        return std::get<1>(it->second);
      } else {
        auto casted_arg = arg.to(to_type);
        cached_casts.emplace(arg.unsafeGetTensorImpl(), val_type{weakref_type(arg.getIntrusivePtr()), casted_arg});
        return casted_arg;
      }
    } else {
      return arg.to(to_type);
    }
  } else {
    return arg;
  }
}

/*******************************
Banned functions
*******************************/

static Tensor binary_cross_entropy_banned(const Tensor &, const Tensor &, const c10::optional<Tensor>&, int64_t) {
  AT_ERROR("torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.\n"
           "Many models use a sigmoid layer right before the binary cross entropy layer.\n"
           "In this case, combine the two layers using torch.nn.functional.binary_cross_entropy_with_logits\n"
           "or torch.nn.BCEWithLogitsLoss.  binary_cross_entropy_with_logits and BCEWithLogits are\n"
           "safe to autocast.");
}

namespace {

/*****************************************
Explicit registration for out-of-place ops
*****************************************/
TORCH_LIBRARY_IMPL(_, Autocast, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, Autocast, m) {
  // lower_precision_fp
  KERNEL_CUDA2(_convolution, deprecated, lower_precision_fp)
  KERNEL_CUDA(_convolution, lower_precision_fp)
  KERNEL_CUDA(conv1d, lower_precision_fp)
  KERNEL_CUDA(conv2d, lower_precision_fp)
  KERNEL_CUDA(conv3d, lower_precision_fp)
  KERNEL_CUDA(conv_tbc, lower_precision_fp)
  KERNEL_CUDA(conv_transpose1d, lower_precision_fp)
  KERNEL_CUDA2(conv_transpose2d, input, lower_precision_fp)
  KERNEL_CUDA2(conv_transpose3d, input, lower_precision_fp)
  KERNEL_CUDA(convolution, lower_precision_fp)
  KERNEL_CUDA(cudnn_convolution, lower_precision_fp)
  KERNEL_CUDA(cudnn_convolution_transpose, lower_precision_fp)
  KERNEL_CUDA(prelu, lower_precision_fp)
  KERNEL_CUDA(addmm, lower_precision_fp)
  KERNEL_CUDA(addmv, lower_precision_fp)
  KERNEL_CUDA(addr, lower_precision_fp)
  KERNEL_CUDA(matmul, lower_precision_fp)
  KERNEL_CUDA(einsum, lower_precision_fp)
  KERNEL_CUDA(mm, lower_precision_fp)
  KERNEL_CUDA(mv, lower_precision_fp)
  KERNEL_CUDA(linear, lower_precision_fp)
  KERNEL_CUDA(addbmm, lower_precision_fp)
  KERNEL_CUDA(baddbmm, lower_precision_fp)
  KERNEL_CUDA(bmm, lower_precision_fp)
  KERNEL_CUDA(chain_matmul, lower_precision_fp)
  KERNEL_CUDA(linalg_multi_dot, lower_precision_fp)
  KERNEL_CUDA(_thnn_fused_lstm_cell, lower_precision_fp)
  KERNEL_CUDA(_thnn_fused_gru_cell, lower_precision_fp)
  KERNEL_CUDA(lstm_cell, lower_precision_fp)
  KERNEL_CUDA(gru_cell, lower_precision_fp)
  KERNEL_CUDA(rnn_tanh_cell, lower_precision_fp)
  KERNEL_CUDA(rnn_relu_cell, lower_precision_fp)
  KERNEL_CUDA(_scaled_dot_product_flash_attention, lower_precision_fp)
  KERNEL_CUDA(scaled_dot_product_attention, lower_precision_fp)

  // fp32
  KERNEL_CUDA(acos, fp32)
  KERNEL_CUDA(asin, fp32)
  KERNEL_CUDA(cosh, fp32)
  KERNEL_CUDA(erfinv, fp32)
  KERNEL_CUDA(exp, fp32)
  KERNEL_CUDA(expm1, fp32)
  KERNEL_CUDA(log, fp32)
  KERNEL_CUDA(log10, fp32)
  KERNEL_CUDA(log2, fp32)
  KERNEL_CUDA(log1p, fp32)
  KERNEL_CUDA(reciprocal, fp32)
  KERNEL_CUDA(rsqrt, fp32)
  KERNEL_CUDA(sinh, fp32)
  KERNEL_CUDA(tan, fp32)
  KERNEL_CUDA2(pow, Tensor_Scalar, fp32)
  KERNEL_CUDA2(pow, Tensor_Tensor, fp32)
  KERNEL_CUDA2(pow, Scalar, fp32)
  KERNEL_CUDA(softplus, fp32)
  KERNEL_CUDA(layer_norm, fp32)
  KERNEL_CUDA(native_layer_norm, fp32)
  KERNEL_CUDA(group_norm, fp32)
  KERNEL_CUDA2(frobenius_norm, dim, fp32)
  KERNEL_CUDA(nuclear_norm, fp32)
  KERNEL_CUDA2(nuclear_norm, dim, fp32)
  KERNEL_CUDA(cosine_similarity, fp32)
  KERNEL_CUDA(poisson_nll_loss, fp32)
  KERNEL_CUDA(cosine_embedding_loss, fp32)
  KERNEL_CUDA(nll_loss, fp32)
  KERNEL_CUDA(nll_loss2d, fp32)
  KERNEL_CUDA(hinge_embedding_loss, fp32)
  KERNEL_CUDA(kl_div, fp32)
  KERNEL_CUDA(l1_loss, fp32)
  KERNEL_CUDA(smooth_l1_loss, fp32)
  KERNEL_CUDA(huber_loss, fp32)
  KERNEL_CUDA(mse_loss, fp32)
  KERNEL_CUDA(margin_ranking_loss, fp32)
  KERNEL_CUDA(multilabel_margin_loss, fp32)
  KERNEL_CUDA(soft_margin_loss, fp32)
  KERNEL_CUDA(triplet_margin_loss, fp32)
  KERNEL_CUDA(multi_margin_loss, fp32)
  KERNEL_CUDA(binary_cross_entropy_with_logits, fp32)
  KERNEL_CUDA(dist, fp32)
  KERNEL_CUDA(pdist, fp32)
  KERNEL_CUDA(cdist, fp32)
  KERNEL_CUDA(renorm, fp32)
  KERNEL_CUDA(logsumexp, fp32)
  // fp32_set_opt_dtype
  KERNEL_CUDA(prod, fp32_set_opt_dtype)
  KERNEL_CUDA2(prod, dim_int, fp32_set_opt_dtype)
  KERNEL_CUDA2(prod, dim_Dimname, fp32_set_opt_dtype)
  KERNEL_CUDA2(softmax, int, fp32_set_opt_dtype)
  KERNEL_CUDA2(softmax, Dimname, fp32_set_opt_dtype)
  KERNEL_CUDA2(log_softmax, int, fp32_set_opt_dtype)
  KERNEL_CUDA2(log_softmax, Dimname, fp32_set_opt_dtype)
  KERNEL_CUDA(cumprod, fp32_set_opt_dtype)
  KERNEL_CUDA2(cumprod, dimname, fp32_set_opt_dtype)
  KERNEL_CUDA(cumsum, fp32_set_opt_dtype)
  KERNEL_CUDA2(cumsum, dimname, fp32_set_opt_dtype)
  KERNEL_CUDA(linalg_vector_norm, fp32_set_opt_dtype)
  KERNEL_CUDA(linalg_matrix_norm, fp32_set_opt_dtype)
  KERNEL_CUDA2(linalg_matrix_norm, str_ord, fp32_set_opt_dtype)
  // commenting these out because they accept an explicit (not-optional) dtype, and we shouldn't try to flip that even
  // when autocasting.
  // KERNEL_CUDA2(norm, ScalarOpt_dtype, fp32_set_opt_dtype)
  // KERNEL_CUDA2(norm, ScalarOpt_dim_dtype, fp32_set_opt_dtype)
  // KERNEL_CUDA2(norm, names_ScalarOpt_dim_dtype, fp32_set_opt_dtype)
  KERNEL_CUDA(sum, fp32_set_opt_dtype)
  KERNEL_CUDA2(sum, dim_IntList, fp32_set_opt_dtype)
  KERNEL_CUDA2(sum, dim_DimnameList, fp32_set_opt_dtype)
  // fp32_append_dtype
  // The fp32_append_dtype wrapper overrides implicit promotion behavior.
  // norm does not implicitly promote, but be aware when adding new ops to this policy.
  KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_CUDA(ADD_NS(norm), "norm.Scalar", Tensor (const Tensor &, const Scalar&), Tensor (const Tensor &, const c10::optional<Scalar>&, ScalarType), fp32_append_dtype)
  KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_CUDA(ADD_NS(norm), "norm.ScalarOpt_dim", Tensor (const Tensor &, const c10::optional<Scalar>&, IntArrayRef, bool), Tensor (const Tensor &, const c10::optional<Scalar>&, IntArrayRef, bool, ScalarType), fp32_append_dtype)
  KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_CUDA(ADD_NS(norm), "norm.names_ScalarOpt_dim", Tensor (const Tensor &, const c10::optional<Scalar>&, DimnameList, bool), Tensor (const Tensor &, const c10::optional<Scalar>&, DimnameList, bool, ScalarType), fp32_append_dtype)
  // promote
  KERNEL_CUDA(addcdiv, promote)
  KERNEL_CUDA(addcmul, promote)
  KERNEL_CUDA(atan2, promote)
  KERNEL_CUDA(bilinear, promote)
  KERNEL_CUDA(cross, promote)
  KERNEL_CUDA(dot, promote)
  KERNEL_CUDA(grid_sampler, promote)
  KERNEL_CUDA(index_put, promote)
  KERNEL_CUDA(tensordot, promote)
  KERNEL_CUDA(scatter_add, promote)

  m.impl(TORCH_SELECTIVE_NAME("aten::binary_cross_entropy"),
         TORCH_FN((&at::autocast::binary_cross_entropy_banned)));
}

TORCH_LIBRARY_IMPL(_, AutocastCPU, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}


TORCH_LIBRARY_IMPL(aten, AutocastCPU, m) {
  // lower_precision_fp cast policy
  KERNEL_CPU(conv1d, lower_precision_fp)
  KERNEL_CPU2(conv1d, padding, lower_precision_fp)
  KERNEL_CPU(conv2d, lower_precision_fp)
  KERNEL_CPU2(conv2d, padding, lower_precision_fp)
  KERNEL_CPU(conv3d, lower_precision_fp)
  KERNEL_CPU2(conv3d, padding, lower_precision_fp)
  KERNEL_CPU(bmm, lower_precision_fp)
  KERNEL_CPU(mm, lower_precision_fp)
  KERNEL_CPU(baddbmm, lower_precision_fp)
  KERNEL_CPU(addmm, lower_precision_fp)
  KERNEL_CPU(addbmm, lower_precision_fp)
  KERNEL_CPU(linear, lower_precision_fp)
  KERNEL_CPU2(_convolution, deprecated, lower_precision_fp)
  KERNEL_CPU(matmul, lower_precision_fp)
  KERNEL_CPU(conv_tbc, lower_precision_fp)
  KERNEL_CPU(mkldnn_rnn_layer, lower_precision_fp)
  KERNEL_CPU(conv_transpose1d, lower_precision_fp)
  KERNEL_CPU2(conv_transpose2d, input, lower_precision_fp)
  KERNEL_CPU2(conv_transpose3d, input, lower_precision_fp)
  KERNEL_CPU(prelu, lower_precision_fp)
  KERNEL_CPU(scaled_dot_product_attention, lower_precision_fp)

  // fp32 cast policy
  KERNEL_CPU(avg_pool3d, fp32)
  KERNEL_CPU(binary_cross_entropy, fp32)
  KERNEL_CPU(grid_sampler, fp32)
  KERNEL_CPU(polar, fp32)
  KERNEL_CPU(prod, fp32)
  KERNEL_CPU2(prod, dim_int, fp32)
  KERNEL_CPU2(prod, dim_Dimname, fp32)
  KERNEL_CPU(quantile, fp32)
  KERNEL_CPU2(quantile, scalar, fp32)
  KERNEL_CPU(nanquantile, fp32)
  KERNEL_CPU2(nanquantile, scalar, fp32)
  KERNEL_CPU(stft, fp32)
  KERNEL_CPU2(stft, center, fp32)
  KERNEL_CPU(cdist, fp32)
  KERNEL_CPU(grid_sampler_2d, fp32)
  KERNEL_CPU(_grid_sampler_2d_cpu_fallback, fp32)
  KERNEL_CPU(grid_sampler_3d, fp32)
  KERNEL_CPU(trace, fp32)
  KERNEL_CPU(view_as_complex, fp32)
  KERNEL_CPU(cholesky, fp32)
  KERNEL_CPU(cholesky_inverse, fp32)
  KERNEL_CPU(cholesky_solve, fp32)
  KERNEL_CPU(inverse, fp32)
  KERNEL_CPU(lu_solve, fp32)
  KERNEL_CPU(orgqr, fp32)
  KERNEL_CPU(ormqr, fp32)
  KERNEL_CPU(pinverse, fp32)
  KERNEL_CPU(max_pool3d, fp32)
  KERNEL_CPU(max_unpool2d, fp32)
  KERNEL_CPU(max_unpool3d, fp32)
  KERNEL_CPU(adaptive_avg_pool3d, fp32)
  KERNEL_CPU(reflection_pad1d, fp32)
  KERNEL_CPU(reflection_pad2d, fp32)
  KERNEL_CPU(replication_pad1d, fp32)
  KERNEL_CPU(replication_pad2d, fp32)
  KERNEL_CPU(replication_pad3d, fp32)
  KERNEL_CPU(mse_loss, fp32)
  KERNEL_CPU(cosine_embedding_loss, fp32)
  KERNEL_CPU(nll_loss, fp32)
  KERNEL_CPU(nll_loss2d, fp32)
  KERNEL_CPU(hinge_embedding_loss, fp32)
  KERNEL_CPU(poisson_nll_loss, fp32)
  KERNEL_CPU(smooth_l1_loss, fp32)
  KERNEL_CPU(cross_entropy_loss, fp32)
  KERNEL_CPU(l1_loss, fp32)
  KERNEL_CPU(huber_loss, fp32)
  KERNEL_CPU(margin_ranking_loss, fp32)
  KERNEL_CPU(soft_margin_loss, fp32)
  KERNEL_CPU(triplet_margin_loss, fp32)
  KERNEL_CPU(multi_margin_loss, fp32)
  KERNEL_CPU2(ctc_loss, IntList, fp32)
  KERNEL_CPU2(ctc_loss, Tensor, fp32)
  KERNEL_CPU(kl_div, fp32)
  KERNEL_CPU(multilabel_margin_loss, fp32)
  KERNEL_CPU(binary_cross_entropy_with_logits, fp32)
  KERNEL_CPU(fft_fft, fp32)
  KERNEL_CPU(fft_ifft, fp32)
  KERNEL_CPU(fft_fft2, fp32)
  KERNEL_CPU(fft_ifft2, fp32)
  KERNEL_CPU(fft_fftn, fp32)
  KERNEL_CPU(fft_ifftn, fp32)
  KERNEL_CPU(fft_rfft, fp32)
  KERNEL_CPU(fft_irfft, fp32)
  KERNEL_CPU(fft_rfft2, fp32)
  KERNEL_CPU(fft_irfft2, fp32)
  KERNEL_CPU(fft_rfftn, fp32)
  KERNEL_CPU(fft_irfftn, fp32)
  KERNEL_CPU(fft_hfft, fp32)
  KERNEL_CPU(fft_ihfft, fp32)
  KERNEL_CPU(linalg_cond, fp32)
  KERNEL_CPU2(linalg_cond, p_str, fp32)
  KERNEL_CPU(linalg_matrix_rank, fp32)
  KERNEL_CPU2(linalg_matrix_rank, tol_tensor, fp32)
  KERNEL_CPU2(linalg_matrix_rank, atol_rtol_tensor, fp32)
  KERNEL_CPU2(linalg_matrix_rank, atol_rtol_float, fp32)
  KERNEL_CPU(linalg_solve, fp32)
  KERNEL_CPU(linalg_cholesky, fp32)
  KERNEL_CPU(linalg_svdvals, fp32)
  KERNEL_CPU(linalg_eigvals, fp32)
  KERNEL_CPU(linalg_eigvalsh, fp32)
  KERNEL_CPU(linalg_inv, fp32)
  KERNEL_CPU(linalg_householder_product, fp32)
  KERNEL_CPU(linalg_tensorinv, fp32)
  KERNEL_CPU(linalg_tensorsolve, fp32)
  KERNEL_CPU(fake_quantize_per_tensor_affine, fp32)
  KERNEL_CPU(geqrf, fp32)
  KERNEL_CPU(_lu_with_info, fp32)
  KERNEL_CPU(qr, fp32)
  KERNEL_CPU(svd, fp32)
  KERNEL_CPU(triangular_solve, fp32)
  KERNEL_CPU(fractional_max_pool2d, fp32)
  KERNEL_CPU(fractional_max_pool3d, fp32)
  KERNEL_CPU(adaptive_max_pool3d, fp32)
  KERNEL_CPU(multilabel_margin_loss_forward, fp32)
  KERNEL_CPU(linalg_qr, fp32)
  KERNEL_CPU(linalg_cholesky_ex, fp32)
  KERNEL_CPU(linalg_svd, fp32)
  KERNEL_CPU(linalg_eig, fp32)
  KERNEL_CPU(linalg_eigh, fp32)
  KERNEL_CPU(linalg_lstsq, fp32)
  KERNEL_CPU(linalg_inv_ex, fp32)

  // promote
  KERNEL_CPU(stack, promote)
  KERNEL_CPU(cat, promote)
  KERNEL_CPU(index_copy, promote)
  KERNEL_CPU2(index_copy, dimname, promote)

}

} // namespace
} // namespace autocast
} // namespace at
