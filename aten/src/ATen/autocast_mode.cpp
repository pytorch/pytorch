#include <ATen/autocast_mode.h>

#include <mutex>
#include <ATen/CachedTensorUtils.h>
#include <c10/util/flat_hash_map.h>

namespace at::autocast {

bool is_autocast_enabled(at::DeviceType device_type) {
  at::DispatchKey dispatch_key = get_autocast_dispatch_key_from_device_type(device_type);
  return !c10::impl::tls_is_dispatch_key_excluded(dispatch_key);
}

void set_autocast_enabled(at::DeviceType device_type, bool enabled) {
  at::DispatchKey dispatch_key = get_autocast_dispatch_key_from_device_type(device_type);
  c10::impl::tls_set_dispatch_key_excluded(dispatch_key, !enabled);
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

static ska::flat_hash_map<TensorImpl*, val_type>& get_cached_casts() {
  static ska::flat_hash_map<TensorImpl*, val_type> cached_casts;
  return cached_casts;
}
std::mutex cached_casts_mutex;


// nesting tracks the nesting depth of the Python-side context manager.
// When the autocast context manager exits to a nesting level that's outside
// any instance of autocast (which should occur at the end of each forward pass)
// it calls clear_cache() to ensure cached Tensors don't leak outside the autocasting region.
thread_local int nesting = 0;

// The order of this array MUST exactly match the definition order of DeviceType
// in c10/core/DeviceType.h.
static_assert(
    at::COMPILE_TIME_MAX_DEVICE_TYPES == 21,
    "The definition of the default autocast data type per device backend doesn't match with the definition of the device type.");
thread_local std::array<at::ScalarType, at::COMPILE_TIME_MAX_DEVICE_TYPES>
    autocast_dtype = {
        at::kBFloat16, // CPU
        at::kHalf, // CUDA.
        at::ScalarType::Undefined, // Reserved for explicit MKLDNN
        at::ScalarType::Undefined, // OpenGL
        at::ScalarType::Undefined, // OpenCL
        at::ScalarType::Undefined, // IDEEP.
        at::kHalf, // AMD HIP
        at::ScalarType::Undefined, // FPGA
        at::ScalarType::Undefined, // ONNX Runtime / Microsoft
        at::kBFloat16, // XLA / TPU
        at::ScalarType::Undefined, // Vulkan
        at::ScalarType::Undefined, // Metal
        at::kHalf, // XPU
        at::kHalf, // MPS
        at::ScalarType::Undefined, // Meta (tensors with no data)
        at::kBFloat16, // HPU / HABANA
        at::ScalarType::Undefined, // SX-Aurora / NEC
        at::ScalarType::Undefined, // Lazy Tensors
        at::kHalf, // Graphcore IPU
        at::ScalarType::Undefined, // Meta training and inference devices
        at::kHalf, // PrivateUse1 device
};

// should we enabled the cache inside autocast.
thread_local bool cache_enabled = true;

} // anonymous namespace

void clear_cache() {
  const std::lock_guard<std::mutex> lock(cached_casts_mutex);
  get_cached_casts().clear();
}

int increment_nesting() {
  return ++nesting;
}

int decrement_nesting() {
  return --nesting;
}

at::ScalarType get_autocast_dtype(at::DeviceType device_type) {
  return autocast_dtype[static_cast<int>(device_type)];
}

void set_autocast_dtype(at::DeviceType device_type, at::ScalarType dtype) {
  autocast_dtype[static_cast<int>(device_type)] = dtype;
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
      auto it = get_cached_casts().find(arg.unsafeGetTensorImpl());
      if (it != get_cached_casts().end()) {
        return std::get<1>(it->second);
      } else {
        auto casted_arg = arg.to(to_type);
        get_cached_casts().emplace(arg.unsafeGetTensorImpl(), val_type{weakref_type(arg.getIntrusivePtr()), casted_arg});
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

static Tensor binary_cross_entropy_banned(const Tensor &, const Tensor &, const std::optional<Tensor>&, int64_t) {
  TORCH_CHECK(false, "torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.\n"
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
#define _KERNEL_CUDA_LOW_PRECISION_FP(...) \
  KERNEL_CUDA(__VA_ARGS__, lower_precision_fp)

  AT_FORALL_LOWER_PRECISION_FP(_KERNEL_CUDA_LOW_PRECISION_FP)
  KERNEL_CUDA(cudnn_convolution, lower_precision_fp)
  KERNEL_CUDA(cudnn_convolution_transpose, lower_precision_fp)

  // fp32
#define _KERNEL_CUDA_FP32(...) KERNEL_CUDA(__VA_ARGS__, fp32)

  AT_FORALL_FP32(_KERNEL_CUDA_FP32)

  // fp32_set_opt_dtype
#define _KERNEL_CUDA_FP32_SET_OPT_DTYPE(...) \
  KERNEL_CUDA(__VA_ARGS__, fp32_set_opt_dtype)

  AT_FORALL_FP32_SET_OPT_DTYPE(_KERNEL_CUDA_FP32_SET_OPT_DTYPE)
  // commenting these out because they accept an explicit (not-optional) dtype, and we shouldn't try to flip that even
  // when autocasting.
  // KERNEL_CUDA(norm, ScalarOpt_dtype, fp32_set_opt_dtype)
  // KERNEL_CUDA(norm, ScalarOpt_dim_dtype, fp32_set_opt_dtype)
  // KERNEL_CUDA(norm, names_ScalarOpt_dim_dtype, fp32_set_opt_dtype)

  // fp32_append_dtype
  // The fp32_append_dtype wrapper overrides implicit promotion behavior.
  // norm does not implicitly promote, but be aware when adding new ops to this policy.
  AT_FORALL_DIFFERENT_REDISPATCH_SIGNATURE(
      KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_CUDA)

  // promote
#define _KERNEL_CUDA_PROMOTE(...) KERNEL_CUDA(__VA_ARGS__, promote)

  AT_FORALL_PROMOTE(_KERNEL_CUDA_PROMOTE)

  m.impl(TORCH_SELECTIVE_NAME("aten::binary_cross_entropy"),
         TORCH_FN((&at::autocast::binary_cross_entropy_banned)));
}

TORCH_LIBRARY_IMPL(_, AutocastMPS, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, AutocastMPS, m) {
  // lower_precision_fp
  KERNEL_MPS(_convolution, deprecated, lower_precision_fp)
  KERNEL_MPS(_convolution, lower_precision_fp)
  KERNEL_MPS(conv1d, lower_precision_fp)
  KERNEL_MPS(conv2d, lower_precision_fp)
  KERNEL_MPS(conv_tbc, lower_precision_fp)
  KERNEL_MPS(conv_transpose1d, lower_precision_fp)
  KERNEL_MPS(conv_transpose2d, input, lower_precision_fp)
  KERNEL_MPS(convolution, lower_precision_fp)
  KERNEL_MPS(_mps_convolution, lower_precision_fp)
  KERNEL_MPS(prelu, lower_precision_fp)
  KERNEL_MPS(addmm, lower_precision_fp)
  KERNEL_MPS(addmv, lower_precision_fp)
  KERNEL_MPS(addr, lower_precision_fp)
  KERNEL_MPS(matmul, lower_precision_fp)
  KERNEL_MPS(einsum, lower_precision_fp)
  KERNEL_MPS(mm, lower_precision_fp)
  KERNEL_MPS(mv, lower_precision_fp)
  KERNEL_MPS(linear, lower_precision_fp)
  KERNEL_MPS(addbmm, lower_precision_fp)
  KERNEL_MPS(baddbmm, lower_precision_fp)
  KERNEL_MPS(bmm, lower_precision_fp)
  KERNEL_MPS(chain_matmul, lower_precision_fp)
  KERNEL_MPS(linalg_multi_dot, lower_precision_fp)
  KERNEL_MPS(lstm_cell, lower_precision_fp)
  KERNEL_MPS(scaled_dot_product_attention, lower_precision_fp)

  // fp32
  KERNEL_MPS(acos, fp32)
  KERNEL_MPS(asin, fp32)
  KERNEL_MPS(cosh, fp32)
  KERNEL_MPS(erfinv, fp32)
  KERNEL_MPS(exp, fp32)
  KERNEL_MPS(expm1, fp32)
  KERNEL_MPS(log, fp32)
  KERNEL_MPS(log10, fp32)
  KERNEL_MPS(log2, fp32)
  KERNEL_MPS(log1p, fp32)
  KERNEL_MPS(reciprocal, fp32)
  KERNEL_MPS(rsqrt, fp32)
  KERNEL_MPS(sinh, fp32)
  KERNEL_MPS(tan, fp32)
  KERNEL_MPS(pow, Tensor_Scalar, fp32)
  KERNEL_MPS(pow, Tensor_Tensor, fp32)
  KERNEL_MPS(pow, Scalar, fp32)
  KERNEL_MPS(softplus, fp32)
  KERNEL_MPS(layer_norm, fp32)
  KERNEL_MPS(native_layer_norm, fp32)
  KERNEL_MPS(group_norm, fp32)
  KERNEL_MPS(frobenius_norm, dim, fp32)
  KERNEL_MPS(nuclear_norm, fp32)
  KERNEL_MPS(nuclear_norm, dim, fp32)
  KERNEL_MPS(batch_norm, fp32)
  KERNEL_MPS(cosine_similarity, fp32)
  KERNEL_MPS(poisson_nll_loss, fp32)
  KERNEL_MPS(cosine_embedding_loss, fp32)
  KERNEL_MPS(nll_loss, fp32)
  KERNEL_MPS(nll_loss2d, fp32)
  KERNEL_MPS(hinge_embedding_loss, fp32)
  KERNEL_MPS(kl_div, fp32)
  KERNEL_MPS(l1_loss, fp32)
  KERNEL_MPS(smooth_l1_loss, fp32)
  KERNEL_MPS(huber_loss, fp32)
  KERNEL_MPS(mse_loss, fp32)
  KERNEL_MPS(margin_ranking_loss, fp32)
  KERNEL_MPS(multilabel_margin_loss, fp32)
  KERNEL_MPS(soft_margin_loss, fp32)
  KERNEL_MPS(triplet_margin_loss, fp32)
  KERNEL_MPS(multi_margin_loss, fp32)
  KERNEL_MPS(binary_cross_entropy_with_logits, fp32)
  KERNEL_MPS(dist, fp32)
  KERNEL_MPS(pdist, fp32)
  KERNEL_MPS(cdist, fp32)
  KERNEL_MPS(renorm, fp32)
  KERNEL_MPS(logsumexp, fp32)

  // fp32_set_opt_dtype
  KERNEL_MPS(prod, fp32)
  KERNEL_MPS(prod, dim_int, fp32)
  KERNEL_MPS(prod, dim_Dimname, fp32)
  KERNEL_MPS(softmax, int, fp32)
  KERNEL_MPS(softmax, Dimname, fp32)
  KERNEL_MPS(log_softmax, int, fp32)
  KERNEL_MPS(log_softmax, Dimname, fp32)
  KERNEL_MPS(cumprod, fp32)
  KERNEL_MPS(cumprod, dimname, fp32)
  KERNEL_MPS(cumsum, fp32)
  KERNEL_MPS(cumsum, dimname, fp32)
  KERNEL_MPS(linalg_vector_norm, fp32)
  KERNEL_MPS(linalg_matrix_norm, fp32)
  KERNEL_MPS(linalg_matrix_norm, str_ord, fp32)
  KERNEL_MPS(sum, fp32)
  KERNEL_MPS(sum, dim_IntList, fp32)
  KERNEL_MPS(sum, dim_DimnameList, fp32)
  //
  // promote
  KERNEL_MPS(addcdiv, promote)
  KERNEL_MPS(addcmul, promote)
  KERNEL_MPS(atan2, promote)
  KERNEL_MPS(bilinear, promote)
  KERNEL_MPS(cross, promote)
  KERNEL_MPS(dot, promote)
  KERNEL_MPS(grid_sampler, promote)
  KERNEL_MPS(index_put, promote)
  KERNEL_MPS(tensordot, promote)
  KERNEL_MPS(scatter_add, promote)
}

TORCH_LIBRARY_IMPL(_, AutocastCPU, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}


TORCH_LIBRARY_IMPL(aten, AutocastCPU, m) {
  // lower_precision_fp cast policy
  KERNEL_CPU(conv1d, lower_precision_fp)
  KERNEL_CPU(conv1d, padding, lower_precision_fp)
  KERNEL_CPU(conv2d, lower_precision_fp)
  KERNEL_CPU(conv2d, padding, lower_precision_fp)
  KERNEL_CPU(conv3d, lower_precision_fp)
  KERNEL_CPU(conv3d, padding, lower_precision_fp)
  KERNEL_CPU(bmm, lower_precision_fp)
  KERNEL_CPU(mm, lower_precision_fp)
  KERNEL_CPU(linalg_vecdot, lower_precision_fp)
  KERNEL_CPU(baddbmm, lower_precision_fp)
  KERNEL_CPU(addmm, lower_precision_fp)
  KERNEL_CPU(_addmm_activation, lower_precision_fp)
  KERNEL_CPU(addbmm, lower_precision_fp)
  KERNEL_CPU(linear, lower_precision_fp)
  KERNEL_CPU(mkldnn_reorder_linear_weight, lower_precision_fp)
  KERNEL_CPU(_convolution, deprecated, lower_precision_fp)
  KERNEL_CPU(matmul, lower_precision_fp)
  KERNEL_CPU(conv_tbc, lower_precision_fp)
  KERNEL_CPU(mkldnn_rnn_layer, lower_precision_fp)
  KERNEL_CPU(conv_transpose1d, lower_precision_fp)
  KERNEL_CPU(conv_transpose2d, input, lower_precision_fp)
  KERNEL_CPU(conv_transpose3d, input, lower_precision_fp)
  KERNEL_CPU(prelu, lower_precision_fp)
  KERNEL_CPU(scaled_dot_product_attention, lower_precision_fp)
  KERNEL_CPU(_native_multi_head_attention, lower_precision_fp)

  // fp32 cast policy
  KERNEL_CPU(avg_pool3d, fp32)
  KERNEL_CPU(binary_cross_entropy, fp32)
  KERNEL_CPU(grid_sampler, fp32)
  KERNEL_CPU(polar, fp32)
  KERNEL_CPU(prod, fp32)
  KERNEL_CPU(prod, dim_int, fp32)
  KERNEL_CPU(prod, dim_Dimname, fp32)
  KERNEL_CPU(quantile, fp32)
  KERNEL_CPU(quantile, scalar, fp32)
  KERNEL_CPU(nanquantile, fp32)
  KERNEL_CPU(nanquantile, scalar, fp32)
  KERNEL_CPU(stft, fp32)
  KERNEL_CPU(stft, center, fp32)
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
  KERNEL_CPU(ctc_loss, IntList, fp32)
  KERNEL_CPU(ctc_loss, Tensor, fp32)
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
  KERNEL_CPU(linalg_cond, p_str, fp32)
  KERNEL_CPU(linalg_matrix_rank, fp32)
  KERNEL_CPU(linalg_matrix_rank, tol_tensor, fp32)
  KERNEL_CPU(linalg_matrix_rank, atol_rtol_tensor, fp32)
  KERNEL_CPU(linalg_matrix_rank, atol_rtol_float, fp32)
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
  KERNEL_CPU(index_copy, dimname, promote)

}

TORCH_LIBRARY_IMPL(_, AutocastXPU, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, AutocastXPU, m) {
  // lower_precision_fp
#define _KERNEL_XPU_LOW_PRECISION_FP(...) \
  KERNEL_XPU(__VA_ARGS__, lower_precision_fp)

  AT_FORALL_LOWER_PRECISION_FP(_KERNEL_XPU_LOW_PRECISION_FP)

  // fp32
#define _KERNEL_XPU_FP32(...) KERNEL_XPU(__VA_ARGS__, fp32)

  AT_FORALL_FP32(_KERNEL_XPU_FP32)

  // fp32_set_opt_dtype
#define _KERNEL_XPU_FP32_SET_OPT_DTYPE(...) \
  KERNEL_XPU(__VA_ARGS__, fp32_set_opt_dtype)

  AT_FORALL_FP32_SET_OPT_DTYPE(_KERNEL_XPU_FP32_SET_OPT_DTYPE)

  // fp32_append_dtype
  // The fp32_append_dtype wrapper overrides implicit promotion behavior.
  // norm does not implicitly promote, but be aware when adding new ops to this policy.
  AT_FORALL_DIFFERENT_REDISPATCH_SIGNATURE(
      KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_XPU)

  // promote
#define _KERNEL_XPU_PROMOTE(...) KERNEL_XPU(__VA_ARGS__, promote)

  AT_FORALL_PROMOTE(_KERNEL_XPU_PROMOTE)

  m.impl(TORCH_SELECTIVE_NAME("aten::binary_cross_entropy"),
         TORCH_FN((&at::autocast::binary_cross_entropy_banned)));
}

} // namespace
} // namespace at::autocast
