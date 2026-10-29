#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/GroupedMMUtils.h>
#include <ATen/native/xpu/grouped_mm/grouped_mm_api.h>

#include <c10/util/StringUtil.h>

#include <cctype>
#include <cstdlib>
#include <string>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_grouped_mm_native.h>
#endif

namespace at::native {

namespace {

// TORCH_XPU_GROUPED_MM_BACKEND = auto (default) | sycltla | onednn | fallback,
// read on every call.
//   auto      SYCL-TLA if it can serve the call, else oneDNN, else the
//             per-group fallback (the order of the checks in _grouped_mm_xpu)
//   sycltla   force SYCL-TLA, onednn force oneDNN: the per-group fallback
//             (with a one-time warning) for calls the backend cannot serve
//   fallback  always the per-group fallback
// Like _grouped_mm_cuda, the checks use only dtype, layout and alignment,
// never the routed row counts, so every rank of an expert-parallel job runs
// the same kernel for the same op.
enum class Backend { Auto, SyclTla, OneDnn, Fallback };

Backend requested_backend() {
  const char* v = std::getenv("TORCH_XPU_GROUPED_MM_BACKEND");
  const std::string s = (v == nullptr) ? "" : v;
  if (s.empty() || s == "auto") {
    return Backend::Auto;
  }
  if (s == "sycltla" || s == "sycl-tla") {
    return Backend::SyclTla;
  }
  if (s == "onednn") {
    return Backend::OneDnn;
  }
  if (s == "fallback") {
    return Backend::Fallback;
  }
  TORCH_WARN_ONCE(
      "Ignoring unknown TORCH_XPU_GROUPED_MM_BACKEND='", s,
      "'; expected auto, sycltla, onednn or fallback.");
  return Backend::Auto;
}

// SYCL-TLA grouped GEMM (torch-xpu-ops): "" if it can serve the call, else
// the reason.
std::string sycltla_why_not(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    c10::ScalarType out_dtype,
    const Tensor& out) {
  if (!offs.has_value()) {
    return "no offs (3D x 3D batched)";
  }
  if (bias.has_value()) {
    return "bias";
  }
  if (out_dtype != mat_a.scalar_type()) {
    return "out_dtype differs from input dtype";
  }
  std::string why;
  sycltla::can_use_grouped_mm(mat_a, mat_b, *offs, out, &why);
  return why;
}

// oneDNN grouped matmul slot, for pytorch/pytorch#195427 (split from #181557,
// by LuFinch). To wire it in, include <ATen/native/mkldnn/xpu/detail/oneDNN.h>,
// accept 2D x 3D with offs and no bias here, and in run_onednn() call
//   at::native::onednn::scaled_grouped_matmul(mat_a, mat_b, std::nullopt,
//       std::nullopt, std::nullopt, std::nullopt, *offs, out);
std::string onednn_why_not(
    const Tensor& /*mat_a*/,
    const Tensor& /*mat_b*/,
    const std::optional<at::Tensor>& /*offs*/,
    const std::optional<at::Tensor>& /*bias*/) {
  return "oneDNN grouped matmul not in this build (pending pytorch/pytorch#195427)";
}

void run_onednn(
    const Tensor& /*mat_a*/,
    const Tensor& /*mat_b*/,
    const std::optional<at::Tensor>& /*offs*/,
    Tensor& /*out*/) {
  TORCH_INTERNAL_ASSERT(false, "oneDNN grouped matmul backend is not built");
}

// Log de-duplication key for a fallback decision: the reason without its
// numbers, plus the dims that do not change with routing.
std::string fallback_log_key(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const std::string& reason) {
  std::string key = "fallback|";
  for (char ch : reason) {
    if (!std::isdigit(static_cast<unsigned char>(ch))) {
      key += ch;
    }
  }
  return c10::str(
      key, "|", mat_a.dim(), "x", mat_b.dim(), "|", mat_a.scalar_type(), "|",
      (mat_a.dim() == 2 && mat_b.dim() == 2) ? mat_a.size(0) : mat_a.size(-1),
      "|", mat_b.size(-1));
}

} // namespace

// torch._grouped_mm on XPU, structured like _grouped_mm_cuda: validate,
// allocate the output, then a grouped GEMM kernel or the generic per-group
// fallback. The kernels serve the shapes autograd produces for a MoE expert
// layer:
//   forward      mat_a [M, K] x mat_b [E, K, N]              -> [M, N]
//   grad_input   mat_a [M, N] x mat_b.transpose(-2, -1)      -> [M, K]
//   grad_weight  mat_a.t() [K, M] x grad [M, N]              -> [E, K, N]
//
// The registration and test scaffolding for XPU grouped GEMM mirrors
// pytorch/pytorch#195428 (split from #181557).
//
// TORCH_XPU_GROUPED_MM_LOG=1 (each distinct decision once) or =2 (every call)
// prints which kernel each call takes, and why, to stderr.
Tensor _grouped_mm_xpu(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    std::optional<c10::ScalarType> out_dtype) {
  _grouped_mm_validate_inputs(mat_a, mat_b, offs, bias, out_dtype);
  const auto out_dtype_ = _resolve_grouped_mm_out_dtype(mat_a, mat_b, out_dtype);
  Tensor out = create_grouped_gemm_output_tensor(mat_a, mat_b, offs, out_dtype_);

  const Backend req = requested_backend();
  std::string reason;
  if (req == Backend::Fallback) {
    reason = "TORCH_XPU_GROUPED_MM_BACKEND=fallback";
  } else {
    // fast paths, no d2h sync needed
    if (req == Backend::Auto || req == Backend::SyclTla) {
      const std::string why =
          sycltla_why_not(mat_a, mat_b, offs, bias, out_dtype_, out);
      if (why.empty()) {
        sycltla::grouped_mm(mat_a, mat_b, *offs, out); // logs its own plan
        return out;
      }
      reason = "sycltla: " + why;
    }
    if (req == Backend::Auto || req == Backend::OneDnn) {
      const std::string why = onednn_why_not(mat_a, mat_b, offs, bias);
      if (why.empty()) {
        run_onednn(mat_a, mat_b, offs, out);
        return out;
      }
      reason += (reason.empty() ? "" : "; ") + ("onednn: " + why);
    }
    if (req != Backend::Auto) {
      TORCH_WARN_ONCE(
          "TORCH_XPU_GROUPED_MM_BACKEND: the forced grouped GEMM cannot serve "
          "some calls (", reason, "); using the per-group fallback for those.");
    }
  }
  if (sycltla::grouped_mm_log_level() > 0) {
    sycltla::log_grouped_mm(
        fallback_log_key(mat_a, mat_b, reason),
        c10::str(
            "fallback per-group mm: ", reason, "  a", mat_a.sizes(), " b",
            mat_b.sizes(), " -> out", out.sizes()));
  }
  _grouped_mm_fallback(mat_a, mat_b, offs, bias, out_dtype, out);
  return out;
}

} // namespace at::native
