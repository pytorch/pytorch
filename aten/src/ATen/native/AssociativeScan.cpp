#include <ATen/WrapDimUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/AssociativeScanKernel.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/associative_scan.h>
#include <ATen/ops/associative_scan_native.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/flip.h>
#include <ATen/ops/movedim.h>
#endif

#include <c10/util/Exception.h>
#include <c10/util/string_view.h>

#include <string>
#include <vector>

namespace at::native {

DEFINE_DISPATCH(associative_scan_stub);
DEFINE_DISPATCH(associative_scan_tensor_list_stub);

namespace {

bool is_supported_combine_mode(c10::string_view combine_mode) {
  return combine_mode == "add" || combine_mode == "mul" ||
      combine_mode == "max" || combine_mode == "min" ||
      combine_mode == "linear_recurrence";
}

void check_combine_mode(c10::string_view combine_mode) {
  TORCH_CHECK(
      is_supported_combine_mode(combine_mode),
      "associative_scan: unsupported combine_mode '",
      combine_mode,
      "'. Supported modes are 'add', 'mul', 'max', 'min', 'linear_recurrence'.");
}

// Inclusive scan over `wrap_dim` of a single tensor. The scan dimension is
// moved to the innermost position so the kernels operate on contiguous
// memory; `reverse` is implemented with a flip of the scan dimension before
// and after the scan, which exactly matches jax.lax.associative_scan
// semantics.
Tensor scan_single(
    const Tensor& self,
    c10::string_view combine_mode,
    int64_t wrap_dim,
    bool reverse) {
  const int64_t ndim = self.dim();
  if (ndim == 0 || self.size(wrap_dim) <= 1) {
    return self.clone();
  }
  Tensor input = reverse ? at::flip(self, {wrap_dim}) : self;
  Tensor moved = at::movedim(input, wrap_dim, ndim - 1);
  Tensor contig = moved.contiguous();
  Tensor result_contig = at::empty_like(contig, MemoryFormat::Contiguous);
  associative_scan_stub(
      contig.device().type(),
      result_contig,
      contig,
      std::string(combine_mode));
  Tensor result = at::movedim(result_contig, ndim - 1, wrap_dim);
  if (reverse) {
    result = at::flip(result, {wrap_dim});
  }
  // The movedim/flip views are non-contiguous for scan dims other than the
  // innermost; return contiguous so the output layout matches the meta
  // implementation and stays compatible with Inductor's fallback path.
  return result.contiguous();
}

} // namespace

Tensor associative_scan(
    const Tensor& self,
    c10::string_view combine_mode,
    int64_t dim,
    bool reverse) {
  check_combine_mode(combine_mode);
  TORCH_CHECK(
      combine_mode != "linear_recurrence",
      "associative_scan: combine_mode 'linear_recurrence' operates on two "
      "input tensors (a, b); call torch.associative_scan([a, b], "
      "'linear_recurrence', dim) instead.");
  if (self.numel() == 0) {
    return self.clone();
  }
  if (self.dim() == 0) {
    return self.clone();
  }
  const int64_t wrap_dim = maybe_wrap_dim(dim, self.dim());
  return scan_single(self, combine_mode, wrap_dim, reverse);
}

std::vector<Tensor> associative_scan_tensor_list(
    TensorList xs,
    c10::string_view combine_mode,
    int64_t dim,
    bool reverse) {
  check_combine_mode(combine_mode);
  TORCH_CHECK(
      combine_mode == "linear_recurrence",
      "associative_scan: the tensor-list overload only supports "
      "combine_mode 'linear_recurrence', got '",
      combine_mode,
      "'.");
  TORCH_CHECK(
      xs.size() == 2,
      "associative_scan: combine_mode 'linear_recurrence' requires exactly 2 "
      "input tensors (a, b), got ",
      xs.size(),
      ".");
  const Tensor& a = xs[0];
  const Tensor& b = xs[1];
  TORCH_CHECK(
      a.sizes() == b.sizes(),
      "associative_scan: all inputs must have the same shape.");
  TORCH_CHECK(
      a.scalar_type() == b.scalar_type(),
      "associative_scan: all inputs must have the same dtype.");
  TORCH_CHECK(
      a.device() == b.device(),
      "associative_scan: all inputs must be on the same device.");

  if (a.numel() == 0) {
    return {a.clone(), b.clone()};
  }
  if (a.dim() == 0) {
    return {a.clone(), b.clone()};
  }
  const int64_t wrap_dim = maybe_wrap_dim(dim, a.dim());
  if (a.size(wrap_dim) <= 1) {
    return {a.clone(), b.clone()};
  }

  const int64_t ndim = a.dim();
  Tensor a_moved = at::movedim(
      reverse ? at::flip(a, {wrap_dim}) : a, wrap_dim, ndim - 1);
  Tensor b_moved = at::movedim(
      reverse ? at::flip(b, {wrap_dim}) : b, wrap_dim, ndim - 1);
  Tensor a_contig = a_moved.contiguous();
  Tensor b_contig = b_moved.contiguous();
  Tensor a_out = at::empty_like(a_contig, MemoryFormat::Contiguous);
  Tensor b_out = at::empty_like(b_contig, MemoryFormat::Contiguous);

  std::vector<TensorBase> outs{a_out, b_out};
  std::vector<TensorBase> ins{a_contig, b_contig};
  associative_scan_tensor_list_stub(
      a.device().type(), outs, ins, std::string(combine_mode));

  Tensor a_result = at::movedim(a_out, ndim - 1, wrap_dim);
  Tensor b_result = at::movedim(b_out, ndim - 1, wrap_dim);
  if (reverse) {
    a_result = at::flip(a_result, {wrap_dim});
    b_result = at::flip(b_result, {wrap_dim});
  }
  return {a_result.contiguous(), b_result.contiguous()};
}

Tensor associative_scan_meta(
    const Tensor& self,
    c10::string_view combine_mode,
    int64_t dim,
    bool reverse) {
  return at::empty_like(self, MemoryFormat::Contiguous);
}

std::vector<Tensor> associative_scan_tensor_list_meta(
    TensorList xs,
    c10::string_view combine_mode,
    int64_t dim,
    bool reverse) {
  std::vector<Tensor> out;
  out.reserve(xs.size());
  for (const auto& x : xs) {
    out.push_back(at::empty_like(x, MemoryFormat::Contiguous));
  }
  return out;
}

} // namespace at::native
