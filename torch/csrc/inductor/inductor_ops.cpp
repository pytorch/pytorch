#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/mm.h>
#endif

#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/inductor/inductor_ops.h>
#include <torch/library.h>

#include <optional>

namespace torch::inductor {
using namespace at;

Tensor _mm_plus_mm_out(
    Tensor& out,
    const Tensor& a,
    const Tensor& b,
    const Tensor& c,
    const Tensor& d) {
  at::mm_out(out, a, b);
  out.addmm_(c, d);
  return out;
}

Tensor _mm_plus_mm(
    const Tensor& a,
    const Tensor& b,
    const Tensor& c,
    const Tensor& d,
    Tensor& out) {
  return _mm_plus_mm_out(out, a, b, c, d);
}

Tensor _alloc_from_pool(
    const Tensor& self,
    int64_t offset_bytes,
    ScalarType dtype,
    IntArrayRef size,
    IntArrayRef stride) {
  TORCH_CHECK(self.storage_offset() == 0);
  // based on alias_with_sizes_and_strides from TensorShape.cpp
  Tensor self_ = at::detail::make_tensor<TensorImpl>(
      // c10::TensorImpl::VIEW,
      Storage(self.storage()),
      self.key_set(),
      caffe2::TypeMeta::fromScalarType(dtype));
  auto* self_tmp_ = self_.unsafeGetTensorImpl();
  self_tmp_->set_storage_offset(
      offset_bytes / static_cast<int64_t>(c10::elementSize(dtype)));
  self_tmp_->set_sizes_and_strides(size, stride);
  return self_;
}

// Similar to as_strided with the following differences
// - offset is added to the existing offset (rather than replacing it)
// - view tracking is disabled similar to unsafe_view
Tensor _reinterpret_tensor(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    int64_t offset_increment) {
  Tensor self_ = at::detail::make_tensor<TensorImpl>(
      Storage(self.storage()), self.key_set(), self.dtype());
  auto* self_tmp_ = self_.unsafeGetTensorImpl();
  self_tmp_->set_storage_offset(self.storage_offset() + offset_increment);
  self_tmp_->set_sizes_and_strides(size, stride);
  return self_;
}

static std::optional<Tensor> accumulate_grad_(
    const Tensor& variable,
    const std::optional<Tensor>& variable_grad,
    const std::optional<Tensor>& new_grad) {
  if (!new_grad.has_value()) {
    if (!variable_grad.has_value() || !variable_grad->defined()) {
      return std::nullopt;
    }
    at::Tensor grad = variable_grad->clone();
    variable.mutable_grad() = grad;
    return grad;
  }

  at::Tensor grad = variable_grad.has_value() && variable_grad->defined()
      ? variable_grad->clone()
      : Tensor();
  if (new_grad->device() != kMeta && !grad.defined()) {
    // Unlike eager AccumulateGrad, this op's schema does not allow the returned
    // grad to alias any input. Clone when initializing grad so
    // functionalization can safely model the output as fresh.
    if (new_grad->is_sparse() || new_grad->is_sparse_csr() ||
        new_grad->is_nested() || new_grad->is_mkldnn()) {
      grad = new_grad->clone();
    } else {
      grad = torch::autograd::utils::clone_obey_contract(*new_grad, variable);
    }
  } else if (new_grad->device() != kMeta) {
    // Do not call into this codepath from C++ frontend, instead call directly
    // into accumulateGrad. The refcount argument only affects no-existing-grad
    // steal paths, which are handled above to avoid input aliasing.
    torch::autograd::AccumulateGrad::accumulateGrad(
        variable,
        grad,
        *new_grad,
        2 /* num_expected_refs */,
        [&grad](at::Tensor&& grad_update) { grad = std::move(grad_update); });
  } else {
    // no shape checking for `device="meta"` to workaround FSDP inplace mutation
    if (!grad.defined()) {
      grad = new_grad->clone();
    }
  }
  if (!grad.defined()) {
    return std::nullopt;
  }
  // Compiled autograd graphs use this op as the grad-accumulation side effect,
  // but functionalization still requires the returned grad to be fresh.
  variable.mutable_grad() = grad;
  return grad;
}

TORCH_LIBRARY_FRAGMENT(inductor, m) {
  m.def(
      "_mm_plus_mm(Tensor a, Tensor b, Tensor c, Tensor d, Tensor(t!) out) -> Tensor(t!)",
      dispatch(c10::DispatchKey::CompositeExplicitAutograd, _mm_plus_mm),
      {at::Tag::pt2_compliant_tag});
  m.def(
      "_alloc_from_pool(Tensor self, int offset_bytes, ScalarType dtype, int[] size, int[] stride) -> Tensor",
      _alloc_from_pool,
      {at::Tag::pt2_compliant_tag});
  m.def(
      "_reinterpret_tensor(Tensor self, int[] size, int[] stride, int offset_increment=0) -> Tensor",
      dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, _reinterpret_tensor),
      {at::Tag::pt2_compliant_tag});
  m.def(
      "accumulate_grad_(Tensor variable, Tensor? variable_grad, Tensor? new_grad) -> Tensor?",
      dispatch(c10::DispatchKey::CompositeExplicitAutograd, accumulate_grad_),
      {at::Tag::pt2_compliant_tag});
}

TORCH_LIBRARY_FRAGMENT(inductor_prims, m) {
  m.def(
      "inductor_reserve_rng_state(Generator? generator, SymInt increment) "
      "-> (Tensor, Tensor, Tensor)",
      {at::Tag::pt2_compliant_tag});
}

} // namespace torch::inductor
