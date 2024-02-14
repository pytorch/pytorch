#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/mm.h>
#endif

#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/inductor/inductor_ops.h>
#include <torch/library.h>

#include <ATen/native/Resize.h>

#ifdef USE_CUDA
#include <ATen/native/cuda/Resize.h>
#endif

namespace torch {
namespace inductor {
using namespace at;

Tensor _mm_plus_mm(
    const Tensor& a,
    const Tensor& b,
    const Tensor& c,
    const Tensor& d,
    Tensor& out) {
  at::mm_out(out, a, b);
  out.addmm_(c, d);
  return out;
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
  self_tmp_->set_storage_offset(offset_bytes / c10::elementSize(dtype));
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

static void accumulate_grad_(const Tensor& variable, const Tensor& new_grad) {
  at::Tensor& grad = variable.mutable_grad();
  if (new_grad.device() != kMeta) {
    // Do not call into this codepath from C++ frontend, instead call directly
    // into accumulateGrad with num_expected_refs set to 1 Here,
    // num_expected_refs is set to 2 to steal the gradient when this is called
    // from Python
    torch::autograd::AccumulateGrad::accumulateGrad(
        variable,
        grad,
        new_grad,
        2 /* num_expected_refs */,
        [&grad](at::Tensor&& grad_update) { grad = std::move(grad_update); });
  } else {
    // no shape checking for `device="meta"` to workaround FSDP inplace mutation
    if (!grad.defined()) {
      grad = new_grad;
    }
  }
}

static void resize_storage_bytes_(const Tensor& variable, SymInt new_size) {
  // similar to THPStorage_resize_ in StorageMethods.cpp, but is traceable
  if (variable.storage().device_type() == at::kCUDA) {
    // rocm build has undefined reference to resize_bytes_cuda
#if defined(USE_CUDA) && !defined(USE_ROCM)
    at::native::resize_bytes_cuda(
        variable.storage().unsafeGetStorageImpl(), new_size.expect_int());
#else
    TORCH_CHECK(false, "built without cuda");
#endif
  } else {
    at::native::resize_bytes_nocuda(variable.storage(), new_size);
  }
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
      "accumulate_grad_(Tensor variable, Tensor new_grad) -> ()",
      dispatch(c10::DispatchKey::CompositeExplicitAutograd, accumulate_grad_),
      {at::Tag::pt2_compliant_tag});
  m.def(
      "resize_storage_bytes_(Tensor variable, SymInt new_size) -> ()",
      dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, resize_storage_bytes_),
      {at::Tag::pt2_compliant_tag});
}

} // namespace inductor
} // namespace torch
