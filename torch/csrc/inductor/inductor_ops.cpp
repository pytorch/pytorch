#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/mm.h>
#endif

#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/inductor/inductor_ops.h>
#include <torch/library.h>

#if defined(USE_CUDA) || defined(USE_ROCM)
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/ops/from_blob.h>
#include <c10/util/Optional.h>
#include <tuple>
#endif

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
}

#if defined(USE_CUDA) || defined(USE_ROCM)
// Reserves RNG state for Inductor with CUDA Graph support.
//
// This function allows Inductor to reserve a specific amount of RNG offset
// (increment) for a kernel. It is designed to be safe for CUDA Graph capture
// by explicitly handling the internal generator state via public APIs.
//
// Behavior:
// - Graph Mode: Advances the generator state and returns pointers (wrapped as
// tensors) to the extragraph state. These tensors effectively point to the
// GPU memory that will be updated by `replay_prologue`.
// - Eager Mode: Advances the generator state and returns concrete values
// wrapped in 1D tensors to maintain shape consistency.
//
// -param gen The CUDA generator to use.
// -param increment The number of RNG values to reserve.
// -return A tuple of (Seed Tensor, Offset Tensor, Intragraph Offset CPU
// Tensor).
static Generator _get_or_default_cuda_generator(
    const c10::optional<Generator>& gen_opt) {
  if (gen_opt.has_value()) {
    return *gen_opt;
  }
  const int device_index = at::cuda::current_device();
  return at::cuda::detail::getDefaultCUDAGenerator(device_index);
}

static std::tuple<Tensor, Tensor, Tensor> inductor_reserve_rng_state(
    const c10::optional<Generator>& generator,
    int64_t increment) {
  const auto gen = _get_or_default_cuda_generator(generator);
  auto* gen_impl = at::check_generator<at::CUDAGeneratorImpl>(gen);

  const auto dev_opts =
      at::TensorOptions().dtype(at::kLong).device(gen.device());
  const auto cpu_opts =at::TensorOptions().dtype(at::kLong).device(at::kCPU);

  const at::PhiloxCudaState st =
      gen_impl->philox_cuda_state(static_cast<uint64_t>(increment));

  if (st.captured_) {
    auto seed_t = at::from_blob(
        static_cast<void*>(st.seed_.ptr), {1}, [](void*) {}, dev_opts);
    auto off_t = at::from_blob(
        static_cast<void*>(st.offset_.ptr), {1}, [](void*) {}, dev_opts);
    auto intra_t =
        at::tensor({static_cast<int64_t>(st.offset_intragraph_)}, cpu_opts);
    return {seed_t, off_t, intra_t};
  }

  auto seed_t =at::scalar_tensor(static_cast<int64_t>(st.seed_.val), dev_opts)
.unsqueeze(0);
  auto off_t =at::scalar_tensor(static_cast<int64_t>(st.offset_.val), dev_opts)
.unsqueeze(0);
  auto intra_t = at::zeros({1}, cpu_opts);
  return {seed_t, off_t, intra_t};
}

TORCH_LIBRARY_FRAGMENT(inductor_prims, m) {
  m.def(
      "inductor_reserve_rng_state(Generator? generator, int increment) "
      "-> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(inductor_prims, BackendSelect, m) {
  m.impl("inductor_reserve_rng_state", TORCH_FN(inductor_reserve_rng_state));
}

TORCH_LIBRARY_IMPL(inductor_prims, CUDA, m) {
  m.impl("inductor_reserve_rng_state", TORCH_FN(inductor_reserve_rng_state));
}

TORCH_LIBRARY_IMPL(inductor_prims, HIP, m) {
  m.impl("inductor_reserve_rng_state", TORCH_FN(inductor_reserve_rng_state));
}

#endif

} // namespace torch::inductor
