#include "OpenReg.h"

#include <ATen/EmptyTensor.h>
#include <ATen/ops/as_strided_cpu_dispatch.h>
#include <ATen/ops/set_cpu_dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>

#include <c10/core/Allocator.h>

#include <torch/library.h>

namespace openreg {
namespace {

struct OpenRegAllocator final : at::Allocator {
  OpenRegAllocator() = default;

  at::DataPtr allocate(size_t nbytes) override {
    py::gil_scoped_acquire acquire;
    auto curr_device_idx = get_method("getDevice")().cast<c10::DeviceIndex>();
    auto curr_device =
        c10::Device(c10::DeviceType::PrivateUse1, curr_device_idx);
    void* data = nullptr;
    if (nbytes > 0) {
      data = reinterpret_cast<void*>(
          get_method("malloc")(nbytes).cast<openreg_ptr_t>());
      TORCH_CHECK(
          data, "Failed to allocator ", nbytes, " bytes on openreg device.");
    }
    return {data, data, &ReportAndDelete<kFreeMethod>, curr_device};
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete<kFreeMethod>;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    py::gil_scoped_acquire acquire;
    get_method("copy_data")(
        reinterpret_cast<openreg_ptr_t>(dest),
        reinterpret_cast<openreg_ptr_t>(src),
        count);
  }
};

static OpenRegAllocator global_openreg_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_openreg_alloc);

// Empty op needs C++ code and cannot be handled by python side fallback
at::Tensor empty_openreg(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(
      c10::layout_or_default(layout_opt) == c10::Layout::Strided,
      "Non strided layout not supported");
  TORCH_CHECK(
      !c10::pinned_memory_or_default(pin_memory_opt),
      "Pin memory can only be on CPU");
  const c10::DeviceGuard device_guard(device);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_generic(
      size, &global_openreg_alloc, pu1_dks, dtype, memory_format_opt);
}

at::Tensor empty_strided_openreg(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(
      c10::layout_or_default(layout_opt) == c10::Layout::Strided,
      "Non strided layout not supported");
  TORCH_CHECK(
      !c10::pinned_memory_or_default(pin_memory_opt),
      "Pin memory can only be on CPU");
  const c10::DeviceGuard device_guard(device);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_strided_generic(
      size, stride, &global_openreg_alloc, pu1_dks, dtype);
}

at::Tensor as_strided_openreg(
    const at::Tensor& self,
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<int64_t> storage_offset_) {
  // Metadata-only change so we re-use the cpu impl
  return at::cpu::as_strided(self, size, stride, storage_offset_);
}

at::Tensor& set_openreg(
    at::Tensor& result,
    at::Storage storage,
    int64_t storage_offset,
    c10::IntArrayRef size,
    c10::IntArrayRef stride) {
  return at::cpu::set_(result, storage, storage_offset, size, stride);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::SymInt, c10::SymInt, at::Tensor, at::Tensor, at::Tensor>
custom_scaled_dot_product_fused_attention_overrideable(
    const at::Tensor & query,
    const at::Tensor & key,
    const at::Tensor & value,
    const std::optional<at::Tensor> & attn_bias,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale) {
  const int64_t batch_size = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t head_dim_v = value.size(3);
  const int64_t max_seqlen_q = query.size(2);
  const int64_t max_seqlen_kv = key.size(2);

  auto opts = query.options();
  auto output = at::empty({batch_size, num_heads, max_seqlen_q, head_dim_v}, opts);
  auto logsumexp = at::empty({batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
  auto debug_attn_mask = at::empty({batch_size, num_heads, max_seqlen_q, max_seqlen_kv},
                                   opts.dtype(at::kFloat));
  auto philox_seed = at::empty({}, at::dtype(at::kLong));
  auto philox_offset = at::empty({}, at::dtype(at::kLong));

  return std::make_tuple(output, logsumexp, at::Tensor(), at::Tensor(), max_seqlen_q, max_seqlen_kv, philox_seed, philox_offset, debug_attn_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
custom_scaled_dot_product_fused_attention_overrideable_backward(
    const at::Tensor & grad_out,
    const at::Tensor & query,
    const at::Tensor & key,
    const at::Tensor & value,
    const at::Tensor & attn_bias,
    std::array<bool,4> grad_input_mask,
    const at::Tensor & out,
    const at::Tensor & logsumexp,
    const at::Tensor & cum_seq_q,
    const at::Tensor & cum_seq_k,
    int64_t max_q,
    int64_t max_k,
    double dropout_p,
    bool is_causal,
    const at::Tensor & philox_seed,
    const at::Tensor & philox_offset,
    std::optional<double> scale) {
  return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
          at::empty_like(query),
          at::empty_like(key),
          at::empty_like(value),
          at::empty_like(attn_bias));
}
}

int64_t _fused_sdp_choice_privateuse1(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  auto backend = sdp::SDPBackend::overrideable;
  return static_cast<int64_t>(backend);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", empty_openreg);
  m.impl("empty_strided", empty_strided_openreg);
  m.impl("as_strided", as_strided_openreg);
  m.impl("set_.source_Storage_storage_offset", set_openreg);
  m.impl("_fused_sdp_choice", &_fused_sdp_choice_privateuse1);
  m.impl("_scaled_dot_product_fused_attention_overrideable", &custom_scaled_dot_product_fused_attention_overrideable);
  m.impl("_scaled_dot_product_fused_attention_overrideable_backward", &custom_scaled_dot_product_fused_attention_overrideable_backward);
}
} // namespace openreg

namespace at::native {
REGISTER_PRIVATEUSE1_DISPATCH(
    _fused_sdp_choice_stub,
    &openreg::_fused_sdp_choice_privateuse1);
} // namespace at::native
