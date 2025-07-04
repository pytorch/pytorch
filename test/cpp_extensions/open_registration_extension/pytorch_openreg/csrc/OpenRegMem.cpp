#include "OpenReg.h"

#include <ATen/EmptyTensor.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/ops/as_strided_cpu_dispatch.h>
#include <ATen/ops/quantize_per_tensor_native.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/set_cpu_dispatch.h>
#include <ATen/ops/set_native.h>

#include <c10/core/Allocator.h>

#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/function_hook.h>
#include <torch/csrc/jit/serialization/pickler.h>

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

const at::Tensor& resize__openreg(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    ::std::optional<at::MemoryFormat> memory_format) {
  return at::native::resize_(
      self, C10_AS_INTARRAYREF_SLOW(size), memory_format);
}

at::Tensor& set_source_Storage_storage_offsetset_openreg(
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

// Using the simplest way to obtain continuous Tensor data and process it.
// This is a demo for using operand API, and you can add more complex logic
// for input and output tensor based on your custom device kernel.
void abs_kernel(at::TensorIteratorBase& iter) {
  // Abs only have a input tensor and a output tensor.
  auto& output_operand = iter.operand(0);
  auto& input_operand = iter.operand(1);
  auto& output_tensor_base = output_operand.tensor_base();
  auto& input_tensor_base = input_operand.tensor_base();
  TORCH_CHECK(!input_operand.original_tensor_base().defined(),
    "input original tensor is defined.");
  TORCH_CHECK(!output_operand.original_tensor_base().defined(),
    "output original tensor is defined.");
  // For easy test, only accept contiguous input tensor for calculate.
  auto memory_format = input_tensor_base.suggest_memory_format();
  TORCH_CHECK(input_tensor_base.is_contiguous(memory_format),
    "Input tensor need be contiguous.");
  // Add necessary restrictions to ensure the security of the demo.
  TORCH_CHECK(input_tensor_base.sizes() == output_tensor_base.sizes(),
    "Intput and output tensor size are not equal.");
  // Common dtype is calculate in TensorIteratorBase.
  TORCH_CHECK(iter.common_dtype() == at::ScalarType::Float,
    "Only support float type.")
  // Using for loop for abs calculate.
  auto abs_function = [](float* output_ptr, const float* input_ptr,
                         const int64_t NUM) {
    for (int64_t i = 0; i < NUM; ++i) {
      *(output_ptr + i) = std::abs(*(input_ptr + i));
    }
  };
  // To simplify the logic of the test demo code,
  // we only use contiguous tensor to calculate on device side.
  // And using input tensor memory format.
  if (iter.is_contiguous()) {
    // Add for will_resize flag check. You can convert to differernt
    // tensor memory format when will_resize is True.
    // If TensorIteratorConfig resize_outputs_ flag is true, and there are two
    // situations:
    // 1) Out tensor is undefined, and TensorIterator set will_resize to true;
    // 2) Out tensor is defined and tensor size is not equal to input tensor size;
    //    TensorIterator set will_resize to true, and call set_output_raw_strided
    //    to resize output tensor.
    // When output operand will_resize flag is ture, dummy
    // device can convert tensor to dummy device preferred memory format.
    // Here we don't convert tensor memory format, because it will become complex
    // when dummy device want keep same memory format for training network.
    TORCH_CHECK(output_operand.will_resize,
      "output operand will_resize flag need be True.");
    abs_function((float*)iter.data_ptr(0), (float*)iter.data_ptr(1), iter.numel());
  } else {
    // Stride copy is not support for foo device, using cpu device instead.
    // For abs op, the last situation is: output tensor is not contiguous with
    // operand will_resize is False.
    TORCH_CHECK(!output_operand.will_resize, "output operand will_resize is True.");
    // Get a contiguous tensor with input memory format.
    at::Tensor output = at::empty(output_tensor_base.sizes(),
                                  input_tensor_base.options()
                                                   .memory_format(memory_format));
    // For structured op which inheried from TensorIteratorBase, maybe you need to
    // call set_output_raw_strided function to update output stored in op sturctured.
    // abs op is no need to do this.
    output_operand.exchange_tensor(c10::MaybeOwned<at::TensorBase>::owned(std::in_place, output));
    abs_function((float*)output_operand.tensor_base().mutable_data_ptr(),
                 (float*)iter.data_ptr(1), iter.numel());
    // Copy tensor base to original tensor base, and keep same scalar type and
    // stride with cpu and gpu.
    if (output_operand.original_tensor_base().defined() &&
        !output_operand.original_tensor_base().is_same(output_operand.tensor_base())) {
      output_operand.original_tensor().copy_(output_operand.tensor());
      output_operand.restore_original_tensor();
    }
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

void quantize_tensor_per_tensor_affine_privateuse1(
    const at::Tensor& rtensor,
    at::Tensor& qtensor,
    double scale,
    int64_t zero_point) {
    // Just test the process, so do nothing
}

struct CustomAutogradFnReturnsSelf
    : public torch::autograd::Function<CustomAutogradFnReturnsSelf> {
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor self) {
    return self;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    return {grad_output[0] * 0.5};
  }
};

struct CustomAutogradFnAliasing
    : public torch::autograd::Function<CustomAutogradFnAliasing> {
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      at::Tensor self) {
    return self.view_symint(self.sym_sizes());
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    return {grad_output[0] * 0.5};
  }
};

at::Tensor custom_autograd_fn_returns_self(at::Tensor x) {
  return CustomAutogradFnReturnsSelf::apply(x);
}

at::Tensor custom_autograd_fn_aliasing(at::Tensor x) {
  return CustomAutogradFnAliasing::apply(x);
}

/* Notes:
 *
 * OpenReg is currently designed to simulate device memory through multiple
 * subprocesses on purpose to ensure we don't mistakenly poke at the "device's
 * memory" from the main process. And be able to simulate the same thing that
 * happens with other accelerators: any metadata-only change is cpu-only
 * (main process), any data change must go through to the device (other process)
 * and any data transfer between the two is expensive (serializing the whole
 * Tensor).
 *
 * Currently, for the efficiency of IPC, most operations are to pass the Tensor
 * metadata, and only a small number of operations involving copy will serialize
 * and pass the Tensor body by custom pickler provided by torch.multiprocess.
 *
 * Therefore, in principle, only operations related to Metadata modification can
 * be directly implemented at the C++ level and registered in PrivateUse1; but
 * if memory access is involved, the relevant operations must be implemented at
 * the Python level, otherwise invalid memory access will result.
 */

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", empty_openreg);
  m.impl("empty_strided", empty_strided_openreg);
  m.impl("as_strided", as_strided_openreg);
  m.impl("resize_", resize__openreg);
  m.impl("set_.source_Storage", at::native::set_);
  m.impl("set_.source_Storage_storage_offset", set_source_Storage_storage_offsetset_openreg);
  m.impl("quantize_per_tensor", at::native::quantize_per_tensor);
  m.impl("_fused_sdp_choice", &_fused_sdp_choice_privateuse1);
  m.impl("_scaled_dot_product_fused_attention_overrideable", &custom_scaled_dot_product_fused_attention_overrideable);
  m.impl("_scaled_dot_product_fused_attention_overrideable_backward", &custom_scaled_dot_product_fused_attention_overrideable_backward);
}

struct OpenRegBackendMeta : public c10::BackendMeta {
  OpenRegBackendMeta(int version_number, int format_number)
      : version_number_(version_number), format_number_(format_number) {}

  int version_number_{-1};
  int format_number_{-1};
};

void for_serialization(
    const at::Tensor& t,
    std::unordered_map<std::string, bool>& m) {
  auto meta_ptr = t.unsafeGetTensorImpl()->get_backend_meta();

  if (meta_ptr != nullptr) {
    auto o_meta_ptr = dynamic_cast<OpenRegBackendMeta*>(meta_ptr);
    if (o_meta_ptr->version_number_ == 1) {
      m["version_number"] = true;
    }
    if (o_meta_ptr->format_number_ == 29) {
      m["format_number"] = true;
    }
  }
}

void for_deserialization(
    const at::Tensor& t,
    std::unordered_map<std::string, bool>& m) {
  int version_number{-1};
  int format_number{-1};

  if (m.find("version_number") != m.end()) {
    version_number = 1;
  }
  if (m.find("format_number") != m.end()) {
    format_number = 29;
  }

  c10::intrusive_ptr<c10::BackendMeta> meta{std::unique_ptr<c10::BackendMeta>(
      new OpenRegBackendMeta(version_number, format_number))};
  t.unsafeGetTensorImpl()->set_backend_meta(meta);
}

REGISTER_PRIVATEUSE1_SERIALIZATION(&for_serialization, &for_deserialization)
} // namespace openreg

namespace at::native {
REGISTER_PRIVATEUSE1_DISPATCH(abs_stub, &openreg::abs_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(
    quantize_tensor_per_tensor_affine_stub,
    &openreg::quantize_tensor_per_tensor_affine_privateuse1);
REGISTER_PRIVATEUSE1_DISPATCH(
    _fused_sdp_choice_stub,
    &openreg::_fused_sdp_choice_privateuse1);
} // namespace at::native

TORCH_LIBRARY(openreg, m) {
  m.def("custom_autograd_fn_returns_self(Tensor input)-> Tensor");
  m.def("custom_autograd_fn_aliasing(Tensor(a) input)-> Tensor(a)");
}

TORCH_LIBRARY_IMPL(openreg, AutogradPrivateUse1, m) {
  m.impl("custom_autograd_fn_aliasing", &openreg::custom_autograd_fn_aliasing);
  m.impl(
      "custom_autograd_fn_returns_self",
      &openreg::custom_autograd_fn_returns_self);
}
