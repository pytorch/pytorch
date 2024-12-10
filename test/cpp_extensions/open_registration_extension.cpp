#include <unordered_map>
#include <c10/core/impl/alloc_cpu.h>
#include <c10/core/Allocator.h>
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>

#include <torch/csrc/Device.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#include <torch/extension.h>

#include <ATen/native/cpu/Loops.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Resize.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/ops/abs_native.h>
#include <ATen/EmptyTensor.h>
#include <ATen/core/GeneratorForPrivateuseone.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/ops/view.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/native/transformers/attention.h>

static uint64_t add_counter = 0;
static uint64_t last_saved_value = 0;
static c10::DeviceIndex custom_device_index = 0;

static uint64_t abs_counter = 0;
static uint64_t last_abs_saved_value = 0;

static uint64_t storageImpl_counter = 0;
static uint64_t last_storageImpl_saved_value = 0;

namespace {

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

void quantize_tensor_per_tensor_affine_privateuse1(
    const at::Tensor& rtensor,
    at::Tensor& qtensor,
    double scale,
    int64_t zero_point) {
    // do nothing
}

int64_t _fused_sdp_choice_privateuse1(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value,
    const std::optional<at::Tensor> & attn_mask, double dropout_p, bool is_causal, std::optional<double> scale, bool enable_gqa){
  auto backend = sdp::SDPBackend::overrideable;
  return static_cast<int64_t>(backend);
}
} // namespace

namespace at::native {

REGISTER_PRIVATEUSE1_DISPATCH(abs_stub, &abs_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(quantize_tensor_per_tensor_affine_stub, &quantize_tensor_per_tensor_affine_privateuse1);
REGISTER_PRIVATEUSE1_DISPATCH(_fused_sdp_choice_stub, &_fused_sdp_choice_privateuse1);

} // namespace at::native
struct CustomBackendMetadata : public c10::BackendMeta {
  // for testing this field will mutate when clone() is called by shallow_copy_from.
  int backend_version_format_{-1};
  int format_number_{-1};
  mutable bool cloned_{false};
  // define the constructor
  CustomBackendMetadata(int backend_version_format, int format_number) :
      backend_version_format_(backend_version_format), format_number_(format_number) {}
  c10::intrusive_ptr<c10::BackendMeta> clone(
      const c10::intrusive_ptr<c10::BackendMeta>& ptr) const override {
    cloned_ = true;
    return c10::BackendMeta::clone(ptr);
  }
};

// we need to register two functions for serialization
void for_serialization(const at::Tensor& t, std::unordered_map<std::string, bool>& m) {
  if (t.unsafeGetTensorImpl()->get_backend_meta_intrusive_ptr() == nullptr) {
    return;
  }
  auto tmeta = dynamic_cast<CustomBackendMetadata*>(t.unsafeGetTensorImpl()->get_backend_meta());
  if (tmeta->backend_version_format_ == 1) {
    m["backend_version_format"] = true;
  }
  if (tmeta->format_number_ == 29) {
    m["format_number"] = true;
  }
}

void for_deserialization(const at::Tensor& t, std::unordered_map<std::string, bool>& m) {
  int backend_version_format{-1};
  int format_number{-1};
  if (m.find("backend_version_format") != m.end()) {
    backend_version_format = 1;
  }
  if (m.find("format_number") != m.end()) {
    format_number = 29;
  }
  c10::intrusive_ptr<c10::BackendMeta> new_tmeta{std::unique_ptr<c10::BackendMeta>(
      new CustomBackendMetadata(backend_version_format, format_number))};
  t.unsafeGetTensorImpl()->set_backend_meta(new_tmeta);
}

void custom_serialization_registry() {
  torch::jit::TensorBackendMetaRegistry(c10::DeviceType::PrivateUse1,
                                        &for_serialization,
                                        &for_deserialization);
}

//check if BackendMeta serialization correctly
bool check_backend_meta(const at::Tensor& t) {
  if (t.unsafeGetTensorImpl()->get_backend_meta_intrusive_ptr()) {
    CustomBackendMetadata* tmeta = dynamic_cast<CustomBackendMetadata*>(
        t.unsafeGetTensorImpl()->get_backend_meta());
    if (tmeta->backend_version_format_==1 && tmeta->format_number_==29) {
      return true;
    }
  }
  return false;
}

// a fake set function is exposed to the Python side
void custom_set_backend_meta(const at::Tensor& t) {
  int backend_version_format{1};
  int format_number{29};
  c10::intrusive_ptr<c10::BackendMeta> new_tmeta{std::unique_ptr<c10::BackendMeta>(
      new CustomBackendMetadata(backend_version_format, format_number))};
  t.unsafeGetTensorImpl()->set_backend_meta(new_tmeta);
}

// A dummy storageImpl for our custom device, that secretly uses the CPU
c10::intrusive_ptr<c10::StorageImpl> make_custom_storage_impl(c10::StorageImpl::use_byte_size_t,
                                                              c10::SymInt size_bytes,
                                                              c10::DataPtr data_ptr,
                                                              c10::Allocator* allocator,
                                                              bool resizable) {
  c10::intrusive_ptr<c10::StorageImpl> custom_storage_impl;
  if (data_ptr == nullptr){
    custom_storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(), size_bytes, allocator, resizable);
  } else {
    custom_storage_impl = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(), size_bytes, std::move(data_ptr), allocator, resizable);
  }
  storageImpl_counter += 1;
  return custom_storage_impl;
}

// Register our dummy storageImpl create method.
void custom_storage_registry() {
  c10::SetStorageImplCreate(c10::DeviceType::PrivateUse1, &make_custom_storage_impl);
}

bool custom_storageImpl_called() {
  if (storageImpl_counter > last_storageImpl_saved_value) {
    last_storageImpl_saved_value = storageImpl_counter;
    return true;
  }
  return false;
}

// basic dummy add function
at::Tensor custom_add_Tensor(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  add_counter += 1;
  // Since this custom device is just for testing, not bothering to implement kernels.
  return at::empty(self.sizes(), self.options());
}

at::Tensor custom__copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
    return dst.copy_(self, false);
}

// Some set operations for the basic use case
at::Tensor& custom_set_source_Storage(at::Tensor& result, c10::Storage src) {
  int64_t new_size = static_cast<int64_t>(src.nbytes() / result.dtype().itemsize());
  c10::IntArrayRef stride = {};
  result.unsafeGetTensorImpl()->set_storage_offset(0);
  at::OptionalIntArrayRef stride_opt = stride.data() != nullptr ? at::OptionalIntArrayRef(stride) : std::nullopt;
  at::native::resize_impl_cpu_(result.unsafeGetTensorImpl(),
                               new_size, stride_opt,
                               /*resize_storage=*/!result.is_meta());
  return result;
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
  const int64_t head_dim_qk = query.size(3);
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

// This macro does the heavy lifting.
// With TORCH_LIBRARY_IMPL, you can register custom kernels for your backend.
// For open registration, we're registering all of our kernels to the PrivateUse1 dispatch key.
// Later in this file, we map a custom device to the PrivateUse1 device type,
// which allows user code that puts a tensor on your custom_device to eventually get plumbed
// into the kernels registered here.
//
// This macro registers your kernels to the PyTorch Dispatcher.
// More details on the dispatcher can be found at http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/.
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("add.Tensor", &custom_add_Tensor);
  m.impl("_copy_from_and_resize", &custom__copy_from_and_resize);
  m.impl("set_.source_Storage", &custom_set_source_Storage);
  m.impl("quantize_per_tensor", at::native::quantize_per_tensor);
  m.impl("_fused_sdp_choice", &_fused_sdp_choice_privateuse1);
  m.impl("_scaled_dot_product_fused_attention_overrideable", &custom_scaled_dot_product_fused_attention_overrideable);
  m.impl("_scaled_dot_product_fused_attention_overrideable_backward", &custom_scaled_dot_product_fused_attention_overrideable_backward);
}

void custom_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  at::native::cpu_fallback(op, stack);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_foreach_add.List", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("_fused_adamw_", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  m.impl("triu_indices", torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
}

// This basic implementation doesn't bother dealing with different device indices
// (e.g. custom_device:0 vs. custom_device:1).
// We could do that by letting the user pass in a device index in our exposed device function.
// Note that if you do that, you'll also need to register a device guard to core.
// See `c10/core/impl/DeviceGuardImplInterface.h:C10_REGISTER_GUARD_IMPL`.
c10::Device get_custom_device() {
  return c10::Device(c10::DeviceType::PrivateUse1, 0);
}

bool custom_add_called() {
  bool called = false;
  if (add_counter > last_saved_value) {
    called = true;
    last_saved_value = add_counter;
  }
  return called;
}

class PrivateGeneratorImpl : public at::CPUGeneratorImpl {
public:
  // Constructors
  PrivateGeneratorImpl(c10::DeviceIndex device_index) {
    device_ = c10::Device(c10::DeviceType::PrivateUse1, device_index);
    key_set_ = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
  }
  ~PrivateGeneratorImpl() override = default;
};

// this is used to register generator
at::Generator make_generator_privateuse1(c10::DeviceIndex device_index) {
  return at::make_generator<PrivateGeneratorImpl>(device_index);
}

void register_generator_second() {
  REGISTER_GENERATOR_PRIVATEUSE1(make_generator_privateuse1)
}

void set_custom_device_index(c10::DeviceIndex device_index) {
  custom_device_index = device_index;
}

const at::Generator& default_generator(c10::DeviceIndex device_index) {
  return at::globalContext().defaultGenerator(at::Device(c10::DeviceType::PrivateUse1, device_index));;
}

void fallback_with_undefined_tensor() {
  at::Tensor first = at::empty((2,3)).to(at::DeviceType::PrivateUse1);
  at::Tensor second = at::Tensor();
  at::Tensor step = at::empty({}).fill_(2).to(at::DeviceType::PrivateUse1);
  at::Tensor grad_scale = at::empty({}).fill_(0.00001).to(at::DeviceType::PrivateUse1);
  at::Tensor found_inf = at::empty({}).fill_(1).to(at::DeviceType::PrivateUse1);
  at::TensorList tensors = {first, first};
  at::TensorList undefined_tensors = {first, second};
  at::TensorList steps = {step, step};
  return at::_fused_adamw_(tensors, tensors, tensors, tensors, undefined_tensors,
                           steps, 0.001, 0.9, 0.999, 1e-2, 1e-8, false, false,
                           grad_scale, found_inf);
}

struct CustomAutogradFnReturnsSelf : public torch::autograd::Function<CustomAutogradFnReturnsSelf> {

  static at::Tensor forward(torch::autograd::AutogradContext* ctx, at::Tensor self) {
    return self;
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_output) {
    return {grad_output[0] * 0.5};
  }
};

struct CustomAutogradFnAliasing : public torch::autograd::Function<CustomAutogradFnAliasing> {

  static at::Tensor forward(torch::autograd::AutogradContext* ctx, at::Tensor self) {
    return self.view_symint(self.sym_sizes());
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_output) {
    return {grad_output[0] * 0.5};
  }
};

at::Tensor custom_autograd_fn_returns_self(at::Tensor x) {
  return CustomAutogradFnReturnsSelf::apply(x);
}

at::Tensor custom_autograd_fn_aliasing(at::Tensor x) {
  return CustomAutogradFnAliasing::apply(x);
}

// Here, we're exposing a custom device object that corresponds to our custom backend.
// We do this using pybind: exposing an "extension_name.custom_device()" function in python,
// that's implemented in C++.
// The implementation in this file maps directly to the `PrivateUse1` device type.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_device", &get_custom_device, "get custom device object");
    m.def("custom_add_called", &custom_add_called, "check if our custom add function was called");
    m.def("register_generator_second", &register_generator_second, "register generator for custom device secondly");
    m.def("set_custom_device_index", &set_custom_device_index, "set custom device index");
    m.def("custom_storage_registry", &custom_storage_registry, "set custom storageImpl creat method");
    m.def("custom_storageImpl_called", &custom_storageImpl_called, "check if our custom abs function was called");
    m.def("custom_set_backend_meta", &custom_set_backend_meta, "a fake set tensor BackendMeta function");
    m.def("check_backend_meta", &check_backend_meta, "check if BackendMeta serialization correctly");
    m.def("custom_serialization_registry", &custom_serialization_registry, "register custom serialization function");
    m.def("default_generator", &default_generator, "default_generator for privateuse1");
    m.def("fallback_with_undefined_tensor", &fallback_with_undefined_tensor, "fallback_with_undefined_tensor for privateuse1");

    // Co-opting this file to more easily test torch.compile'ing of custom autograd functions in C++
    m.def("custom_autograd_fn_returns_self", &custom_autograd_fn_returns_self);
}

TORCH_LIBRARY(_test_funcs, m) {
  m.def("custom_autograd_fn_aliasing(Tensor(a) input)-> Tensor(a)");
}
TORCH_LIBRARY_IMPL(_test_funcs, AutogradCPU, m) {
  m.impl("custom_autograd_fn_aliasing", &custom_autograd_fn_aliasing);
}
