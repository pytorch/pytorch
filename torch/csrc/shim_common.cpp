#include <c10/core/Device.h>
#include <c10/core/DispatchKey.h>
#include <c10/util/Exception.h>
#include <torch/csrc/inductor/aoti_runtime/utils.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/stable/library.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/from_blob.h>
#endif // AT_PER_OPERATOR_HEADERS
#include <ATen/Parallel.h>
#include <torch/csrc/shim_conversion_utils.h>
#include <torch/csrc/stable/c/shim.h>

AOTITorchError torch_new_list_reserve_size(size_t size, StableListHandle* ret) {
  auto list_ptr = std::make_unique<std::vector<StableIValue>>();
  list_ptr->reserve(size);
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret = list_pointer_to_list_handle(list_ptr.release()); });
}

AOTI_TORCH_EXPORT AOTITorchError
torch_list_size(StableListHandle list_handle, size_t* size) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    std::vector<StableIValue>* list = list_handle_to_list_pointer(list_handle);
    *size = list->size();
  });
}

AOTI_TORCH_EXPORT AOTITorchError torch_list_get_item(
    StableListHandle list_handle,
    size_t index,
    StableIValue* element) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    std::vector<StableIValue>* list = list_handle_to_list_pointer(list_handle);
    *element = list->at(index);
  });
}

AOTI_TORCH_EXPORT AOTITorchError torch_list_set_item(
    StableListHandle list_handle,
    size_t index,
    StableIValue element) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    std::vector<StableIValue>* list = list_handle_to_list_pointer(list_handle);
    list->at(index) = element;
  });
}

AOTITorchError torch_list_push_back(
    StableListHandle list_handle,
    StableIValue element) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    std::vector<StableIValue>* list = list_handle_to_list_pointer(list_handle);
    list->push_back(element);
  });
}

AOTI_TORCH_EXPORT AOTITorchError
torch_delete_list(StableListHandle list_handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    std::vector<StableIValue>* list_ptr =
        list_handle_to_list_pointer(list_handle);
    delete list_ptr;
  });
}

static StableIValue from_ivalue(
    const c10::TypePtr& type,
    const c10::IValue& ivalue,
    uint64_t extension_build_version) {
  switch (type->kind()) {
    case c10::TypeKind::TensorType: {
      AtenTensorHandle ath = torch::aot_inductor::new_tensor_handle(
          std::move(const_cast<at::Tensor&>(ivalue.toTensor())));
      return torch::stable::detail::_from(ath, extension_build_version);
    }
    case c10::TypeKind::IntType: {
      return torch::stable::detail::_from(
          ivalue.toInt(), extension_build_version);
    }
    case c10::TypeKind::FloatType: {
      return torch::stable::detail::_from(
          ivalue.toDouble(), extension_build_version);
    }
    case c10::TypeKind::BoolType: {
      return torch::stable::detail::_from(
          ivalue.toBool(), extension_build_version);
    }
    case c10::TypeKind::ScalarTypeType: {
      return torch::stable::detail::_from(
          ivalue.toScalarType(), extension_build_version);
    }
    case c10::TypeKind::DeviceObjType: {
      // Pack device type and index into StableIValue in platform-independent
      // format Lower 32 bits = device index, upper 32 bits = device type
      const auto& device = ivalue.toDevice();
      uint64_t device_index_bits =
          static_cast<uint64_t>(static_cast<uint32_t>(device.index()));
      uint64_t device_type_bits =
          static_cast<uint64_t>(static_cast<int8_t>(device.type())) << 32;
      return device_index_bits | device_type_bits;
    }
    case c10::TypeKind::LayoutType: {
      return torch::stable::detail::_from(
          ivalue.toLayout(), extension_build_version);
    }
    case c10::TypeKind::MemoryFormatType: {
      return torch::stable::detail::_from(
          ivalue.toMemoryFormat(), extension_build_version);
    }
    case c10::TypeKind::OptionalType: {
      auto inner_type = type->castRaw<at::OptionalType>()->getElementType();

      // ideally, if we had the C++ type corresponding to inner_type, which we
      // will denote as inner_type::t (does not actually exist), we would be
      // able to follow the patterned semantic of every other case here in one
      // line:
      //
      // return
      // torch::stable::detail::from<std::optional<inner_type::t>>(ivalue.toInnerTypeT()));
      //
      // BUT we do NOT have that type inner_type::t readily available, so we
      // will manually unwrap and recursively call. This implementation MUST
      // be kept in sync with torch::stable::detail::from<std::optional<T>>
      // function in torch/csrc/stable/stableivalue_conversions.h
      if (ivalue.isNone()) {
        return torch::stable::detail::_from(
            std::nullopt, extension_build_version);
      }
      StableIValue* sivp = new StableIValue(
          from_ivalue(inner_type, ivalue, extension_build_version));
      return torch::stable::detail::_from(sivp, extension_build_version);
    }
    case c10::TypeKind::ListType: {
      auto inner_type = type->castRaw<c10::ListType>()->getElementType();
      auto ivalue_list = ivalue.toList();
      auto stableivalue_list = std::make_unique<std::vector<StableIValue>>();
      stableivalue_list->reserve(ivalue_list.size());
      for (const auto& elem : ivalue_list) {
        stableivalue_list->emplace_back(
            from_ivalue(inner_type, elem, extension_build_version));
      }
      return torch::stable::detail::_from(
          list_pointer_to_list_handle(stableivalue_list.release()),
          extension_build_version);
    }
    case c10::TypeKind::StringType: {
      return torch::stable::detail::_from(
          ivalue.toStringRef(), extension_build_version);
    }
    case c10::TypeKind::SymIntType: {
      // Treat SymInt as Int for StableIValue <-> IValue conversion
      return from_ivalue(c10::IntType::get(), ivalue, extension_build_version);
    }
    default: {
      TORCH_CHECK(
          false,
          "Not yet supported conversion from IValue to StableIValue for schema type: ",
          type->str());
    }
  }
}

static c10::IValue to_ivalue(
    const c10::TypePtr& type,
    const StableIValue stable_ivalue,
    uint64_t extension_build_version) {
  switch (type->kind()) {
    case c10::TypeKind::TensorType: {
      auto ret_raiiath = torch::aot_inductor::RAIIAtenTensorHandle(
          torch::stable::detail::_to<AtenTensorHandle>(
              stable_ivalue, extension_build_version));
      return (c10::IValue(*torch::aot_inductor::tensor_handle_to_tensor_pointer(
          ret_raiiath.get())));
    }
    case c10::TypeKind::IntType: {
      return c10::IValue(torch::stable::detail::_to<int64_t>(
          stable_ivalue, extension_build_version));
    }
    case c10::TypeKind::FloatType: {
      return c10::IValue(torch::stable::detail::_to<double>(
          stable_ivalue, extension_build_version));
    }
    case c10::TypeKind::BoolType: {
      return c10::IValue(torch::stable::detail::_to<bool>(
          stable_ivalue, extension_build_version));
    }
    case c10::TypeKind::ScalarTypeType: {
      return c10::IValue(torch::stable::detail::_to<c10::ScalarType>(
          stable_ivalue, extension_build_version));
    }
    case c10::TypeKind::DeviceObjType: {
      // Unpack device type and index from StableIValue
      // Lower 32 bits = device index, upper 32 bits = device type
      int32_t device_index = static_cast<int32_t>(
          static_cast<uint32_t>(stable_ivalue & 0xFFFFFFFF));
      c10::DeviceType device_type =
          static_cast<c10::DeviceType>(static_cast<int8_t>(
              static_cast<uint32_t>((stable_ivalue >> 32) & 0xFFFFFFFF)));
      TORCH_CHECK(
          device_index >= std::numeric_limits<int8_t>::min() &&
              device_index <= std::numeric_limits<int8_t>::max(),
          "Device index ",
          device_index,
          " is out of range for int8_t [",
          static_cast<int>(std::numeric_limits<int8_t>::min()),
          ", ",
          static_cast<int>(std::numeric_limits<int8_t>::max()),
          "]");
      return c10::IValue(
          c10::Device(device_type, static_cast<int8_t>(device_index)));
    }
    case c10::TypeKind::LayoutType: {
      return c10::IValue(torch::stable::detail::_to<c10::Layout>(
          stable_ivalue, extension_build_version));
    }
    case c10::TypeKind::MemoryFormatType: {
      return c10::IValue(torch::stable::detail::_to<c10::MemoryFormat>(
          stable_ivalue, extension_build_version));
    }
    case c10::TypeKind::OptionalType: {
      auto inner_type = type->castRaw<at::OptionalType>()->getElementType();

      // ideally, if we had the C++ type corresponding to inner_type, which we
      // will denote as inner_type::t (does not actually exist), we would be
      // able to follow the patterned semantic of every other case here in one
      // line:
      //
      // return
      // c10::IValue(torch::stable::detail::to<std::optional<inner_type::t>>(stable_ivalue));
      //
      // BUT we do NOT have that type inner_type::t readily available, so we
      // will manually unwrap and recursively call. This implementation MUST
      // be kept in sync with the torch::stable::detail::_to<T> function in
      // torch/csrc/stable/library.h
      if (stable_ivalue ==
          torch::stable::detail::_from(std::nullopt, extension_build_version)) {
        return c10::IValue();
      }
      auto sivp = torch::stable::detail::_to<StableIValue*>(
          stable_ivalue, extension_build_version);
      auto ival = to_ivalue(inner_type, *sivp, extension_build_version);
      delete sivp;
      return ival;
    }
    case c10::TypeKind::ListType: {
      auto inner_type = type->castRaw<c10::ListType>()->getElementType();
      auto list_handle = torch::stable::detail::_to<StableListHandle>(
          stable_ivalue, extension_build_version);
      std::vector<StableIValue>* stableivalue_list =
          list_handle_to_list_pointer(list_handle);
      auto ivalue_list = c10::impl::GenericList(inner_type);
      ivalue_list.reserve(stableivalue_list->size());
      for (const auto& elem : *stableivalue_list) {
        ivalue_list.emplace_back(
            to_ivalue(inner_type, elem, extension_build_version));
      }
      TORCH_ERROR_CODE_CHECK(torch_delete_list(list_handle));
      return ivalue_list;
    }
    case c10::TypeKind::StringType: {
      return c10::IValue(torch::stable::detail::_to<std::string>(
          stable_ivalue, extension_build_version));
    }
    case c10::TypeKind::SymIntType: {
      // Treat SymInt as Int for StableIValue <-> IValue conversion
      return to_ivalue(
          c10::IntType::get(), stable_ivalue, extension_build_version);
    }
    default: {
      TORCH_CHECK(
          false,
          "Not yet supported conversion from StableIValue to IValue for schema type: ",
          type->str());
    }
  }
}

class StableIValueBoxedKernel : public c10::OperatorKernel {
 public:
  StableIValueBoxedKernel(
      void (*fn)(StableIValue*, uint64_t, uint64_t),
      uint64_t extension_build_version)
      : fn_(fn), extension_build_version_(extension_build_version) {}

  void operator()(
      const c10::OperatorHandle& op,
      c10::DispatchKeySet keyset,
      torch::jit::Stack* stack) {
    const auto& schema = op.schema();
    const auto num_returns = schema.returns().size();
    const auto num_arguments = schema.arguments().size();

    auto ministack =
        std::make_unique<StableIValue[]>(std::max(num_arguments, num_returns));

    for (const auto idx : c10::irange(num_arguments)) {
      const auto ministack_idx = num_arguments - idx - 1;
      const c10::TypePtr& arg_type =
          schema.arguments()[ministack_idx].real_type();
      ministack[ministack_idx] = from_ivalue(
          arg_type, torch::jit::pop(stack), extension_build_version_);
    }

    // boxed function is going to take a stack of StableIValues, cast them to
    // our schema values, and run the function and modify the StableIValue stack
    fn_(ministack.get(), num_arguments, num_returns);

    // read the output from the end of the stack and wrap that back into
    // IValue from StableIValue
    for (size_t idx = 0; idx < num_returns; idx++) {
      const c10::TypePtr& ret_type = schema.returns()[idx].real_type();
      torch::jit::push(
          stack, to_ivalue(ret_type, ministack[idx], extension_build_version_));
    }
  }

 private:
  void (*fn_)(StableIValue*, uint64_t, uint64_t);
  uint64_t extension_build_version_;
};

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_library_impl(
    TorchLibraryHandle self,
    const char* name,
    void (*fn)(StableIValue*, uint64_t, uint64_t)) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    reinterpret_cast<torch::Library*>(self)->impl(
        name,
        torch::CppFunction::makeFromBoxedFunctor(
            std::make_unique<StableIValueBoxedKernel>(fn, TORCH_ABI_VERSION)));
  });
}

// Helper function to parse device string using c10::Device
// Returns device type and index
AOTI_TORCH_EXPORT AOTITorchError torch_parse_device_string(
    const char* device_string,
    uint32_t* out_device_type,
    int32_t* out_device_index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    c10::Device device{std::string(device_string)};
    *out_device_type = static_cast<uint32_t>(device.type());
    *out_device_index = static_cast<int32_t>(device.index());
  });
}

// Version-aware variant of aoti_torch_library_impl that takes an
// extension_build_version parameter for backward compatibility
AOTI_TORCH_EXPORT AOTITorchError torch_library_impl(
    TorchLibraryHandle self,
    const char* name,
    void (*fn)(StableIValue*, uint64_t, uint64_t),
    uint64_t extension_build_version) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    reinterpret_cast<torch::Library*>(self)->impl(
        name,
        torch::CppFunction::makeFromBoxedFunctor(
            std::make_unique<StableIValueBoxedKernel>(
                fn, extension_build_version)));
  });
}

AOTITorchError aoti_torch_call_dispatcher(
    const char* opName,
    const char* overloadName,
    StableIValue* stack) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    const auto op =
        c10::Dispatcher::singleton().findSchemaOrThrow(opName, overloadName);
    const auto& schema = op.schema();
    const auto num_returns = schema.returns().size();
    const auto num_arguments = schema.arguments().size();

    torch::jit::Stack ivalue_stack;
    // we will only need max(num_args, num_returns)
    ivalue_stack.reserve(std::max(num_arguments, num_returns));

    // convert StableIValue stack to c10::IValue stack
    for (const auto idx : c10::irange(num_arguments)) {
      auto stable_ivalue = stack[idx];
      auto arg_type = schema.arguments()[idx].real_type();
      torch::jit::push(
          ivalue_stack, to_ivalue(arg_type, stable_ivalue, TORCH_ABI_VERSION));
    }

    op.callBoxed(ivalue_stack);

    // there should then be num_returns IValues on the stack, which
    // we will convert to StableIValue and repopulate user input stack
    for (const auto idx : c10::irange(num_returns)) {
      const auto stack_idx = num_returns - idx - 1;
      const c10::TypePtr& ret_type = schema.returns()[idx].real_type();
      stack[stack_idx] = from_ivalue(
          ret_type, torch::jit::pop(ivalue_stack), TORCH_ABI_VERSION);
    }
  });
}

// Schema Adapter Infrastructure
// SchemaAdapterRegistry contains the adapters registered via
// register_schema_adapter that define how to convert the StableIValue argument
// stack to an IValue stack when changes are made to the schema of an ATen
// function. This should only be relevant in the context of calling
// torch_call_dispatcher.

// Currently this only adapts the argument stack.
// C++ default argument resolution will happen at compile time in the
// torch/csrc/stable/ops.h header, so extensions always pass complete argument
// lists for the version they build against's schema. As such, this is only
// needed if a new argument is added to the schema
//
// This is not declared in the stable shim.h,
// so we **do not make any guarantees that the signature of this will not
// change**. If there is a need to define similar infrastructure for the returns
// of an aten function we can update this.

namespace {
using SchemaAdapterFn = std::function<torch::jit::Stack(
    const c10::FunctionSchema& current_schema,
    const StableIValue* extension_stack,
    uint64_t extension_build_version)>;

// Global registry for schema adapters
class SchemaAdapterRegistry {
 private:
  std::unordered_map<
      std::string,
      std::vector<std::pair<uint64_t, SchemaAdapterFn>>>
      adapters_;

 public:
  static SchemaAdapterRegistry& instance() {
    static SchemaAdapterRegistry registry;
    return registry;
  }

  void register_adapter(
      const std::string& op_name,
      uint64_t
          applies_to_versions_below, // versions below this need the adapter
      SchemaAdapterFn adapter) {
    adapters_[op_name].emplace_back(applies_to_versions_below, adapter);
    // Sort by version ascending - this allows us to find the first (most
    // specific) match
    std::sort(
        adapters_[op_name].begin(),
        adapters_[op_name].end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });
  }

  std::optional<SchemaAdapterFn> get_adapter(
      const std::string& op_name,
      uint64_t extension_version) {
    auto it = adapters_.find(op_name);
    if (it == adapters_.end())
      return std::nullopt;

    // Find the first adapter that applies (most specific due to ascending sort)
    for (const auto& [applies_to_versions_below, adapter] : it->second) {
      if (extension_version < applies_to_versions_below) {
        return adapter;
      }
    }
    return std::nullopt;
  }
};

// Internal API for registering adapters that define how to convert the
// StableIValue  **argument** stack to an IValue stack when changes are
// made to the schema of a function. adapter_fn will be used if
// extension_build_version < applies_to_versions_below.
[[maybe_unused]] AOTITorchError register_schema_adapter(
    const char* op_name,
    uint64_t applies_to_versions_below,
    SchemaAdapterFn adapter_fn) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto& registry = SchemaAdapterRegistry::instance();
    registry.register_adapter(
        std::string(op_name), applies_to_versions_below, std::move(adapter_fn));
  });
}

} // namespace

// Function to register test schema adapters for _test_schema_upgrader
// This demonstrates the adapter registration pattern (internal use only)
static AOTITorchError _register_adapters() {
  // ** Schema adapters should be registered here**
  // Refer to https://github.com/pytorch/pytorch/pull/165284/ for an example.
  //
  // if (auto err = register_schema_adapter(
  //         "aten::your_op",
  //         VERSION_FOO, // applies to versions < VERSION_FOO
  //         adapt_v1_to_vfoo)) {
  //   return err;
  // }
  return AOTI_TORCH_SUCCESS;
}

// Static initialization to automatically register test adapters
static struct AdapterInitializer {
  AdapterInitializer() {
    // Register the test adapters when the library loads
    _register_adapters();
  }
} adapter_initializer;

AOTI_TORCH_EXPORT AOTITorchError torch_call_dispatcher(
    const char* opName,
    const char* overloadName,
    StableIValue* stack,
    // version of stable headers used to build the extension: necessary for
    // applying schema adapters
    uint64_t extension_build_version) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    const auto op =
        c10::Dispatcher::singleton().findSchemaOrThrow(opName, overloadName);
    const auto& schema = op.schema();
    const auto num_returns = schema.returns().size();
    const auto num_arguments = schema.arguments().size();

    torch::jit::Stack ivalue_stack;
    auto& registry = SchemaAdapterRegistry::instance();

    // Check if we need an adapter for this operation
    if (auto adapter = registry.get_adapter(opName, extension_build_version)) {
      // Use adapter to create IValue stack
      ivalue_stack = (*adapter)(schema, stack, extension_build_version);
    } else {
      // No adapter needed - implementation matches aoti_torch_call_dispatcher
      ivalue_stack.reserve(std::max(num_arguments, num_returns));
      for (const auto idx : c10::irange(num_arguments)) {
        auto stable_ivalue = stack[idx];
        auto arg_type = schema.arguments()[idx].real_type();
        torch::jit::push(
            ivalue_stack,
            to_ivalue(arg_type, stable_ivalue, extension_build_version));
      }
    }

    op.callBoxed(ivalue_stack);

    // there should then be num_returns IValues on the stack, which
    // we will convert to StableIValue and repopulate user input stack
    for (const auto idx : c10::irange(num_returns)) {
      const auto stack_idx = num_returns - idx - 1;
      const c10::TypePtr& ret_type = schema.returns()[idx].real_type();
      stack[stack_idx] = from_ivalue(
          ret_type, torch::jit::pop(ivalue_stack), extension_build_version);
    }
  });
}

AOTI_TORCH_EXPORT AOTITorchError torch_parallel_for(
    int64_t begin,
    int64_t end,
    int64_t grain_size,
    ParallelFunc func,
    void* ctx) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::parallel_for(
        begin, end, grain_size, [func, ctx](int64_t begin, int64_t end) {
          func(begin, end, ctx);
        });
  });
}

AOTI_TORCH_EXPORT AOTITorchError
torch_get_thread_idx(uint32_t* out_thread_idx) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *out_thread_idx = static_cast<uint32_t>(at::get_thread_num()); });
}

AOTI_TORCH_EXPORT AOTITorchError
torch_get_num_threads(uint32_t* out_num_threads) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *out_num_threads = static_cast<uint32_t>(at::get_num_threads()); });
}

AOTI_TORCH_EXPORT AOTITorchError
torch_get_const_data_ptr(AtenTensorHandle tensor, const void** ret_data_ptr) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t =
        torch::aot_inductor::tensor_handle_to_tensor_pointer(tensor);
    *ret_data_ptr = t->const_data_ptr();
  });
}

AOTI_TORCH_EXPORT AOTITorchError
torch_get_mutable_data_ptr(AtenTensorHandle tensor, void** ret_data_ptr) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t =
        torch::aot_inductor::tensor_handle_to_tensor_pointer(tensor);
    *ret_data_ptr = t->mutable_data_ptr();
  });
}

AOTI_TORCH_EXPORT AOTITorchError
torch_new_string_handle(const char* data, size_t length, StringHandle* handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto str_ptr = new std::string(data, length);
    *handle = reinterpret_cast<StringHandle>(str_ptr);
  });
}

AOTI_TORCH_EXPORT AOTITorchError torch_delete_string(StringHandle handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto str_ptr = reinterpret_cast<std::string*>(handle);
    delete str_ptr;
  });
}

AOTI_TORCH_EXPORT AOTITorchError
torch_string_length(StringHandle handle, size_t* length) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto str_ptr = reinterpret_cast<std::string*>(handle);
    *length = str_ptr->length();
  });
}

AOTI_TORCH_EXPORT AOTITorchError
torch_string_c_str(StringHandle handle, const char** data) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto str_ptr = reinterpret_cast<std::string*>(handle);
    *data = str_ptr->c_str();
  });
}

AOTI_TORCH_EXPORT AOTITorchError
torch_set_requires_grad(AtenTensorHandle tensor, bool requires_grad) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::Tensor* t =
        torch::aot_inductor::tensor_handle_to_tensor_pointer(tensor);
    t->set_requires_grad(requires_grad);
  });
}

// Most other dtypes defined in torch/csrc/inductor/aoti_torch/shim_common.cpp
#define TORCH_DTYPE_IMPL(dtype, stype)                    \
  AOTI_TORCH_EXPORT int32_t torch_dtype_##dtype() {       \
    return (int32_t)torch::headeronly::ScalarType::stype; \
  }

TORCH_DTYPE_IMPL(float8_e8m0fnu, Float8_e8m0fnu)
TORCH_DTYPE_IMPL(float4_e2m1fn_x2, Float4_e2m1fn_x2)

#undef TORCH_DTYPE_IMPL

AOTI_TORCH_EXPORT AOTITorchError torch_from_blob(
    void* data,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorHandle* ret_new_tensor,
    int32_t layout,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size,
    void (*deleter)(void*)) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    c10::IntArrayRef sizes(sizes_ptr, ndim);
    c10::IntArrayRef strides(strides_ptr, ndim);
    c10::Device device(static_cast<c10::DeviceType>(device_type), device_index);
    c10::TensorOptions options = c10::TensorOptions().device(device).dtype(
        static_cast<c10::ScalarType>(dtype));
    at::Tensor tensor;
    if (data != nullptr) {
      if (deleter != nullptr) {
        tensor = at::for_blob(data, sizes)
                     .strides(strides)
                     .storage_offset(storage_offset)
                     .deleter(deleter)
                     .options(options)
                     .make_tensor();
      } else {
        tensor = at::for_blob(data, sizes)
                     .strides(strides)
                     .storage_offset(storage_offset)
                     .options(options)
                     .make_tensor();
      }
    } else {
      tensor = at::empty_strided(sizes, strides, options);
    }
    *ret_new_tensor = torch::aot_inductor::new_tensor_handle(std::move(tensor));
  });
}

AOTI_TORCH_EXPORT AOTITorchError torch_from_blob_v2(
    void* data,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    AtenTensorHandle* ret_new_tensor,
    int32_t layout,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size,
    void (*deleter_callback)(void* data, void* ctx),
    void* deleter_ctx) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    c10::IntArrayRef sizes(sizes_ptr, ndim);
    c10::IntArrayRef strides(strides_ptr, ndim);
    c10::Device device(static_cast<c10::DeviceType>(device_type), device_index);
    c10::TensorOptions options = c10::TensorOptions().device(device).dtype(
        static_cast<c10::ScalarType>(dtype));
    at::Tensor tensor;
    if (data != nullptr) {
      if (deleter_callback != nullptr) {
        // Combine the two-arg C callback and its context into a single-arg
        // C++ callable that at::for_blob().deleter() expects.
        auto wrapped_deleter = [deleter_callback, deleter_ctx](void* data) {
          deleter_callback(data, deleter_ctx);
        };
        tensor = at::for_blob(data, sizes)
                     .strides(strides)
                     .storage_offset(storage_offset)
                     .deleter(wrapped_deleter)
                     .options(options)
                     .make_tensor();
      } else {
        tensor = at::for_blob(data, sizes)
                     .strides(strides)
                     .storage_offset(storage_offset)
                     .options(options)
                     .make_tensor();
      }
    } else {
      tensor = at::empty_strided(sizes, strides, options);
    }
    *ret_new_tensor = torch::aot_inductor::new_tensor_handle(std::move(tensor));
  });
}
