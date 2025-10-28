#include <c10/core/DispatchKey.h>
#include <c10/util/Exception.h>
#include <torch/csrc/inductor/aoti_runtime/utils.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/stable/library.h>
#include <torch/library.h>

static StableIValue from_ivalue(
    const c10::TypePtr& type,
    const c10::IValue& ivalue) {
  switch (type->kind()) {
    case c10::TypeKind::TensorType: {
      AtenTensorHandle ath = torch::aot_inductor::new_tensor_handle(
          std::move(const_cast<at::Tensor&>(ivalue.toTensor())));
      return torch::stable::detail::from(ath);
    }
    case c10::TypeKind::IntType: {
      return torch::stable::detail::from(ivalue.toInt());
    }
    case c10::TypeKind::FloatType: {
      return torch::stable::detail::from(ivalue.toDouble());
    }
    case c10::TypeKind::BoolType: {
      return torch::stable::detail::from(ivalue.toBool());
    }
    case c10::TypeKind::ScalarTypeType: {
      return torch::stable::detail::from(ivalue.toScalarType());
    }
    case c10::TypeKind::DeviceObjType: {
      return torch::stable::detail::from(ivalue.toDevice());
    }
    case c10::TypeKind::LayoutType: {
      return torch::stable::detail::from(ivalue.toLayout());
    }
    case c10::TypeKind::MemoryFormatType: {
      return torch::stable::detail::from(ivalue.toMemoryFormat());
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
        return torch::stable::detail::from(std::nullopt);
      }
      StableIValue* sivp = new StableIValue(from_ivalue(inner_type, ivalue));
      return torch::stable::detail::from(sivp);
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
    const StableIValue stable_ivalue) {
  switch (type->kind()) {
    case c10::TypeKind::TensorType: {
      auto ret_raiiath = torch::aot_inductor::RAIIAtenTensorHandle(
          torch::stable::detail::to<AtenTensorHandle>(stable_ivalue));
      return (c10::IValue(*torch::aot_inductor::tensor_handle_to_tensor_pointer(
          ret_raiiath.get())));
    }
    case c10::TypeKind::IntType: {
      return c10::IValue(torch::stable::detail::to<int64_t>(stable_ivalue));
    }
    case c10::TypeKind::FloatType: {
      return c10::IValue(torch::stable::detail::to<double>(stable_ivalue));
    }
    case c10::TypeKind::BoolType: {
      return c10::IValue(torch::stable::detail::to<bool>(stable_ivalue));
    }
    case c10::TypeKind::ScalarTypeType: {
      return c10::IValue(
          torch::stable::detail::to<c10::ScalarType>(stable_ivalue));
    }
    case c10::TypeKind::DeviceObjType: {
      return c10::IValue(torch::stable::detail::to<c10::Device>(stable_ivalue));
    }
    case c10::TypeKind::LayoutType: {
      return c10::IValue(torch::stable::detail::to<c10::Layout>(stable_ivalue));
    }
    case c10::TypeKind::MemoryFormatType: {
      return c10::IValue(
          torch::stable::detail::to<c10::MemoryFormat>(stable_ivalue));
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
      // be kept in sync with the torch::stable::detail::to<T> function in
      // torch/csrc/stable/stableivalue_conversions.h
      if (stable_ivalue == torch::stable::detail::from(std::nullopt)) {
        return c10::IValue();
      }
      auto sivp = torch::stable::detail::to<StableIValue*>(stable_ivalue);
      auto ival = to_ivalue(inner_type, *sivp);
      delete sivp;
      return ival;
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
  StableIValueBoxedKernel(void (*fn)(StableIValue*, uint64_t, uint64_t))
      : fn_(fn) {}

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
      const c10::TypePtr& arg_type = schema.arguments()[ministack_idx].type();
      ministack[ministack_idx] = from_ivalue(arg_type, torch::jit::pop(stack));
    }

    // boxed function is going to take a stack of StableIValues, cast them to
    // our schema values, and run the function and modify the StableIValue stack
    fn_(ministack.get(), num_arguments, num_returns);

    // read the output from the end of the stack and wrap that back into
    // IValue from StableIValue
    for (size_t idx = 0; idx < num_returns; idx++) {
      const c10::TypePtr& ret_type = schema.returns()[idx].type();
      torch::jit::push(stack, to_ivalue(ret_type, ministack[idx]));
    }
  }

 private:
  void (*fn_)(StableIValue*, uint64_t, uint64_t);
};

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_library_impl(
    TorchLibraryHandle self,
    const char* name,
    void (*fn)(StableIValue*, uint64_t, uint64_t)) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    reinterpret_cast<torch::Library*>(self)->impl(
        name,
        torch::CppFunction::makeFromBoxedFunctor(
            std::make_unique<StableIValueBoxedKernel>(fn)));
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
      auto arg_type = schema.arguments()[idx].type();
      torch::jit::push(ivalue_stack, to_ivalue(arg_type, stable_ivalue));
    }

    op.callBoxed(ivalue_stack);

    // there should then be num_returns IValues on the stack, which
    // we will convert to StableIValue and repopulate user input stack
    for (const auto idx : c10::irange(num_returns)) {
      const auto stack_idx = num_returns - idx - 1;
      const c10::TypePtr& ret_type = schema.returns()[idx].type();
      stack[stack_idx] = from_ivalue(ret_type, torch::jit::pop(ivalue_stack));
    }
  });
}
