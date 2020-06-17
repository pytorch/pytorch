#pragma once

#include <c10/util/Metaprogramming.h>
#include <c10/util/TypeList.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/CompileTimeFunctionPointer.h>

// This file defines hacky_wrapper_for_legacy_signatures, which takes a kernel written in a legacy way
// (e.g. with TensorOptions packed) and wraps it into a kernel with the signature expected by
// the PyTorch operator library. The intention is to ultimately rewrite kernels to take the new signature
// and then delete this file. This transition process can happen kernel-by-kernel, since this wrapper
// is a no-op for kernels that already have a non-legacy signature.

namespace c10 {
namespace impl {

inline c10::optional<MemoryFormat> process_memory_format(const TensorOptions& options, c10::optional<MemoryFormat> memory_format) {
    TORCH_CHECK(
        !(options.has_memory_format() && memory_format.has_value()),
        "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
        "the redundant setter.");
    if (memory_format.has_value()) {
        return memory_format;
    } else {
        return options.memory_format_opt();
    }
}

namespace detail {

// with_scattered_tensor_options takes a function pointer that potentially takes a TensorOptions argument.
// If it does, then it creates a new function pointer that takes scattered arguments, internally
// gathers those arguments, and then calls the underlying function pointer. If the underlying
// function pointer does not take a TensorOptions argument, it is passed through unmodified.

template<class Type, class Enable = void> struct is_tensoroptions_arg : std::false_type {};
template<class Type> struct is_tensoroptions_arg<Type, std::enable_if_t<std::is_same<TensorOptions, std::decay_t<Type>>::value>> : std::true_type {};
template<class Type>
using is_tensoroptions_arg_t = typename is_tensoroptions_arg<Type>::type;

template<class FuncType>
inline constexpr bool has_tensoroptions_arg() {
    using parameter_types = typename guts::infer_function_traits_t<FuncType>::parameter_types;
    constexpr size_t num_tensoroptions_args = guts::typelist::count_if<is_tensoroptions_arg_t, parameter_types>::value;
    static_assert(num_tensoroptions_args <= 1, "Function has multiple TensorOptions parameters. We support at most one.");
    return num_tensoroptions_args > 0;
}

static_assert(has_tensoroptions_arg<int (int64_t, const TensorOptions&)>(), "");
static_assert(has_tensoroptions_arg<int (int64_t, TensorOptions)>(), "");
static_assert(!has_tensoroptions_arg<int (int64_t, std::string)>(), "");

template<class FuncPtr, class ParametersBeforeTensorOptions, class ParametersAfterTensorOptions> struct with_scattered_tensor_options_;

template<class FuncPtr, class Enable = void>
struct with_scattered_tensor_options final {};

template<class UnderlyingFuncPtr>
struct with_scattered_tensor_options<UnderlyingFuncPtr, std::enable_if_t<!has_tensoroptions_arg<typename UnderlyingFuncPtr::FuncType>()>> final {
    // FuncType does not have TensorOptions arguments.
    // Don't wrap anything but just return the base pointer.
    using FuncPtr = UnderlyingFuncPtr;
};

template<class UnderlyingFuncPtr>
struct with_scattered_tensor_options<UnderlyingFuncPtr, std::enable_if_t<has_tensoroptions_arg<typename UnderlyingFuncPtr::FuncType>()>> final {
private:
    // FuncType has TensorOptions arguments.
    // Return a function pointer to a wrapper function that replaces those with expanded arguments.
    using gathered_parameter_types = typename guts::infer_function_traits_t<typename UnderlyingFuncPtr::FuncType>::parameter_types;
    static constexpr size_t tensoroptions_arg_index =
        guts::typelist::find_if<
            gathered_parameter_types,
            is_tensoroptions_arg_t
        >::value;

    using parameters_before_tensoroptions =
        guts::typelist::take_t<gathered_parameter_types, tensoroptions_arg_index>;
    using parameters_after_tensoroptions =
        guts::typelist::drop_t<gathered_parameter_types, tensoroptions_arg_index + 1>;

    using wrapper = with_scattered_tensor_options_<UnderlyingFuncPtr, parameters_before_tensoroptions, parameters_after_tensoroptions>;
public:
    using FuncPtr = TORCH_FN_TYPE(&wrapper::wrapper);
};

template<class FuncPtr, class... ParametersBeforeTensorOptions, class... ParametersAfterTensorOptions>
struct with_scattered_tensor_options_<FuncPtr, guts::typelist::typelist<ParametersBeforeTensorOptions...>, guts::typelist::typelist<ParametersAfterTensorOptions...>> final {
    static decltype(auto) wrapper(
                ParametersBeforeTensorOptions... parameters_before,
                optional<ScalarType> scalar_type,
                optional<Layout> layout,
                optional<Device> device,
                optional<bool> pin_memory,
                ParametersAfterTensorOptions... parameters_after) {
        return (*FuncPtr::func_ptr())(
            std::forward<ParametersBeforeTensorOptions>(parameters_before)...,
            TensorOptions().dtype(scalar_type).device(device).layout(layout).pinned_memory(pin_memory),
            std::forward<ParametersAfterTensorOptions>(parameters_after)...
        );
    }
};

}

template<class FuncPtr>
constexpr auto hacky_wrapper_for_legacy_signatures(FuncPtr) {
    return typename detail::with_scattered_tensor_options<FuncPtr>::FuncPtr();
};

}
}
