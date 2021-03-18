#pragma once

#include <c10/util/Metaprogramming.h>
#include <c10/util/TypeList.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/CompileTimeFunctionPointer.h>
#include <ATen/Tensor.h>

// This file defines hacky_wrapper_for_legacy_signatures, which takes a kernel written in a legacy way
// (e.g. with TensorOptions packed) and wraps it into a kernel with the signature expected by
// the PyTorch operator library. The intention is to ultimately rewrite kernels to take the new signature
// and then delete this file. This transition process can happen kernel-by-kernel, since this wrapper
// is a no-op for kernels that already have a non-legacy signature.

namespace c10 {
namespace impl {

inline c10::optional<MemoryFormat> check_tensor_options_and_extract_memory_format(const TensorOptions& options, c10::optional<MemoryFormat> memory_format) {
    TORCH_CHECK(options.requires_grad_opt() == c10::nullopt || options.requires_grad_opt().value() == false,
        "Operators taking TensorOptions cannot take a TensorOptions with "
        "options.requires_grad set as true. This isn't implemented yet.");
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

// sanity checks
static_assert(has_tensoroptions_arg<int (int64_t, const TensorOptions&)>(), "");
static_assert(has_tensoroptions_arg<int (int64_t, TensorOptions)>(), "");
static_assert(!has_tensoroptions_arg<int (int64_t, std::string)>(), "");

template<class FuncPtr, class ParametersBeforeTensorOptions, class ParametersAfterTensorOptions> struct with_scattered_tensor_options_impl_;

template<class FuncPtr, class Enable = void>
struct with_scattered_tensor_options_impl final {};

template<class UnderlyingFuncPtr>
struct with_scattered_tensor_options_impl<UnderlyingFuncPtr, std::enable_if_t<!has_tensoroptions_arg<typename UnderlyingFuncPtr::FuncType>()>> final {
    // FuncType does not have TensorOptions arguments.
    // Don't wrap anything but just return the base pointer.
    using FuncPtr = UnderlyingFuncPtr;
};

template<class UnderlyingFuncPtr>
struct with_scattered_tensor_options_impl<UnderlyingFuncPtr, std::enable_if_t<has_tensoroptions_arg<typename UnderlyingFuncPtr::FuncType>()>> final {
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

    using wrapper = with_scattered_tensor_options_impl_<UnderlyingFuncPtr, parameters_before_tensoroptions, parameters_after_tensoroptions>;
public:
    using FuncPtr = TORCH_FN_TYPE(&wrapper::wrapper);
};

// This template generates the actual wrapper. It is only invoked when we
// already know that we have an op with a TensorOptions argument and the
// argument list is already parsed and passed in to this template separately
// using argument packs ParametersBeforeTensorOptions and ParametersAfterTensorOptions.
template<class FuncPtr, class... ParametersBeforeTensorOptions, class... ParametersAfterTensorOptions>
struct with_scattered_tensor_options_impl_<FuncPtr, guts::typelist::typelist<ParametersBeforeTensorOptions...>, guts::typelist::typelist<ParametersAfterTensorOptions...>> final {
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

/**
 * Take a kernel function that has a `TensorOptions` argument and
 * return a new kernel function that has `optional<ScalarType>,
 * optional<Layout>, optional<Device>, optional<bool>` arguments
 * instead, packs them into a `TensorOptions` struct and
 * calls the original kernel function.
 */
template<class FuncPtr>
constexpr auto with_scattered_tensor_options(FuncPtr) {
    return typename with_scattered_tensor_options_impl<FuncPtr>::FuncPtr();
}

// make_optional_tensor_explicit takes an argument of any type T and
// a KernelType.
//  - T: The type the op gets called with
//  - KernelType: The type the kernel function expects
// Those types are usually the same but they are allowed to differ when
// KernelType == `Tensor` and T == `optional<Tensor>` because that just
// means the kernel is written in the legacy way and we want to wrap it.
// In this case, make_optional_tensor_explicit maps any `optional<Tensor>`
// to a `Tensor` (mapping `nullopt` to undefined tensor).
// Everything else is passed through unmodified.
template<class KernelType>
struct make_optional_tensor_explicit final {
    // SFINAE for KernelType != `Tensor`
    template<class T>
    static decltype(auto) call(T&& arg) {
        // pass through everything unmodified
        return std::forward<T>(arg);
    }
};

template<>
struct make_optional_tensor_explicit<at::Tensor> final {
    // SFINAE for KernelType == `Tensor`
    template<class _Tensor, std::enable_if_t<std::is_same<std::remove_cv_t<std::remove_reference_t<_Tensor>>, at::Tensor>::value, int> = 0>
    static decltype(auto) call(_Tensor&& arg) {
        // pass through arguments that already are `Tensor` unmodified
        return std::forward<_Tensor>(arg);
    }

    template<class _OptTensor, std::enable_if_t<std::is_same<std::remove_cv_t<std::remove_reference_t<_OptTensor>>, optional<at::Tensor>>::value, int> = 0>
    static at::Tensor call(_OptTensor&& arg) {
        // map `optional<Tensor>` to `Tensor`
        if (arg.has_value()) {
            return *std::forward<_OptTensor>(arg);
        } else {
            return at::Tensor(); // undefined tensor
        }
    }
};

// This template generates the actual wrapper. It is only invoked when we
// already know that we have an op with an optional<Tensor> argument and
// we need to wrap it.
template<class TargetSignature, class KernelSignature, class KernelFunc>
struct with_explicit_optional_tensors_ final {};

template<class Return, class... TargetSignatureArgs, class... KernelSignatureArgs, Return(*KernelFunc)(KernelSignatureArgs...)>
struct with_explicit_optional_tensors_<Return (TargetSignatureArgs...), Return(KernelSignatureArgs...), TORCH_FN_TYPE(KernelFunc)> final {
    static Return wrapper(TargetSignatureArgs... args) {
        return (*KernelFunc)(make_optional_tensor_explicit<std::remove_cv_t<std::remove_reference_t<KernelSignatureArgs>>>::call(
                std::forward<TargetSignatureArgs>(args)
            )...);
    }
};

template<class T>
constexpr bool _is_optional_tensor_arg() {
    return std::is_same<c10::optional<at::Tensor>, std::decay_t<T>>::value;
}
template<class T> using is_optional_tensor_arg = guts::bool_constant<_is_optional_tensor_arg<T>()>;

/**
 * Take a kernel function that has a number of `Tensor` arguments
 * and take in a `TargetSignature` that must match, but is allowed
 * to take `optional<Tensor>` in place of some or all of the `Tensor`
 * arguments. Returns a new kernel function that has `optional<Tensor>`
 * in those locations, unwraps them to `Tensor` (potentially undefined tensor)
 * and calls the original kernel function.
 */
template<class TargetSignature, class KernelFunc, std::enable_if_t<
    guts::typelist::true_for_any_type<is_optional_tensor_arg, typename guts::infer_function_traits_t<TargetSignature>::parameter_types>::value, int> = 0>
constexpr auto with_explicit_optional_tensors(KernelFunc kernel_func) {
    // SFINAE case for kernels that have optional tensor arguments.
    // Wrap them to unpack the optionals before calling the kernel
    return TORCH_FN((&with_explicit_optional_tensors_<TargetSignature, typename KernelFunc::FuncType, KernelFunc>::wrapper));
}

template<class TargetSignature, class KernelFunc, std::enable_if_t<
    !guts::typelist::true_for_any_type<is_optional_tensor_arg, typename guts::infer_function_traits_t<TargetSignature>::parameter_types>::value, int> = 0>
constexpr auto with_explicit_optional_tensors(KernelFunc kernel_func) {
    // SFINAE case for kernels that don't have optional tensor arguments.
    // Don't wrap them but just use the kernel directly.
    return kernel_func;
}

template<class Arg> constexpr bool is_out_argument_() {
    return std::is_same<at::Tensor&, Arg>::value;
}
template<class Arg> using is_out_argument = guts::bool_constant<is_out_argument_<Arg>()>;

template<size_t NumOutParameters, class KernelFunc>
struct with_out_arguments_reordered_impl final {
private:
    // For an example op
    //   > aten::example(Tensor a, int64_t b, int64_t c, Tensor(a!) out_d, Tensor(b!) out_e) -> (Tensor(a!), Tensor(b!))
    // we get a KernelFunc
    //   > KernelFunc = std::tuple<Tensor, Tensor> example(Tensor& out_d, Tensor& out_e, const Tensor& a, int64_t b, int64_t c)
    //   > NumOutParameters = 2
    // with the out arguments at the front, and reorder that into
    //   > std::tuple<Tensor, Tensor> example(const Tensor& a, int64_t b, int64_t c, Tensor& out_d, Tensor& out_e)
    // where the out arguments are in the back.

    using kernel_signature_traits = guts::infer_function_traits_t<typename KernelFunc::FuncType>;

    // Assert that the KernelFunc is what we expect. The following block is
    // not strictly necessary for the metaprogramming here, it's just a check.
    static_assert(
        guts::typelist::all<
            is_out_argument,
            guts::typelist::take_t<
                typename kernel_signature_traits::parameter_types,
                NumOutParameters
            >
        >::value,
        "The kernel function has the wrong number of leading Tensor& arguments to match the out arguments in the JIT signature"
    );

    static constexpr size_t num_parameters = kernel_signature_traits::number_of_parameters;
    static constexpr size_t num_nonout_parameters = num_parameters - NumOutParameters;

    // kernel_to_schema_permutation_indices contains a mapping from argument index in KernelFunc to the corresponding
    // argument index in the schema.
    // For the aten::example op, that'll be
    //  > kernel_to_schema_permutation_indices = [3, 4, 0, 1, 2]
    // Interpreted as a mapping, this means
    //  - argument 0 in KernelFunc maps to argument 3 in the schema,
    //  - argument 1 in KernelFunc maps to argument 4 in the schema,
    //  - argument 2 in KernelFunc maps to argument 0 in the schema,
    //  - ...
    // We can use this as a permutation function to reorder types or values correspondingly
    using kernel_to_schema_permutation_indices = guts::concat_iseq_t<
        guts::make_offset_index_sequence<num_nonout_parameters, NumOutParameters>,
        std::make_index_sequence<num_nonout_parameters>
    >;

    // For types, we need the inverse permutation because parameters (i.e. types) and arguments (i.e. values)
    // need to be mapped in inverted directions. For types, we generate the schema order types from
    // the KernelFunction types, but for arguments we get schema order arguments and need to generate
    // the KernelFunction arguments.
    // That's why in this reordering, we use NumOutParameters instead of the num_nonout_parameters we used above.
    using schema_parameters = guts::typelist::concat_t<
        guts::typelist::drop_t<typename kernel_signature_traits::parameter_types, NumOutParameters>,
        guts::typelist::take_t<typename kernel_signature_traits::parameter_types, NumOutParameters>
    >;

    template<class Return, class SchemaParameterList, class KernelParameterList, class IndexPermutation>
    struct wrapper_;
    template<class Return, class... SchemaParameters, class... KernelParameters, size_t... Indices>
    struct wrapper_<Return, guts::typelist::typelist<SchemaParameters...>, guts::typelist::typelist<KernelParameters...>, std::index_sequence<Indices...>> {
        static Return call(SchemaParameters... args) {
            // call through to KernelFunc but reorder arguments as determined
            // by the permutation we calculated above.
            return (*KernelFunc::func_ptr())(
                std::forward<KernelParameters>(
                    std::get<Indices>(
                        std::tuple<std::add_lvalue_reference_t<SchemaParameters>...>(args...)
                    )
                )...
            );
        }
    };

public:
    using wrapper = wrapper_<typename kernel_signature_traits::return_type, schema_parameters, typename kernel_signature_traits::parameter_types, kernel_to_schema_permutation_indices>;
};


/**
 * Take a kernel function that has a number of `Tensor`, `const Tensor&` or `Tensor&` arguments
 * where all `Tensor&` arguments are at the beginning, and take NumOutParameters.
 * Create a wrapper function that has `NumOutParameters` `Tensor&` arguments at the end
 * and calls through the underlying kernel function by reordering them to the front.
 */
template<size_t NumOutParameters, class KernelFunc, std::enable_if_t<(NumOutParameters > 0), int> = 0>
constexpr auto with_out_arguments_reordered(KernelFunc kernel_func) {
    // SFINAE case for kernels that have out tensor arguments.
    // Wrap them and reorder the arguments.
    using impl = with_out_arguments_reordered_impl<NumOutParameters, KernelFunc>;
    return TORCH_FN((&impl::wrapper::call));
}

template<size_t NumOutParameters, class KernelFunc, std::enable_if_t<(NumOutParameters == 0), int> = 0>
constexpr auto with_out_arguments_reordered(KernelFunc kernel_func) {
    // SFINAE case for kernels that don't have out tensor arguments.
    // Don't wrap them but just use the kernel directly.
    return kernel_func;
}

}

template<class TargetSignature, size_t NumOutParameters, class FuncPtr>
constexpr auto hacky_wrapper_for_legacy_signatures(FuncPtr kernel_func) {
    auto with_scattered_tensor_options = detail::with_scattered_tensor_options(kernel_func);
    auto with_out_arguments_reordered = detail::with_out_arguments_reordered<NumOutParameters>(with_scattered_tensor_options);
    auto result = detail::with_explicit_optional_tensors<TargetSignature>(with_out_arguments_reordered);
    static_assert(std::is_same<TargetSignature, typename decltype(result)::FuncType>::value, "Generated signature doesn't match the expected one.");
    return result;
};

}
}
