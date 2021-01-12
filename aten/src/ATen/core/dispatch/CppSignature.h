#pragma once

#include <typeindex>
#include <c10/macros/Macros.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/Type.h>

namespace c10 {
namespace impl {

// A CppSignature object holds RTTI information about a C++ function signature at runtime
// and can compare them or get a debug-printable name.
class TORCH_API CppSignature final {
public:
    CppSignature(const CppSignature&) = default;
    CppSignature(CppSignature&&) noexcept = default;
    CppSignature& operator=(const CppSignature&) = default;
    CppSignature& operator=(CppSignature&&) noexcept = default;

    template<class FuncType>
    static CppSignature make(
          std::enable_if_t<!std::is_same<
            DispatchKeySet,
            guts::typelist::head_with_default_t<nullptr_t, typename c10::guts::infer_function_traits_t<std::decay_t<FuncType>>::parameter_types>
          >::value, std::nullptr_t> = nullptr) {
        // Normalize functors, lambdas, function pointers, etc. into the plain function type
        using decayed_function_type = typename guts::infer_function_traits_t<std::decay_t<FuncType>>::func_type;

        return CppSignature(std::type_index(typeid(decayed_function_type)));
    }

    // Specialization for when the first argument is of type DispatchKeySet
    // Hide the DispatchKeySet argument from the schema.
    // We do this to guarantee that all CppSignature's for an operator will match, even if they're registered
    // with different calling conventions.
    // See Note [Plumbing Keys Through The Dispatcher]
    template<class FuncType>
    static CppSignature make(
          std::enable_if_t<std::is_same<
            DispatchKeySet,
            guts::typelist::head_with_default_t<nullptr_t, typename c10::guts::infer_function_traits_t<std::decay_t<FuncType>>::parameter_types>
          >::value, std::nullptr_t> = nullptr) {
        // Normalize functors, lambdas, function pointers, etc. into the plain function type
        using decayed_function_type = typename guts::infer_function_traits_t<std::decay_t<FuncType>>::func_type_skip_first_arg;

        return CppSignature(std::type_index(typeid(decayed_function_type)));
    }

    std::string name() const {
        return c10::demangle(signature_.name());
    }

    friend bool operator==(const CppSignature& lhs, const CppSignature& rhs) {
        if (lhs.signature_ == rhs.signature_) {
            return true;
        }
        // Without RTLD_GLOBAL, the type_index comparison could yield false because
        // they point to different instances of the RTTI data, but the types would
        // still be the same. Let's check for that case too.
        // Note that there still is a case where this might not work, i.e. when
        // linking libraries of different compilers together, they might have
        // different ways to serialize a type name. That, together with a missing
        // RTLD_GLOBAL, would still fail this.
        if (lhs.name() == rhs.name()) {
            return true;
        }

        return false;
    }

private:
    explicit CppSignature(std::type_index signature): signature_(std::move(signature)) {}
    std::type_index signature_;
};

inline bool operator!=(const CppSignature& lhs, const CppSignature& rhs) {
    return !(lhs == rhs );
}

}
}
