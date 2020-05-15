#pragma once

#include <typeindex>
#include <c10/macros/Macros.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/Type.h>

namespace c10 {
namespace impl {

// A CppSignature object holds RTTI information about a C++ function signature at runtime
// and can compare them or get a debug-printable name.
class CAFFE2_API CppSignature final {
public:
    CppSignature(const CppSignature&) = default;
    CppSignature(CppSignature&&) noexcept = default;
    CppSignature& operator=(const CppSignature&) = default;
    CppSignature& operator=(CppSignature&&) noexcept = default;

    template<class FuncType>
    static CppSignature make() {
        // Normalize functors, lambdas, function pointers, etc. into the plain function type
        using decayed_function_type = typename guts::infer_function_traits_t<std::decay_t<FuncType>>::func_type;

        return CppSignature(std::type_index(typeid(decayed_function_type)));
    }

    std::string name() const {
        return c10::demangle(signature_.name());
    }

    friend bool operator==(const CppSignature& lhs, const CppSignature& rhs) {
        return lhs.signature_ == rhs.signature_;
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
