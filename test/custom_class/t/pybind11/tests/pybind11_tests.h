#pragma once
#include <pybind11/pybind11.h>

#if defined(_MSC_VER) && _MSC_VER < 1910
// We get some really long type names here which causes MSVC 2015 to emit warnings
#  pragma warning(disable: 4503) // warning C4503: decorated name length exceeded, name was truncated
#endif

namespace py = pybind11;
using namespace pybind11::literals;

class test_initializer {
    using Initializer = void (*)(py::module &);

public:
    test_initializer(Initializer init);
    test_initializer(const char *submodule_name, Initializer init);
};

#define TEST_SUBMODULE(name, variable)                   \
    void test_submodule_##name(py::module &);            \
    test_initializer name(#name, test_submodule_##name); \
    void test_submodule_##name(py::module &variable)


/// Dummy type which is not exported anywhere -- something to trigger a conversion error
struct UnregisteredType { };

/// A user-defined type which is exported and can be used by any test
class UserType {
public:
    UserType() = default;
    UserType(int i) : i(i) { }

    int value() const { return i; }
    void set(int set) { i = set; }

private:
    int i = -1;
};

/// Like UserType, but increments `value` on copy for quick reference vs. copy tests
class IncType : public UserType {
public:
    using UserType::UserType;
    IncType() = default;
    IncType(const IncType &other) : IncType(other.value() + 1) { }
    IncType(IncType &&) = delete;
    IncType &operator=(const IncType &) = delete;
    IncType &operator=(IncType &&) = delete;
};

/// Custom cast-only type that casts to a string "rvalue" or "lvalue" depending on the cast context.
/// Used to test recursive casters (e.g. std::tuple, stl containers).
struct RValueCaster {};
NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)
template<> class type_caster<RValueCaster> {
public:
    PYBIND11_TYPE_CASTER(RValueCaster, _("RValueCaster"));
    static handle cast(RValueCaster &&, return_value_policy, handle) { return py::str("rvalue").release(); }
    static handle cast(const RValueCaster &, return_value_policy, handle) { return py::str("lvalue").release(); }
};
NAMESPACE_END(detail)
NAMESPACE_END(pybind11)
