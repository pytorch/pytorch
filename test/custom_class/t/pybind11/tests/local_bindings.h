#pragma once
#include "pybind11_tests.h"

/// Simple class used to test py::local:
template <int> class LocalBase {
public:
    LocalBase(int i) : i(i) { }
    int i = -1;
};

/// Registered with py::module_local in both main and secondary modules:
using LocalType = LocalBase<0>;
/// Registered without py::module_local in both modules:
using NonLocalType = LocalBase<1>;
/// A second non-local type (for stl_bind tests):
using NonLocal2 = LocalBase<2>;
/// Tests within-module, different-compilation-unit local definition conflict:
using LocalExternal = LocalBase<3>;
/// Mixed: registered local first, then global
using MixedLocalGlobal = LocalBase<4>;
/// Mixed: global first, then local
using MixedGlobalLocal = LocalBase<5>;

/// Registered with py::module_local only in the secondary module:
using ExternalType1 = LocalBase<6>;
using ExternalType2 = LocalBase<7>;

using LocalVec = std::vector<LocalType>;
using LocalVec2 = std::vector<NonLocal2>;
using LocalMap = std::unordered_map<std::string, LocalType>;
using NonLocalVec = std::vector<NonLocalType>;
using NonLocalVec2 = std::vector<NonLocal2>;
using NonLocalMap = std::unordered_map<std::string, NonLocalType>;
using NonLocalMap2 = std::unordered_map<std::string, uint8_t>;

PYBIND11_MAKE_OPAQUE(LocalVec);
PYBIND11_MAKE_OPAQUE(LocalVec2);
PYBIND11_MAKE_OPAQUE(LocalMap);
PYBIND11_MAKE_OPAQUE(NonLocalVec);
//PYBIND11_MAKE_OPAQUE(NonLocalVec2); // same type as LocalVec2
PYBIND11_MAKE_OPAQUE(NonLocalMap);
PYBIND11_MAKE_OPAQUE(NonLocalMap2);


// Simple bindings (used with the above):
template <typename T, int Adjust = 0, typename... Args>
py::class_<T> bind_local(Args && ...args) {
    return py::class_<T>(std::forward<Args>(args)...)
        .def(py::init<int>())
        .def("get", [](T &i) { return i.i + Adjust; });
};

// Simulate a foreign library base class (to match the example in the docs):
namespace pets {
class Pet {
public:
    Pet(std::string name) : name_(name) {}
    std::string name_;
    const std::string &name() { return name_; }
};
}

struct MixGL { int i; MixGL(int i) : i{i} {} };
struct MixGL2 { int i; MixGL2(int i) : i{i} {} };
