#pragma once

#include <Python.h>

#include "torch/csrc/utils/pybind.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/THP.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/interned_strings.h"
#include "torch/csrc/jit/tracer.h"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace pybind11 { namespace detail {

template <> struct type_caster<torch::jit::Symbol> {
public:
  PYBIND11_TYPE_CASTER(torch::jit::Symbol, _("Symbol"));

  bool load(handle src, bool) {
    // TODO: Is there a way to py::cast that doesn't raise an exception on
    // failure?  Can we catch pybind11::cast_error here instead?
    std::string src_str;
    try {
      src_str = py::cast<std::string>(src);
    } catch (std::exception& e) {
      return false;
    }
    value = torch::jit::Symbol::fromQualString(src_str);
    return true;
  }

  static handle cast(torch::jit::Symbol src, return_value_policy /* policy */, handle /* parent */) {
    return py::cast(std::string(src.toQualString()), return_value_policy::copy).release();
  }
};

template <> struct type_caster<torch::jit::AttributeKind> {
public:
  PYBIND11_TYPE_CASTER(torch::jit::AttributeKind, _("AttributeKind"));

  bool load(handle src, bool) {
    return false;
  }

  static handle cast(torch::jit::AttributeKind src, return_value_policy /* policy */, handle /* parent */) {
    return py::cast(std::string(torch::jit::toString(src)), return_value_policy::copy).release();
  }
};

// See https://github.com/pybind/pybind11/issues/637
using ListCasterBase = pybind11::detail::list_caster<std::vector<torch::jit::Node *>, torch::jit::Node *>;
template<> struct type_caster<std::vector<torch::jit::Node *>> : ListCasterBase {
    static handle cast(const std::vector<torch::jit::Node *> &src, return_value_policy, handle parent) {
        return ListCasterBase::cast(src, return_value_policy::reference, parent);
    }
    static handle cast(const std::vector<torch::jit::Node *> *src, return_value_policy pol, handle parent) {
        return cast(*src, pol, parent);
    }
};

}} // namespace pybind11::detail

namespace torch { namespace jit {

static inline py::tuple tuple_tail(const py::tuple & tup) {
  py::tuple r(tup.size() - 1);
  for(std::size_t i = 1; i < tup.size(); i++) {
    r[i-1] = tup[i];
  }
  return r;
}

}}
