#pragma once

#include "torch/csrc/jit/pybind.h"
#include "torch/csrc/autograd/variable.h"

#include <tuple>

namespace torch { namespace jit { namespace python {

struct ParsedArgs {
  // Flat vector of Variables found in arguments
  autograd::variable_list vars;
  // Description of argument structure. Variables are replaced with
  // different characters, depending on their flags, beginnings and
  // ends of tuples and lists are denoted by a pair of parenthesis
  // of their corresponding kind. They should always be paired.
  // Example desc: (rn[n(r)r]). Would be (vv[v(v)v]) if **any**
  // input Variable was volatile (even non-volatile ones are marked with v).
  std::string desc;
  // True iff any of vars is volatile
  bool is_volatile = false;
};


ParsedArgs flatten(py::handle obj);
PyObject* unflatten(autograd::variable_list outputs, std::string descriptor);

}}} // namespace torch::jit::python

namespace pybind11 { namespace detail {

template<> struct type_caster<torch::jit::python::ParsedArgs> {
public:
  PYBIND11_TYPE_CASTER(torch::jit::python::ParsedArgs, _("torch::jit::python::ParsedArgs"));
  bool load(handle src, bool) {
    throw std::runtime_error("ParsedArgs can only be casted to Python");
  }
  static handle cast(torch::jit::python::ParsedArgs src, return_value_policy /* policy */, handle /* parent */) {
    py::bytes desc = bytes(src.desc);
    return py::cast(std::tie(src.vars,
                             desc,
                             src.is_volatile)).release();
  }
};

}} // namespace pybind11::detail

