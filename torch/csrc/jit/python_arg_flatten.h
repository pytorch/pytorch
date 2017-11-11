#pragma once

#include "torch/csrc/jit/pybind.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/utils/hash.h"

#include <tuple>
#include <vector>
#include <functional>

namespace torch { namespace jit { namespace python {

struct IODescriptor {
  struct VariableMetadata {
    VariableMetadata(const autograd::Variable& var)
      : sizes(var.sizes())
      , type(var.type().scalarType())
      , device(var.type().is_cuda() ? var.get_device() : -1)
      , requires_grad(var.requires_grad())
      , is_volatile(var.is_volatile()) {}

    bool operator==(const VariableMetadata& o) const {
      return std::tie(  device,   requires_grad,   is_volatile,   type,  sizes) ==
             std::tie(o.device, o.requires_grad, o.is_volatile, o.type, o.sizes);
    }

    static std::size_t hash(const VariableMetadata& m) {
      return get_hash(m.sizes, m.device, m.requires_grad, m.type, m.is_volatile);
    }

    std::vector<int64_t> sizes;
    at::ScalarType type;
    int device;
    bool requires_grad;
    bool is_volatile;
  };

  bool operator==(const IODescriptor& o) const {
    return std::tie(  structure,   metadata) ==
           std::tie(o.structure, o.metadata);
  }

  static std::size_t hash(const IODescriptor& o) {
    return get_hash(o.structure, o.metadata);
  }

  // Description of argument structure. Variables are replaced with
  // different characters, depending on their flags, beginnings and
  // ends of tuples and lists are denoted by a pair of parenthesis
  // of their corresponding kind. They should always be paired.
  // Example desc: (vv[v(v)v])
  std::string structure;
  std::vector<VariableMetadata> metadata;
};

struct ParsedArgs {
  // Flat vector of Variables found in arguments
  autograd::variable_list vars;
  // Metadata describing nesting of objects received from Python and
  // metadata of vars.
  IODescriptor desc;
  // True iff any of vars is volatile
  bool is_volatile = false;
};


ParsedArgs flatten(py::handle obj);
PyObject* unflatten(autograd::variable_list&& outputs, const IODescriptor& structure);

}}} // namespace torch::jit::python
