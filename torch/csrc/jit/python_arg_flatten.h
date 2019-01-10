#pragma once

#include "torch/csrc/jit/pybind.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/utils/hash.h"

#include <ATen/ATen.h>
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
      , requires_grad(var.requires_grad()) {}

    bool operator==(const VariableMetadata& o) const {
      return std::tie(  device,   requires_grad,   type,  sizes) ==
             std::tie(o.device, o.requires_grad, o.type, o.sizes);
    }

    static std::size_t hash(const VariableMetadata& m) {
      return get_hash(m.sizes, m.device, m.requires_grad, m.type);
    }

    std::vector<int64_t> sizes;
    at::ScalarType type;
    int device;
    bool requires_grad;
  };

  bool operator==(const IODescriptor& o) const {
    return std::tie(  structure,   metadata,   grad_enabled) ==
           std::tie(o.structure, o.metadata, o.grad_enabled);
  }

  static std::size_t hash(const IODescriptor& o) {
    return get_hash(o.structure, o.metadata, o.grad_enabled);
  }

  void extend(const autograd::variable_list& list) {
    metadata.reserve(metadata.size() + list.size());
    for (auto & var : list)
      metadata.emplace_back(var);
  }

  // Description of argument structure. Variables are replaced with
  // different characters, depending on their flags, beginnings and
  // ends of tuples and lists are denoted by a pair of parenthesis
  // of their corresponding kind. They should always be paired.
  // Example desc: (vv[v(v)v])
  // NOTE: if extend() was ever called then metadata.size() can be
  // different than the number of 'v's in structure.
  std::string structure;
  std::vector<VariableMetadata> metadata;
  bool grad_enabled;
};

static inline std::ostream& operator<<(std::ostream& out, const IODescriptor::VariableMetadata& meta) {
  auto & t = at::getType(meta.device < 0 ? at::kCPU : at::kCUDA, meta.type);
  out << t << "(requires_grad=" << meta.requires_grad;
  if (meta.device > 0) {
    out << ", device=" << meta.device;
  }
  out << ") {";
  for(size_t i = 0; i < meta.sizes.size(); ++i) {
    if(i > 0)
      out << ", ";
    out << meta.sizes[i];
  }
  out << "}";
  return out;
}

static inline std::ostream& operator<<(std::ostream & out, const IODescriptor & desc) {
  out << desc.structure << "\n";
  out << "  with grad_enabled=" << desc.grad_enabled << "\n";
  for(size_t i = 0; i < desc.metadata.size(); ++i) {
    out << "  with v" << i << " having type " << desc.metadata[i] << "\n";
  }
  return out;
}

struct ParsedArgs {
  // Flat vector of Variables found in arguments
  autograd::variable_list vars;
  // Metadata describing nesting of objects received from Python and
  // metadata of vars and whether grad is enabled.
  IODescriptor desc;

  void extend(const autograd::variable_list& list) {
    if (list.empty()) return;
    vars.reserve(vars.size() + list.size());
    for (auto & var : list)
      vars.emplace_back(var);
    desc.extend(list);
  }
};


ParsedArgs flatten(py::handle obj);
PyObject* unflatten(at::ArrayRef<autograd::Variable> outputs,
                    const IODescriptor& structure);

}}} // namespace torch::jit::python
