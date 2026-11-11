#pragma once

// C++ side of torch.cfg_compile.

#include <torch/csrc/python_headers.h>

#include <cstdint>

extern "C" {

enum TcfgPyOpKind : int32_t {
  // operands = [lhs, rhs]
  TCFG_BINARY = 2,
  // operands = [self]: unary operators, truthiness, attribute getters
  TCFG_UNARY = 3,
};

// One Python-level operation, as the VM executed it.
struct TcfgPyOp {
  int32_t kind;
  int32_t n_operands;
  PyObject* const* operands;
  PyObject* result;
};

} // extern "C"

namespace torch::cfg {

void initCfgBindings(PyObject* module);

} // namespace torch::cfg
