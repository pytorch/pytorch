#include "ir.h"

#include "torch/csrc/utils/auto_gil.h"

namespace torch { namespace autograd {

std::string PyNode::name() {
  AutoGIL gil;
  return std::string(Py_TYPE(obj)->tp_name);
}

}}
