#include <torch/csrc/jit/python/pybind_utils.h>

namespace torch {
namespace jit {

void initStaticRuntimeBindings(PyObject* module);

} // namespace jit
} // namespace torch
