#include <torch/csrc/jit/python/pybind_utils.h>

namespace torch {
namespace jit {

void initStaticModuleBindings(PyObject* module);

} // namespace jit
} // namespace torch
