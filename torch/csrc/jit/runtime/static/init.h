#include <torch/csrc/jit/python/pybind_utils.h>

namespace torch::jit {

void initStaticModuleBindings(PyObject* module);

} // namespace torch::jit
