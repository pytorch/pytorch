#include <torch/csrc/utils/PythonWrapper.h>

namespace torch::functorch::impl {

void initFuncTorchBindings(PyObject* module);
PyObject* unwrap_dead_wrappers(PyObject* args);

} // namespace torch::functorch::impl
