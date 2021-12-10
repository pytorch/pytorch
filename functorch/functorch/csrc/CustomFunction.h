#include <torch/extension.h>
#include <torch/library.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/autograd/python_variable.h>

namespace at { namespace functorch {

void initDispatchBindings(PyObject* module);

}}
