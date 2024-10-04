#include <ATen/DeviceAccelerator.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::accelerator {

void initModule(PyObject* module);

} // torch::accelerator
