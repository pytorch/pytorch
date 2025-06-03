#ifndef ITT_H
#define ITT_H
#include <torch/csrc/utils/pybind.h>

namespace torch::profiler {
void initIttBindings(PyObject* module); // namespace torch::profiler
}
#endif // ITT_H
