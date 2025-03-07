#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace torch {
namespace runtime {

void initModelRunnerPybind(pybind11::module& m);

} // namespace runtime
} // namespace torch
