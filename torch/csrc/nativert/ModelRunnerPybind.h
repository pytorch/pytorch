#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace torch {
namespace nativert {

void initModelRunnerPybind(pybind11::module& m);

} // namespace nativert
} // namespace torch
