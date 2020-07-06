#include <pybind11/pybind11.h>

namespace torch {
namespace impl {
namespace dispatch {

void initDispatchBindings(PyObject* module);

}}} // namespace torch::impl::dispatch
