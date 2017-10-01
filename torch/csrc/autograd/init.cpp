#include <Python.h>
#include "torch/csrc/utils/pybind.h"
#include "torch/csrc/autograd/profiler.h"

#include "THP.h"

namespace pybind11 { namespace detail {

template <> struct type_caster<torch::autograd::profiler::EventKind> {
public:
  PYBIND11_TYPE_CASTER(torch::autograd::profiler::EventKind, _("torch::autograd::profiler::EventKind"));

  bool load(handle src, bool) {
    try {
      auto str = py::cast<std::string>(src);
      if (str == "push") {
        value = torch::autograd::profiler::EventKind::PushRange;
      } else if (str == "pop") {
        value = torch::autograd::profiler::EventKind::PopRange;
      } else if (str == "mark") {
        value = torch::autograd::profiler::EventKind::Mark;
      } else {
        return false;
      }
    } catch (std::exception& e) {
      return false;
    }
    return true;
  }
  static handle cast(torch::autograd::profiler::EventKind src, return_value_policy /* policy */, handle /* parent */) {
    switch (src) {
      case torch::autograd::profiler::EventKind::PushRange:
        return py::cast("push").release();
      case torch::autograd::profiler::EventKind::PopRange:
        return py::cast("pop").release();
      case torch::autograd::profiler::EventKind::Mark:
        return py::cast("mark").release();
    }
    __builtin_unreachable();
  }
};

}} // namespace pybind11::detail

PyObject * THPAutograd_initExtension(PyObject *_unused)
{
  THPUtils_assert_PyImport("torch.autograd", autograd_module);
  PyObject *autograd_dict = PyModule_GetDict(autograd_module);

  THPVariableClass      = PyMapping_GetItemString(autograd_dict,(char*)"Variable");
  THPFunctionClass      = PyMapping_GetItemString(autograd_dict,(char*)"Function");

  THPUtils_assert_PyImport("torch.nn._functions.thnn", thnn_functions);
  THPBatchNormBackwardBackwardFunction = PyObject_GetAttrString(thnn_functions,(char*)"batchnorm_double_backwards_fn");

  THPStochasticFunctionClass = PyMapping_GetItemString(autograd_dict,(char*)"StochasticFunction");
  THPUtils_assert(THPVariableClass, "couldn't find Variable class in "
          "torch.autograd module");
  THPUtils_assert(THPFunctionClass, "couldn't find Function class in "
          "torch.autograd module");
  THPUtils_assert(THPStochasticFunctionClass, "couldn't find "
          "StochasticFunction class in torch.autograd module");

  auto m = py::handle(autograd_module).cast<py::module>();
  m.def("_enable_profiler", torch::autograd::profiler::enableProfiler);
  m.def("_disable_profiler", torch::autograd::profiler::disableProfiler);

  Py_RETURN_TRUE;
}
