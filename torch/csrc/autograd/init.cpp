#include <Python.h>
#include "torch/csrc/utils/pybind.h"
#include "torch/csrc/autograd/profiler.h"

#include "THP.h"

#ifdef _MSC_VER
#define ENSURE_UNREACHABLE __assume(0);
#else
#define ENSURE_UNREACHABLE __builtin_unreachable();
#endif

PyObject * THPAutograd_initExtension(PyObject *_unused)
{
  THPUtils_assert_PyImport("torch.autograd", autograd_module);
  PyObject *autograd_dict = PyModule_GetDict(autograd_module);

  THPVariableClass      = PyMapping_GetItemString(autograd_dict,(char*)"Variable");
  THPFunctionClass      = PyMapping_GetItemString(autograd_dict,(char*)"Function");

  THPUtils_assert_PyImport("torch.nn._functions.thnn", thnn_functions);
  THPBatchNormBackwardBackwardFunction = PyObject_GetAttrString(thnn_functions,(char*)"batchnorm_double_backwards_fn");

  THPUtils_assert(THPVariableClass, "couldn't find Variable class in "
          "torch.autograd module");
  THPUtils_assert(THPFunctionClass, "couldn't find Function class in "
          "torch.autograd module");

  auto m = py::handle(autograd_module).cast<py::module>();

  py::class_<torch::autograd::profiler::Event>(m,"ProfilerEvent")
  .def("kind",&torch::autograd::profiler::Event::kind)
  .def("name",&torch::autograd::profiler::Event::name)
  .def("thread_id",&torch::autograd::profiler::Event::thread_id)
  .def("cpu_elapsed_us",&torch::autograd::profiler::Event::cpu_elapsed_us)
  .def("cuda_elapsed_us",&torch::autograd::profiler::Event::cuda_elapsed_us)
  .def("has_cuda",&torch::autograd::profiler::Event::has_cuda);
  py::enum_<torch::autograd::profiler::ProfilerState>(m,"ProfilerState")
  .value("Disabled", torch::autograd::profiler::ProfilerState::Disabled)
  .value("CPU", torch::autograd::profiler::ProfilerState::CPU)
  .value("CUDA", torch::autograd::profiler::ProfilerState::CUDA)
  .value("NVTX", torch::autograd::profiler::ProfilerState::NVTX);

  m.def("_enable_profiler", torch::autograd::profiler::enableProfiler);
  m.def("_disable_profiler", torch::autograd::profiler::disableProfiler);

  m.def("_push_range", [](const char *name) {
    using namespace torch::autograd::profiler;
    if (state  == ProfilerState::Disabled) return;
    pushRange(name);
  });
  m.def("_pop_range", []() {
    using namespace torch::autograd::profiler;
    if (state  == ProfilerState::Disabled) return;
    popRange();
  });

  Py_RETURN_TRUE;
}
