#include "torch/csrc/python_headers.h"

#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils/pybind.h"
#include "torch/csrc/autograd/grad_mode.h"
#include "torch/csrc/autograd/profiler.h"
#include "torch/csrc/autograd/python_function.h"

PyObject * THPAutograd_initExtension(PyObject *_unused)
{
  auto tensor_module = THPObjectPtr(PyImport_ImportModule("torch.tensor"));
  if (!tensor_module) throw python_error();

  // NOTE: "leaks" THPVariableClass
  THPVariableClass = PyObject_GetAttrString(tensor_module, "Tensor");
  if (!THPVariableClass) throw python_error();

  auto autograd_module = THPObjectPtr(PyImport_ImportModule("torch.autograd"));
  if (!autograd_module) throw python_error();

  // NOTE: "leaks" Function
  THPFunctionClass = PyObject_GetAttrString(autograd_module, "Function");
  if (!THPFunctionClass) throw python_error();

  auto m = py::handle(autograd_module).cast<py::module>();

  py::class_<torch::autograd::profiler::Event>(m,"ProfilerEvent")
  .def("kind",&torch::autograd::profiler::Event::kind)
  .def("name",&torch::autograd::profiler::Event::name)
  .def("thread_id",&torch::autograd::profiler::Event::thread_id)
  .def("device",&torch::autograd::profiler::Event::device)
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

namespace torch { namespace autograd {

static PyObject * set_grad_enabled(PyObject* _unused, PyObject *arg) {
  HANDLE_TH_ERRORS
  if (!PyBool_Check(arg)) {
    throw TypeError("enabled must be a bool (got %s)", Py_TYPE(arg)->tp_name);
  }
  GradMode::set_enabled(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * is_grad_enabled(PyObject* _unused, PyObject *arg) {
  HANDLE_TH_ERRORS
  if (GradMode::is_enabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

// autograd methods on torch._C
static PyMethodDef methods[] = {
  {"set_grad_enabled", (PyCFunction)set_grad_enabled, METH_O, nullptr},
  {"is_grad_enabled", (PyCFunction)is_grad_enabled, METH_NOARGS, nullptr},
  {nullptr, nullptr, 0, nullptr}
};

PyMethodDef* python_functions() {
  return methods;
}

}} // namespace torch::autograd
