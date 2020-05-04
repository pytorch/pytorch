#include <torch/csrc/python_headers.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <ATen/autocast_mode.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/autograd/record_function_ops.h>
#include <torch/csrc/autograd/python_function.h>
#include <torch/csrc/autograd/function.h>
#ifdef USE_DISTRIBUTED
#include <torch/csrc/distributed/rpc/message.h>
#endif

PyObject* THPAutograd_initExtension(PyObject* _unused, PyObject *unused) {
  using namespace torch::autograd::profiler;
  auto tensor_module = THPObjectPtr(PyImport_ImportModule("torch.tensor"));
  if (!tensor_module)
    throw python_error();

  // NOTE: "leaks" THPVariableClass
  THPVariableClass = PyObject_GetAttrString(tensor_module, "Tensor");
  if (!THPVariableClass)
    throw python_error();

  auto autograd_module = THPObjectPtr(PyImport_ImportModule("torch.autograd"));
  if (!autograd_module)
    throw python_error();

  // NOTE: "leaks" Function
  THPFunctionClass = PyObject_GetAttrString(autograd_module, "Function");
  if (!THPFunctionClass)
    throw python_error();

  auto m = py::handle(autograd_module).cast<py::module>();

  py::enum_<ProfilerState>(m, "ProfilerState")
      .value("Disabled", ProfilerState::Disabled)
      .value("CPU", ProfilerState::CPU)
      .value("CUDA", ProfilerState::CUDA)
      .value("NVTX", ProfilerState::NVTX);

  py::class_<ProfilerConfig>(m, "ProfilerConfig")
      .def(py::init<ProfilerState, bool, bool>());

  py::class_<Event>(m, "ProfilerEvent")
      .def("kind", &Event::kind)
      .def("name", [](const Event& e) { return e.name(); })
      .def("thread_id", &Event::thread_id)
      .def("device", &Event::device)
      .def("cpu_elapsed_us", &Event::cpu_elapsed_us)
      .def("cuda_elapsed_us", &Event::cuda_elapsed_us)
      .def("has_cuda", &Event::has_cuda)
      .def("shapes", &Event::shapes)
      .def("cpu_memory_usage", &Event::cpu_memory_usage)
      .def("cuda_memory_usage", &Event::cuda_memory_usage);

  m.def("_enable_profiler", enableProfiler);
  m.def("_disable_profiler", disableProfiler);
  m.def("_profiler_enabled", profilerEnabled);

  // TODO: remove when jit future can hold PyObject (https://github.com/pytorch/pytorch/issues/34999)
#ifdef USE_DISTRIBUTED
  m.def(
      "_call_end_callbacks_on_fut",
      [](const at::Tensor& handle,
         const std::shared_ptr<torch::distributed::rpc::FutureIValue>& fut) {
        torch::autograd::profiler::_call_end_callbacks_on_fut(handle, fut);
      });
#endif

  Py_RETURN_TRUE;
}

namespace torch { namespace autograd {

static PyObject * set_autocast_enabled(PyObject* _unused, PyObject *arg) {
  HANDLE_TH_ERRORS
  if (!PyBool_Check(arg)) {
    throw TypeError("enabled must be a bool (got %s)", Py_TYPE(arg)->tp_name);
  }
  at::autocast::set_enabled(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * is_autocast_enabled(PyObject* _unused, PyObject *arg) {
  HANDLE_TH_ERRORS
  if (at::autocast::is_enabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject * clear_autocast_cache(PyObject* _unused, PyObject *arg) {
  HANDLE_TH_ERRORS
  at::autocast::clear_cache();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * autocast_increment_nesting(PyObject* _unused, PyObject *arg) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(at::autocast::increment_nesting());
  END_HANDLE_TH_ERRORS
}

static PyObject * autocast_decrement_nesting(PyObject* _unused, PyObject *arg) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(at::autocast::decrement_nesting());
  END_HANDLE_TH_ERRORS
}

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

static PyObject * set_anomaly_mode_enabled(PyObject* _unused, PyObject *arg) {
  HANDLE_TH_ERRORS
  if (!PyBool_Check(arg)) {
    throw TypeError("enabled must be a bool (got %s)", Py_TYPE(arg)->tp_name);
  }
  AnomalyMode::set_enabled(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * is_anomaly_mode_enabled(PyObject* _unused, PyObject *arg) {
  HANDLE_TH_ERRORS
  if (AnomalyMode::is_enabled()) {
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
  {"set_autocast_enabled", (PyCFunction)set_autocast_enabled, METH_O, nullptr},
  {"is_autocast_enabled", (PyCFunction)is_autocast_enabled, METH_NOARGS, nullptr},
  {"clear_autocast_cache", (PyCFunction)clear_autocast_cache, METH_NOARGS, nullptr},
  {"autocast_increment_nesting", (PyCFunction)autocast_increment_nesting, METH_NOARGS, nullptr},
  {"autocast_decrement_nesting", (PyCFunction)autocast_decrement_nesting, METH_NOARGS, nullptr},
  {"set_anomaly_enabled", (PyCFunction)set_anomaly_mode_enabled, METH_O, nullptr},
  {"is_anomaly_enabled", (PyCFunction)is_anomaly_mode_enabled, METH_NOARGS, nullptr},
  {nullptr, nullptr, 0, nullptr}
};

PyMethodDef* python_functions() {
  return methods;
}

}} // namespace torch::autograd
