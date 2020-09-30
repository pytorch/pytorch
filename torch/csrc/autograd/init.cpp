#include <torch/csrc/python_headers.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <ATen/autocast_mode.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/autograd/python_function.h>
#include <torch/csrc/autograd/function.h>

PyObject* THPAutograd_initExtension(PyObject* _unused, PyObject *unused) {
  using namespace torch::autograd::profiler;
  auto tensor_module = THPObjectPtr(PyImport_ImportModule("torch.tensor"));
  if (!tensor_module)
    return nullptr;

  // NOTE: "leaks" THPVariableClass
  THPVariableClass = PyObject_GetAttrString(tensor_module, "Tensor");
  if (!THPVariableClass)
    return nullptr;

  auto autograd_module = THPObjectPtr(PyImport_ImportModule("torch.autograd"));
  if (!autograd_module)
    return nullptr;

  // NOTE: "leaks" Function
  THPFunctionClass = PyObject_GetAttrString(autograd_module, "Function");
  if (!THPFunctionClass)
    return nullptr;

  auto torch_C_module = THPObjectPtr(PyImport_ImportModule("torch._C"));
  if (!torch_C_module)
    return nullptr;
  auto _C_m = py::handle(torch_C_module).cast<py::module>();
  auto m = _C_m.def_submodule("_autograd", "autograd bindings");


  py::enum_<ProfilerState>(m, "ProfilerState")
      .value("Disabled", ProfilerState::Disabled)
      .value("CPU", ProfilerState::CPU)
      .value("CUDA", ProfilerState::CUDA)
      .value("NVTX", ProfilerState::NVTX);

  py::class_<ProfilerConfig>(m, "ProfilerConfig")
      .def(py::init<ProfilerState, bool, bool, bool>());

  py::class_<Event>(m, "ProfilerEvent")
      .def("kind", &Event::kind)
      .def("name", [](const Event& e) { return e.name(); })
      .def("thread_id", &Event::threadId)
      .def("fwd_thread_id", &Event::fwdThreadId)
      .def("device", &Event::device)
      .def("cpu_elapsed_us", &Event::cpuElapsedUs)
      .def("cuda_elapsed_us", &Event::cudaElapsedUs)
      .def("has_cuda", &Event::hasCuda)
      .def("shapes", &Event::shapes)
      .def("cpu_memory_usage", &Event::cpuMemoryUsage)
      .def("cuda_memory_usage", &Event::cudaMemoryUsage)
      .def("handle", &Event::handle)
      .def("node_id", &Event::nodeId)
      .def("is_remote", &Event::isRemote)
      .def("sequence_nr", &Event::sequenceNr)
      .def("stack", &Event::stack)
      .def("scope", &Event::scope);

  py::class_<ProfilerDisableOptions>(m, "_ProfilerDisableOptions")
    .def(py::init<bool, bool>());

  m.def("_enable_profiler", enableProfiler);
  m.def(
      "_disable_profiler",
      disableProfiler,
      py::arg("profiler_disable_options") = ProfilerDisableOptions());
  m.def("_profiler_enabled", profilerEnabled);
  m.def("_enable_record_function", [](bool enable) {
    at::enableRecordFunction(enable);
  });
  m.def("_set_empty_test_observer", [](bool is_global, double sampling_prob) {
    auto cb = at::RecordFunctionCallback(
        [](const at::RecordFunction&) {},
        [](const at::RecordFunction&) {})
      .needsInputs(true)
      .samplingProb(sampling_prob);
    if (is_global) {
      at::addGlobalCallback(cb);
    } else {
      at::addThreadLocalCallback(cb);
    }
  });
  m.def("_clear_callbacks", []() {
    at::clearCallbacks();
  });

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
static PyMethodDef methods[] = { // NOLINT
  {"_set_grad_enabled", (PyCFunction)set_grad_enabled, METH_O, nullptr},
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
