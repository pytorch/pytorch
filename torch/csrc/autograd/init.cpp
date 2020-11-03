#include <torch/csrc/python_headers.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <ATen/autocast_mode.h>
#include <torch/csrc/autograd/profiler_kineto.h>
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
      .value("NVTX", ProfilerState::NVTX)
      .value("KINETO", ProfilerState::KINETO);

  py::enum_<ActivityType>(m, "ProfilerActivity")
      .value("CPU", ActivityType::CPU)
      //.value("CUDA_RUNTIME", ActivityType::CUDA_RUNTIME)
      .value("CUDA", ActivityType::CUDA);

  py::class_<ProfilerConfig>(m, "ProfilerConfig")
      .def(py::init<ProfilerState, bool, bool, bool>());

  py::class_<LegacyEvent>(m, "ProfilerEvent")
      .def("kind", &LegacyEvent::kindStr)
      .def("name", [](const LegacyEvent& e) { return e.name(); })
      .def("thread_id", &LegacyEvent::threadId)
      .def("fwd_thread_id", &LegacyEvent::fwdThreadId)
      .def("device", &LegacyEvent::device)
      .def("cpu_elapsed_us", &LegacyEvent::cpuElapsedUs)
      .def("cuda_elapsed_us", &LegacyEvent::cudaElapsedUs)
      .def("has_cuda", &LegacyEvent::hasCuda)
      .def("shapes", &LegacyEvent::shapes)
      .def("cpu_memory_usage", &LegacyEvent::cpuMemoryUsage)
      .def("cuda_memory_usage", &LegacyEvent::cudaMemoryUsage)
      .def("handle", &LegacyEvent::handle)
      .def("node_id", &LegacyEvent::nodeId)
      .def("is_remote", &LegacyEvent::isRemote)
      .def("sequence_nr", &LegacyEvent::sequenceNr)
      .def("stack", &LegacyEvent::stack)
      .def("scope", &LegacyEvent::scope)
      .def("correlation_id", &LegacyEvent::correlationId);

#ifdef USE_KINETO
  py::class_<KinetoEvent>(m, "KinetoEvent")
      .def("name", &KinetoEvent::name)
      .def("start_thread_id", [](const KinetoEvent& e) { return e.startThreadId(); })
      .def("end_thread_id", [](const KinetoEvent& e) { return e.endThreadId(); })
      .def("device_index", &KinetoEvent::deviceIndex)
      .def("start_us", &KinetoEvent::startUs)
      .def("duration_us", &KinetoEvent::durationUs)
      .def("correlation_id", [](const KinetoEvent& e) { return e.correlationId(); })
      .def("fwd_thread_id", [](const KinetoEvent& e) { return e.fwdThreadId(); })
      .def("shapes", [](const KinetoEvent& e) { return e.shapes(); })
      .def("sequence_nr", [](const KinetoEvent& e) { return e.sequenceNr(); })
      .def("stack", [](const KinetoEvent& e) { return e.stack(); })
      .def("scope", [](const KinetoEvent& e) { return e.scope(); });

  py::class_<ProfilerResult>(m, "ProfilerResult")
      .def("events", &ProfilerResult::events)
      .def("legacy_events", &ProfilerResult::legacy_events);

  m.def("_enable_profiler", enableProfiler);
  m.def("_disable_profiler", disableProfiler);
  m.def("_prepare_profiler", prepareProfiler);
#endif

  m.def("kineto_available", kinetoAvailable);

  m.def("_enable_profiler_legacy", enableProfilerLegacy);
  py::class_<ProfilerDisableOptions>(m, "_ProfilerDisableOptions")
      .def(py::init<bool, bool>());
  m.def(
      "_disable_profiler_legacy",
      disableProfilerLegacy,
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
  {"_set_grad_enabled", set_grad_enabled, METH_O, nullptr},
  {"is_grad_enabled", is_grad_enabled, METH_NOARGS, nullptr},
  {"set_autocast_enabled", set_autocast_enabled, METH_O, nullptr},
  {"is_autocast_enabled", is_autocast_enabled, METH_NOARGS, nullptr},
  {"clear_autocast_cache", clear_autocast_cache, METH_NOARGS, nullptr},
  {"autocast_increment_nesting", autocast_increment_nesting, METH_NOARGS, nullptr},
  {"autocast_decrement_nesting", autocast_decrement_nesting, METH_NOARGS, nullptr},
  {"set_anomaly_enabled", set_anomaly_mode_enabled, METH_O, nullptr},
  {"is_anomaly_enabled", is_anomaly_mode_enabled, METH_NOARGS, nullptr},
  {nullptr, nullptr, 0, nullptr}
};

PyMethodDef* python_functions() {
  return methods;
}

}} // namespace torch::autograd
