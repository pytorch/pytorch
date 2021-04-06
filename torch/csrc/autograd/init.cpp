#include <torch/csrc/python_headers.h>

#include <c10/core/DeviceType.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <ATen/autocast_mode.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/autograd/python_function.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/autograd/utils/python_arg_parsing.h>
#include <torch/csrc/utils/pycfunction_helpers.h>

PyObject* THPAutograd_initExtension(PyObject* _unused, PyObject *unused) {
  using namespace torch::autograd::profiler;
  auto tensor_module = THPObjectPtr(PyImport_ImportModule("torch._tensor"));
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

  auto parameter_module = THPObjectPtr(PyImport_ImportModule("torch.nn.parameter"));
  if (!parameter_module)
    return nullptr;

  // NOTE: "leaks" ParameterClass
  ParameterClass = PyObject_GetAttrString(parameter_module, "Parameter");
  if (!ParameterClass)
    return nullptr;

  py::enum_<ProfilerState>(m, "ProfilerState")
      .value("Disabled", ProfilerState::Disabled)
      .value("CPU", ProfilerState::CPU)
      .value("CUDA", ProfilerState::CUDA)
      .value("NVTX", ProfilerState::NVTX)
      .value("KINETO", ProfilerState::KINETO);

  py::enum_<ActivityType>(m, "ProfilerActivity")
      .value("CPU", ActivityType::CPU)
      .value("CUDA", ActivityType::CUDA);

  py::class_<ProfilerConfig>(m, "ProfilerConfig")
      .def(py::init<ProfilerState, bool, bool, bool, bool>());

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
      .def("correlation_id", &LegacyEvent::correlationId)
      .def("start_us", &LegacyEvent::cpuUs)
      .def("flops", &LegacyEvent::flops);

  py::enum_<c10::DeviceType>(m, "DeviceType")
      .value("CPU", c10::DeviceType::CPU)
      .value("CUDA", c10::DeviceType::CUDA)
      .value("MKLDNN", c10::DeviceType::MKLDNN)
      .value("OPENGL", c10::DeviceType::OPENGL)
      .value("OPENCL", c10::DeviceType::OPENCL)
      .value("IDEEP", c10::DeviceType::IDEEP)
      .value("HIP", c10::DeviceType::HIP)
      .value("FPGA", c10::DeviceType::FPGA)
      .value("MSNPU", c10::DeviceType::MSNPU)
      .value("XLA", c10::DeviceType::XLA)
      .value("MLC", c10::DeviceType::MLC)
      .value("Meta", c10::DeviceType::Meta)
      .value("Vulkan", c10::DeviceType::Vulkan)
      .value("Metal", c10::DeviceType::Metal);

#ifdef USE_KINETO
  py::class_<KinetoEvent>(m, "KinetoEvent")
      // name of the event
      .def("name", &KinetoEvent::name)
      // PyTorch thread id of the start callback
      .def("start_thread_id", [](const KinetoEvent& e) {
        return e.startThreadId();
      })
      // PyTorch thread id of the end callback
      .def("end_thread_id", [](const KinetoEvent& e) {
        return e.endThreadId();
      })
      // for events of scope BACKWARD_FUNCTION - PyTorch thread id
      // of the corresponding forward op
      .def("fwd_thread_id", [](const KinetoEvent& e) {
        return e.fwdThreadId();
      })
      // together with fwd_thread_id, used to uniquely identify
      // the forward op
      .def("sequence_nr", [](const KinetoEvent& e) {
        return e.sequenceNr();
      })
      // absolute start time (since unix epoch) in us
      .def("start_us", &KinetoEvent::startUs)
      // duration in us
      .def("duration_us", &KinetoEvent::durationUs)
      // used for correlation between high-level PyTorch events
      // and low-level device events
      .def("correlation_id", [](const KinetoEvent& e) {
        return e.correlationId();
      })
      // shapes of input tensors
      .def("shapes", [](const KinetoEvent& e) {
        if (e.hasShapes()) {
          return e.shapes();
        } else {
          return std::vector<std::vector<int64_t>>();
        }
      })
      // stack traces of the PyTorch CPU events
      .def("stack", [](const KinetoEvent& e) {
        if (e.hasStack()) {
          return e.stack();
        } else {
          return std::vector<std::string>();
        }
      })
      // type of the RecordFunction that generated a PyTorch CPU event
      // (op, torchscript function, user label, etc)
      .def("scope", [](const KinetoEvent& e) {
        return e.scope();
      })
      // device number, for CPU - process id
      .def("device_index", &KinetoEvent::deviceIndex)
      // for CUDA - stream id, for CPU - start thread id
      .def("device_resource_id", &KinetoEvent::deviceResourceId)
      // device type
      .def("device_type", [](const KinetoEvent& e) {
        return e.deviceType();
      })
      // correlation id of a linked event
      .def("linked_correlation_id", &KinetoEvent::linkedCorrelationId)
      // compute flops
      .def("flops", [](const KinetoEvent& e) {
        return e.flops();
      });

  py::class_<ProfilerResult>(m, "ProfilerResult")
    .def("events", &ProfilerResult::events)
    .def("legacy_events", &ProfilerResult::legacy_events)
    .def("save", &ProfilerResult::save);

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
    auto cb = at::RecordFunctionCallback(nullptr)
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

static PyObject * set_forward_AD_enabled(PyObject* _unused, PyObject *arg) {
  HANDLE_TH_ERRORS
  if (!PyBool_Check(arg)) {
    throw TypeError("enabled must be a bool (got %s)", Py_TYPE(arg)->tp_name);
  }
  setForwardADEnabled(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject * is_forward_AD_enabled(PyObject* _unused, PyObject *arg) {
  HANDLE_TH_ERRORS
  if (isForwardADEnabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
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

static PyObject * python_enter_dual_level(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // It is unlikely that the depth of forward nesting will overflow int64_t so we
  // just static cast here.
  return utils::wrap(static_cast<int64_t>(forward_ad::enter_dual_level()));
  END_HANDLE_TH_ERRORS
}

static PyObject * python_exit_dual_level(PyObject* _unused, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "exit_dual_level(int64_t level)"
  });

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  forward_ad::exit_dual_level(_r.toInt64(0));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// autograd methods on torch._C
static PyMethodDef methods[] = { // NOLINT
  {"_set_grad_enabled", set_grad_enabled, METH_O, nullptr},
  {"is_grad_enabled", is_grad_enabled, METH_NOARGS, nullptr},
  {"_set_forward_AD_enabled", set_forward_AD_enabled, METH_O, nullptr},
  {"_is_forward_AD_enabled", is_forward_AD_enabled, METH_NOARGS, nullptr},
  {"set_autocast_enabled", set_autocast_enabled, METH_O, nullptr},
  {"is_autocast_enabled", is_autocast_enabled, METH_NOARGS, nullptr},
  {"clear_autocast_cache", clear_autocast_cache, METH_NOARGS, nullptr},
  {"autocast_increment_nesting", autocast_increment_nesting, METH_NOARGS, nullptr},
  {"autocast_decrement_nesting", autocast_decrement_nesting, METH_NOARGS, nullptr},
  {"set_anomaly_enabled", set_anomaly_mode_enabled, METH_O, nullptr},
  {"is_anomaly_enabled", is_anomaly_mode_enabled, METH_NOARGS, nullptr},
  {"_enter_dual_level", python_enter_dual_level, METH_NOARGS, nullptr},
  {"_exit_dual_level", castPyCFunctionWithKeywords(python_exit_dual_level), METH_VARARGS | METH_KEYWORDS, nullptr},
  {nullptr, nullptr, 0, nullptr}
};

PyMethodDef* python_functions() {
  return methods;
}

}} // namespace torch::autograd
