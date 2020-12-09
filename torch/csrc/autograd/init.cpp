#include <torch/csrc/python_headers.h>

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

static PyObject * python_make_dual(PyObject* _unused, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "make_dual(Tensor tensor, Tensor tangent, *, int64_t level)"
  });

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  return utils::wrap(forward_ad::make_dual(_r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  END_HANDLE_TH_ERRORS
}

static PyObject * python_unpack_dual(PyObject* _unused, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "unpack_dual(Tensor tensor, *, int64_t level)"
  });

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  return utils::wrap(forward_ad::unpack_dual(_r.tensor(0), _r.toInt64(1)));
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
  {"make_dual", castPyCFunctionWithKeywords(python_make_dual), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"unpack_dual", castPyCFunctionWithKeywords(python_unpack_dual), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"enter_dual_level", python_enter_dual_level, METH_NOARGS, nullptr},
  {"exit_dual_level", castPyCFunctionWithKeywords(python_exit_dual_level), METH_VARARGS | METH_KEYWORDS, nullptr},
  {nullptr, nullptr, 0, nullptr}
};

PyMethodDef* python_functions() {
  return methods;
}

}} // namespace torch::autograd
