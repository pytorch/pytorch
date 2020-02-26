#include <torch/csrc/python_headers.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/autograd/python_function.h>
#include <torch/csrc/autograd/function.h>

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
      .def(py::init<ProfilerState, bool>());

  py::class_<Event>(m, "ProfilerEvent")
      .def("kind", &Event::kind)
      .def("name", [](const Event& e) { return e.name(); })
      .def("thread_id", &Event::thread_id)
      .def("device", &Event::device)
      .def("cpu_elapsed_us", &Event::cpu_elapsed_us)
      .def("cuda_elapsed_us", &Event::cuda_elapsed_us)
      .def("has_cuda", &Event::has_cuda)
      .def("shapes", &Event::shapes);

  m.def("_enable_profiler", enableProfiler);
  m.def("_disable_profiler", disableProfiler);
  m.def("_profiler_enabled", profilerEnabled);

  m.def("_push_range", [](std::string name) { pushRange(std::move(name)); });
  m.def("_pop_range", []() { popRange(); });
  m.def("_run_before_callbacks", runBeforeCallbacks);

  py::class_<RecordFunction, std::shared_ptr<RecordFunction>>(
      m, "_RecordFunction")
      .def(py::init<>())

  py::class_<
      RecordFunctionAsync,
      std::shared_ptr<RecordFunctionAsync>,
      RecordFunction>(m, "_RecordFunctionAsync")
      .def(py::init<>())
      .def(
          "before",
          [](RecordFunctionAsync& recordFunctionAsync, std::string name) {
            recordFunctionAsync.before(std::move(name));
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "exit_scope",
          &RecordFunctionAsync::exitScope,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "end",
          &RecordFunctionAsync::end,
          py::call_guard<py::gil_scoped_release>());

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
  {"set_anomaly_enabled", (PyCFunction)set_anomaly_mode_enabled, METH_O, nullptr},
  {"is_anomaly_enabled", (PyCFunction)is_anomaly_mode_enabled, METH_NOARGS, nullptr},
  {nullptr, nullptr, 0, nullptr}
};

PyMethodDef* python_functions() {
  return methods;
}

}} // namespace torch::autograd
