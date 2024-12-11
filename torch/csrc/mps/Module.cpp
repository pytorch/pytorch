#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <ATen/ATen.h>
#include <c10/util/CallOnce.h>
#include <pybind11/pytypes.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>
#include <memory>

// pthread.h is included for tracking bad forks
#ifndef WIN32
#include <pthread.h>
#endif

#ifdef USE_MPS
#include <ATen/native/mps/MetalShaderLibrary.h>
#endif

namespace torch::mps {

namespace {
// True for children forked after mps init
static bool in_bad_fork = false;

// Called in the forked child if mps has already been initialized
static void forked_mps_child() {
  in_bad_fork = true;
}

// Should be called before the first mps call.
static void track_bad_mps_fork() {
#ifndef WIN32
  static c10::once_flag flag;
  c10::call_once(
      flag, [] { pthread_atfork(nullptr, nullptr, forked_mps_child); });
#endif
}
} // namespace

static PyObject* MPSModule_isInBadFork(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(in_bad_fork);
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_getDefaultMPSGenerator(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  track_bad_mps_fork();
  return THPGenerator_initDefaultGenerator(
      at::detail::getMPSHooks().getDefaultGenerator());
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_isAvailable(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  track_bad_mps_fork();
  if (at::detail::getMPSHooks().hasMPS()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_isMacOSorNewer(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  size_t major = 0;
  size_t minor = 0;
  if (!PyArg_ParseTuple(args, "LL", &major, &minor)) {
    return nullptr;
  }
  if (at::detail::getMPSHooks().isOnMacOSorNewer(major, minor)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_deviceSynchronize(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  at::detail::getMPSHooks().deviceSynchronize();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_emptyCache(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  at::detail::getMPSHooks().emptyCache();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_setMemoryFraction(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkDouble(args), "invalid argument to setMemoryFraction()");
  double fraction = THPUtils_unpackDouble(args);
  at::detail::getMPSHooks().setMemoryFraction(fraction);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_currentAllocatedMemory(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packUInt64(
      at::detail::getMPSHooks().getCurrentAllocatedMemory());
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_driverAllocatedMemory(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packUInt64(
      at::detail::getMPSHooks().getDriverAllocatedMemory());
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_recommendedMaxMemory(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packUInt64(
      at::detail::getMPSHooks().getRecommendedMaxMemory());
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_profilerStartTrace(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* mode_string_o = nullptr;
  PyObject* wait_until_completed_string_o = nullptr;
  if (!PyArg_ParseTuple(
          args, "OO", &mode_string_o, &wait_until_completed_string_o)) {
    return nullptr;
  }
  const std::string mode = THPUtils_unpackString(mode_string_o);
  const bool waitUntilCompleted =
      THPUtils_unpackBool(wait_until_completed_string_o);
  at::detail::getMPSHooks().profilerStartTrace(mode, waitUntilCompleted);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_profilerStopTrace(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  at::detail::getMPSHooks().profilerStopTrace();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_acquireEvent(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  const bool enable_timing = THPUtils_unpackBool(args);
  return THPUtils_packUInt32(
      at::detail::getMPSHooks().acquireEvent(enable_timing));
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_releaseEvent(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  const uint32_t event_id = THPUtils_unpackUInt32(args);
  at::detail::getMPSHooks().releaseEvent(event_id);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_recordEvent(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  const uint32_t event_id = THPUtils_unpackUInt32(args);
  at::detail::getMPSHooks().recordEvent(event_id);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_waitForEvent(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  const uint32_t event_id = THPUtils_unpackUInt32(args);
  at::detail::getMPSHooks().waitForEvent(event_id);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_synchronizeEvent(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  const uint32_t event_id = THPUtils_unpackUInt32(args);
  at::detail::getMPSHooks().synchronizeEvent(event_id);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_queryEvent(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  const uint32_t event_id = THPUtils_unpackUInt32(args);

  if (at::detail::getMPSHooks().queryEvent(event_id)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* MPSModule_elapsedTimeOfEvents(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* start_event_o = nullptr;
  PyObject* end_event_o = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &start_event_o, &end_event_o)) {
    return nullptr;
  }
  const uint32_t start_event_id = THPUtils_unpackUInt32(start_event_o);
  const uint32_t end_event_id = THPUtils_unpackUInt32(end_event_o);
  return PyFloat_FromDouble(at::detail::getMPSHooks().elapsedTimeOfEvents(
      start_event_id, end_event_id));
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(*-c-arrays, *-global-variables)
static struct PyMethodDef _MPSModule_methods[] = {
    {"_mps_deviceSynchronize",
     MPSModule_deviceSynchronize,
     METH_NOARGS,
     nullptr},
    {"_mps_is_in_bad_fork", MPSModule_isInBadFork, METH_NOARGS, nullptr},
    {"_mps_is_available", MPSModule_isAvailable, METH_NOARGS, nullptr},
    {"_mps_is_on_macos_or_newer",
     MPSModule_isMacOSorNewer,
     METH_VARARGS,
     nullptr},
    {"_mps_get_default_generator",
     MPSModule_getDefaultMPSGenerator,
     METH_NOARGS,
     nullptr},
    {"_mps_emptyCache", MPSModule_emptyCache, METH_NOARGS, nullptr},
    {"_mps_setMemoryFraction", MPSModule_setMemoryFraction, METH_O, nullptr},
    {"_mps_currentAllocatedMemory",
     MPSModule_currentAllocatedMemory,
     METH_NOARGS,
     nullptr},
    {"_mps_driverAllocatedMemory",
     MPSModule_driverAllocatedMemory,
     METH_NOARGS,
     nullptr},
    {"_mps_recommendedMaxMemory",
     MPSModule_recommendedMaxMemory,
     METH_NOARGS,
     nullptr},
    {"_mps_profilerStartTrace",
     MPSModule_profilerStartTrace,
     METH_VARARGS,
     nullptr},
    {"_mps_profilerStopTrace",
     MPSModule_profilerStopTrace,
     METH_NOARGS,
     nullptr},
    {"_mps_acquireEvent", MPSModule_acquireEvent, METH_O, nullptr},
    {"_mps_releaseEvent", MPSModule_releaseEvent, METH_O, nullptr},
    {"_mps_recordEvent", MPSModule_recordEvent, METH_O, nullptr},
    {"_mps_waitForEvent", MPSModule_waitForEvent, METH_O, nullptr},
    {"_mps_synchronizeEvent", MPSModule_synchronizeEvent, METH_O, nullptr},
    {"_mps_queryEvent", MPSModule_queryEvent, METH_O, nullptr},
    {"_mps_elapsedTimeOfEvents",
     MPSModule_elapsedTimeOfEvents,
     METH_VARARGS,
     nullptr},
    {nullptr}};

PyMethodDef* python_functions() {
  return _MPSModule_methods;
}

#ifdef USE_MPS
namespace {
template <typename T = uint64_t>
std::optional<std::vector<T>> optional_vec_from_pyobject(
    const py::object& py_value) {
  if (py_value.is_none()) {
    return std::nullopt;
  }
  if (py::isinstance<py::int_>(py_value)) {
    return std::vector({py_value.cast<T>()});
  }
  auto vec = py_value.cast<std::vector<T>>();
  TORCH_CHECK(vec.size() > 0 && vec.size() < 4);
  return vec;
}

struct OptionalArgCaster {
 public:
  OptionalArgCaster(const py::object& arg) {
    if (arg.is_none()) {
    } else if (py::isinstance<py::str>(arg)) {
      default_cast = arg.cast<std::string>();
    } else if (py::isinstance<py::dict>(arg)) {
      cast_map = arg.cast<std::unordered_map<unsigned, std::string>>();
    } else {
      TORCH_CHECK(
          false,
          "Unexpected caster arg type ",
          arg.attr("__class__").attr("__name__").cast<const std::string>());
    }
  }
  template <typename T>
  void setValue(
      ::at::native::mps::MetalKernelFunction& f,
      unsigned idx,
      const std::vector<T>& values) {
    auto cast_str =
        cast_map.find(idx) != cast_map.end() ? cast_map[idx] : default_cast;
    if (cast_str.size() == 0) {
      f.setArg(idx, values);
    } else if (cast_str == "fp16") {
      std::vector<c10::Half> cast_values(values.begin(), values.end());
      f.setArg(idx, cast_values);
    } else if (cast_str == "bf16") {
      std::vector<c10::BFloat16> cast_values(values.begin(), values.end());
      f.setArg(idx, cast_values);
    } else if (cast_str == "int32") {
      std::vector<int32_t> cast_values(values.begin(), values.end());
      f.setArg(idx, cast_values);
    } else if (cast_str == "int16") {
      std::vector<int16_t> cast_values(values.begin(), values.end());
      f.setArg(idx, cast_values);
    } else if (cast_str == "int8") {
      std::vector<int8_t> cast_values(values.begin(), values.end());
      f.setArg(idx, cast_values);
    } else if (cast_str == "uint8") {
      std::vector<uint8_t> cast_values(values.begin(), values.end());
      f.setArg(idx, cast_values);
    } else {
      TORCH_CHECK(false, "Unsupported cast instruction ", default_cast);
    }
  }

  template <
      typename T,
      typename = std::enable_if_t<
          std::is_same_v<float, T> || std::is_same_v<int64_t, T>>>
  void setValue(
      ::at::native::mps::MetalKernelFunction& f,
      unsigned idx,
      const T& value) {
    auto cast_str =
        cast_map.find(idx) != cast_map.end() ? cast_map[idx] : default_cast;
    if (cast_str.size() == 0) {
      f.setArg(idx, value);
    } else if (cast_str == "fp16") {
      f.setArg(idx, static_cast<c10::Half>(value));
    } else if (cast_str == "bf16") {
      f.setArg(idx, static_cast<c10::BFloat16>(value));
    } else if (cast_str == "int32") {
      f.setArg(idx, static_cast<int32_t>(value));
    } else if (cast_str == "int16") {
      f.setArg(idx, static_cast<int16_t>(value));
    } else if (cast_str == "int8") {
      f.setArg(idx, static_cast<int8_t>(value));
    } else if (cast_str == "uint8") {
      f.setArg(idx, static_cast<uint8_t>(value));
    } else {
      TORCH_CHECK(false, "Unsupported cast instruction ", default_cast);
    }
  }

  void setValue(
      ::at::native::mps::MetalKernelFunction& f,
      unsigned idx,
      const py::object& arg) {
    if (py::isinstance<py::tuple>(arg) || py::isinstance<py::list>(arg)) {
      auto len = arg.attr("__len__")().cast<uint64_t>();
      TORCH_CHECK(
          len > 0, "Empty list/tuple can not be an argument to metal kernel")
      auto element = arg.attr("__getitem__")(0);
      if (py::isinstance<py::int_>(element)) {
        auto values = arg.cast<std::vector<int64_t>>();
        setValue(f, idx, values);
      } else if (py::isinstance<py::float_>(element)) {
        auto values = arg.cast<std::vector<float>>();
        setValue(f, idx, values);
      } else {
        TORCH_CHECK(false, "Unexpected argument types");
      }
    } else if (py::isinstance<py::float_>(arg)) {
      auto value = arg.cast<float>();
      setValue(f, idx, value);
    } else if (py::isinstance<py::int_>(arg)) {
      auto value = arg.cast<int64_t>();
      setValue(f, idx, value);
    } else {
      TORCH_CHECK(false, "Unsupported argument type");
    }
  }

 private:
  std::string default_cast;
  std::unordered_map<unsigned, std::string> cast_map;
};

} // namespace

void initModule(PyObject* module) {
  using namespace at::native::mps;
  auto m = py::handle(module).cast<py::module>();
  py::class_<
      DynamicMetalShaderLibrary,
      std::shared_ptr<DynamicMetalShaderLibrary>>(m, "_mps_ShaderLibrary")
      .def(
          "__getattr__",
          [](DynamicMetalShaderLibrary& self, const std::string& name) {
            return self.getKernelFunction(name);
          })
      .def("__dir__", [](DynamicMetalShaderLibrary& self) {
        return self.getFunctionNames();
      });
  py::class_<MetalKernelFunction, std::shared_ptr<MetalKernelFunction>>(
      m, "_mps_MetalKernel")
      .def(
          "__call__",
          [](MetalKernelFunction& self,
             const py::args& args,
             const py::object& py_threads,
             const py::object& py_group_size,
             const py::object& arg_casts) {
            auto threads = optional_vec_from_pyobject(py_threads);
            auto group_size = optional_vec_from_pyobject(py_group_size);
            OptionalArgCaster caster(arg_casts);
            self.runCommandBlock([&] {
              self.startEncoding();
              for (auto idx : c10::irange(args.size())) {
                if (THPVariable_Check(args[idx].ptr())) {
                  auto t = THPVariable_Unpack(args[idx].ptr());
                  self.setArg(idx, t);
                  if (!threads) {
                    threads = {static_cast<uint64_t>(t.numel())};
                  }
                  continue;
                }
                caster.setValue(self, idx, args[idx]);
              }
              TORCH_CHECK(
                  threads.has_value() && threads->size() < 4,
                  "Number of threads is undefined or has wrong dimention");
              TORCH_CHECK(
                  !group_size.has_value() ||
                  threads->size() == group_size->size());
              if (threads->size() == 1) {
                if (group_size.has_value()) {
                  self.dispatch(threads->at(0), group_size->at(0));
                } else {
                  self.dispatch(threads->at(0));
                }
              } else if (threads->size() == 2) {
                if (group_size.has_value()) {
                  self.dispatch(
                      {threads->at(0), threads->at(1)},
                      {group_size->at(0), group_size->at(1)});
                } else {
                  self.dispatch({threads->at(0), threads->at(1)});
                }
              } else {
                if (group_size.has_value()) {
                  self.dispatch(
                      {threads->at(0), threads->at(1), threads->at(2)},
                      {group_size->at(0),
                       group_size->at(1),
                       group_size->at(2)});
                } else {
                  self.dispatch(
                      {threads->at(0), threads->at(1), threads->at(2)});
                }
              }
            });
          },
          py::kw_only(),
          py::arg("threads") = py::none(),
          py::arg("group_size") = py::none(),
          py::arg("arg_casts") = py::none())
      .def_property_readonly(
          "max_threads_per_threadgroup",
          &MetalKernelFunction::getMaxThreadsPerThreadgroup)
      .def_property_readonly(
          "thread_execution_width",
          &MetalKernelFunction::getThreadExecutionWidth)
      .def_property_readonly(
          "static_thread_group_memory_length",
          &MetalKernelFunction::getStaticThreadGroupMemoryLength);
  m.def("_mps_compileShader", [](const std::string& source) {
    return std::make_shared<DynamicMetalShaderLibrary>(source);
  });
}
#endif /* USE_MPS */

} // namespace torch::mps
