#include <torch/csrc/python_headers.h>

#include <ATen/PythonTorchFunctionTLS.h>
#include <ATen/SavedTensorHooks.h>
#include <ATen/SequenceNumber.h>
#include <ATen/autocast_mode.h>
#include <ATen/core/PythonFallbackKernel.h>
#include <ATen/record_function.h>
#include <c10/core/DeviceType.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/ScalarType.h>
#include <c10/core/impl/PythonDispatcherTLS.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/autograd_not_implemented_fallback.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/autograd/profiler_python.h>
#include <torch/csrc/autograd/python_function.h>
#include <torch/csrc/autograd/python_saved_variable_hooks.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/record_function_ops.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/autograd/utils/python_arg_parsing.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/profiler/collection.h>
#include <torch/csrc/profiler/kineto_shim.h>
#include <torch/csrc/utils.h>
#include <torch/csrc/utils/disable_torch_function.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_raii.h>
#include <torch/csrc/utils/python_torch_function_mode.h>

#include <set>
#include <unordered_set>
#include <utility>

using torch::impl::py_context_manager;
using torch::impl::py_context_manager_DEPRECATED;

namespace {

struct DisableFuncTorch {
  DisableFuncTorch()
      : front_guard_(c10::DispatchKey::FuncTorchDynamicLayerFrontMode),
        back_guard_(c10::DispatchKey::FuncTorchDynamicLayerBackMode) {}
  c10::impl::ExcludeDispatchKeyGuard front_guard_;
  c10::impl::ExcludeDispatchKeyGuard back_guard_;
};

struct MultithreadingEnabled {
  MultithreadingEnabled(bool enabled)
      : old_(c10::AutogradState::get_tls_state().get_multithreading_enabled()) {
    c10::AutogradState::get_tls_state().set_multithreading_enabled(enabled);
  }
  ~MultithreadingEnabled() {
    c10::AutogradState::get_tls_state().set_multithreading_enabled(old_);
  }
  bool old_;
};

struct ViewReplayEnabled {
  ViewReplayEnabled(bool enabled)
      : old_(c10::AutogradState::get_tls_state().get_view_replay_enabled()) {
    c10::AutogradState::get_tls_state().set_view_replay_enabled(enabled);
  }
  ~ViewReplayEnabled() {
    c10::AutogradState::get_tls_state().set_view_replay_enabled(old_);
  }
  bool old_;
};

struct DisableAutocast {
  c10::impl::ExcludeDispatchKeyGuard guard_{c10::autocast_dispatch_keyset};
};

struct EnableTorchFunction {
  EnableTorchFunction()
      : old_(at::impl::PythonTorchFunctionTLS::get_disabled_state()) {
    at::impl::PythonTorchFunctionTLS::set_disabled_state(
        at::impl::TorchFunctionDisabledState::ENABLED);
  }
  ~EnableTorchFunction() {
    at::impl::PythonTorchFunctionTLS::set_disabled_state(old_);
  }
  at::impl::TorchFunctionDisabledState old_;
};

struct EnablePythonDispatcher {
  EnablePythonDispatcher() : old_(c10::impl::PythonDispatcherTLS::get_state()) {
    c10::impl::PythonDispatcherTLS::set_state(getPyInterpreter());
  }
  ~EnablePythonDispatcher() {
    c10::impl::PythonDispatcherTLS::set_state(old_);
  }
  c10::impl::PyInterpreter* old_;
};

struct EnablePreDispatch {
  EnablePreDispatch() : guard_(c10::DispatchKey::PreDispatch) {}
  c10::impl::IncludeDispatchKeyGuard guard_;
};

} // namespace

PyObject* THPAutograd_initExtension(PyObject* _unused, PyObject* unused) {
  using namespace torch::autograd::profiler;
  using namespace torch::profiler::impl;
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

  auto parameter_module =
      THPObjectPtr(PyImport_ImportModule("torch.nn.parameter"));
  if (!parameter_module)
    return nullptr;

  // NOTE: "leaks" ParameterClass
  ParameterClass = PyObject_GetAttrString(parameter_module, "Parameter");
  if (!ParameterClass)
    return nullptr;

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
      .def("flops", &LegacyEvent::flops)
      .def("is_async", &LegacyEvent::isAsync);

  py::enum_<c10::DeviceType>(m, "DeviceType")
      .value("CPU", c10::DeviceType::CPU)
      .value("CUDA", c10::DeviceType::CUDA)
      .value("MKLDNN", c10::DeviceType::MKLDNN)
      .value("OPENGL", c10::DeviceType::OPENGL)
      .value("OPENCL", c10::DeviceType::OPENCL)
      .value("IDEEP", c10::DeviceType::IDEEP)
      .value("HIP", c10::DeviceType::HIP)
      .value("FPGA", c10::DeviceType::FPGA)
      .value("ORT", c10::DeviceType::ORT)
      .value("XLA", c10::DeviceType::XLA)
      .value("Vulkan", c10::DeviceType::Vulkan)
      .value("Metal", c10::DeviceType::Metal)
      .value("XPU", c10::DeviceType::XPU)
      .value("MPS", c10::DeviceType::MPS)
      .value("Meta", c10::DeviceType::Meta)
      .value("HPU", c10::DeviceType::HPU)
      .value("VE", c10::DeviceType::VE)
      .value("Lazy", c10::DeviceType::Lazy)
      .value("IPU", c10::DeviceType::IPU)
      .value("PrivateUse1", c10::DeviceType::PrivateUse1);

  using torch::autograd::CreationMeta;
  py::enum_<CreationMeta>(m, "CreationMeta")
      .value("DEFAULT", CreationMeta::DEFAULT)
      .value("IN_CUSTOM_FUNCTION", CreationMeta::IN_CUSTOM_FUNCTION)
      .value("MULTI_OUTPUT_NODE", CreationMeta::MULTI_OUTPUT_NODE)
      .value("NO_GRAD_MODE", CreationMeta::NO_GRAD_MODE)
      .value("INFERENCE_MODE", CreationMeta::INFERENCE_MODE);

  py::class_<KinetoEvent>(m, "_KinetoEvent")
      // name of the event
      .def("name", [](const KinetoEvent& e) { return e.name(); })
      // PyTorch thread id of the start callback
      .def(
          "start_thread_id",
          [](const KinetoEvent& e) { return e.startThreadId(); })
      // PyTorch thread id of the end callback
      .def(
          "end_thread_id", [](const KinetoEvent& e) { return e.endThreadId(); })
      // for events of scope BACKWARD_FUNCTION - PyTorch thread id
      // of the corresponding forward op
      .def(
          "fwd_thread_id", [](const KinetoEvent& e) { return e.fwdThreadId(); })
      // together with fwd_thread_id, used to uniquely identify
      // the forward op
      .def("sequence_nr", [](const KinetoEvent& e) { return e.sequenceNr(); })
      // absolute start time (since unix epoch) in us
      .def("start_us", [](const KinetoEvent& e) { return e.startUs(); })
      // duration in us
      .def("duration_us", [](const KinetoEvent& e) { return e.durationUs(); })
      // used for correlation between high-level PyTorch events
      // and low-level device events
      .def(
          "correlation_id",
          [](const KinetoEvent& e) { return e.correlationId(); })
      // shapes of input tensors
      .def("shapes", [](const KinetoEvent& e) { return e.shapes().vec(); })
      .def("dtypes", [](const KinetoEvent& e) { return e.dtypes().vec(); })
      .def(
          "concrete_inputs",
          [](const KinetoEvent& e) {
            std::vector<py::object> as_pyobj;
            std::transform(
                e.concreteInputs().begin(),
                e.concreteInputs().end(),
                std::back_inserter(as_pyobj),
                [](const c10::IValue& val) {
                  return torch::jit::toPyObject(val);
                });
            return as_pyobj;
          })
      // stack traces of the PyTorch CPU events
      .def("stack", [](const KinetoEvent& e) { return e.stack().vec(); })
      // type of the RecordFunction that generated a PyTorch CPU event
      // (op, torchscript function, user label, etc)
      .def("scope", [](const KinetoEvent& e) { return e.scope(); })
      // device number, for CPU - process id
      .def("device_index", [](const KinetoEvent& e) { return e.deviceIndex(); })
      // for CUDA - stream id, for CPU - start thread id
      .def(
          "device_resource_id",
          [](const KinetoEvent& e) { return e.deviceResourceId(); })
      // device type
      .def("device_type", [](const KinetoEvent& e) { return e.deviceType(); })
      // correlation id of a linked event
      .def(
          "linked_correlation_id",
          [](const KinetoEvent& e) { return e.linkedCorrelationId(); })
      // compute flops
      .def("flops", [](const KinetoEvent& e) { return e.flops(); })
      // Whether this is async event or not
      .def("is_async", [](const KinetoEvent& e) { return e.isAsync(); })
      .def("cuda_elapsed_us", &KinetoEvent::cudaElapsedUs)
      .def("privateuse1_elapsed_us", &KinetoEvent::privateuse1ElapsedUs)
      .def("nbytes", [](const KinetoEvent& e) { return e.nBytes(); });

  m.def("_soft_assert_raises", &setSoftAssertRaises);
  m.def("_get_sequence_nr", &at::sequence_number::peek);

  py::class_<ProfilerResult>(m, "_ProfilerResult")
      .def("trace_start_us", &ProfilerResult::trace_start_us)
      .def("events", &ProfilerResult::events)
      .def("experimental_event_tree", &ProfilerResult::event_tree)
#ifdef USE_KINETO
      .def("save", &ProfilerResult::save)
#endif // USE_KINETO
      ;

  m.def(
      "_enable_profiler",
      &enableProfiler,
      py::arg("config"),
      py::arg("activities"),
      py::arg("scopes") = std::unordered_set<at::RecordScope>());
  m.def("_disable_profiler", disableProfiler);
  m.def("_prepare_profiler", prepareProfiler);
  m.def("_add_metadata_json", addMetadataJson); // Only if `USE_KINETO` is set
  m.def("_kineto_step", profilerStep); // Only if `USE_KINETO` is set
  m.def("kineto_available", []() { return torch::profiler::kKinetoAvailable; });

  // NOTICE: These record functions are not torch operators and may not show up
  // in TorchScript tracing, FX transforms, or operator serialization. For these
  // use cases, please use `torch.profiler.record_function`.
  // Creates a new profiling scope using RecordFunction and invokes its starting
  // callbacks.
  m.def(
      "_record_function_with_args_enter",
      [](const std::string& name, py::args args) {
        using torch::autograd::profiler::PythonRecordFunction;
        auto python_rec = c10::make_intrusive<PythonRecordFunction>(
            at::RecordScope::USER_SCOPE);
        auto* rec = &python_rec->record;
        if (rec->isActive()) {
          if (rec->needsInputs()) {
            auto iv_inputs = std::vector<c10::IValue>();
            for (const auto& arg : args) {
              iv_inputs.push_back(torch::jit::toTypeInferredIValue(arg));
            }
            rec->before(
                name,
                c10::ArrayRef<const c10::IValue>(
                    iv_inputs.data(), iv_inputs.size()));
          } else {
            rec->before(name);
          }
        }
        return torch::jit::toPyObject(std::move(python_rec));
      });

  // Ends the profiling scope created with record_function_with_param_enter.
  m.def("_record_function_with_args_exit", [](const py::object& obj) {
    using torch::autograd::profiler::PythonRecordFunction;
    auto python_record = torch::jit::toCustomClass<PythonRecordFunction>(obj);

    // We don't actually need to do anything with handle just need to persist
    // the lifetime until now.
    python_record->record.end();
  });

  m.def("_supported_activities", []() {
    std::set<ActivityType> activities{ActivityType::CPU};
#if defined(USE_KINETO) && \
    (!defined(LIBKINETO_NOCUPTI) || !defined(LIBKINETO_NOROCTRACER))
    if (at::getNumGPUs() > 0) {
      activities.insert(ActivityType::CUDA);
    }
#elif defined(USE_KINETO)
    if (at::hasXPU()) {
      activities.insert(ActivityType::XPU);
    }
    if (at::hasMTIA()) {
      activities.insert(ActivityType::MTIA);
    }
#endif
    return activities;
  });

  m.def("_unsafe_set_version_counter", [](at::Tensor t, int64_t i) {
    auto vc = torch::autograd::impl::version_counter(t);
    vc.set_version(i);
  });

  m.def("_enable_profiler_legacy", enableProfilerLegacy);
  py::class_<ProfilerDisableOptions>(m, "_ProfilerDisableOptions")
      .def(py::init<bool, bool>());
  m.def(
      "_disable_profiler_legacy",
      disableProfilerLegacy,
      py::arg("profiler_disable_options") = ProfilerDisableOptions());
  m.def("_profiler_enabled", profilerEnabled);
  m.def("_profiler_type", torch::profiler::impl::profilerType);
  m.def("_enable_record_function", [](bool enable) {
    at::enableRecordFunction(enable);
  });
  m.def("_set_empty_test_observer", [](bool is_global, double sampling_prob) {
    auto cb =
        at::RecordFunctionCallback(nullptr).needsInputs(true).samplingProb(
            sampling_prob);
    if (is_global) {
      at::addGlobalCallback(cb);
    } else {
      at::addThreadLocalCallback(cb);
    }
  });
  m.def("_clear_callbacks", []() { at::clearCallbacks(); });
  m.def(
      "_saved_tensors_hooks_is_enabled",
      at::SavedTensorDefaultHooks::is_enabled);
  m.def("_saved_tensors_hooks_enable", at::SavedTensorDefaultHooks::enable);
  m.def("_saved_tensors_hooks_disable", at::SavedTensorDefaultHooks::disable);
  m.def(
      "_saved_tensors_hooks_get_disabled_error_message",
      at::SavedTensorDefaultHooks::get_disabled_error_message);
  m.def(
      "_push_saved_tensors_default_hooks",
      [](py::function& pack_hook, py::function& unpack_hook) {
        torch::autograd::PyDefaultSavedVariableHooks::push_hooks(
            pack_hook, unpack_hook);
      });
  m.def("_pop_saved_tensors_default_hooks", []() {
    torch::autograd::PyDefaultSavedVariableHooks::pop_hooks();
  });

  m.def("_get_creation_meta", [](const at::Tensor& t) {
    auto* meta = torch::autograd::impl::get_view_autograd_meta(t);
    TORCH_CHECK(meta != nullptr);
    return meta->get_creation_meta();
  });

  m.def(
      "_set_creation_meta",
      [](const at::Tensor& t, CreationMeta new_creation_meta) {
        auto* meta = torch::autograd::impl::get_view_autograd_meta(t);
        TORCH_CHECK(meta != nullptr);
        meta->set_creation_meta(new_creation_meta);
      });

  _C_m.def(
      "_register_py_class_for_device",
      [](const std::string& device, py::object python_type_class) {
        auto cls = python_type_class.ptr();
        registerPythonTensorClass(device, cls);
      });
  _C_m.def("_set_autograd_fallback_mode", [](const std::string& mode) {
    if (mode == "nothing") {
      torch::autograd::setAutogradFallbackMode(
          torch::autograd::AutogradFallbackMode::Nothing);
      return;
    }
    if (mode == "warn") {
      torch::autograd::setAutogradFallbackMode(
          torch::autograd::AutogradFallbackMode::Warn);
      return;
    }
    if (mode == "error") {
      torch::autograd::setAutogradFallbackMode(
          torch::autograd::AutogradFallbackMode::Error);
      return;
    }
    TORCH_INTERNAL_ASSERT(false, "Unsupported AutogradFallbackMode: ", mode);
  });
  _C_m.def("_get_autograd_fallback_mode", []() {
    auto mode = torch::autograd::getAutogradFallbackMode();
    switch (mode) {
      case torch::autograd::AutogradFallbackMode::Nothing:
        return "nothing";
      case torch::autograd::AutogradFallbackMode::Warn:
        return "warn";
      case torch::autograd::AutogradFallbackMode::Error:
        return "error";
      default:
        TORCH_INTERNAL_ASSERT(false, "Unsupported AutogradFallbackMode");
    }
  });

  _C_m.def("_activate_cuda_trace", []() { activateCUDATrace(); });

  py_context_manager_DEPRECATED<c10::InferenceMode, bool>(
      _C_m, "_InferenceMode");
  py_context_manager<at::impl::RestorePythonTLSSnapshot>(
      _C_m, "_RestorePythonTLSSnapshot");

  py_context_manager_DEPRECATED<torch::DisableTorchDispatch>(
      _C_m, "_DisableTorchDispatch");
  py_context_manager_DEPRECATED<EnableTorchFunction>(
      _C_m, "_EnableTorchFunction");
  py_context_manager_DEPRECATED<EnablePythonDispatcher>(
      _C_m, "_EnablePythonDispatcher");
  py_context_manager<c10::impl::DisablePythonDispatcher>(
      _C_m, "_DisablePythonDispatcher");
  py_context_manager<EnablePreDispatch>(_C_m, "_EnablePreDispatch");
  py_context_manager_DEPRECATED<DisableFuncTorch>(_C_m, "_DisableFuncTorch");
  py_context_manager_DEPRECATED<MultithreadingEnabled, bool>(
      _C_m, "_MultithreadingEnabled");
  py_context_manager<DisableAutocast>(_C_m, "_DisableAutocast");
  py_context_manager<ViewReplayEnabled, bool>(_C_m, "_ViewReplayEnabled");
  py::class_<torch::autograd::SavedVariable>(std::move(m), "SavedTensor")
      .def(py::init([]() -> torch::autograd::SavedVariable {
        TORCH_CHECK(
            false,
            "Trying to create a SavedTensor object from Python is forbidden.");
      }))
      .def(
          "register_hooks",
          [](torch::autograd::SavedVariable& s,
             py::function& pack_hook,
             py::function& unpack_hook) {
            // Because we use a py::object, pybind will increment the refcount
            // of the hook functions for us
            s.register_hooks(
                std::make_unique<torch::autograd::PySavedVariableHooks>(
                    pack_hook, unpack_hook));
          });

  torch::autograd::profiler::python_tracer::init();
  Py_RETURN_TRUE;
}

namespace torch {
namespace autograd {

static PyObject* set_autocast_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (!PyBool_Check(arg)) {
    throw TypeError("enabled must be a bool (got %s)", Py_TYPE(arg)->tp_name);
  }
  at::autocast::set_enabled(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* is_autocast_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (at::autocast::is_enabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* is_any_autocast_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (at::autocast::is_enabled() || at::autocast::is_cpu_enabled() ||
      at::autocast::is_xpu_enabled() || at::autocast::is_ipu_enabled() ||
      at::autocast::is_xla_enabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* set_autocast_cpu_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (!PyBool_Check(arg)) {
    throw TypeError("enabled must be a bool (got %s)", Py_TYPE(arg)->tp_name);
  }
  at::autocast::set_cpu_enabled(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* is_autocast_cpu_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (at::autocast::is_cpu_enabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* set_autocast_ipu_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (!PyBool_Check(arg)) {
    throw TypeError("enabled must be a bool (got %s)", Py_TYPE(arg)->tp_name);
  }
  at::autocast::set_ipu_enabled(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* is_autocast_ipu_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (at::autocast::is_ipu_enabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* set_autocast_xla_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (!PyBool_Check(arg)) {
    throw TypeError("enabled must be a bool (got %s)", Py_TYPE(arg)->tp_name);
  }
  at::autocast::set_xla_enabled(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* is_autocast_xla_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (at::autocast::is_xla_enabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* set_autocast_gpu_dtype(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (!THPDtype_Check(arg)) {
    throw TypeError(
        "dtype must be a torch.dtype (got %s)", Py_TYPE(arg)->tp_name);
  }
  at::ScalarType targetType = reinterpret_cast<THPDtype*>(arg)->scalar_type;
  at::autocast::set_autocast_gpu_dtype(targetType);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* set_autocast_cpu_dtype(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (!THPDtype_Check(arg)) {
    throw TypeError(
        "dtype must be a torch.dtype (got %s)", Py_TYPE(arg)->tp_name);
  }
  at::ScalarType targetType = reinterpret_cast<THPDtype*>(arg)->scalar_type;
  at::autocast::set_autocast_cpu_dtype(targetType);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* set_autocast_ipu_dtype(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (!THPDtype_Check(arg)) {
    throw TypeError(
        "dtype must be a torch.dtype (got %s)", Py_TYPE(arg)->tp_name);
  }
  at::ScalarType targetType = reinterpret_cast<THPDtype*>(arg)->scalar_type;
  at::autocast::set_autocast_ipu_dtype(targetType);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* set_autocast_xla_dtype(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (!THPDtype_Check(arg)) {
    throw TypeError(
        "dtype must be a torch.dtype (got %s)", Py_TYPE(arg)->tp_name);
  }
  at::ScalarType targetType = reinterpret_cast<THPDtype*>(arg)->scalar_type;
  at::autocast::set_autocast_xla_dtype(targetType);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* get_autocast_gpu_dtype(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  at::ScalarType current_dtype = at::autocast::get_autocast_gpu_dtype();
  auto dtype = (PyObject*)torch::getTHPDtype(current_dtype);
  Py_INCREF(dtype);
  return dtype;
  END_HANDLE_TH_ERRORS
}

static PyObject* get_autocast_cpu_dtype(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  at::ScalarType current_dtype = at::autocast::get_autocast_cpu_dtype();
  auto dtype = (PyObject*)torch::getTHPDtype(current_dtype);
  Py_INCREF(dtype);
  return dtype;
  END_HANDLE_TH_ERRORS
}

static PyObject* get_autocast_ipu_dtype(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  at::ScalarType current_dtype = at::autocast::get_autocast_ipu_dtype();
  auto dtype = (PyObject*)torch::getTHPDtype(current_dtype);
  Py_INCREF(dtype);
  return dtype;
  END_HANDLE_TH_ERRORS
}

static PyObject* get_autocast_xla_dtype(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  at::ScalarType current_dtype = at::autocast::get_autocast_xla_dtype();
  auto dtype = (PyObject*)torch::getTHPDtype(current_dtype);
  Py_INCREF(dtype);
  return dtype;
  END_HANDLE_TH_ERRORS
}

static PyObject* clear_autocast_cache(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  at::autocast::clear_cache();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* autocast_increment_nesting(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(at::autocast::increment_nesting());
  END_HANDLE_TH_ERRORS
}

static PyObject* autocast_decrement_nesting(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(at::autocast::decrement_nesting());
  END_HANDLE_TH_ERRORS
}

static PyObject* is_autocast_cache_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (at::autocast::is_autocast_cache_enabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* set_autocast_cache_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (!PyBool_Check(arg)) {
    throw TypeError("enabled must be a bool (got %s)", Py_TYPE(arg)->tp_name);
  }
  at::autocast::set_autocast_cache_enabled(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* set_grad_enabled(
    PyObject* _unused,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
      "set_grad_enabled(bool enabled)",
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (at::impl::torch_function_mode_enabled()) {
    auto torch_C_module = THPObjectPtr(PyImport_ImportModule("torch._C"));
    return handle_torch_function(
        r, args, kwargs, torch_C_module, "torch._C", "_set_grad_enabled");
  }
  auto grad_enabled = r.toBool(0);
  GradMode::set_enabled(grad_enabled);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* is_grad_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (GradMode::is_enabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* set_fwd_grad_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (!PyBool_Check(arg)) {
    throw TypeError("enabled must be a bool (got %s)", Py_TYPE(arg)->tp_name);
  }
  c10::AutogradState::get_tls_state().set_fw_grad_mode(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* is_fwd_grad_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (c10::AutogradState::get_tls_state().get_fw_grad_mode()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* set_multithreading_enabled(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
      "set_multithreading_enabled(bool enabled)",
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  if (at::impl::torch_function_mode_enabled()) {
    auto torch_C_module = THPObjectPtr(PyImport_ImportModule("torch._C"));
    return handle_torch_function(
        r,
        args,
        kwargs,
        torch_C_module,
        "torch._C",
        "_set_multithreading_enabled");
  }
  auto multithreading_enabled = r.toBool(0);
  c10::AutogradState::get_tls_state().set_multithreading_enabled(
      multithreading_enabled);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* is_multithreading_enabled(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  if (c10::AutogradState::get_tls_state().get_multithreading_enabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* is_inference_mode_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (c10::InferenceMode::is_enabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* set_anomaly_mode_enabled(
    PyObject* _unused,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
      "set_anomaly_enabled(bool enabled, bool check_nan=True)",
  });
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  AnomalyMode::set_enabled(r.toBool(0), r.toBool(1));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* is_anomaly_mode_enabled(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  if (AnomalyMode::is_enabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* is_anomaly_check_nan_enabled(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  if (AnomalyMode::should_check_nan()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* python_enter_dual_level(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  // It is unlikely that the depth of forward nesting will overflow int64_t so
  // we just static cast here.
  return utils::wrap(static_cast<int64_t>(forward_ad::enter_dual_level()));
  END_HANDLE_TH_ERRORS
}

static PyObject* python_exit_dual_level(
    PyObject* _unused,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({"exit_dual_level(int64_t level)"});

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  auto idx = _r.toInt64(0);
  // Make sure the given index is valid before casting it
  TORCH_CHECK(idx >= 0, "Dual level must be a positive number.");
  forward_ad::exit_dual_level(static_cast<uint64_t>(idx));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* is_torch_function_mode_enabled(
    PyObject* _unused,
    PyObject* _unused2) {
  HANDLE_TH_ERRORS
  if (at::impl::torch_function_mode_enabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* push_on_torch_function_stack(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  if (arg != Py_None) {
    Py_INCREF(arg);
    at::impl::PythonTorchFunctionTLS::push_onto_stack(
        std::make_shared<c10::SafePyObject>(arg, getPyInterpreter()));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* pop_torch_function_stack(
    PyObject* _unused,
    PyObject* _unused2) {
  HANDLE_TH_ERRORS
  const auto& mode = at::impl::PythonTorchFunctionTLS::pop_stack();
  auto* r = mode->ptr(getPyInterpreter());
  Py_INCREF(r);
  return r;
  END_HANDLE_TH_ERRORS
}

static PyObject* get_function_stack_at(
    PyObject* _unused,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({"get_stack_at(int64_t level)"});

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  auto idx = _r.toInt64(0);
  const auto& mode = at::impl::PythonTorchFunctionTLS::get_stack_at(idx);
  auto* r = mode->ptr(getPyInterpreter());
  Py_INCREF(r);
  return r;
  END_HANDLE_TH_ERRORS
}

static PyObject* len_torch_function_stack(
    PyObject* _unused,
    PyObject* _unused2) {
  HANDLE_TH_ERRORS
  const auto len = at::impl::PythonTorchFunctionTLS::stack_len();
  return utils::wrap(static_cast<int64_t>(len));
  END_HANDLE_TH_ERRORS
}

static PyObject* push_on_torch_dispatch_stack(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  if (arg != Py_None) {
    Py_INCREF(arg);
    c10::impl::TorchDispatchModeTLS::push_onto_stack(
        std::make_shared<c10::SafePyObject>(arg, getPyInterpreter()));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* pop_torch_dispatch_stack(
    PyObject* _unused,
    PyObject* _unused2) {
  HANDLE_TH_ERRORS
  const auto& mode = c10::impl::TorchDispatchModeTLS::pop_stack();
  auto* r = mode->ptr(getPyInterpreter());
  Py_INCREF(r);
  return r;
  END_HANDLE_TH_ERRORS
}

static PyObject* get_dispatch_stack_at(
    PyObject* _unused,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({"get_stack_at(int64_t level)"});

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(args, kwargs, parsed_args);

  auto idx = _r.toInt64(0);
  const auto& mode = c10::impl::TorchDispatchModeTLS::get_stack_at(idx);
  auto* r = mode->ptr(getPyInterpreter());
  Py_INCREF(r);
  return r;
  END_HANDLE_TH_ERRORS
}

static PyObject* len_torch_dispatch_stack(
    PyObject* _unused,
    PyObject* _unused2) {
  HANDLE_TH_ERRORS
  const auto len = c10::impl::TorchDispatchModeTLS::stack_len();
  return utils::wrap(static_cast<int64_t>(len));
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_increment_version(PyObject* _unused, PyObject* tensor) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPVariable_Check(tensor), "increment_version expect a Tensor as input");
  torch::autograd::increment_version((THPVariable_Unpack(tensor)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// autograd methods on torch._C
static PyMethodDef methods[] = { // NOLINT
    {"_set_grad_enabled",
     castPyCFunctionWithKeywords(set_grad_enabled),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"is_grad_enabled", is_grad_enabled, METH_NOARGS, nullptr},
    {"_set_fwd_grad_enabled", set_fwd_grad_enabled, METH_O, nullptr},
    {"_is_fwd_grad_enabled", is_fwd_grad_enabled, METH_NOARGS, nullptr},
    {"is_inference_mode_enabled",
     is_inference_mode_enabled,
     METH_NOARGS,
     nullptr},
    {"set_autocast_enabled", set_autocast_enabled, METH_O, nullptr},
    {"is_autocast_enabled", is_autocast_enabled, METH_NOARGS, nullptr},
    {"_is_any_autocast_enabled", is_any_autocast_enabled, METH_NOARGS, nullptr},
    {"clear_autocast_cache", clear_autocast_cache, METH_NOARGS, nullptr},
    {"set_autocast_cpu_enabled", set_autocast_cpu_enabled, METH_O, nullptr},
    {"is_autocast_cpu_enabled", is_autocast_cpu_enabled, METH_NOARGS, nullptr},
    {"set_autocast_cpu_dtype", set_autocast_cpu_dtype, METH_O, nullptr},
    {"get_autocast_cpu_dtype", get_autocast_cpu_dtype, METH_NOARGS, nullptr},
    {"set_autocast_gpu_dtype", set_autocast_gpu_dtype, METH_O, nullptr},
    {"get_autocast_gpu_dtype", get_autocast_gpu_dtype, METH_NOARGS, nullptr},
    {"set_autocast_xla_enabled", set_autocast_xla_enabled, METH_O, nullptr},
    {"is_autocast_xla_enabled", is_autocast_xla_enabled, METH_NOARGS, nullptr},
    {"set_autocast_xla_dtype", set_autocast_xla_dtype, METH_O, nullptr},
    {"get_autocast_xla_dtype", get_autocast_xla_dtype, METH_NOARGS, nullptr},
    {"set_autocast_ipu_enabled", set_autocast_ipu_enabled, METH_O, nullptr},
    {"is_autocast_ipu_enabled", is_autocast_ipu_enabled, METH_NOARGS, nullptr},
    {"set_autocast_ipu_dtype", set_autocast_ipu_dtype, METH_O, nullptr},
    {"get_autocast_ipu_dtype", get_autocast_ipu_dtype, METH_NOARGS, nullptr},
    {"autocast_increment_nesting",
     autocast_increment_nesting,
     METH_NOARGS,
     nullptr},
    {"autocast_decrement_nesting",
     autocast_decrement_nesting,
     METH_NOARGS,
     nullptr},
    {"is_autocast_cache_enabled",
     is_autocast_cache_enabled,
     METH_NOARGS,
     nullptr},
    {"set_autocast_cache_enabled", set_autocast_cache_enabled, METH_O, nullptr},
    {"_increment_version", THPModule_increment_version, METH_O, nullptr},
    {"set_anomaly_enabled",
     castPyCFunctionWithKeywords(set_anomaly_mode_enabled),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"is_anomaly_enabled", is_anomaly_mode_enabled, METH_NOARGS, nullptr},
    {"is_anomaly_check_nan_enabled",
     is_anomaly_check_nan_enabled,
     METH_NOARGS,
     nullptr},
    {"_is_multithreading_enabled",
     is_multithreading_enabled,
     METH_NOARGS,
     nullptr},
    {"_set_multithreading_enabled",
     castPyCFunctionWithKeywords(set_multithreading_enabled),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_enter_dual_level", python_enter_dual_level, METH_NOARGS, nullptr},
    {"_exit_dual_level",
     castPyCFunctionWithKeywords(python_exit_dual_level),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_is_torch_function_mode_enabled",
     is_torch_function_mode_enabled,
     METH_NOARGS,
     nullptr},
    {"_push_on_torch_function_stack",
     push_on_torch_function_stack,
     METH_O,
     nullptr},
    {"_pop_torch_function_stack",
     pop_torch_function_stack,
     METH_NOARGS,
     nullptr},
    {"_get_function_stack_at",
     castPyCFunctionWithKeywords(get_function_stack_at),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_len_torch_function_stack",
     len_torch_function_stack,
     METH_NOARGS,
     nullptr},
    {"_push_on_torch_dispatch_stack",
     push_on_torch_dispatch_stack,
     METH_O,
     nullptr},
    {"_pop_torch_dispatch_stack",
     pop_torch_dispatch_stack,
     METH_NOARGS,
     nullptr},
    {"_get_dispatch_stack_at",
     castPyCFunctionWithKeywords(get_dispatch_stack_at),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_len_torch_dispatch_stack",
     len_torch_dispatch_stack,
     METH_NOARGS,
     nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* python_functions() {
  return methods;
}

} // namespace autograd
} // namespace torch
