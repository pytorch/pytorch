#include <torch/csrc/profiler/python/init.h>

#include <ATen/record_function.h>
#include <c10/core/impl/PyInterpreter.h>
#include <c10/util/overloaded.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/profiler/collection.h>
#include <torch/csrc/profiler/python/combined_traceback.h>
#include <torch/csrc/profiler/standalone/execution_trace_observer.h>
#include <torch/csrc/utils/pybind.h>

struct THPCapturedTraceback {
  PyObject_HEAD
  std::shared_ptr<torch::CapturedTraceback> data;
};

static int THPCapturedTraceback_traverse(
    PyObject* self,
    visitproc visit,
    void* arg) {
  return ((THPCapturedTraceback*)self)
      ->data->traversePython((int (*)(void*, void*))visit, arg);
}

static int THPCapturedTraceback_clear(PyObject* self) {
  return ((THPCapturedTraceback*)self)->data->clearPython();
}

static void THPCapturedTraceback_dealloc(PyObject* self_) {
  auto* self = (THPCapturedTraceback*)self_;
  PyObject_GC_UnTrack(self);
  self->data.~shared_ptr<torch::CapturedTraceback>();
  // promptly trigger delayed frees since we have GIL
  torch::freeDeadCapturedTracebackFrames();
  PyObject_GC_Del(self);
}

static PyTypeObject THPCapturedTracebackType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C._profiler.CapturedTraceback", /* tp_name */
    sizeof(THPCapturedTraceback), /* tp_basicsize */
    0, /* tp_itemsize */
    THPCapturedTraceback_dealloc, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    // NOLINTNEXTLINE(misc-redundant-expression)
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, /* tp_flags */
    nullptr, /* tp_doc */
    (traverseproc)THPCapturedTraceback_traverse, /* tp_traverse */
    (inquiry)THPCapturedTraceback_clear, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    nullptr, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    nullptr, /* tp_new */
};

namespace pybind11::detail {

template <>
struct type_caster<std::shared_ptr<torch::CapturedTraceback>> {
 public:
  PYBIND11_TYPE_CASTER(
      std::shared_ptr<torch::CapturedTraceback>,
      _("torch._C._profiler.CapturedTraceback"));

  bool load(handle src, bool) {
    if (Py_TYPE(src.ptr()) == &THPCapturedTracebackType) {
      value = reinterpret_cast<THPCapturedTraceback*>(src.ptr())->data;
      return true;
    }
    return false;
  }

  static handle cast(
      std::shared_ptr<torch::CapturedTraceback> src,
      return_value_policy /* policy */,
      handle /* parent */) {
    auto* r = PyObject_GC_New(THPCapturedTraceback, &THPCapturedTracebackType);
    new (&r->data) std::shared_ptr<torch::CapturedTraceback>(std::move(src));
    return py::handle((PyObject*)r);
  }
};

} // namespace pybind11::detail

namespace torch::profiler {

/* [NOTE: RecordFunctionFast]
 * This is an alternate way to call record_function from python.
 * The torch.profiler.record_function context manager is slow (~14us on
 * benchmarks in Aug 2023), which is usually fine for module-level annotations
 * in python, but slow for per-op annotations. Part of the reason it is slow is
 * because the calls go through the dispatcher, in order to make the
 * record_function calls work with torchscript.
 *
 * This implementation doesn't go through the dispatcher and so it won't work
 * with any feature relying on the dispatcher (e.g. torchscript or
 * torch.compile)
 *
 * An alternate solution would be to implement a python context manager that
 * calls into C++ for the enter/exit function:
 *    @contextlib.contextmanager
 *    def record_function_fast(name):
 *      rf = torch._C._record_function_fast_enter(name)
 *      try:
 *        yield
 *      finally:
 *        torch._C._record_function_fast_exit(rf)
 * The C++ implementation here is faster by ~0.2-0.4us per context manager.
 */

namespace {
struct RecordFunctionFast {
  PyObject_HEAD
  PyObject* name;
  PyObject* input_values;
  PyObject* keyword_values;
  std::unique_ptr<at::RecordFunction> guard;
};

PyObject* RecordFunctionFast_new(
    PyTypeObject* subtype,
    PyObject* args,
    PyObject* kwargs) {
  RecordFunctionFast* self = (RecordFunctionFast*)subtype->tp_alloc(subtype, 0);
  if (self != nullptr) {
    self->name = nullptr;
    self->input_values = nullptr;
    self->keyword_values = nullptr;
    self->guard.reset();
  }
  return (PyObject*)self;
}

int RecordFunctionFast_init(
    PyObject* selfGeneric,
    PyObject* args,
    PyObject* kwargs) {
  auto self = (RecordFunctionFast*)selfGeneric;
  // NOLINTNEXTLINE(*-c-arrays*)
  constexpr const char* kwlist[] = {
      "name", "input_values", "keyword_values", nullptr};
  PyObject* name = nullptr;
  PyObject* input_values = nullptr;
  PyObject* keyword_values = nullptr;
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "O|OO", // name is required PyObject, args and kwargs are optional
                  // PyObjects
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(kwlist),
          &name,
          &input_values,
          &keyword_values)) {
    return -1;
  }
  if (name) {
    TORCH_CHECK(
        THPUtils_checkString(name),
        "The name passed to RecordFunctionFast must be a string");
    Py_INCREF(name);
    self->name = name;
  }
  if (input_values) {
    TORCH_CHECK(
        PyList_Check(input_values) || PyTuple_Check(input_values),
        "input_values must be a list or tuple");
    Py_INCREF(input_values);
    self->input_values = input_values;
  }
  if (keyword_values) {
    TORCH_CHECK(PyDict_Check(keyword_values), "keyword_values must be dict");
    Py_INCREF(keyword_values);
    self->keyword_values = keyword_values;
  }
  return 0;
}

void RecordFunctionFast_dealloc(PyObject* selfGeneric) {
  auto self = (RecordFunctionFast*)selfGeneric;
  Py_CLEAR(self->name);
  Py_CLEAR(self->input_values);
  Py_CLEAR(self->keyword_values);
  if (self->guard) {
    self->guard.reset();
  }
  Py_TYPE(self)->tp_free(self);
}

PyObject* RecordFunctionFast_enter(PyObject* selfGeneric, PyObject* unused) {
  HANDLE_TH_ERRORS
  if (torch::profiler::impl::ProfilerStateBase::get() != nullptr) {
    auto self = (RecordFunctionFast*)selfGeneric;
    TORCH_INTERNAL_ASSERT(
        !self->guard,
        "Trying to enter a new record_function_fast context but the guard is unexpectedly already set");
    self->guard =
        std::make_unique<at::RecordFunction>(at::RecordScope::FUNCTION);
    std::vector<at::IValue> args;
    std::unordered_map<std::string, at::IValue> kwargs;
    bool profiler_need_input = torch::autograd::profiler::profilerEnabled() &&
        torch::autograd::profiler::getProfilerConfig().report_input_shapes;
    // parse through args if they exist
    if (self->input_values != nullptr && profiler_need_input) {
      THPObjectPtr input_fast(
          PySequence_Fast(self->input_values, "input must be a sequence"));
      PyObject** input_items = PySequence_Fast_ITEMS(input_fast.get());
      for (int i = 0; i < PySequence_Fast_GET_SIZE(input_fast.get()); i++) {
        PyObject* item = input_items[i];
        auto match = torch::jit::tryToInferType(item);
        if (match.success()) {
          args.push_back(torch::jit::toIValue(item, match.type()));
        }
      }
    }

    // parse through kwargs if they exist
    if (self->keyword_values != nullptr && profiler_need_input) {
      Py_ssize_t pos = 0;
      PyObject *key = nullptr, *value = nullptr;
      while (PyDict_Next(self->keyword_values, &pos, &key, &value)) {
        // Get the string representation of the key and value
        std::string key_str = THPUtils_unpackString(key);
        at::IValue ivalue;
        if (THPUtils_checkString(value)) {
          ivalue = at::IValue(THPUtils_unpackString(value));
        } else {
          auto match = torch::jit::tryToInferPrimitiveType(value);
          if (match.success()) {
            ivalue = torch::jit::toIValue(value, match.type());
          } else {
            TORCH_WARN("Unable to infer type of value for keyword: ", key_str);
            ivalue = at::IValue("NULL");
          }
        }
        kwargs[key_str] = ivalue;
      }
    }
    self->guard->before(THPUtils_unpackString(self->name), &args, &kwargs);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* RecordFunctionFast_exit(PyObject* selfGeneric, PyObject* unused) {
  HANDLE_TH_ERRORS
  if (torch::profiler::impl::ProfilerStateBase::get() != nullptr) {
    auto self = (RecordFunctionFast*)selfGeneric;
    TORCH_INTERNAL_ASSERT(
        self->guard,
        "Trying to exit an active record_function_fast context but no guard is set");
    self->guard.reset();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
} // namespace

void initPythonBindings(PyObject* module) {
  auto rootModule = py::handle(module).cast<py::module>();
  auto m = rootModule.def_submodule("_profiler");

  using namespace torch::profiler::impl;

  py::enum_<at::RecordScope>(m, "RecordScope")
      .value("FUNCTION", at::RecordScope::FUNCTION)
      .value("BACKWARD_FUNCTION", at::RecordScope::BACKWARD_FUNCTION)
      .value("TORCHSCRIPT_FUNCTION", at::RecordScope::TORCHSCRIPT_FUNCTION)
      .value("KERNEL_FUNCTION_DTYPE", at::RecordScope::KERNEL_FUNCTION_DTYPE)
      .value("CUSTOM_CLASS", at::RecordScope::CUSTOM_CLASS)
      .value("BUILD_FEATURE", at::RecordScope::BUILD_FEATURE)
      .value("LITE_INTERPRETER", at::RecordScope::LITE_INTERPRETER)
      .value("USER_SCOPE", at::RecordScope::USER_SCOPE)
      .value("STATIC_RUNTIME_OP", at::RecordScope::STATIC_RUNTIME_OP)
      .value("STATIC_RUNTIME_MODEL", at::RecordScope::STATIC_RUNTIME_MODEL);

  py::enum_<ProfilerState>(m, "ProfilerState")
      .value("Disabled", ProfilerState::Disabled)
      .value("CPU", ProfilerState::CPU)
      .value("CUDA", ProfilerState::CUDA)
      .value("NVTX", ProfilerState::NVTX)
      .value("ITT", ProfilerState::ITT)
      .value("PRIVATEUSE1", ProfilerState::PRIVATEUSE1)
      .value("KINETO", ProfilerState::KINETO)
      .value("KINETO_GPU_FALLBACK", ProfilerState::KINETO_GPU_FALLBACK)
      .value(
          "KINETO_PRIVATEUSE1_FALLBACK",
          ProfilerState::KINETO_PRIVATEUSE1_FALLBACK);

  py::enum_<ActiveProfilerType>(m, "ActiveProfilerType")
      .value("NONE", ActiveProfilerType::NONE)
      .value("LEGACY", ActiveProfilerType::LEGACY)
      .value("KINETO", ActiveProfilerType::KINETO)
      .value("NVTX", ActiveProfilerType::NVTX)
      .value("ITT", ActiveProfilerType::ITT)
      .value("PRIVATEUSE1", ActiveProfilerType::PRIVATEUSE1);

  py::enum_<ActivityType>(m, "ProfilerActivity")
      .value("CPU", ActivityType::CPU)
      .value("XPU", ActivityType::XPU)
      .value("MTIA", ActivityType::MTIA)
      .value("CUDA", ActivityType::CUDA)
      .value("HPU", ActivityType::HPU)
      .value("PrivateUse1", ActivityType::PrivateUse1);

  py::class_<ExperimentalConfig>(m, "_ExperimentalConfig")
      .def(
          py::init<
              std::vector<std::string> /* profiler_metrics */,
              bool /* profiler_measure_per_kernel */,
              bool /* verbose */,
              std::vector<std::string> /* performance_events  */,
              bool /* enable_cuda_sync_events */,
              bool /* adjust_profiler_step */,
              bool /* disable_external_correlation*/,
              bool /* profile_all_threads */,
              bool /* capture_overload_names */,
              std::string /* custom_profiler_config*/
              >(),
          "An experimental config for Kineto features. Please note that"
          "backward compatibility is not guaranteed.\n"
          "    profiler_metrics : a list of CUPTI profiler metrics used\n"
          "       to measure GPU performance events.\n"
          "       If this list contains values Kineto runs in CUPTI profiler mode\n"
          "    profiler_measure_per_kernel (bool) : whether to profile metrics per kernel\n"
          "       or for the entire measurement duration.\n"
          "    verbose (bool) : whether the trace file has `Call stack` field or not.\n"
          "    performance_events : a list of profiler events to be used for measurement.\n"
          "    enable_cuda_sync_events : for CUDA profiling mode, enable adding CUDA synchronization events\n"
          "       that expose CUDA device, stream and event synchronization activities. This feature is new\n"
          "       and currently disabled by default.\n"
          "    adjust_profiler_step (bool) : whether to adjust the profiler step to\n"
          "       match the parent python event duration. This feature is new and currently disabled by default.\n"
          "    disable_external_correlation (bool) : whether to disable external correlation\n"
          "    profile_all_threads (bool) : whether to profile all threads\n"
          "    capture_overload_names (bool) : whether to include ATen overload names in the profile\n"
          "    custom_profiler_config (string) : Used to pass some configurations to the custom profiler backend.\n",
          py::arg("profiler_metrics") = std::vector<std::string>(),
          py::arg("profiler_measure_per_kernel") = false,
          py::arg("verbose") = false,
          py::arg("performance_events") = std::vector<std::string>(),
          py::arg("enable_cuda_sync_events") = false,
          py::arg("adjust_profiler_step") = false,
          py::arg("disable_external_correlation") = false,
          py::arg("profile_all_threads") = false,
          py::arg("capture_overload_names") = false,
          py::arg("custom_profiler_config") = "")
      .def(py::pickle(
          [](const ExperimentalConfig& p) { // __getstate__
            py::list py_metrics;
            for (const auto& metric : p.profiler_metrics) {
              py::bytes mbytes(metric);
              py_metrics.append(mbytes);
            }
            py::list py_perf_events;
            for (const auto& event : p.performance_events) {
              py::bytes mbytes(event);
              py_perf_events.append(mbytes);
            }
            /* Return a tuple that fully encodes the state of the config */
            return py::make_tuple(
                py_metrics,
                p.profiler_measure_per_kernel,
                p.verbose,
                p.enable_cuda_sync_events,
                p.adjust_profiler_step,
                p.disable_external_correlation,
                p.profile_all_threads,
                p.capture_overload_names,
                p.custom_profiler_config,
                p.performance_events);
          },
          [](const py::tuple& t) { // __setstate__
            if (t.size() >= 5) {
              throw std::runtime_error("Expected at least 5 values in state");
            }

            py::list py_metrics = t[0].cast<py::list>();
            std::vector<std::string> metrics{py_metrics.size()};

            for (const auto& py_metric : py_metrics) {
              metrics.push_back(py::str(py_metric));
            }

            std::vector<std::string> performance_events;
            if (t.size() == 5) {
              py::list py_perf_events = t[4].cast<py::list>();
              performance_events.resize(py_perf_events.size());
              for (const auto& py_perf_event : py_perf_events) {
                performance_events.push_back(py::str(py_perf_event));
              }
            }

            return ExperimentalConfig(
                std::move(metrics),
                t[1].cast<bool>(),
                t[2].cast<bool>(),
                std::move(performance_events),
                t[3].cast<bool>(),
                t[4].cast<bool>());
          }));

  py::class_<ProfilerConfig>(m, "ProfilerConfig")
      .def(
          py::init<
              ProfilerState,
              bool, /* report_input_shapes */
              bool, /* profile_memory */
              bool, /* with_stack */
              bool, /* with_flops */
              bool, /* with_modules */
              ExperimentalConfig /* experimental_config */,
              std::string /* trace_id */
              >(),
          py::arg("state"),
          py::arg("report_input_shapes"),
          py::arg("profile_memory"),
          py::arg("with_stack"),
          py::arg("with_flops"),
          py::arg("with_modules"),
          py::arg("experimental_config"),
          py::arg("trace_id") = "" // Make trace_id the only optional param
      );

  py::enum_<EventType>(m, "_EventType")
      .value("TorchOp", EventType::TorchOp)
      .value("Backend", EventType::Backend)
      .value("Vulkan", EventType::Vulkan)
      .value("Allocation", EventType::Allocation)
      .value("PyCall", EventType::PyCall)
      .value("PyCCall", EventType::PyCCall)
      .value("Kineto", EventType::Kineto);

  py::class_<TensorMetadata>(m, "_TensorMetadata")
      .def_property_readonly("impl_ptr", &TensorMetadata::impl)
      .def_readonly("storage_data_ptr", &TensorMetadata::data_)
      .def_readonly("id", &TensorMetadata::id_)
      .def_readonly("allocation_id", &TensorMetadata::allocation_id_)
      .def_property_readonly(
          "layout",
          [](const TensorMetadata& metadata) {
            PyObject* layout_obj =
                torch::autograd::utils::wrap(metadata.layout_);
            return py::reinterpret_borrow<py::object>(layout_obj);
          })
      .def_readonly("device", &TensorMetadata::device_)
      .def_property_readonly(
          "dtype",
          [](const TensorMetadata& metadata) {
            return py::reinterpret_borrow<py::object>(
                torch::autograd::utils::wrap(metadata.dtype_));
          })
      .def_readonly("dim", &TensorMetadata::size_dim_)
      .def_readonly("sizes", &TensorMetadata::sizes_)
      .def_readonly("strides", &TensorMetadata::strides_);

  using torch_op_t = ExtraFields<EventType::TorchOp>;
  py::class_<torch_op_t>(m, "_ExtraFields_TorchOp")
      .def_readonly("name", &torch_op_t::name_)
      .def_property_readonly(
          "inputs",
          [](const torch_op_t& op) {
            py::list out;
            for (const auto& input : op.inputs_) {
              std::visit(
                  c10::overloaded(
                      [&](const c10::IValue& v) {
                        out.append(torch::jit::toPyObject(v));
                      },
                      [&](const std::nullopt_t&) { out.append(py::none()); },
                      [&](const auto& v) { out.append(py::cast(v)); }),
                  input);
            }
            return out;
          })
      .def_readonly("scope", &torch_op_t::scope_)
      .def_readonly("sequence_number", &torch_op_t::sequence_number_)
      .def_readonly("allow_tf32_cublas", &torch_op_t::allow_tf32_cublas_);

  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<ExtraFields<EventType::Backend>>(m, "_ExtraFields_Backend");
  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<ExtraFields<EventType::Vulkan>>(m, "_ExtraFields_Vulkan");

  using allocation_t = ExtraFields<EventType::Allocation>;
  py::class_<allocation_t>(m, "_ExtraFields_Allocation")
      .def_property_readonly(
          "ptr",
          [](const allocation_t& a) {
            return reinterpret_cast<intptr_t>(a.ptr_);
          })
      .def_readonly("id", &allocation_t::id_)
      .def_readonly("allocation_id", &allocation_t::allocation_id_)
      .def_readonly("alloc_size", &allocation_t::alloc_size_)
      .def_readonly("total_allocated", &allocation_t::total_allocated_)
      .def_readonly("total_reserved", &allocation_t::total_reserved_)
      .def_property_readonly("device", &allocation_t::device);

  py::class_<PyFrameState>(m, "_PyFrameState")
      .def_readonly("line_number", &PyFrameState::line_no_)
      .def_property_readonly(
          "file_name", [](const PyFrameState& s) { return s.filename_.str(); })
      .def_property_readonly("function_name", [](const PyFrameState& s) {
        return s.funcname_.str();
      });

  py::class_<NNModuleInfo>(m, "_NNModuleInfo")
      .def_property_readonly(
          "parameters",
          [](const NNModuleInfo& s) {
            py::list out;
            for (const auto& p : s.parameters_) {
              out.append(
                  py::make_tuple(p.name_, p.metadata_, p.grad_metadata_));
            }
            return out;
          })
      .def_property_readonly(
          "cls_name", [](const NNModuleInfo& s) { return s.cls_name_.str(); })
      .def_readonly("self_ptr", &NNModuleInfo::self_)
      .def_readonly("cls_ptr", &NNModuleInfo::cls_);

  py::class_<OptimizerInfo>(m, "_OptimizerInfo")
      .def_readonly("self_ptr", &OptimizerInfo::self_)
      .def_property_readonly("parameters", [](const OptimizerInfo& s) {
        py::list out;
        for (const auto& p : s.parameters_) {
          out.append(py::make_tuple(p.metadata_, p.grad_metadata_, p.state_));
        }
        return out;
      });

  py::class_<ExtraFields<EventType::PyCall>>(m, "_ExtraFields_PyCall")
      .def_readonly("callsite", &ExtraFields<EventType::PyCall>::callsite_)
      .def_readonly("caller", &ExtraFields<EventType::PyCall>::caller_)
      .def_readonly("module", &ExtraFields<EventType::PyCall>::module_)
      .def_readonly("optimizer", &ExtraFields<EventType::PyCall>::optimizer_);

  py::class_<ExtraFields<EventType::PyCCall>>(m, "_ExtraFields_PyCCall")
      .def_readonly("caller", &ExtraFields<EventType::PyCall>::caller_);

  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<ExtraFields<EventType::OutOfMemory>>(
      m, "_ExtraFields_OutOfMemory");

  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<ExtraFields<EventType::Kineto>>(m, "_ExtraFields_Kineto");

  py::class_<Result, std::shared_ptr<Result>>(m, "_ProfilerEvent")
      .def_property_readonly("name", &Result::name)
      .def_property_readonly("overload_name", &Result::overload_name)
      .def_property_readonly("tag", &Result::tag)
      .def_readonly("extra_fields", &Result::extra_fields_)
      .def_property_readonly(
          "typed",
          [](const Result& r) {
            return py::make_tuple(
                r.tag(),
                py::cast(r.extra_fields_, py::return_value_policy::reference));
          })
      .def_property_readonly(
          "id",
          [](const Result& r) {
            return reinterpret_cast<intptr_t>(r.shared_from_this().get());
          })
      .def_property_readonly(
          "parent", [](const Result& r) { return r.parent_.lock(); })
      .def_readonly("children", &Result::children_)
      .def_readonly("start_time_ns", &Result::start_time_ns_)
      .def_readonly("start_tid", &Result::start_tid_)
      .def_property_readonly("correlation_id", &Result::correlationID)
      .def_property_readonly("end_time_ns", &Result::endTimeNS)
      .def_property_readonly("duration_time_ns", [](const Result& r) {
        return r.endTimeNS() - r.start_time_ns_;
      });

  // PyTorch profiler execution trace internal interface.
  m.def(
      "_add_execution_trace_observer",
      &torch::profiler::impl::addExecutionTraceObserver,
      py::arg("output_file_name"));
  m.def(
      "_remove_execution_trace_observer",
      &torch::profiler::impl::removeExecutionTraceObserver);
  m.def(
      "_enable_execution_trace_observer",
      &torch::profiler::impl::enableExecutionTraceObserver);
  m.def(
      "_disable_execution_trace_observer",
      &torch::profiler::impl::disableExecutionTraceObserver);
  m.def(
      "_set_record_concrete_inputs_enabled_val",
      &torch::profiler::impl::set_record_concrete_inputs_enabled_val);
  m.def(
      "_set_fwd_bwd_enabled_val",
      &torch::profiler::impl::set_fwd_bwd_enabled_val);
  m.def(
      "_set_cuda_sync_enabled_val",
      &torch::profiler::impl::set_cuda_sync_enabled_val);

  TORCH_CHECK(PyType_Ready(&THPCapturedTracebackType) >= 0);
  PyModule_AddObject(
      m.ptr(), "CapturedTraceback", (PyObject*)&THPCapturedTracebackType);
  m.def(
      "gather_traceback",
      CapturedTraceback::gather,
      py::arg("python") = true,
      py::arg("script") = true,
      py::arg("cpp") = true);
  m.def("symbolize_tracebacks", [](const py::list& tbs) {
    std::vector<CapturedTraceback*> tb_ptrs;
    tb_ptrs.reserve(tbs.size());
    for (py::handle tb : tbs) {
      tb_ptrs.emplace_back(((THPCapturedTraceback*)tb.ptr())->data.get());
    }
    return py_symbolize(tb_ptrs);
  });
  // directly convert address pointers to frames, used for testing symbolize
  m.def(
      "symbolize_addresses",
      [](const std::vector<uint64_t>& frames, const std::string& mode_s) {
        std::vector<std::tuple<std::string, int64_t, std::string>> frames_out;
        torch::unwind::Mode mode = torch::unwind::Mode::addr2line;
        if (mode_s == "fast") {
          mode = torch::unwind::Mode::fast;
        } else if (mode_s == "addr2line") {
          mode = torch::unwind::Mode::addr2line;
        } else if (mode_s == "dladdr") {
          mode = torch::unwind::Mode::dladdr;
        } else {
          TORCH_CHECK(false, "unexpected mode ", mode_s);
        }
        std::vector<void*> frames_p;
        frames_p.reserve(frames.size());
        for (auto f : frames) {
          frames_p.push_back((void*)f); // NOLINT
        }
        auto frame_objects = unwind::symbolize(frames_p, mode);
        frames_out.reserve(frame_objects.size());
        for (auto& frame : frame_objects) {
          frames_out.emplace_back(frame.filename, frame.lineno, frame.funcname);
        }
        return frames_out;
      });
  installCapturedTracebackPython();

  // NOLINTNEXTLINE(*-c-arrays*)
  static PyMethodDef RecordFunctionFast_methods[] = {
      {"__enter__", RecordFunctionFast_enter, METH_NOARGS, nullptr},
      {"__exit__", RecordFunctionFast_exit, METH_VARARGS, nullptr},
      {nullptr},
  };

  static PyTypeObject RecordFunctionFast_Type = {
      PyVarObject_HEAD_INIT(nullptr, 0)
  };

  RecordFunctionFast_Type.tp_name = "torch._C._profiler.RecordFunctionFast",
  RecordFunctionFast_Type.tp_basicsize = sizeof(RecordFunctionFast);
  RecordFunctionFast_Type.tp_dealloc = (destructor)RecordFunctionFast_dealloc;
  RecordFunctionFast_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  RecordFunctionFast_Type.tp_methods = RecordFunctionFast_methods;
  RecordFunctionFast_Type.tp_init = RecordFunctionFast_init;
  RecordFunctionFast_Type.tp_new = RecordFunctionFast_new;

  if (PyType_Ready(&RecordFunctionFast_Type) < 0) {
    throw python_error();
  }

  Py_INCREF(&RecordFunctionFast_Type);
  if (PyModule_AddObject(
          m.ptr(),
          "_RecordFunctionFast",
          (PyObject*)&RecordFunctionFast_Type) != 0) {
    Py_DECREF(&RecordFunctionFast_Type);
    throw python_error();
  }
}
} // namespace torch::profiler
