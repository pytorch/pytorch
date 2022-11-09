#include <torch/csrc/profiler/python/init.h>

#include <ATen/record_function.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/profiler/collection.h>
#include <torch/csrc/profiler/standalone/execution_graph_observer.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace profiler {

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
      .value("KINETO", ProfilerState::KINETO)
      .value("KINETO_GPU_FALLBACK", ProfilerState::KINETO_GPU_FALLBACK);

  py::enum_<ActiveProfilerType>(m, "ActiveProfilerType")
      .value("NONE", ActiveProfilerType::NONE)
      .value("LEGACY", ActiveProfilerType::LEGACY)
      .value("KINETO", ActiveProfilerType::KINETO)
      .value("NVTX", ActiveProfilerType::NVTX)
      .value("ITT", ActiveProfilerType::ITT);

  py::enum_<ActivityType>(m, "ProfilerActivity")
      .value("CPU", ActivityType::CPU)
      .value("CUDA", ActivityType::CUDA);

  py::class_<ExperimentalConfig>(m, "_ExperimentalConfig")
      .def(
          py::init<
              std::vector<std::string> /* profiler_metrics */,
              bool /* profiler_measure_per_kernel */,
              bool /* verbose */,
              std::vector<std::string> /* performance_events  */
              >(),
          "An experimental config for Kineto features. Please note that"
          "backward compatibility is not guaranteed.\n"
          "    profiler_metrics : a list of CUPTI profiler metrics used\n"
          "       to measure GPU performance events.\n"
          "       If this list contains values Kineto runs in CUPTI profiler mode\n"
          "    profiler_measure_per_kernel (bool) : whether to profile metrics per kernel\n"
          "       or for the entire measurement duration.\n"
          "    verbose (bool) : whether the trace file has `Call stack` field or not.\n"
          "    performance_events : a list of profiler events to be used for measurement",
          py::arg("profiler_metrics") = std::vector<std::string>(),
          py::arg("profiler_measure_per_kernel") = false,
          py::arg("verbose") = false,
          py::arg("performance_events") = std::vector<std::string>())
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
                p.performance_events);
          },
          [](py::tuple t) { // __setstate__
            if (t.size() >= 3) {
              throw std::runtime_error("Expected atleast 3 values in state");
            }

            py::list py_metrics = t[0].cast<py::list>();
            std::vector<std::string> metrics{py_metrics.size()};

            for (const auto& py_metric : py_metrics) {
              metrics.push_back(py::str(py_metric));
            }

            std::vector<std::string> performance_events;
            if (t.size() == 4) {
              py::list py_perf_events = t[3].cast<py::list>();
              performance_events.resize(py_perf_events.size());
              for (const auto& py_perf_event : py_perf_events) {
                performance_events.push_back(py::str(py_perf_event));
              }
            }

            return ExperimentalConfig(
                std::move(metrics),
                t[1].cast<bool>(),
                t[2].cast<bool>(),
                std::move(performance_events));
          }));

  py::class_<ProfilerConfig>(m, "ProfilerConfig")
      .def(py::init<
           ProfilerState,
           bool, /* record_input_shapes */
           bool, /* profile_memory */
           bool, /* with_stack */
           bool, /* with_flops */
           bool, /* with_modules */
           ExperimentalConfig /* experimental_config */
           >());

  py::enum_<EventType>(m, "_EventType")
      .value("TorchOp", EventType::TorchOp)
      .value("Backend", EventType::Backend)
      .value("Vulkan", EventType::Vulkan)
      .value("Allocation", EventType::Allocation)
      .value("PyCall", EventType::PyCall)
      .value("PyCCall", EventType::PyCCall)
      .value("Kineto", EventType::Kineto);

  py::class_<Inputs>(m, "_Inputs")
      .def_readonly("shapes", &Inputs::shapes_)
      .def_readonly("dtypes", &Inputs::dtypes_)
      .def_readonly("strides", &Inputs::strides_)
      .def_property_readonly(
          "ivalues",
          [](const Inputs& inputs) {
            py::list list;
            for (auto& v : inputs.ivalues_) {
              list.append(torch::jit::toPyObject(v));
            }
            return list;
          })
      .def_readonly("tensor_metadata", &Inputs::tensor_metadata_);

  py::class_<TensorMetadata>(m, "_TensorMetadata")
      .def_property_readonly("impl_ptr", &TensorMetadata::impl)
      .def_readonly("storage_data_ptr", &TensorMetadata::data_)
      .def_readonly("id", &TensorMetadata::id_)
      .def_property_readonly(
          "layout",
          [](const TensorMetadata& metadata) {
            PyObject* layout_obj =
                torch::autograd::utils::wrap(metadata.layout_);
            return py::reinterpret_borrow<py::object>(layout_obj);
          })
      .def_property_readonly("device", &TensorMetadata::device)
      .def_property_readonly(
          "dtype",
          [](const TensorMetadata& metadata) {
            return py::reinterpret_borrow<py::object>(
                torch::autograd::utils::wrap(
                    torch::getTHPDtype(metadata.dtype_)));
          })
      .def_readonly("dim", &TensorMetadata::dim_);

  using torch_op_t = ExtraFields<EventType::TorchOp>;
  py::class_<torch_op_t>(m, "_ExtraFields_TorchOp")
      .def_readonly("inputs", &torch_op_t::inputs_)
      .def_readonly("scope", &torch_op_t::scope_)
      .def_readonly("sequence_number", &torch_op_t::sequence_number_)
      .def_readonly("allow_tf32_cublas", &torch_op_t::allow_tf32_cublas_);

  py::class_<ExtraFields<EventType::Backend>>(m, "_ExtraFields_Backend");
  py::class_<ExtraFields<EventType::Vulkan>>(m, "_ExtraFields_Vulkan");

  using allocation_t = ExtraFields<EventType::Allocation>;
  py::class_<allocation_t>(m, "_ExtraFields_Allocation")
      .def_property_readonly(
          "ptr",
          [](const allocation_t& a) {
            return reinterpret_cast<intptr_t>(a.ptr_);
          })
      .def_readonly("id", &allocation_t::id_)
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

  py::class_<ExtraFields<EventType::OutOfMemory>>(
      m, "_ExtraFields_OutOfMemory");

  py::class_<ExtraFields<EventType::Kineto>>(m, "_ExtraFields_Kineto");

  py::class_<Result, std::shared_ptr<Result>>(m, "_ProfilerEvent")
      .def_property_readonly("name", &Result::name)
      .def_property_readonly("tag", &Result::tag)
      .def_readonly("extra_fields", &Result::extra_fields_)
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

  // PyTorch profiler execution graph internal interface.
  m.def(
      "_add_execution_graph_observer",
      &torch::profiler::impl::addExecutionGraphObserver,
      py::arg("output_file_name"));
  m.def(
      "_remove_execution_graph_observer",
      &torch::profiler::impl::removeExecutionGraphObserver);
  m.def(
      "_enable_execution_graph_observer",
      &torch::profiler::impl::enableExecutionGraphObserver);
  m.def(
      "_disable_execution_graph_observer",
      &torch::profiler::impl::disableExecutionGraphObserver);
}

} // namespace profiler
} // namespace torch
