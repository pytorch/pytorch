#include <torch/csrc/profiler/python/init.h>

#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/profiler/collection.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace profiler {

void initPythonBindings(PyObject* module) {
  auto rootModule = py::handle(module).cast<py::module>();
  auto m = rootModule.def_submodule("_profiler");

  using namespace torch::profiler::impl;

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
      .value("NVTX", ActiveProfilerType::NVTX);

  py::enum_<ActivityType>(m, "ProfilerActivity")
      .value("CPU", ActivityType::CPU)
      .value("CUDA", ActivityType::CUDA);

  py::class_<ExperimentalConfig>(m, "_ExperimentalConfig")
      .def(
          py::init<
              std::vector<std::string> /* profiler_metrics */,
              bool /* profiler_measure_per_kernel */
              >(),
          "An experimental config for Kineto features. Please note that"
          "backward compatibility is not guaranteed.\n"
          "    profiler_metrics : a list of CUPTI profiler metrics used\n"
          "       to measure GPU performance events.\n"
          "       If this list contains values Kineto runs in CUPTI profiler mode\n"
          "    profiler_measure_per_kernel (bool) : whether to profile metrics per kernel\n"
          "       or for the entire measurement duration.",
          py::arg("profiler_metrics") = std::vector<std::string>(),
          py::arg("profiler_measure_per_kernel") = false)
      .def(py::pickle(
          [](const ExperimentalConfig& p) { // __getstate__
            py::list py_metrics;
            for (const auto& metric : p.profiler_metrics) {
              py::bytes mbytes(metric);
              py_metrics.append(mbytes);
            }
            /* Return a tuple that fully encodes the state of the config */
            return py::make_tuple(py_metrics, p.profiler_measure_per_kernel);
          },
          [](py::tuple t) { // __setstate__
            if (t.size() != 2) {
              throw std::runtime_error("Expected 2 values in state");
            }

            py::list py_metrics = t[0].cast<py::list>();
            std::vector<std::string> metrics{py_metrics.size()};

            for (const auto& py_metric : py_metrics) {
              metrics.push_back(py::str(py_metric));
            }

            return ExperimentalConfig(std::move(metrics), t[1].cast<bool>());
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
      .value("Allocation", EventType::Allocation)
      .value("PyCall", EventType::PyCall)
      .value("PyCCall", EventType::PyCCall)
      .value("Kineto", EventType::Kineto);

  py::class_<Inputs>(m, "_Inputs")
      .def_readonly("shapes", &Inputs::shapes_)
      .def_readonly("dtypes", &Inputs::dtypes_)
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
      .def_property_readonly(
          "layout",
          [](const TensorMetadata& metadata) {
            PyObject* layout_obj =
                torch::autograd::utils::wrap(metadata.layout_);
            return py::reinterpret_borrow<py::object>(layout_obj);
          })
      .def_property_readonly("device", [](const TensorMetadata& metadata) {
        // Have to pull a copy of the existing Python Device object.
        PyObject* thp_device = THPDevice_New(
            c10::Device(metadata.device_type_, metadata.device_index_));
        return py::reinterpret_borrow<py::object>(thp_device);
      });

  py::class_<ExtraFields<EventType::TorchOp>>(m, "_ExtraFields_TorchOp")
      .def_readonly("inputs", &ExtraFields<EventType::TorchOp>::inputs_)
      .def_readonly(
          "allow_tf32_cublas",
          &ExtraFields<EventType::TorchOp>::allow_tf32_cublas_);

  py::class_<ExtraFields<EventType::Backend>>(m, "_ExtraFields_Backend");

  py::class_<ExtraFields<EventType::Allocation>>(m, "_ExtraFields_Allocation");

  py::class_<ExtraFields<EventType::PyCall>>(m, "_ExtraFields_PyCall")
      .def_readonly("callsite", &ExtraFields<EventType::PyCall>::callsite_)
      .def_readonly("caller", &ExtraFields<EventType::PyCall>::caller_);

  py::class_<ExtraFields<EventType::PyCCall>>(m, "_ExtraFields_PyCCall")
      .def_readonly("caller", &ExtraFields<EventType::PyCall>::caller_);

  py::class_<PyFrameState>(m, "_PyFrameState")
      .def_readonly("line_number", &PyFrameState::line_no_)
      .def_property_readonly(
          "file_name", [](const PyFrameState& s) { return s.filename_.str(); })
      .def_property_readonly("function_name", [](const PyFrameState& s) {
        return s.funcname_.str();
      });

  py::class_<ExtraFields<EventType::Kineto>>(m, "_ExtraFields_Kineto");

  py::class_<Result, std::shared_ptr<Result>>(m, "_ProfilerEvent")
      .def("name", &Result::name)
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
}

} // namespace profiler
} // namespace torch
