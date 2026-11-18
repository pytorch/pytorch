#include <torch/csrc/python_headers.h>

#include <pybind11/chrono.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/xpu/XPUGraph.h>
#include <c10/xpu/XPUGraphsC10Utils.h>

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

void THXPGraph_init(PyObject* module) {
  // Pybind11 patch notes say "py::module_" is more up-to-date syntax,
  // but CI linter and some builds prefer "module".
  auto torch_C_m = py::handle(module).cast<py::module>();

  torch_C_m.def("_xpu_graph_pool_handle", &::at::xpu::graph_pool_handle);

  shared_ptr_class_<::at::xpu::XPUGraph>(torch_C_m, "_XPUGraph")
      .def(py::init<bool>(), py::arg("keep_graph") = false)
      .def(
          "capture_begin",
          [](::at::xpu::XPUGraph& self,
             std::optional<c10::xpu::MempoolId_t> pool_opt) {
            c10::xpu::MempoolId_t pool = pool_opt.has_value()
                ? pool_opt.value()
                : c10::xpu::MempoolId_t{0, 0};
            return self.capture_begin(pool);
          },
          py::arg("pool"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "capture_end",
          torch::wrap_pybind_function_no_gil(&at::xpu::XPUGraph::capture_end))
      .def(
          "instantiate",
          torch::wrap_pybind_function_no_gil(&at::xpu::XPUGraph::instantiate))
      .def(
          "register_generator_state",
          [](::at::xpu::XPUGraph& self, py::handle raw_generator) {
            auto generator = THPGenerator_Unwrap(raw_generator.ptr());
            // We've unwrapped Python object to C++ object,
            // so we could release GIL before calling into C++
            py::gil_scoped_release release;
            return self.register_generator_state(generator);
          },
          py::arg("generator"))
      .def(
          "replay",
          torch::wrap_pybind_function_no_gil(&at::xpu::XPUGraph::replay))
      .def(
          "reset",
          torch::wrap_pybind_function_no_gil(&at::xpu::XPUGraph::reset))
      .def("pool", torch::wrap_pybind_function_no_gil(&at::xpu::XPUGraph::pool))
      .def(
          "enable_debug_mode",
          torch::wrap_pybind_function_no_gil(
              &at::xpu::XPUGraph::enable_debug_mode))
      .def(
          "debug_dump",
          torch::wrap_pybind_function_no_gil(&at::xpu::XPUGraph::debug_dump),
          py::arg("debug_path"))
      .def(
          "raw_xpu_graph",
          [](at::xpu::XPUGraph& self) {
            at::xpu::xpuGraph_t* graph = self.raw_xpu_graph();
            return reinterpret_cast<uintptr_t>(graph);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "raw_xpu_graph_exec",
          [](at::xpu::XPUGraph& self) {
            at::xpu::xpuGraphExec_t* graph_exec = self.raw_xpu_graph_exec();
            return reinterpret_cast<uintptr_t>(graph_exec);
          },
          py::call_guard<py::gil_scoped_release>());
}
