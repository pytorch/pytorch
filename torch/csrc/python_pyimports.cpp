#include <torch/csrc/PyInterpreter.h>
#include <torch/csrc/python_pyimports.h>
#include <torch/csrc/utils/pybind.h>
#include <iostream>

namespace torch {
namespace detail {

struct PyImportsImpl : c10::impl::PyImportsInterface {
  void do_pyimport(const char* module, const char* context) const override {
    try {
      py::module::import("torch._cpp_pyimports")
          .attr("do_import")(module, context);
    } catch (py::error_already_set& ex) {
      ex.restore();
    }
  }
};

c10::impl::IgnoredPyImports initialize_pyimports_handler() {
  if (!isMainPyInterpreter()) {
    return {};
  }
  std::lock_guard<std::mutex> lock(c10::impl::kPyImportsHandlerMutex());

  TORCH_INTERNAL_ASSERT(
      !c10::impl::unsafe_has_pyimports_handler(),
      "pyimports was already initialized?");
  c10::impl::unsafe_set_pyimports_handler(std::make_unique<PyImportsImpl>());

  const auto& ignored_pyimports = c10::impl::unsafe_get_ignored_pyimports();
  return ignored_pyimports;
}

} // namespace detail
} // namespace torch
