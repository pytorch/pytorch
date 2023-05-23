#include <c10/util/Optional.h>
#include <torch/csrc/utils/pybind.h>
#include <tuple>

namespace torch {
namespace impl {

template <typename GuardT, typename... Args>
struct RAIIContextManager {
  explicit RAIIContextManager(Args&&... args)
      : args_(std::forward<Args>(args)...) {}

  void enter() {
    auto emplace = [&](Args... args) {
      return guard_.emplace(std::forward<Args>(args)...);
    };
    std::apply(std::move(emplace), args_);
  }

  void exit() {
    guard_ = c10::nullopt;
  }

 private:
  c10::optional<GuardT> guard_;
  std::tuple<Args...> args_;
};

// Turns a C++ RAII guard into a Python context manager.
// See _ExcludeDispatchKeyGuard in python_dispatch.cpp for example.
template <typename GuardT, typename... GuardArgs>
void py_context_manager(const py::module& m, const char* name) {
  using ContextManagerT = RAIIContextManager<GuardT, GuardArgs...>;
  py::class_<ContextManagerT>(m, name)
      .def(py::init<GuardArgs...>())
      .def("__enter__", [](ContextManagerT& guard) { guard.enter(); })
      .def(
          "__exit__",
          [](ContextManagerT& guard,
             py::object exc_type,
             py::object exc_value,
             py::object traceback) { guard.exit(); });
}

} // namespace impl
} // namespace torch
