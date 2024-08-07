#include <torch/csrc/utils/pybind.h>
#include <optional>
#include <tuple>

namespace torch::impl {

template <typename GuardT, typename... Args>
struct RAIIContextManager {
  explicit RAIIContextManager(Args&&... args)
      : args_(std::forward<Args>(args)...) {}

  void enter() {
    auto emplace = [&](Args... args) {
      guard_.emplace(std::forward<Args>(args)...);
    };
    std::apply(std::move(emplace), args_);
  }

  void exit() {
    guard_ = std::nullopt;
  }

 private:
  std::optional<GuardT> guard_;
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
             const py::object& exc_type,
             const py::object& exc_value,
             const py::object& traceback) { guard.exit(); });
}

template <typename GuardT, typename... Args>
struct DeprecatedRAIIContextManager {
  explicit DeprecatedRAIIContextManager(Args&&... args) {
    guard_.emplace(std::forward<Args>(args)...);
  }

  void enter() {}

  void exit() {
    guard_ = std::nullopt;
  }

 private:
  std::optional<GuardT> guard_;
  std::tuple<Args...> args_;
};

// Definition: a "Python RAII guard" is an object in Python that acquires
// a resource on init and releases the resource on deletion.
//
// This API turns a C++ RAII guard into an object can be used either as a
// Python context manager or as a "Python RAII guard".
//
// Please prefer `py_context_manager` to this API if you are binding a new
// RAII guard into Python because "Python RAII guards" don't work as expected
// in Python (Python makes no guarantees about when an object gets deleted)
template <typename GuardT, typename... GuardArgs>
void py_context_manager_DEPRECATED(const py::module& m, const char* name) {
  using ContextManagerT = DeprecatedRAIIContextManager<GuardT, GuardArgs...>;
  py::class_<ContextManagerT>(m, name)
      .def(py::init<GuardArgs...>())
      .def("__enter__", [](ContextManagerT& guard) { guard.enter(); })
      .def(
          "__exit__",
          [](ContextManagerT& guard,
             const py::object& exc_type,
             const py::object& exc_value,
             const py::object& traceback) { guard.exit(); });
}

} // namespace torch::impl
