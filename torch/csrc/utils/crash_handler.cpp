#include <client/linux/handler/exception_handler.h>
#include <torch/csrc/utils/crash_handler.h>

#include <pybind11/pybind11.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <algorithm>
#include <memory>

static bool dumpCallback(
    const google_breakpad::MinidumpDescriptor& descriptor,
    void* context,
    bool succeeded) {
  if (succeeded) {
    std::cout << "Wrote minidump to " << descriptor.path() << std::endl;
  }
  return succeeded;
}

std::unique_ptr<google_breakpad::ExceptionHandler> handler;
std::string minidump_directory;

namespace torch {
namespace crash_handler {

void initCrashHandlerBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  m.def(
       "_enable_minidump_collection",
       [](const std::string& dir) {
         if (handler == nullptr) {
           minidump_directory = dir;
           handler = std::make_unique<google_breakpad::ExceptionHandler>(
               google_breakpad::MinidumpDescriptor(minidump_directory),
               nullptr,
               dumpCallback,
               nullptr,
               true,
               -1);
         }
       })
      .def("_get_minidump_directory", []() { return minidump_directory; })
      .def("_crash", []() {
        // TODO: Testing only, remove before landing
        volatile int* bad = nullptr;
        return *bad;
      });
}

} // namespace crash_handler
} // namespace torch
