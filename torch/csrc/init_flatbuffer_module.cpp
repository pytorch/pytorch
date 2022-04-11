#include <torch/csrc/python_headers.h>

#include <libshm.h>
#include <cstdlib>

#include <pybind11/detail/common.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <Python.h> // NOLINT
#include <torch/csrc/jit/mobile/flatbuffer_loader.h>
#include <torch/csrc/jit/python/module_python.h>
#include <torch/csrc/jit/python/python_ivalue.h>
#include <torch/csrc/jit/python/python_sugared_value.h>
#include <torch/csrc/jit/serialization/flatbuffer_serializer.h>

namespace py = pybind11;

static std::shared_ptr<char> copyStr(const std::string& bytes) {
  size_t size = (bytes.size() / FLATBUFFERS_MAX_ALIGNMENT + 1) *
      FLATBUFFERS_MAX_ALIGNMENT;
#ifdef _WIN32
  std::shared_ptr<char> bytes_copy(
      static_cast<char*>(_aligned_malloc(size, FLATBUFFERS_MAX_ALIGNMENT)),
      _aligned_free);
#else
  std::shared_ptr<char> bytes_copy(
      static_cast<char*>(aligned_alloc(FLATBUFFERS_MAX_ALIGNMENT, size)), free);
#endif
  memcpy(bytes_copy.get(), bytes.data(), bytes.size());
  return bytes_copy;
}

extern "C"
#ifdef _WIN32
    __declspec(dllexport)
#endif
        PyObject* initModuleFlatbuffer() {
  using namespace torch::jit;
  PyMethodDef m[] = {{nullptr, nullptr, 0, nullptr}}; // NOLINT
  static struct PyModuleDef torchmodule = {
      PyModuleDef_HEAD_INIT,
      "torch._C_flatbuffer",
      nullptr,
      -1,
      m,
  }; // NOLINT
  PyObject* module = PyModule_Create(&torchmodule);
  auto pym = py::handle(module).cast<py::module>();
  pym.def("_load_mobile_module_from_file", [](const std::string& filename) {
    return torch::jit::load_mobile_module_from_file(filename);
  });
  pym.def("_load_mobile_module_from_bytes", [](const std::string& bytes) {
    auto bytes_copy = copyStr(bytes);
    return torch::jit::parse_and_initialize_mobile_module(
        bytes_copy, bytes.size());
  });
  pym.def("_load_jit_module_from_file", [](const std::string& filename) {
    ExtraFilesMap extra_files = ExtraFilesMap();
    return torch::jit::load_jit_module_from_file(filename, extra_files);
  });
  pym.def("_load_jit_module_from_bytes", [](const std::string& bytes) {
    auto bytes_copy = copyStr(bytes);
    ExtraFilesMap extra_files = ExtraFilesMap();
    return torch::jit::parse_and_initialize_jit_module(
        bytes_copy, bytes.size(), extra_files);
  });
  pym.def(
      "_save_mobile_module",
      [](const torch::jit::mobile::Module& module,
         const std::string& filename) {
        return torch::jit::save_mobile_module(module, filename);
      });
  pym.def(
      "_save_jit_module",
      [](const torch::jit::Module& module, const std::string& filename) {
        return torch::jit::save_jit_module(module, filename);
      });
  pym.def(
      "_save_mobile_module_to_bytes",
      [](const torch::jit::mobile::Module& module) {
        auto detached_buffer = torch::jit::save_mobile_module_to_bytes(module);
        return py::bytes(
            reinterpret_cast<char*>(detached_buffer.data()),
            detached_buffer.size());
      });
  pym.def("_save_jit_module_to_bytes", [](const torch::jit::Module& module) {
    auto detached_buffer = torch::jit::save_jit_module_to_bytes(module);
    return py::bytes(
        reinterpret_cast<char*>(detached_buffer.data()),
        detached_buffer.size());
  });
  return module;
}
