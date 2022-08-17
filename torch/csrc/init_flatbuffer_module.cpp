#include <torch/csrc/python_headers.h>

#include <libshm.h>
#include <cstdlib>

#include <pybind11/detail/common.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <torch/csrc/utils/pybind.h>

#include <Python.h> // NOLINT
#include <torch/csrc/jit/mobile/flatbuffer_loader.h>
#include <torch/csrc/jit/python/module_python.h>
#include <torch/csrc/jit/python/python_ivalue.h>
#include <torch/csrc/jit/python/python_sugared_value.h>
#include <torch/csrc/jit/serialization/flatbuffer_serializer.h>
#include <torch/csrc/jit/serialization/flatbuffer_serializer_jit.h>

namespace py = pybind11;

using torch::jit::kFlatbufferDataAlignmentBytes;

static std::shared_ptr<char> copyStr(const std::string& bytes) {
  size_t size = (bytes.size() / kFlatbufferDataAlignmentBytes + 1) *
      kFlatbufferDataAlignmentBytes;
#ifdef _WIN32
  std::shared_ptr<char> bytes_copy(
      static_cast<char*>(_aligned_malloc(size, kFlatbufferDataAlignmentBytes)),
      _aligned_free);
#elif defined(__APPLE__)
  void* p;
  ::posix_memalign(&p, kFlatbufferDataAlignmentBytes, size);
  TORCH_INTERNAL_ASSERT(p, "Could not allocate memory for flatbuffer");
  std::shared_ptr<char> bytes_copy(static_cast<char*>(p), free);
#else
  std::shared_ptr<char> bytes_copy(
      static_cast<char*>(aligned_alloc(kFlatbufferDataAlignmentBytes, size)),
      free);
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
         const std::string& filename,
         const ExtraFilesMap& _extra_files = ExtraFilesMap()) {
        return torch::jit::save_mobile_module(module, filename, _extra_files);
      });
  pym.def(
      "_save_jit_module",
      [](const torch::jit::Module& module,
         const std::string& filename,
         const ExtraFilesMap& _extra_files = ExtraFilesMap()) {
        return torch::jit::save_jit_module(module, filename, _extra_files);
      });
  pym.def(
      "_save_mobile_module_to_bytes",
      [](const torch::jit::mobile::Module& module,
         const ExtraFilesMap& _extra_files = ExtraFilesMap()) {
        auto detached_buffer =
            torch::jit::save_mobile_module_to_bytes(module, _extra_files);
        return py::bytes(
            reinterpret_cast<char*>(detached_buffer->data()),
            detached_buffer->size());
      });
  pym.def(
      "_save_jit_module_to_bytes",
      [](const torch::jit::Module& module,
         const ExtraFilesMap& _extra_files = ExtraFilesMap()) {
        auto detached_buffer =
            torch::jit::save_jit_module_to_bytes(module, _extra_files);
        return py::bytes(
            reinterpret_cast<char*>(detached_buffer->data()),
            detached_buffer->size());
      });
  pym.def(
      "_get_module_info_from_flatbuffer", [](std::string flatbuffer_content) {
        py::gil_scoped_acquire acquire;
        py::dict result;
        mobile::ModuleInfo minfo =
            torch::jit::get_module_info_from_flatbuffer(&flatbuffer_content[0]);
        result["bytecode_version"] = minfo.bytecode_version;
        result["operator_version"] = minfo.operator_version;
        result["function_names"] = minfo.function_names;
        result["type_names"] = minfo.type_names;
        result["opname_to_num_args"] = minfo.opname_to_num_args;
        return result;
      });

  return module;
}
