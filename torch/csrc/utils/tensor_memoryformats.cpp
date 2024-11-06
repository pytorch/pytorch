#include <torch/csrc/utils/tensor_memoryformats.h>

#include <c10/core/MemoryFormat.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/MemoryFormat.h>

#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>

namespace torch::utils {

namespace {
// Intentionally leaked
std::array<PyObject*, static_cast<int>(at::MemoryFormat::NumOptions)>
    memory_format_registry = {};
} // anonymous namespace

PyObject* getTHPMemoryFormat(at::MemoryFormat memory_format) {
  auto py_memory_format =
      memory_format_registry[static_cast<int>(memory_format)];
  if (!py_memory_format) {
    throw std::invalid_argument("unsupported memory_format");
  }
  return py_memory_format;
}

void initializeMemoryFormats() {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module) {
    throw python_error();
  }

  auto add_memory_format = [&](at::MemoryFormat format, const char* name) {
    std::string module_name = "torch.";
    PyObject* memory_format = THPMemoryFormat_New(format, module_name + name);
    Py_INCREF(memory_format);
    if (PyModule_AddObject(torch_module, name, memory_format) != 0) {
      Py_DECREF(memory_format);
      throw python_error();
    }
    Py_INCREF(memory_format);
    memory_format_registry[static_cast<size_t>(format)] = memory_format;
  };

  add_memory_format(at::MemoryFormat::Preserve, "preserve_format");
  add_memory_format(at::MemoryFormat::Contiguous, "contiguous_format");
  add_memory_format(at::MemoryFormat::ChannelsLast, "channels_last");
  add_memory_format(at::MemoryFormat::ChannelsLast3d, "channels_last_3d");
}

} // namespace torch::utils
