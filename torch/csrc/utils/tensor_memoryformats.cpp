#include <torch/csrc/utils/tensor_memoryformats.h>

#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/MemoryFormat.h>
#include <c10/core/MemoryFormat.h>

#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>


namespace torch {
namespace utils {

#define _ADD_MEMORY_FORMAT(format, name)                                       \
  {                                                                            \
    std::string module_name = "torch.";                                        \
    PyObject* memory_format = THPMemoryFormat_New(format, module_name + name); \
    Py_INCREF(memory_format);                                                  \
    if (PyModule_AddObject(torch_module, name, memory_format) != 0) {          \
      throw python_error();                                                    \
    }                                                                          \
  }

void initializeMemoryFormats() {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module) {
    throw python_error();
  }

  _ADD_MEMORY_FORMAT(at::MemoryFormat::Preserve, "preserve_format");
  _ADD_MEMORY_FORMAT(at::MemoryFormat::Contiguous, "contiguous_format");
  _ADD_MEMORY_FORMAT(at::MemoryFormat::ChannelsLast, "channels_last");
  _ADD_MEMORY_FORMAT(at::MemoryFormat::ChannelsLast3d, "channels_last_3d");

}

} // namespace utils
} // namespace torch
