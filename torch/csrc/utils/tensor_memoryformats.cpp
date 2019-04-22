#include <torch/csrc/python_headers.h>

#include <torch/csrc/utils/tensor_memoryformats.h>

#include <torch/csrc/MemoryFormat.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>

#include <ATen/MemoryFormat.h>

namespace torch { namespace utils {

void initializeMemoryFormats() {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module) throw python_error();

  PyObject *cf_memory_format = THPMemoryFormat_New(at::MemoryFormat::ChannelsFirst, "torch.channels_first");
  Py_INCREF(cf_memory_format);
  if (PyModule_AddObject(torch_module, "channels_first", cf_memory_format) != 0) {
    throw python_error();
  }
}

}} // namespace torch::utils
