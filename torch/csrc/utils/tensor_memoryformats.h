#pragma once

#include <c10/core/MemoryFormat.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/utils/python_stub.h>

namespace torch::utils {

void initializeMemoryFormats();

// This methods returns a borrowed reference!
TORCH_PYTHON_API PyObject* getTHPMemoryFormat(c10::MemoryFormat);

} // namespace torch::utils
