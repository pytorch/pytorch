#pragma once

#include <c10/core/MemoryFormat.h>
#include <torch/csrc/utils/python_stub.h>

namespace torch {
namespace utils {

void initializeMemoryFormats();
PyObject* getTHPMemoryFormat(c10::MemoryFormat);

} // namespace utils
} // namespace torch
