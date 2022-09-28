#pragma once

#include <c10/core/MemoryFormat.h>

namespace torch {
namespace utils {

void initializeMemoryFormats();
PyObject* getTHPMemoryFormat(c10::MemoryFormat);

} // namespace utils
} // namespace torch
