#pragma once

#include <c10/core/MemoryFormat.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace utils {

void initializeMemoryFormats();
py::object getTHPMemoryFormat(c10::MemoryFormat);

} // namespace utils
} // namespace torch
