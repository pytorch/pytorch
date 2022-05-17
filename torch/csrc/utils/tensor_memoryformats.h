#pragma once

#include <torch/csrc/utils/pybind.h>
#include <c10/core/MemoryFormat.h>

namespace torch { namespace utils {

void initializeMemoryFormats();
py::object getTHPMemoryFormat(c10::MemoryFormat);

}} // namespace torch::utils
