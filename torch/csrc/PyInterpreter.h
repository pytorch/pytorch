#pragma once

#include <c10/core/impl/PyInterpreter.h>
#include <torch/csrc/Export.h>

TORCH_PYTHON_API c10::impl::PyInterpreter* getPyInterpreter();
TORCH_PYTHON_API bool isMainPyInterpreter();
