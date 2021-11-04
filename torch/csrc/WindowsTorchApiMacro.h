#pragma once

#include <c10/macros/Export.h>

#ifdef _WIN32
#define TORCH_PYTHON_API
#else
#define TORCH_PYTHON_API TORCH_API
#endif
