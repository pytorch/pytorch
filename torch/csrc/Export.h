#pragma once

#include <c10/macros/Export.h>

#ifdef THP_BUILD_MAIN_LIB
#define TORCH_PYTHON_API C10_EXPORT
#else
#define TORCH_PYTHON_API C10_IMPORT
#endif
