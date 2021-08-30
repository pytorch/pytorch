#pragma once

#ifdef USE_C10D_NCCL
#include <c10/macros/Export.h>

namespace c10d {

TORCH_API void initCustomClassBindingsNccl();

}

#endif
