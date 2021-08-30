#pragma once

#if defined(USE_C10D_NCCL) || defined(USE_C10D_UCC)
#include <c10/macros/Export.h>

namespace c10d {

TORCH_API void initCustomClassBindingsNccl();

}

#endif
