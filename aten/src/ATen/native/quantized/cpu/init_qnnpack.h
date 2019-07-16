#pragma once

#ifdef USE_QNNPACK
#include "qnnpack.h"

namespace at {
namespace native {

void initQNNPACK();

} // namespace native
} // namespace at
#endif
