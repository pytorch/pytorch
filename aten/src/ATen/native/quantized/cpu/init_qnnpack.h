#pragma once

#ifdef USE_PYTORCH_QNNPACK

namespace at {
namespace native {

void initQNNPACK();

} // namespace native
} // namespace at

#endif
