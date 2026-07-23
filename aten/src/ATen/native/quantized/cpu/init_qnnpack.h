#pragma once

#ifdef USE_PYTORCH_QNNPACK

namespace at::native {

void initQNNPACK();

} // namespace at::native

#endif
