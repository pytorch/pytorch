#pragma once

#include <ATen/cudnn/cudnn-wrapper.h>

// Note: The version below should not actually be 8000. Instead, it should
// be whatever version of cuDNN that v8 API work with PyTorch correctly.
// The version is set to 8000 today for convenience of debugging.
#if defined(USE_EXPERIMENTAL_CUDNN_V8_API) && defined(CUDNN_VERSION) && CUDNN_VERSION >= 8200
#define HAS_CUDNN_V8() true
#else
#define HAS_CUDNN_V8() false
#endif
