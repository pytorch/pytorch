#pragma once

#include <ATen/cudnn/cudnn-wrapper.h>

// Warning: The following macro is guard for enabling cuDNN v8 API,
// which is not finished yet on PyTorch. This macro will be removed
// once the cuDNN v8 binding is done.
// cuDNN v8 API is not finished yet, and is not recommended to use.
// enable this only if you know what you are doing.
#define _ENABLE_CUDNN_V8_API true

// Note: The version below should not actually be 8000. Instead, it should
// be whatever version of cuDNN that v8 API work with PyTorch correctly.
// The version is set to 8000 today for convenience of debugging.
#if _ENABLE_CUDNN_V8_API && defined(CUDNN_VERSION) && CUDNN_VERSION >= 8000
#define HAS_CUDNN_V8() true
#else
#define HAS_CUDNN_V8() false
#endif
