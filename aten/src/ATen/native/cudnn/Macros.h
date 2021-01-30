#pragma once

// Warning: The following macro is guard for enabling cuDNN v8 API,
// which is not finished yet on PyTorch. This macro will be removed
// once the cuDNN v8 binding is done.
// cuDNN v8 API is not finished yet, and is not recommended to use.
// enable this only if you know what you are doing.
#define _ENABLE_CUDNN_V8_API true

#define HAS_CUDNN_V8() (_ENABLE_CUDNN_V8_API && defined(CUDNN_VERSION) && CUDNN_VERSION >= 8000)
