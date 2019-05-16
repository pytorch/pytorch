#pragma once

#include <torch/all.h>

#ifdef TORCH_API_INCLUDE_EXTENSION_H
#include <torch/extension.h>

#define DEPRECATE_MESSAGE \
    "Including torch/torch.h for C++ extensions is deprecated. Please include torch/extension.h"

#ifdef _MSC_VER
#  pragma message ( DEPRECATE_MESSAGE )
#else
#  warning DEPRECATE_MESSAGE
#endif

#endif // defined(TORCH_API_INCLUDE_EXTENSION_H)
