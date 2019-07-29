#pragma once

#include <torch/all.h>

#ifdef TORCH_API_INCLUDE_EXTENSION_H
#include <torch/extension.h>

#ifdef _MSC_VER
#  pragma message ( "Including torch/torch.h for C++ extensions is deprecated. Please include torch/extension.h" )
#else
#  warning "Including torch/torch.h for C++ extensions is deprecated. Please include torch/extension.h"
#endif

#endif // defined(TORCH_API_INCLUDE_EXTENSION_H)
