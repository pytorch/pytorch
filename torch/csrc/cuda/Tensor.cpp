#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif

// Order of these includes matters, which should be fixed.
// clang-format off
#include <torch/csrc/python_headers.h>
#include <structmember.h>

#include <torch/csrc/cuda/THCP.h>

#include <torch/csrc/utils/tensor_numpy.h>
#include <torch/csrc/copy_utils.h>
#include <torch/csrc/DynamicTypes.h>
// clang-format on
