#define __STDC_FORMAT_MACROS

// Order of these includes matters, which should be fixed.
// clang-format off
#include <torch/csrc/python_headers.h>
#include <structmember.h>

#include <stack>
#include <tuple>
#include <vector>
#include <torch/csrc/cuda/THCP.h>

#include <torch/csrc/utils/tensor_numpy.h>
#include <torch/csrc/cuda/override_macros.h>
#include <torch/csrc/copy_utils.h>
#include <torch/csrc/DynamicTypes.h>

// generic_include THC torch/csrc/generic/Tensor.cpp

#include <torch/csrc/cuda/undef_macros.h>
#include <torch/csrc/cuda/restore_macros.h>
// clang-format on
