#define __STDC_FORMAT_MACROS

#include <torch/csrc/python_headers.h>
#include <structmember.h>

#include <vector>
#include <stack>
#include <tuple>
#include <torch/csrc/cuda/THCP.h>

#include <torch/csrc/cuda/override_macros.h>
#include <torch/csrc/copy_utils.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/utils/tensor_numpy.h>

//generic_include THC torch/csrc/generic/Tensor.cpp

#include <torch/csrc/cuda/undef_macros.h>
#include <torch/csrc/cuda/restore_macros.h>
