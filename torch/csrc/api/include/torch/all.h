#pragma once

#if !defined(_MSC_VER) && __cplusplus < 201402L
#error C++14 or later compatible compiler is required to use PyTorch.
#endif

#include <torch/cuda.h>
#include <torch/data.h>
#include <torch/enum.h>
#include <torch/jit.h>
#include <torch/linalg.h>
#include <torch/nn.h>
#include <torch/optim.h>
#include <torch/serialize.h>
#include <torch/types.h>
#include <torch/utils.h>
#include <torch/autograd.h>
