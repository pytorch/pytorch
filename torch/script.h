#pragma once

#include <torch/csrc/api/include/torch/types.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/custom_class.h>

#include <ATen/ATen.h>
