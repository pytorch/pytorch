// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#pragma once

#include <ATen/ATen.h>
#include <ATen/Exceptions.h>

#include <typeinfo>

#define TENSOR_NDIM_EQUALS(ten, dims)      \
  TORCH_CHECK(                             \
      (ten).ndimension() == (dims),        \
      "Tensor '" #ten "' must have " #dims \
      " dimension(s). "                    \
      "Found ",                            \
      (ten).ndimension())

#define TENSOR_ON_CUDA_GPU(x)                                  \
  TORCH_CHECK(                                                 \
      x.is_cuda(),                                             \
      #x " must be a CUDA tensor; it is currently on device ", \
      torch_tensor_device_name(x))
