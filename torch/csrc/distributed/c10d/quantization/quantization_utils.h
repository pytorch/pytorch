// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#pragma once

#include <ATen/ATen.h>

#include <typeinfo>

inline std::string torch_tensor_device_name(const at::Tensor& ten) {
  return c10::DeviceTypeName(ten.device().type());
}

inline bool torch_tensor_on_cpu_check(const at::Tensor& ten) {
  return !ten.is_cuda(); // TODO: Should be a better way to do this
}

inline bool torch_tensor_on_cuda_gpu_check(const at::Tensor& ten) {
  return ten.is_cuda();
}

#define TENSOR_NDIM_EQUALS(ten, dims)      \
  TORCH_CHECK(                             \
      (ten).ndimension() == (dims),        \
      "Tensor '" #ten "' must have " #dims \
      " dimension(s). "                    \
      "Found ",                            \
      (ten).ndimension())

#define TENSOR_ON_CPU(x)                                      \
  TORCH_CHECK(                                                \
      torch_tensor_on_cpu_check(x),                           \
      #x " must be a CPU tensor; it is currently on device ", \
      torch_tensor_device_name(x))

#define TENSOR_ON_CUDA_GPU(x)                                  \
  TORCH_CHECK(                                                 \
      torch_tensor_on_cuda_gpu_check(x),                       \
      #x " must be a CUDA tensor; it is currently on device ", \
      torch_tensor_device_name(x))
