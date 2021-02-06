#pragma once

#include <ATen/ATen.h>

#include <typeinfo>


namespace at {
namespace checks {

inline bool torch_tensor_on_same_device_check(const at::Tensor &ten1, const at::Tensor &ten2){
  return ten1.get_device()==ten2.get_device();
}

inline bool torch_tensor_on_same_device_check(
  const at::Tensor &ten1,
  const c10::optional<at::Tensor> &ten2
){
  return !ten2.has_value() || ten1.get_device()==ten2->get_device();
}

inline bool torch_tensor_on_cpu_check(const at::Tensor &ten){
  return !ten.is_cuda(); // TODO: Should be a better way to do this
}

inline bool torch_tensor_on_cpu_check(const c10::optional<at::Tensor> &ten){
  return !ten.has_value() || !ten->is_cuda(); // TODO: Should be a better way to do this
}

inline bool torch_tensor_on_cuda_gpu_check(const at::Tensor &ten){
  return ten.is_cuda();
}

inline bool torch_tensor_on_cuda_gpu_check(const c10::optional<at::Tensor> &ten){
  return !ten.has_value() || ten->is_cuda();
}

inline std::string torch_tensor_device_name(const at::Tensor &ten){
  return c10::DeviceTypeName(ten.device().type());
}

inline std::string torch_tensor_device_name(const c10::optional<at::Tensor> &ten){
  if(ten.has_value()){
    return c10::DeviceTypeName(ten->device().type());
  } else {
    return "No device: optional tensor unused.";
  }
}

}}

#define TENSORS_HAVE_SAME_NUMEL(x,y) TORCH_CHECK(    \
  (x).numel() == (y).numel(),                        \
  #x " must have the same number of elements as " #y \
  " They had ", (x).numel(), " and ", (y).numel())

#define TENSORS_HAVE_SAME_TYPE(x,y) TORCH_CHECK(                   \
  (x).dtype() == (y).dtype(),                                      \
  #x " must have the same type as " #y                             \
  " types were ", (x).dtype().name(), " and ", (y).dtype().name())

#define TENSOR_TYPE_MUST_BE(ten,typ) TORCH_CHECK( \
  (ten).scalar_type() == typ,                     \
  "Tensor '" #ten "' must have scalar type " #typ \
  " but it had type ", (ten).dtype().name())

#define TENSOR_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

#define TENSOR_NDIM_EQUALS(ten,dims)                     \
  TORCH_CHECK((ten).ndimension() == (dims),              \
  "Tensor '" #ten "' must have " #dims " dimension(s). " \
  "Found ", (ten).ndimension())

#define TENSOR_NDIM_IS_GE(ten,dims)                        \
  TORCH_CHECK((ten).dim() >= (dims),                       \
  "Tensor '" #ten "' must have >=" #dims " dimension(s). " \
  "Found ", (ten).ndimension())

#define TENSOR_NDIM_EXCEEDS(ten,dims)                              \
  TORCH_CHECK((ten).dim() > (dims),                                \
  "Tensor '" #ten "' must have more than " #dims " dimension(s). " \
  "Found ", (ten).ndimension())

#define TENSORS_ON_SAME_DEVICE(x,y) TORCH_CHECK(        \
  at::checks::torch_tensor_on_same_device_check(x,y),   \
  #x " must be on the same device as " #y "! "          \
  #x " is currently on ", torch_tensor_device_name(x),  \
  #y " is currently on ", torch_tensor_device_name(y))

#define TENSOR_ON_CPU(x) TORCH_CHECK(             \
  at::checks::torch_tensor_on_cpu_check(x),       \
  #x " must be a CPU tensor; it is currently on device ", at::checks::torch_tensor_device_name(x))

#define TENSOR_CONTIGUOUS_AND_ON_CPU(x) \
  TENSOR_ON_CPU(x);                     \
  TENSOR_CONTIGUOUS(x)

#define TENSOR_ON_CUDA_GPU(x) TORCH_CHECK(                  \
  at::checks::torch_tensor_on_cuda_gpu_check(x),            \
  #x " must be a CUDA tensor; it is currently on device ", at::checks::torch_tensor_device_name(x))

#define TENSOR_CONTIGUOUS_AND_ON_CUDA_GPU(x) \
  TENSOR_ON_CUDA_GPU(x);                     \
  TENSOR_CONTIGUOUS(x)
