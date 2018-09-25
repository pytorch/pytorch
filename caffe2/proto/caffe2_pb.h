#pragma once
#include <ATen/core/Device.h>
#include <ATen/core/Error.h>
#include <caffe2/proto/caffe2.pb.h>

namespace caffe2 {

using DeviceType = at::DeviceType;
constexpr DeviceType CPU = DeviceType::CPU;
constexpr DeviceType CUDA = DeviceType::CUDA;
constexpr DeviceType OPENGL = DeviceType::OPENGL;
constexpr DeviceType OPENCL = DeviceType::OPENCL;
constexpr DeviceType MKLDNN = DeviceType::MKLDNN;
constexpr DeviceType IDEEP = DeviceType::IDEEP;
constexpr DeviceType HIP = DeviceType::HIP;
constexpr DeviceType COMPILE_TIME_MAX_DEVICE_TYPES =
    DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES;
constexpr DeviceType ONLY_FOR_TEST = DeviceType::ONLY_FOR_TEST;

inline CAFFE2_API DeviceType ProtoToType(const caffe2::DeviceTypeProto p) {
  switch (p) {
    case caffe2::PROTO_CPU:
      return DeviceType::CPU;
    case caffe2::PROTO_CUDA:
      return DeviceType::CUDA;
    case caffe2::PROTO_OPENGL:
      return DeviceType::OPENGL;
    case caffe2::PROTO_OPENCL:
      return DeviceType::OPENCL;
    case caffe2::PROTO_MKLDNN:
      return DeviceType::MKLDNN;
    case caffe2::PROTO_IDEEP:
      return DeviceType::IDEEP;
    case caffe2::PROTO_HIP:
      return DeviceType::HIP;
    case caffe2::PROTO_COMPILE_TIME_MAX_DEVICE_TYPES:
      return DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES;
    case caffe2::PROTO_ONLY_FOR_TEST:
      return DeviceType::ONLY_FOR_TEST;
    default:
      AT_ERROR(
          "Unknown device:",
          static_cast<int32_t>(p),
          ". If you have recently updated the caffe2.proto file to add a new "
          "device type, did you forget to update the ProtoToType() and TypeToProto"
          "function to reflect such recent changes?");
  }
}

inline CAFFE2_API DeviceType ProtoToType(int p) {
  return ProtoToType(static_cast<caffe2::DeviceTypeProto>(p));
}

inline CAFFE2_API DeviceTypeProto TypeToProto(const DeviceType& t) {
  switch (t) {
    case DeviceType::CPU:
      return caffe2::PROTO_CPU;
    case DeviceType::CUDA:
      return caffe2::PROTO_CUDA;
    case DeviceType::OPENGL:
      return caffe2::PROTO_OPENGL;
    case DeviceType::OPENCL:
      return caffe2::PROTO_OPENCL;
    case DeviceType::MKLDNN:
      return caffe2::PROTO_MKLDNN;
    case DeviceType::IDEEP:
      return caffe2::PROTO_IDEEP;
    case DeviceType::HIP:
      return caffe2::PROTO_HIP;
    case DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES:
      return caffe2::PROTO_COMPILE_TIME_MAX_DEVICE_TYPES;
    case DeviceType::ONLY_FOR_TEST:
      return caffe2::PROTO_ONLY_FOR_TEST;
    default:
      AT_ERROR(
          "Unknown device:",
          static_cast<int32_t>(t),
          ". If you have recently updated the caffe2.proto file to add a new "
          "device type, did you forget to update the ProtoToType() and TypeToProto"
          "function to reflect such recent changes?");
  }
}

inline CAFFE2_API caffe2::DeviceOption DeviceToOption(
    const at::Device& device) {
  caffe2::DeviceOption option;
  auto type = device.type();
  option.set_device_type(TypeToProto(type));
  // sets the gpu_id to -1 means we'll use the current gpu id when the function
  // is being called, see context_gpu.cu for more info.
  if (type == at::DeviceType::CUDA) {
    option.set_cuda_gpu_id(device.index());
  } else if (type == at::DeviceType::HIP) {
    option.set_hip_gpu_id(device.index());
  }
  return option;
}

inline CAFFE2_API at::Device OptionToDevice(const caffe2::DeviceOption option) {
  at::Device device(ProtoToType(option.device_type()));
  auto type = device.type();
  if (type == at::DeviceType::CUDA) {
    device.set_index(option.cuda_gpu_id());
  } else if (type == at::DeviceType::HIP) {
    device.set_index(option.hip_gpu_id());
  }
  return device;
}

} // namespace caffe2
