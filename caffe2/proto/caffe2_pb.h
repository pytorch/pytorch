#pragma once
#include <c10/core/Device.h>
#include <c10/util/Exception.h>
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

inline TORCH_API DeviceType ProtoToType(const caffe2::DeviceTypeProto p) {
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
    default:
      AT_ERROR(
          "Unknown device:",
          static_cast<int32_t>(p),
          ". If you have recently updated the caffe2.proto file to add a new "
          "device type, did you forget to update the ProtoToType() and TypeToProto"
          "function to reflect such recent changes?");
  }
}

inline TORCH_API DeviceType ProtoToType(int p) {
  return ProtoToType(static_cast<caffe2::DeviceTypeProto>(p));
}

inline TORCH_API DeviceTypeProto TypeToProto(const DeviceType& t) {
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
    default:
      AT_ERROR(
          "Unknown device:",
          static_cast<int32_t>(t),
          ". If you have recently updated the caffe2.proto file to add a new "
          "device type, did you forget to update the ProtoToType() and TypeToProto"
          "function to reflect such recent changes?");
  }
}

inline TORCH_API caffe2::DeviceOption DeviceToOption(
    const at::Device& device) {
  caffe2::DeviceOption option;
  auto type = device.type();
  option.set_device_type(TypeToProto(type));

  switch (type) {
    case DeviceType::CPU:
      if (device.index() != -1) {
        option.set_numa_node_id(device.index());
      }
      break;
    case DeviceType::CUDA:
    case DeviceType::HIP:
      option.set_device_id(device.index());
      break;
    case DeviceType::OPENGL:
    case DeviceType::OPENCL:
    case DeviceType::MKLDNN:
    case DeviceType::IDEEP:
    case DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES:
      break;
    default:
      AT_ERROR(
          "Unknown device:",
          static_cast<int32_t>(type),
          ". If you have recently updated the caffe2.proto file to add a new "
          "device type, did you forget to update the ProtoToType() and TypeToProto"
          "function to reflect such recent changes?");
  }
  return option;
}

inline TORCH_API at::Device OptionToDevice(const caffe2::DeviceOption option) {
  auto type = option.device_type();
  int32_t id = -1;
  switch (type) {
    case caffe2::PROTO_CPU:
      if (option.has_numa_node_id()) {
        id = option.numa_node_id();
      }
      break;
    case caffe2::PROTO_CUDA:
    case caffe2::PROTO_HIP:
      id = option.device_id();
      break;
  }
  return at::Device(ProtoToType(type), id);
}

inline void ExtractDeviceOption(
    DeviceOption* device_option,
    const at::Device& device) {
  AT_ASSERT(device_option);
  device_option->CopyFrom(DeviceToOption(device));
}

} // namespace caffe2
