#pragma once

#include <functional>
#include <string>

namespace caffe2 {
namespace onnx {

enum class DeviceType { CPU = 0, CUDA = 1 };

struct Device {
  Device(const std::string& spec);
  DeviceType type;
  int device_id{-1};
};

} // namespace onnx
} // namespace caffe2

namespace std {
template <>
struct hash<caffe2::onnx::DeviceType> {
  std::size_t operator()(const caffe2::onnx::DeviceType& k) const {
    return std::hash<int>()(static_cast<int>(k));
  }
};
} // namespace std
