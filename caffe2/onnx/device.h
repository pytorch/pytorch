/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <functional>
#include <string>

namespace caffe2 { namespace onnx {

enum class DeviceType {CPU=0, CUDA=1};

struct Device {
  Device(const std::string& spec);
  DeviceType type;
  int device_id{-1};
};

}}

namespace std {
template <> struct hash<caffe2::onnx::DeviceType> {
  std::size_t operator()(const caffe2::onnx::DeviceType &k) const {
    return std::hash<int>()(static_cast<int>(k));
  }
};
} // namespace std
