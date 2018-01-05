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

#include <exception>

#include "caffe2/core/blob.h"

#include <gloo/config.h>
#include <gloo/context.h>
#include <gloo/transport/device.h>

namespace caffe2 {
namespace gloo {

void signalFailure(Blob* status_blob, std::exception& exception);

struct createDeviceAttr {
    // "tcp" or "ibverbs"
    std::string transport;

    // E.g. "eth0" (tcp), or "mlx5_0" (ibverbs).
    // This may be empty to make Gloo figure it out.
    std::string interface;
};

std::shared_ptr<::gloo::transport::Device> createDevice(
    const createDeviceAttr attr);

// Captures the parameters passed to Gloo.
struct GlooParameters {
  std::shared_ptr<::gloo::Context> context;
  std::vector<const void*> inputs;
  std::vector<void*> outputs;
  size_t size;
  TypeMeta meta;

  template <typename T>
  std::vector<const T*> getInputs() {
    std::vector<const T*> result;
    result.reserve(inputs.size());
    for (auto& input : inputs) {
      result.push_back(reinterpret_cast<const T*>(input));
    }
    return result;
  }

  template <typename T>
  std::vector<T*> getOutputs() {
    std::vector<T*> result;
    result.reserve(outputs.size());
    for (auto& output : outputs) {
      result.push_back(reinterpret_cast<T*>(output));
    }
    return result;
  }

  template <typename T>
  T* getOutput() {
    return reinterpret_cast<T*>(outputs[0]);
  }

  template <typename T>
  bool IsType() const {
    return meta.Match<T>();
  }

  bool operator==(GlooParameters const& other) const {
    return context == other.context && inputs == other.inputs &&
        outputs == other.outputs && size == other.size;
  }
};

} // namespace gloo
} // namespace caffe2
