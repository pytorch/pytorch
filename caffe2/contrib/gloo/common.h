#pragma once

#include <exception>

#include "caffe2/core/blob.h"

#include <gloo/config.h>
#include <gloo/context.h>
#include <gloo/transport/device.h>

namespace caffe2 {
namespace gloo {

TORCH_API void signalFailure(Blob* status_blob, std::exception& exception);

struct createDeviceAttr {
    // "tcp" or "ibverbs"
    std::string transport;

    // E.g. "eth0" (tcp), or "mlx5_0" (ibverbs).
    // This may be empty to make Gloo figure it out.
    std::string interface;
};

TORCH_API std::shared_ptr<::gloo::transport::Device> createDevice(
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
