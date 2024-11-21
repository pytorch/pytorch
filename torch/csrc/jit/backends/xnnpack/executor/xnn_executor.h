// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <xnnpack.h>
#include <memory>
#include <vector>

namespace torch::jit::xnnpack::delegate {

class XNNExecutor {
 private:
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> runtime_{
      nullptr,
      &xnn_delete_runtime};
  std::vector<uint32_t> input_ids_;
  std::vector<uint32_t> output_ids_;
  std::vector<xnn_external_value> externals_;

 public:
  XNNExecutor() = default;

  template <typename T>
  bool set_inputs(std::vector<T*>& inputs, std::vector<T*>& outputs) {
    externals_.clear();

    if (inputs.size() != input_ids_.size()) {
      return false;
    }

    for (int i = 0; i < inputs.size(); i++) {
      externals_.emplace_back(xnn_external_value{input_ids_[i], inputs[i]});
    }

    if (outputs.size() != output_ids_.size()) {
      return false;
    }

    for (int i = 0; i < outputs.size(); i++) {
      externals_.emplace_back(xnn_external_value{output_ids_[i], outputs[i]});
    }

    return true;
  }

  bool forward() {
    xnn_status status =
        xnn_setup_runtime(runtime_.get(), externals_.size(), externals_.data());

    if (status != xnn_status_success) {
      return false;
    }

    status = xnn_invoke_runtime(runtime_.get());

    if (status != xnn_status_success) {
      return false;
    }

    return true;
  }

  friend class XNNCompiler;
};

} // namespace torch::jit::xnnpack::delegate
