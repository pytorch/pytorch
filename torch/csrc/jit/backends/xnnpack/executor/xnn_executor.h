// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <xnnpack.h>
#include <memory>
#include <vector>

namespace torch {
namespace jit {
namespace xnnpack {
namespace delegate {

class XNNExecutor {
 private:
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> runtime_;
  std::vector<uint32_t> input_ids_;
  std::vector<uint32_t> output_ids_;
  std::vector<xnn_external_value> externals_;

 public:
  XNNExecutor(xnn_runtime_t runtime_ptr)
      : runtime_(runtime_ptr, xnn_delete_runtime){};

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
  };

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
  };

  friend class XNNCompiler;
};

} // namespace delegate
} // namespace xnnpack
} // namespace jit
} // namespace torch
