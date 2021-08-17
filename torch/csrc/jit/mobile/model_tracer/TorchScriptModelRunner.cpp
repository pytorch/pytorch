// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#include "caffe2/fb/model_tracer/TorchScriptModelRunner.h"
#include "caffe2/fb/model_tracer/MobileModelRunner.h"

namespace facebook::pytorch {

std::vector<std::vector<at::IValue>> TorchScriptModelRunner::
    get_all_bundled_inputs() {
  auto has_bundled_input = module_->find_method("get_all_bundled_inputs");
  CAFFE_ENFORCE(
      has_bundled_input,
      "Model does not have bundled inputs. ",
      "Use torch.utils.bundled_inputs.augment_model_with_bundled_inputs to add.");

  c10::IValue bundled_inputs = module_->run_method("get_all_bundled_inputs");
  return MobileModelRunner::ivalue_to_bundled_inputs(bundled_inputs);
}

void TorchScriptModelRunner::run_with_inputs(
    std::vector<std::vector<at::IValue>> const& bundled_inputs) {
  for (std::vector<at::IValue> const& input : bundled_inputs) {
    module_->forward(input);
  }
}

} // namespace facebook::pytorch
