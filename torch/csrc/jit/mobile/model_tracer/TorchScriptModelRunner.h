#pragma once
// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#include <iostream>
#include <sstream>

#include "torch/csrc/autograd/grad_mode.h"
#include "torch/csrc/jit/api/module.h"
#include "torch/csrc/jit/serialization/export.h"
#include "torch/script.h"

namespace facebook::pytorch {

/**
 * TorchScriptModelRunner is instantiated by passing in a PyTorch (TorchScript)
 * model file's path (location) on disk.
 */
class TorchScriptModelRunner {
  std::shared_ptr<torch::jit::Module> module_;

 protected:
 public:
  explicit TorchScriptModelRunner(std::string file_path) {
    module_ = std::make_shared<torch::jit::Module>(torch::jit::load(file_path));
  }
  ~TorchScriptModelRunner() = default;

  /**
   * Fetches all the bundled inputs of the loaded model.
   *
   * A bundled input itself is of type std::vector<at::IValue> and the
   * elements of this vector<> are the arguments that the "forward"
   * method of the model accepts. i.e. each of the at::IValue is a
   * single argument to the model's "forward" method.
   *
   * The outer vector holds a bundled input. For models with bundled
   * inputs, the outer most vector will have size > 0.
   */
  std::vector<std::vector<at::IValue>> get_all_bundled_inputs();

  /**
   * Runs the model against all of the provided inputs using the model's
   * "forward" method.
   */
  void run_with_inputs(
      std::vector<std::vector<at::IValue>> const& bundled_inputs);

  std::vector<std::string> get_root_operators() {
    return torch::jit::export_opnames(*module_);
  }

  void save_for_mobile(std::string file_path) {
    module_->_save_for_mobile(file_path);
  }
};

} // namespace facebook::pytorch
