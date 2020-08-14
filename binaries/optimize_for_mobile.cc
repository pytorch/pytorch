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

#include <string>

#include "torch/csrc/jit/api/module.h"
#include "torch/csrc/jit/passes/vulkan_rewrite.h"
#include "torch/csrc/jit/passes/xnnpack_rewrite.h"
#include "torch/csrc/jit/serialization/import.h"

C10_DEFINE_string(model, "", "The given torch script model to transform.");
C10_DEFINE_string(
    output,
    "",
    "Name of the output model to be saved.");
C10_DEFINE_bool(
    save_for_mobile,
    false,
    "Save the model with bytecode format compatible with lite inteprter.");
C10_DEFINE_bool(vulkan, false, "Vulkan optimize_for_mobile");

int main(int argc, char** argv) {
  c10::SetUsageMessage(
    "Run speed benchmark for pytorch model.\n"
    "Example usage:\n"
    "./optimize_for_mobile"
    " --model=<model_file>"
    " --output=<output_file_name>");
  if (!c10::ParseCommandLineFlags(&argc, &argv)) {
    std::cerr << "Failed to parse command line flags!" << std::endl;
    return 1;
  }

  CAFFE_ENFORCE(FLAGS_model != "", "Valid input must be provided.");

  std::string output_model_name =
    FLAGS_model.substr(0, FLAGS_model.find(".")) + "_mobile_optimized.pt";

  if (FLAGS_output != "") {
    output_model_name = FLAGS_output;
  }

  auto module = torch::jit::load(FLAGS_model);

  auto optimized_module = FLAGS_vulkan
      ? torch::jit::vulkanOptimizeForMobile(module)
      : torch::jit::optimizeForMobile(module);

  if (FLAGS_save_for_mobile) {
    optimized_module._save_for_mobile(output_model_name);
  } else {
    optimized_module.save(output_model_name);
  }

  return 0;
}
