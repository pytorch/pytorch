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

#include "torch/script.h"
#include "torch/csrc/jit/api/module.h"
#include "torch/csrc/jit/passes/vulkan_rewrite.h"
#include "torch/csrc/jit/passes/xnnpack_rewrite.h"
#include "torch/csrc/jit/serialization/import.h"
#include "torch/csrc/jit/serialization/export.h"

C10_DEFINE_string(model, "", "The torch script model to optimize.");
C10_DEFINE_string(
    output,
    "",
    "Name of the output model to be saved.");
C10_DEFINE_string(backend, "", "The backend to be optimized");

int main(int argc, char** argv) {
  c10::SetUsageMessage(
    "\nRun optimization pass for pytorch model. Example usage:\n"
    "./optimize_for_mobile"
    " --model=<model_file>"
    " [--output=<output_file_name>]"
    " [--backend=<cpu|vulkan>]"
  );

  if (!c10::ParseCommandLineFlags(&argc, &argv)) {
    std::cerr << "Failed to parse command line flags!" << std::endl;
    std::cout << c10::UsageMessage() << std::endl;
    return 1;
  }

  CAFFE_ENFORCE(FLAGS_model != "", c10::UsageMessage());

  std::string output_model_name =
    FLAGS_model.substr(0, FLAGS_model.find(".")) + "_optimized.bc";

  if (FLAGS_output != "") {
    output_model_name = FLAGS_output;
  }

  auto module = torch::jit::load(FLAGS_model);
  auto ops = torch::jit::export_opnames(module);
  std::cout << "\npt_operator_library(" << std::endl;
  std::cout << "\tname = \"old_op_library\"," << std::endl;
  std::cout << "\tops = [" << std::endl;
  for (auto const& op: ops) {
    std::cout << "\t\t\"" << op << "\"," << std::endl;
  }
  std::cout << "\t],\n)\n" << std::endl;

  torch::jit::Module optimized_module;
  if (FLAGS_backend == "" || FLAGS_backend == "cpu") {
    optimized_module = torch::jit::optimizeForMobile(module);
  } else if (FLAGS_backend == "vulkan") {
    optimized_module = torch::jit::vulkanOptimizeForMobile(module);
  } else {
    CAFFE_ENFORCE(false, "Unknown backend: " + FLAGS_backend);
  }
  auto new_ops = torch::jit::export_opnames(optimized_module);
  std::cout << "\npt_operator_library(" << std::endl;
  std::cout << "\tname = \"new_op_library\"," << std::endl;
  std::cout << "\tops = [" << std::endl;
  for (auto const& op: new_ops) {
    std::cout << "\t\t\"" << op << "\"," << std::endl;
  }
  std::cout << "\t],\n)\n" << std::endl;
  optimized_module._save_for_mobile(output_model_name);
  std::cout << "The optimized model for lite interpreter was saved to " << output_model_name << std::endl;
  return 0;
}
