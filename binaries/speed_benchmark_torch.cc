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
#include <vector>

#include "ATen/ATen.h"
#include "caffe2/core/timer.h"
#include "caffe2/utils/string_utils.h"
#include "torch/csrc/autograd/grad_mode.h"
#include "torch/csrc/jit/import.h"
#include "torch/script.h"

#include <chrono>
using namespace std::chrono;

C10_DEFINE_string(model, "", "The given torch script model to benchmark.");
C10_DEFINE_string(
    input_dims,
    "",
    "Alternate to input_files, if all inputs are simple "
    "float TensorCPUs, specify the dimension using comma "
    "separated numbers. If multiple input needed, use "
    "semicolon to separate the dimension of different "
    "tensors.");
C10_DEFINE_string(input_type, "", "Input type (uint8_t/float)");
C10_DEFINE_bool(
  print_output,
  false,
  "Whether to print output with all one input tensor.");
C10_DEFINE_int(warmup, 0, "The number of iterations to warm up.");
C10_DEFINE_int(iter, 10, "The number of iterations to run.");
C10_DEFINE_bool(
  report_pep,
  false,
  "Whether to print performance stats for AI-PEP.");

C10_DEFINE_int(pytext_len, 0, "Length of input sequence.");

std::vector<std::string>
split(char separator, const std::string& string, bool ignore_empty = true) {
  std::vector<std::string> pieces;
  std::stringstream ss(string);
  std::string item;
  while (getline(ss, item, separator)) {
    if (!ignore_empty || !item.empty()) {
      pieces.push_back(std::move(item));
    }
  }
  return pieces;
}

int main(int argc, char** argv) {
  c10::SetUsageMessage(
    "Run speed benchmark for pytorch model.\n"
    "Example usage:\n"
    "./speed_benchmark_torch"
    " --model=<model_file>"
    " --input_dims=\"1,3,224,224\""
    " --input_type=float"
    " --warmup=5"
    " --iter=20");
  if (!c10::ParseCommandLineFlags(&argc, &argv)) {
    std::cerr << "Failed to parse command line flags!" << std::endl;
    return 1;
  }

  CAFFE_ENFORCE_GE(FLAGS_input_dims.size(), 0, "Input dims must be specified.");
  CAFFE_ENFORCE_GE(FLAGS_input_type.size(), 0, "Input type must be specified.");

  std::vector<std::string> input_dims_list = split(';', FLAGS_input_dims);
  std::vector<std::string> input_type_list = split(';', FLAGS_input_type);
  CAFFE_ENFORCE_EQ(
      input_dims_list.size(),
      input_type_list.size(),
      "Input dims and type should have the same number of items.");

  std::vector<c10::IValue> inputs;
  for (size_t i = 0; i < input_dims_list.size(); ++i) {
    auto input_dims_str = split(',', input_dims_list[i]);
    std::vector<int64_t> input_dims;
    for (const auto& s : input_dims_str) {
      input_dims.push_back(c10::stoi(s));
    }
    if (input_type_list[i] == "float") {
      inputs.push_back(torch::ones(input_dims, at::ScalarType::Float));
    } else if (input_type_list[i] == "uint8_t") {
      inputs.push_back(torch::ones(input_dims, at::ScalarType::Byte));
    } else if (input_type_list[i] == "int64") {
      inputs.push_back(torch::ones(input_dims, torch::kI64));
    } else {
      CAFFE_THROW("Unsupported input type: ", input_type_list[i]);
    }
  }

  if (FLAGS_pytext_len > 0) {
    auto stensor = FLAGS_pytext_len * at::ones({1}, torch::kI64);
    inputs.push_back(stensor);
  }

  auto qengines = at::globalContext().supportedQEngines();
  if (std::find(qengines.begin(), qengines.end(), at::QEngine::QNNPACK) != qengines.end()) {
    at::globalContext().setQEngine(at::QEngine::QNNPACK);
  }
  torch::autograd::AutoGradMode guard(false);
  torch::jit::GraphOptimizerEnabledGuard no_optimizer_guard(false);
  auto module = torch::jit::load(FLAGS_model);

  module.eval();
  if (FLAGS_print_output) {
    std::cout << module.forward(inputs) << std::endl;
  }

  std::cout << "Starting benchmark." << std::endl;
  std::cout << "Running warmup runs." << std::endl;
  CAFFE_ENFORCE(
      FLAGS_warmup >= 0,
      "Number of warm up runs should be non negative, provided ",
      FLAGS_warmup,
      ".");
  for (int i = 0; i < FLAGS_warmup; ++i) {
    module.forward(inputs);
  }

  std::cout << "Main runs." << std::endl;
  CAFFE_ENFORCE(
      FLAGS_iter >= 0,
      "Number of main runs should be non negative, provided ",
      FLAGS_iter,
      ".");
  caffe2::Timer timer;
  std::vector<float> times;
  auto millis = timer.MilliSeconds();
  for (int i = 0; i < FLAGS_iter; ++i) {
    auto start = high_resolution_clock::now();
    module.forward(inputs);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    times.push_back(duration.count());
  }
  millis = timer.MilliSeconds();
  if (FLAGS_report_pep) {
    for (auto t : times) {
      std::cout << "PyTorchObserver {\"type\": \"NET\", \"unit\": \"us\", \"metric\": \"latency\", \"value\": \"" << t << "\"}" << std::endl;
    }
  }
  std::cout << "Main run finished. Milliseconds per iter: "
            << millis / FLAGS_iter
            << ". Iters per second: " << 1000.0 * FLAGS_iter / millis
            << std::endl;

  return 0;
}
