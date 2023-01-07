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

#include <iomanip>
#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <caffe2/core/timer.h>
#include <caffe2/utils/string_utils.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/script.h>

#include <c10/mobile/CPUCachingAllocator.h>

C10_DEFINE_string(
    refmodel,
    "",
    "The reference torch script model to compare against.");
C10_DEFINE_string(
    model,
    "",
    "The torch script model to compare to the reference model.");
C10_DEFINE_string(
    input_dims,
    "",
    "Alternate to input_files, if all inputs are simple "
    "float TensorCPUs, specify the dimension using comma "
    "separated numbers. If multiple input needed, use "
    "semicolon to separate the dimension of different "
    "tensors.");
C10_DEFINE_string(input_type, "", "Input type (uint8_t/float)");
C10_DEFINE_string(
    input_memory_format,
    "contiguous_format",
    "Input memory format (contiguous_format/channels_last)");
C10_DEFINE_int(input_max, 1, "The maximum value inputs should have");
C10_DEFINE_int(input_min, -1, "The minimum value inputs should have");
C10_DEFINE_bool(
    no_inputs,
    false,
    "Whether the model has any input. Will ignore other input arguments if true");
C10_DEFINE_bool(
    use_caching_allocator,
    false,
    "Whether to cache allocations between inference iterations");
C10_DEFINE_bool(
    print_output,
    false,
    "Whether to print output with all one input tensor.");
C10_DEFINE_int(iter, 10, "The number of iterations to run.");
C10_DEFINE_int(report_freq, 1000, "An update will be reported every n iterations");
C10_DEFINE_int(pytext_len, 0, "Length of input sequence.");
C10_DEFINE_string(
    backend,
    "cpu",
    "what backend to use for model (vulkan, cpu, metal) (default=cpu)");
C10_DEFINE_string(
    refbackend,
    "cpu",
    "what backend to use for model (vulkan, cpu, metal) (default=cpu)");
C10_DEFINE_string(tolerance, "1e-5", "tolerance to use for comparison");
C10_DEFINE_int(nthreads, 1, "Number of threads to launch. Useful for checking correct concurrent behaviour.");
C10_DEFINE_bool(
    report_failures,
    true,
    "Whether to report error during failed iterations");

bool checkRtol(
    const at::Tensor& diff,
    const std::vector<at::Tensor>& inputs,
    float tolerance,
    bool report) {
  float maxValue = 0.0f;

  for (const auto& tensor : inputs) {
    maxValue = fmax(tensor.abs().max().item<float>(), maxValue);
  }
  float threshold = tolerance * maxValue;
  float maxDiff = diff.abs().max().item<float>();

  bool passed = maxDiff < threshold;
  if (!passed && report) {
    std::cout << "Check FAILED!      Max diff allowed: "
              << std::setw(10) << std::setprecision(5) << threshold
              << "     max diff: "
              << std::setw(10) << std::setprecision(5) << maxDiff
              << std::endl;
  }

  return passed;
}

void report_pass_rate(int passed, int total) {
  int pass_rate = static_cast<int>(static_cast<float>(passed) / static_cast<float>(total) * 100);
  std::cout << "Output was equal within tolerance " << passed << "/"
            << total
            << " times. Pass rate: " << pass_rate
            << std::setprecision(2) << "%" << std::endl;
}

std::vector<std::string> split(
    char separator,
    const std::string& string,
    bool ignore_empty = true) {
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

std::vector<c10::IValue> create_inputs(
    std::vector<c10::IValue>& refinputs,
    std::vector<c10::IValue>& inputs,
    std::string& refbackend,
    std::string& backend,
    const int range_min,
    const int range_max) {
  if (FLAGS_no_inputs) {
    return {};
  }

  CAFFE_ENFORCE_GE(FLAGS_input_dims.size(), 0, "Input dims must be specified.");
  CAFFE_ENFORCE_GE(FLAGS_input_type.size(), 0, "Input type must be specified.");

  std::vector<std::string> input_dims_list = split(';', FLAGS_input_dims);
  std::vector<std::string> input_type_list = split(';', FLAGS_input_type);
  std::vector<std::string> input_memory_format_list =
      split(';', FLAGS_input_memory_format);

  CAFFE_ENFORCE_GE(
      input_dims_list.size(), 0, "Input dims not specified correctly.");
  CAFFE_ENFORCE_GE(
      input_type_list.size(), 0, "Input type not specified correctly.");
  CAFFE_ENFORCE_GE(
      input_memory_format_list.size(),
      0,
      "Input format list not specified correctly.");

  CAFFE_ENFORCE_EQ(
      input_dims_list.size(),
      input_type_list.size(),
      "Input dims and type should have the same number of items.");
  CAFFE_ENFORCE_EQ(
      input_dims_list.size(),
      input_memory_format_list.size(),
      "Input dims and format should have the same number of items.");

  for (size_t i = 0; i < input_dims_list.size(); ++i) {
    auto input_dims_str = split(',', input_dims_list[i]);
    std::vector<int64_t> input_dims;
    input_dims.reserve(input_dims_str.size());
    for (const auto& s : input_dims_str) {
      input_dims.push_back(c10::stoi(s));
    }

    at::ScalarType input_type;
    if (input_type_list[i] == "float") {
      input_type = at::ScalarType::Float;
    } else if (input_type_list[i] == "uint8_t") {
      input_type = at::ScalarType::Byte;
    } else if (input_type_list[i] == "int64") {
      input_type = at::ScalarType::Long;
    } else {
      CAFFE_THROW("Unsupported input type: ", input_type_list[i]);
    }

    at::MemoryFormat input_memory_format;
    if (input_memory_format_list[i] == "channels_last") {
      if (input_dims.size() != 4u) {
        CAFFE_THROW(
            "channels_last memory format only available on 4D tensors!");
      }
      input_memory_format = at::MemoryFormat::ChannelsLast;
    } else if (input_memory_format_list[i] == "contiguous_format") {
      input_memory_format = at::MemoryFormat::Contiguous;
    } else {
      CAFFE_THROW(
          "Unsupported input memory format: ", input_memory_format_list[i]);
    }

    const auto input_tensor = torch::rand(
        input_dims,
        at::TensorOptions(input_type).memory_format(input_memory_format))*(range_max - range_min) - range_min;

    if (refbackend == "vulkan") {
      refinputs.emplace_back(input_tensor.vulkan());
    } else {
      refinputs.emplace_back(input_tensor);
    }

    if (backend == "vulkan") {
      inputs.emplace_back(input_tensor.vulkan());
    } else {
      inputs.emplace_back(input_tensor);
    }
  }

  if (FLAGS_pytext_len > 0) {
    auto stensor = FLAGS_pytext_len * at::ones({1}, torch::kI64);
    if (refbackend == "vulkan") {
      refinputs.emplace_back(stensor.vulkan());
    } else {
      refinputs.emplace_back(stensor);
    }

    if (backend == "vulkan") {
      inputs.emplace_back(stensor.vulkan());
    } else {
      inputs.emplace_back(stensor);
    }
  }

  return inputs;
}

void run_check(float tolerance) {
  torch::jit::Module module = torch::jit::load(FLAGS_model);
  torch::jit::Module refmodule = torch::jit::load(FLAGS_refmodel);

  module.eval();
  refmodule.eval();

  std::thread::id this_id = std::this_thread::get_id();
  std::cout << "Running check on thread " << this_id << "." << std::endl;

  int passed = 0;
  for (int i = 0; i < FLAGS_iter; ++i) {
    std::vector<c10::IValue> refinputs;
    std::vector<c10::IValue> inputs;
    create_inputs(
        refinputs, inputs,
        FLAGS_refbackend, FLAGS_backend,
        FLAGS_input_min, FLAGS_input_max);

    const auto refoutput = refmodule.forward(refinputs).toTensor().cpu();
    const auto output = module.forward(inputs).toTensor().cpu();

    bool check = checkRtol(
        refoutput-output,
        {refoutput, output},
        tolerance,
        FLAGS_report_failures);

    if (check) {
      passed += 1;
    }
    else if (FLAGS_report_failures) {
      std::cout << " (Iteration " << i << " failed)" << std::endl;
    }

    if (i > 0 && (i+1) % FLAGS_report_freq == 0) {
      report_pass_rate(passed, i+1);
    }
  }
  report_pass_rate(passed, FLAGS_iter);
}

int main(int argc, char** argv) {
  c10::SetUsageMessage(
      "Run accuracy comparison to a reference model for a pytorch model.\n"
      "Example usage:\n"
      "./compare_models_torch"
      " --refmodel=<ref_model_file>"
      " --model=<model_file>"
      " --iter=20");
  if (!c10::ParseCommandLineFlags(&argc, &argv)) {
    std::cerr << "Failed to parse command line flags!" << std::endl;
    return 1;
  }

  if (FLAGS_input_min >= FLAGS_input_max) {
    std::cerr << "Input min: " << FLAGS_input_min
              << " should be less than input max: "
              << FLAGS_input_max << std::endl;
    return 1;
  }

  std::stringstream ss(FLAGS_tolerance);
  float tolerance = 0;
  ss >> tolerance;
  std::cout << "tolerance: " << tolerance << std::endl;

  c10::InferenceMode mode;
  torch::autograd::AutoGradMode guard(false);
  torch::jit::GraphOptimizerEnabledGuard no_optimizer_guard(false);

  c10::CPUCachingAllocator caching_allocator;
  c10::optional<c10::WithCPUCachingAllocatorGuard> caching_allocator_guard;
  if (FLAGS_use_caching_allocator) {
    caching_allocator_guard.emplace(&caching_allocator);
  }

  std::vector<std::thread> check_threads;
  check_threads.reserve(FLAGS_nthreads);
  for (int i = 0; i < FLAGS_nthreads; ++i) {
    check_threads.emplace_back(std::thread(run_check, tolerance));
  }

  for (std::thread& th : check_threads) {
    if (th.joinable()) {
      th.join();
    }
  }

  return 0;
}
