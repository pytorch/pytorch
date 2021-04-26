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

#include <ATen/ATen.h>
#include "caffe2/core/timer.h"
#include "caffe2/utils/string_utils.h"
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/script.h>

#include <c10/mobile/CPUCachingAllocator.h>

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
C10_DEFINE_string(
    input_memory_format,
    "contiguous_format",
    "Input memory format (contiguous_format/channels_last)");
C10_DEFINE_bool(
  no_inputs,
  false,
  "Whether the model has any input. Will ignore other input arugments if true");
C10_DEFINE_bool(
  use_caching_allocator,
  false,
  "Whether to cache allocations between inference iterations");
C10_DEFINE_int(
    use_bundled_input,
    -1,
    "If set, benchmark will expect the model to have bundled inputs "
    "and will run on the input with this index. ");
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
C10_DEFINE_bool(vulkan, false, "Whether to use Vulkan backend (GPU).");

namespace {

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

std::vector<c10::IValue> create_inputs() {
  if (FLAGS_no_inputs) {
    return {};
  }

  if (FLAGS_use_bundled_input >= 0) {
    // Need to get these after the model is loaded.
    return {};
  }

  CAFFE_ENFORCE_GE(FLAGS_input_dims.size(), 0, "Input dims must be specified.");
  CAFFE_ENFORCE_GE(FLAGS_input_type.size(), 0, "Input type must be specified.");

  std::vector<std::string> input_dims_list = split(';', FLAGS_input_dims);
  std::vector<std::string> input_type_list = split(';', FLAGS_input_type);
  std::vector<std::string> input_memory_format_list =
      split(';', FLAGS_input_memory_format);

  CAFFE_ENFORCE_EQ(
      input_dims_list.size(),
      input_type_list.size(),
      "Input dims and type should have the same number of items.");
  CAFFE_ENFORCE_EQ(
      input_dims_list.size(),
      input_memory_format_list.size(),
      "Input dims and format should have the same number of items.");

  std::vector<c10::IValue> inputs;
  for (size_t i = 0; i < input_dims_list.size(); ++i) {
    auto input_dims_str = split(',', input_dims_list[i]);
    std::vector<int64_t> input_dims;
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

    inputs.push_back(
        torch::ones(
            input_dims,
            at::TensorOptions(input_type).
            memory_format(input_memory_format)));
  }

  if (FLAGS_pytext_len > 0) {
    auto stensor = FLAGS_pytext_len * at::ones({1}, torch::kI64);
    inputs.push_back(stensor);
  }

  return inputs;
}

template<class T>
class Runner {
 public:
  virtual ~Runner() = default;
  virtual c10::IValue run(
      T& module,
      const std::vector<c10::IValue>& inputs) {
    return module.forward(inputs);
  }
};

template<class T>
class vkRunner final : public Runner<T> {
 public:
  virtual ~vkRunner() = default;
  virtual c10::IValue run(
      T& module,
      const std::vector<c10::IValue>& inputs) override {
    // Upload the input tensor(s) to GPU memory.
    inputs_.clear();
    inputs_.reserve(inputs.size());
    for (const auto& input : inputs) {
      inputs_.emplace_back(input.toTensor().vulkan());
    }

    // Run, and download the output tensor to system memory.
    return module.forward(inputs_).toTensor().cpu();
  }

 private:
  std::vector<c10::IValue> inputs_;
};

} // namespace

int main(int argc, char** argv) {
  c10::SetUsageMessage(
    "Run speed benchmark for pytorch model.\n"
    "Example usage:\n"
    "./speed_benchmark_torch"
    " --model=<model_file>"
    " --use_bundled_input=0"
    " --warmup=5"
    " --iter=20");
  if (!c10::ParseCommandLineFlags(&argc, &argv)) {
    std::cerr << "Failed to parse command line flags!" << std::endl;
    return 1;
  }

  std::vector<c10::IValue> inputs = create_inputs();

  c10::InferenceMode mode;
#if BUILD_LITE_INTERPRETER
  auto module = torch::jit::_load_for_mobile(FLAGS_model);
#else
  torch::jit::GraphOptimizerEnabledGuard no_optimizer_guard(false);
  auto module = torch::jit::load(FLAGS_model);
#endif

  if (FLAGS_use_bundled_input >= 0) {
    auto get_method = module.find_method("get_all_bundled_inputs");
    if (!get_method) {
      std::cerr << "Model does not have bundled inputs.  Before saving," << std::endl
        << "use torch.utils.bundled_inputs.augment_model_with_bundled_inputs." << std::endl;
      return 1;
    }

    auto all_inputs = (*get_method)({}).toList();
    if (FLAGS_use_bundled_input >= all_inputs.size()) {
      // NOTE: This check is only to make the error message nicer.
      // The get call below does internal bounds checking.
      std::cerr << "Model has only " << all_inputs.size() << " bundled inputs." << std::endl;
      return 1;
    }
    inputs = all_inputs.get(FLAGS_use_bundled_input).toTuple()->elements();
  }

#ifdef BUILD_LITE_INTERPRETER
  using ModuleType = torch::jit::mobile::Module;
#else
  using ModuleType = torch::jit::Module;
#endif

  const auto runner = FLAGS_vulkan ? std::make_unique<vkRunner<ModuleType>>()
                                   : std::make_unique<Runner<ModuleType>>();

#ifndef BUILD_LITE_INTERPRETER
  module.eval();
#endif

  if (FLAGS_print_output) {
    std::cout << runner->run(module, inputs) << std::endl;
  }

  c10::CPUCachingAllocator caching_allocator;
  c10::optional<c10::WithCPUCachingAllocatorGuard> caching_allocator_guard;
  if (FLAGS_use_caching_allocator) {
    caching_allocator_guard.emplace(&caching_allocator);
  }
  std::cout << "Starting benchmark." << std::endl;
  std::cout << "Running warmup runs." << std::endl;
  CAFFE_ENFORCE(
      FLAGS_warmup >= 0,
      "Number of warm up runs should be non negative, provided ",
      FLAGS_warmup,
      ".");
  for (int i = 0; i < FLAGS_warmup; ++i) {
    runner->run(module, inputs);
  }

  std::cout << "Main runs." << std::endl;
  CAFFE_ENFORCE(
      FLAGS_iter >= 0,
      "Number of main runs should be non negative, provided ",
      FLAGS_iter,
      ".");
  caffe2::Timer timer;
  std::vector<float> times;
  auto micros = timer.MicroSeconds();
  for (int i = 0; i < FLAGS_iter; ++i) {
    auto start = high_resolution_clock::now();
    runner->run(module, inputs);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    times.push_back(duration.count());
  }
  micros = timer.MicroSeconds();
  if (FLAGS_report_pep) {
    for (auto t : times) {
      std::cout << "PyTorchObserver {\"type\": \"NET\", \"unit\": \"us\", \"metric\": \"latency\", \"value\": \"" << t << "\"}" << std::endl;
    }
  }
  std::cout << "Main run finished. Microseconds per iter: "
            << micros / FLAGS_iter
            << ". Iters per second: " << 1000.0 * 1000 * FLAGS_iter / micros
            << std::endl;

  return 0;
}
