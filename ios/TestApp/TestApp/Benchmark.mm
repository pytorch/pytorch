#import "Benchmark.h"
#include <string>
#include <vector>

#include "ATen/ATen.h"
#include "caffe2/core/timer.h"
#include "caffe2/utils/string_utils.h"
#include "torch/csrc/autograd/grad_mode.h"
#include "torch/csrc/jit/import.h"
#include "torch/script.h"

C10_DEFINE_string(model, "", "The given torch script model to benchmark.");
C10_DEFINE_string(input_dims, "1,3,224,224",
                  "Alternate to input_files, if all inputs are simple "
                  "float TensorCPUs, specify the dimension using comma "
                  "separated numbers. If multiple input needed, use "
                  "semicolon to separate the dimension of different "
                  "tensors.");
C10_DEFINE_string(input_type, "float", "Input type (uint8_t/float)");
C10_DEFINE_bool(print_output, false, "Whether to print output with all one input tensor.");
C10_DEFINE_int(warmup, 0, "The number of iterations to warm up.");
C10_DEFINE_int(iter, 10, "The number of iterations to run.");

@implementation Benchmark

+ (NSString*)benchmarkWithModel:(NSString*)modelPath {
  std::vector<std::string> logs;
#define UI_LOG(fmt, ...)                                          \
  {                                                               \
    NSString* log = [NSString stringWithFormat:fmt, __VA_ARGS__]; \
    NSLog(@"%@", log);                                            \
    logs.push_back(log.UTF8String);                               \
  }

  FLAGS_model = std::string(modelPath.UTF8String);
  CAFFE_ENFORCE_GE(FLAGS_input_dims.size(), 0, "Input dims must be specified.");
  CAFFE_ENFORCE_GE(FLAGS_input_type.size(), 0, "Input type must be specified.");

  std::vector<std::string> input_dims_list = caffe2::split(';', FLAGS_input_dims);
  std::vector<std::string> input_type_list = caffe2::split(';', FLAGS_input_type);
  CAFFE_ENFORCE_EQ(input_dims_list.size(), input_type_list.size(),
                   "Input dims and type should have the same number of items.");

  std::vector<c10::IValue> inputs;
  for (size_t i = 0; i < input_dims_list.size(); ++i) {
    auto input_dims_str = caffe2::split(',', input_dims_list[i]);
    std::vector<int64_t> input_dims;
    for (const auto& s : input_dims_str) {
      input_dims.push_back(c10::stoi(s));
    }
    if (input_type_list[i] == "float") {
      inputs.push_back(torch::ones(input_dims, at::ScalarType::Float));
    } else if (input_type_list[i] == "uint8_t") {
      inputs.push_back(torch::ones(input_dims, at::ScalarType::Byte));
    } else {
      CAFFE_THROW("Unsupported input type: ", input_type_list[i]);
    }
  }

  auto qengines = at::globalContext().supportedQEngines();
  if (std::find(qengines.begin(), qengines.end(), at::QEngine::QNNPACK) != qengines.end()) {
    at::globalContext().setQEngine(at::QEngine::QNNPACK);
  }
  torch::autograd::AutoGradMode guard(false);
  auto module = torch::jit::load(FLAGS_model);

  at::AutoNonVariableTypeMode non_var_type_mode(true);
  module.eval();
  if (FLAGS_print_output) {
    std::cout << module.forward(inputs) << std::endl;
  }
  UI_LOG(@"Start benchmarking...", nil);
  UI_LOG(@"Running warmup runs")
  CAFFE_ENFORCE(FLAGS_warmup >= 0, "Number of warm up runs should be non negative, provided ",
                FLAGS_warmup, ".");
  for (int i = 0; i < FLAGS_warmup; ++i) {
    module.forward(inputs);
  }
  UI_LOG(@"Main runs", nil);
  CAFFE_ENFORCE(FLAGS_iter >= 0, "Number of main runs should be non negative, provided ",
                FLAGS_iter, ".");
  caffe2::Timer timer;
  auto millis = timer.MilliSeconds();
  for (int i = 0; i < FLAGS_iter; ++i) {
    module.forward(inputs);
  }
  millis = timer.MilliSeconds();
  UI_LOG(@"Main run finished. Milliseconds per iter: %.3f", millis / FLAGS_iter, nil);
  UI_LOG(@"Iters per second: : %.3f", 1000.0 * FLAGS_iter / millis, nil);
  UI_LOG(@"Done.", nil);

  NSString* results = @"";
  for (auto& msg : logs) {
    results = [results stringByAppendingString:[NSString stringWithUTF8String:msg.c_str()]];
    results = [results stringByAppendingString:@"\n"];
  }
  return results;
}

@end
