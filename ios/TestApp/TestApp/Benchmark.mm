#ifdef BUILD_LITE_INTERPRETER

#import "Benchmark.h"
#include <string>
#include <vector>
#include <torch/script.h>
#include <torch/csrc/jit/mobile/function.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/interpreter.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/observer.h>
#include "ATen/ATen.h"
#include "caffe2/core/timer.h"
#include "caffe2/utils/string_utils.h"
#include "torch/csrc/autograd/grad_mode.h"

static std::string model = "model_lite.ptl";
static std::string input_dims = "1,3,224,224";
static std::string input_type = "float";
static BOOL print_output = false;
static int warmup = 10;
static int iter = 10;

@implementation Benchmark

+ (BOOL)setup:(NSDictionary*)config {
  NSString* modelPath = [[NSBundle mainBundle] pathForResource:@"model_lite" ofType:@"ptl"];
  if (![[NSFileManager defaultManager] fileExistsAtPath:modelPath]) {
    NSLog(@"model_lite.ptl doesn't exist!");
    return NO;
  }
  model = std::string(modelPath.UTF8String);
  input_dims = std::string(((NSString*)config[@"input_dims"]).UTF8String);
  input_type = std::string(((NSString*)config[@"input_type"]).UTF8String);
  warmup = ((NSNumber*)config[@"warmup"]).intValue;
  iter = ((NSNumber*)config[@"iter"]).intValue;
  print_output = ((NSNumber*)config[@"print_output"]).boolValue;
  return YES;
}

+ (NSString*)run {
  std::vector<std::string> logs;
#define UI_LOG(fmt, ...)                                          \
  {                                                               \
    NSString* log = [NSString stringWithFormat:fmt, __VA_ARGS__]; \
    NSLog(@"%@", log);                                            \
    logs.push_back(log.UTF8String);                               \
  }

  CAFFE_ENFORCE_GE(input_dims.size(), 0, "Input dims must be specified.");
  CAFFE_ENFORCE_GE(input_type.size(), 0, "Input type must be specified.");

  std::vector<std::string> input_dims_list = caffe2::split(';', input_dims);
  std::vector<std::string> input_type_list = caffe2::split(';', input_type);
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

  c10::InferenceMode mode;
  auto module = torch::jit::_load_for_mobile(model);

//  module.eval();
  if (print_output) {
    std::cout << module.forward(inputs) << std::endl;
  }
  UI_LOG(@"Running warmup runs", nil);
  CAFFE_ENFORCE(warmup >= 0, "Number of warm up runs should be non negative, provided ", warmup,
                ".");
  for (int i = 0; i < warmup; ++i) {
    module.forward(inputs);
  }
  UI_LOG(@"Main runs", nil);
  CAFFE_ENFORCE(iter >= 0, "Number of main runs should be non negative, provided ", iter, ".");
  caffe2::Timer timer;
  auto millis = timer.MilliSeconds();
  for (int i = 0; i < iter; ++i) {
    module.forward(inputs);
  }
  millis = timer.MilliSeconds();
  UI_LOG(@"Main run finished. Milliseconds per iter: %.3f", millis / iter, nil);
  UI_LOG(@"Iters per second: : %.3f", 1000.0 * iter / millis, nil);
  UI_LOG(@"Done.", nil);

  NSString* results = @"";
  for (auto& msg : logs) {
    results = [results stringByAppendingString:[NSString stringWithUTF8String:msg.c_str()]];
    results = [results stringByAppendingString:@"\n"];
  }
  return results;
}

@end
#endif
