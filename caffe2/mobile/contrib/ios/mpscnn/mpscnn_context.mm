
#include "caffe2/core/common.h"

#if CAFFE2_MOBILE

#include "mpscnn_context.h"
#include "mpscnn_kernels.h"

#include "caffe2/core/logging.h"
#include "caffe2/core/timer.h"

#include <array>
#include <mutex>
#include <thread>

#import <Metal/MTLFunctionConstantValues.h>

namespace caffe2 {

MPSCNNContext& getMPSCNNContext() {
  static std::once_flag once;
  static MPSCNNContext ctx;
  std::call_once(once, []() {
    NSError* compileError = nil;
    ctx.device = MTLCreateSystemDefaultDevice();
    ctx.library = [ctx.device newLibraryWithSource:[NSString stringWithUTF8String:MPSCNN_KERNELS]
                                           options:nil
                                             error:&compileError];
    if (compileError != nil || ctx.library == nil) {
      CAFFE_THROW("Failed to load kernels: ", [[compileError localizedDescription] UTF8String]);
    }
    ctx.commandQueue = [ctx.device newCommandQueue];
  });
  return ctx;
}

id<MTLComputePipelineState> MPSCNNContext::getPipelineState(NSString* kernel) {
  std::string kernelStr = std::string([kernel UTF8String]);
  std::lock_guard<std::mutex> g(pipelineCacheMutex_);
  if (pipelineCache_.find(kernelStr) != pipelineCache_.end()) {
    VLOG(1) << "Hit in pipeline cache for: " << kernelStr;
    return pipelineCache_[kernelStr];
  }
  LOG(INFO) << "Miss in pipeline cache for: " << kernelStr;
  id<MTLFunction> func = [library newFunctionWithName:kernel];
  if (!func) {
    CAFFE_THROW("Couldn't get function: ", kernelStr);
    return nullptr;
  }
  NSError* errors;
  id<MTLComputePipelineState> state =
      [device newComputePipelineStateWithFunction:func error:&errors];
  if (!state) {
    CAFFE_THROW("Couldn't get state: ", kernelStr);
    return nullptr;
  }
  pipelineCache_[kernelStr] = state;
  return state;
}

id<MTLComputePipelineState> MPSCNNContext::getSpecializedPipelineState(
    NSString* kernel, const std::vector<ushort>& constants) {
  std::string kernelStr = std::string([kernel UTF8String]);
  for (auto i = 0; i < constants.size(); ++i) {
    kernelStr += "_" + std::to_string(constants[i]);
  }
  std::lock_guard<std::mutex> g(pipelineCacheMutex_);
  if (pipelineCache_.find(kernelStr) != pipelineCache_.end()) {
    VLOG(1) << "Hit in pipeline cache for: " << kernelStr;
    return pipelineCache_[kernelStr];
  }
  MTLFunctionConstantValues* constantValues = [MTLFunctionConstantValues new];
  for (auto i = 0; i < constants.size(); ++i) {
    [constantValues setConstantValue:&constants[i] type:MTLDataTypeUShort atIndex:i];
  }
  NSError* errors;

  LOG(INFO) << "Miss in pipeline cache for: " << kernelStr;
  id<MTLFunction> func =
      [library newFunctionWithName:kernel constantValues:constantValues error:&errors];
  if (!func) {
    CAFFE_THROW("Couldn't get function: ",
                kernelStr,
                " error: ",
                [[errors localizedDescription] UTF8String]);
    return nullptr;
  }
  id<MTLComputePipelineState> state =
      [device newComputePipelineStateWithFunction:func error:&errors];
  if (!state) {
    CAFFE_THROW("Couldn't get function: ",
                kernelStr,
                " error: ",
                [[errors localizedDescription] UTF8String]);
    return nullptr;
  }
  pipelineCache_[kernelStr] = state;
  return state;
}
}

#endif
