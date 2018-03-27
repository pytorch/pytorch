
#pragma once

#import <Metal/MTLBuffer.h>
#import <Metal/MTLDevice.h>
#import <Metal/MTLLibrary.h>

#include <array>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

namespace caffe2 {

struct MPSCNNContext {
 public:
  id<MTLDevice> device;
  id<MTLCommandQueue> commandQueue;
  id<MTLLibrary> library;

  id<MTLComputePipelineState> getPipelineState(NSString* kernel);
  id<MTLComputePipelineState> getSpecializedPipelineState(NSString* kernel,
                                                          const std::vector<ushort>& constants);

 private:
  std::mutex pipelineCacheMutex_;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelineCache_;
};

// get the singleton instance.
MPSCNNContext& getMPSCNNContext();
} // namespace caffe2
