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
