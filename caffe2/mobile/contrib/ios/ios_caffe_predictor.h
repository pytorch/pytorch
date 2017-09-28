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

#include <string>
#include "caffe2/core/net.h"
#include "caffe2/core/predictor.h"
#include "caffe2/mobile/contrib/ios/ios_caffe_defines.h"

struct Tensor {
  std::vector<int64_t> dims;
  uint8_t* data;
};

class IOS_CAFFE_EXPORT Caffe2IOSPredictor final {
 public:
  /**
   @allowMetalOperators Allow converting eligible operators to Metal GPU framework accelerated
   operators. Setting this flag to true doesn't gaurantee predictor will be using Metal operators;
   Client code must check usingMetalOperators flag to determine predictor is using them.
   */
  static Caffe2IOSPredictor* NewCaffe2IOSPredictor(const caffe2::NetDef& init_net,
                                                   const caffe2::NetDef& predict_net,
                                                   bool disableMultithreadProcessing,
                                                   bool allowMetalOperators);
  void run(const Tensor& inData, Tensor& outData, std::string& errorMessage);
  ~Caffe2IOSPredictor(){};

  const bool usingMetalOperators;

 private:
  Caffe2IOSPredictor(const caffe2::NetDef& init_net,
                     const caffe2::NetDef& predict_net,
                     bool disableMultithreadProcessing,
                     bool usingMetalOperators);
  caffe2::Predictor predictor_;
};
