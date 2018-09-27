
#pragma once

#include <string>
#include "caffe2/core/net.h"
#include "caffe2/mobile/contrib/ios/ios_caffe_defines.h"
#include "caffe2/predictor/predictor.h"

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
