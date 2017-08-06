// Copyright 2004-present Facebook. All Rights Reserved.

#ifdef __cplusplus

#include "caffe2/contrib/ios/ios_caffe_defines.h"
#include "caffe2/contrib/ios/ios_caffe_predictor.h"
#include "caffe2/core/predictor.h"
#include <string>
#include <vector>

extern "C" {

IOS_CAFFE_EXPORT Caffe2IOSPredictor* MakeCaffe2Predictor(const std::string& init_net_str,
                                                         const std::string& predict_net_str,
                                                         bool disableMultithreadProcessing,
                                                         bool allowMetalOperators,
                                                         std::string& errorMessage);
IOS_CAFFE_EXPORT void GenerateStylizedImage(std::vector<float>& originalImage,
                                            const std::string& init_net_str,
                                            const std::string& predict_net_str,
                                            int height,
                                            int width,
                                            std::vector<float>& dataOut);
}

#endif
