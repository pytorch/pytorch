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


#ifdef __cplusplus

#include <string>
#include <vector>
#include "caffe2/core/predictor.h"
#include "caffe2/mobile/contrib/ios/ios_caffe_defines.h"
#include "caffe2/mobile/contrib/ios/ios_caffe_predictor.h"

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
