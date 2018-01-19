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

#ifndef CAFFE2_VIDEO_OPTICAL_FLOW_H_
#define CAFFE2_VIDEO_OPTICAL_FLOW_H_

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>

#include <caffe2/core/logging.h>

namespace caffe2 {

// Four different types of optical flow algorithms supported;
// BroxOpticalFlow doesn't have a CPU version;
// DensePyrLKOpticalFlow only has sparse CPU version;
enum FLowAlgType {
  FarnebackOpticalFlow = 0,
  DensePyrLKOpticalFlow = 1,
  BroxOpticalFlow = 2,
  OpticalFlowDual_TVL1 = 3,
};

// Define different types of optical flow data type
// 0: original two channel optical flow
// 1: three channel optical flow with magnitude as the third channel
// 2: two channel optical flow + one channel gray
// 3: two channel optical flow + three channel rgb
enum FlowDataType {
  Flow2C = 0,
  Flow3C = 1,
  FlowWithGray = 2,
  FlowWithRGB = 3,
};

// mean and std for normalizing different optical flow data type;
// Note that the statistics are generated from SOA, and you may
// want to change them if you are running on a different dataset;
const std::vector<float> InputDataMean =
    {0.0046635, 0.0046261, 0.963986, 102.976, 110.201, 100.64, 95.9966};
const std::vector<float> InputDataStd =
    {0.972347, 0.755146, 1.43588, 55.3691, 58.1489, 56.4701, 55.3324};

void OpticalFlowExtractor(
    const cv::Mat& prev_gray,
    const cv::Mat& curr_gray,
    const int optical_flow_alg_type,
    cv::Mat& flow);

void MergeOpticalFlow(cv::Mat& prev_flow, const cv::Mat& curr_flow);

void MultiFrameOpticalFlowExtractor(
    const std::vector<cv::Mat>& grays,
    const int optical_flow_alg_type,
    cv::Mat& flow);

} // namespace caffe2

#endif // CAFFE2_VIDEO_OPTICAL_FLOW_H_
