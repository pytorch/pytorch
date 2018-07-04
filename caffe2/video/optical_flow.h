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
