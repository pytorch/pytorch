#include <caffe2/video/optical_flow.h>

namespace caffe2 {

void OpticalFlowExtractor(
    const cv::Mat& prev_gray,
    const cv::Mat& curr_gray,
    const int flow_alg_type,
    cv::Mat& flow) {
#if CV_MAJOR_VERSION >= 4
  cv::Ptr<cv::DISOpticalFlow> tvl1 = cv::DISOpticalFlow::create();
#else
  cv::Ptr<cv::DualTVL1OpticalFlow> tvl1 = cv::DualTVL1OpticalFlow::create();
#endif
  switch (flow_alg_type) {
    case FLowAlgType::FarnebackOpticalFlow:
      cv::calcOpticalFlowFarneback(
          prev_gray,
          curr_gray,
          flow,
          std::sqrt(2) / 2.0,
          5,
          10,
          2,
          7,
          1.5,
          cv::OPTFLOW_FARNEBACK_GAUSSIAN);
      break;
    case FLowAlgType::DensePyrLKOpticalFlow:
      LOG(ERROR) << "DensePyrLKOpticalFlow only has sparse version on CPU";
      break;
    case FLowAlgType::BroxOpticalFlow:
      LOG(ERROR) << "BroxOpticalFlow on CPU is not available";
      break;
    case FLowAlgType::OpticalFlowDual_TVL1:
      tvl1->calc(prev_gray, curr_gray, flow);
      break;
    default:
      LOG(ERROR) << "Unsupported optical flow type " << flow_alg_type;
      break;
  }
}

void MergeOpticalFlow(cv::Mat& prev_flow, const cv::Mat& curr_flow) {
  const int rows = prev_flow.rows;
  const int cols = prev_flow.cols;

  // merge two optical flows into one
  for (int y = 0; y < rows; y++) {
    for (int x = 0; x < cols; x++) {
      cv::Point2f u = prev_flow.at<cv::Point2f>(y, x);
      // get the new location
      int x_new = std::min(cols - 1, std::max(0, cvRound(u.x + x)));
      int y_new = std::min(rows - 1, std::max(0, cvRound(u.y + y)));
      cv::Point2f u_new = curr_flow.at<cv::Point2f>(y_new, x_new);

      // update the flow
      prev_flow.at<cv::Point2f>(y, x) += u_new;
    }
  }
}

void MultiFrameOpticalFlowExtractor(
    const std::vector<cv::Mat>& grays,
    const int optical_flow_alg_type,
    cv::Mat& flow) {
  int num_frames = grays.size();
  CAFFE_ENFORCE_GE(num_frames, 2, "need at least 2 frames!");

  // compute optical flow for every two frames
  std::vector<cv::Mat> flows;
  for (int i = 0; i < num_frames - 1; i++) {
    cv::Mat tmp;
    OpticalFlowExtractor(grays[i], grays[i + 1], optical_flow_alg_type, tmp);
    flows.push_back(tmp);
  }

  flows[0].copyTo(flow);
  // aggregate optical flow across multiple frame
  for (int i = 1; i < num_frames - 1; i++) {
    MergeOpticalFlow(flow, flows[i]);
  }
}

} // namespace caffe2
