#include <caffe2/video/video_io.h>
#include <caffe2/core/logging.h>
#include <algorithm>
#include <random>
#include <string>

namespace caffe2 {

void ClipTransformRGB(
    const unsigned char* buffer_rgb,
    const int crop_size,
    const int length_rgb,
    const int channels_rgb,
    const int sampling_rate_rgb,
    const int height,
    const int width,
    const int h_off,
    const int w_off,
    const bool mirror_me,
    const std::vector<float>& mean_rgb,
    const std::vector<float>& inv_std_rgb,
    float* transformed_clip) {
  // The order of output dimensions is C, L, H, W
  int orig_index, tran_index;
  for (int c = 0; c < channels_rgb; ++c) {
    for (int l = 0; l < length_rgb; ++l) {
      int orig_index_l = l * sampling_rate_rgb * height * width * channels_rgb;
      int tran_index_l = (c * length_rgb + l) * crop_size;

      for (int h = 0; h < crop_size; ++h) {
        int orig_index_h = orig_index_l + (h + h_off) * width * channels_rgb;
        int tran_index_h = (tran_index_l + h) * crop_size;

        for (int w = 0; w < crop_size; ++w) {
          orig_index = orig_index_h + (w + w_off) * channels_rgb + c;

          // mirror the frame
          if (mirror_me) {
            tran_index = tran_index_h + (crop_size - 1 - w);
          } else {
            tran_index = tran_index_h + w;
          }

          // normalize and transform the clip
          transformed_clip[tran_index] =
              (buffer_rgb[orig_index] - mean_rgb[c]) * inv_std_rgb[c];
        }
      }
    }
  }
}

void ClipTransformOpticalFlow(
    const unsigned char* buffer_rgb,
    const int crop_size,
    const int length_of,
    const int channels_of,
    const int sampling_rate_of,
    const int height,
    const int width,
    const cv::Rect& rect,
    const int channels_rgb,
    const bool mirror_me,
    const int flow_alg_type,
    const int flow_data_type,
    const int frame_gap_of,
    const bool do_flow_aggregation,
    const std::vector<float>& mean_of,
    const std::vector<float>& inv_std_of,
    float* transformed_clip) {
  const int frame_size = crop_size * crop_size;
  const int channel_size_flow = length_of * frame_size;

  // for get the mean and std of the input data
  bool extract_statistics = false;
  static std::vector<double> mean_static(channels_of, 0.f);
  static std::vector<double> std_static(channels_of, 0.f);
  static long long count = 0;
  cv::Scalar mean_img, std_img;

  for (int l = 0; l < length_of; l++) {
    // get the grayscale frames
    std::vector<cv::Mat> grays, rgbs;
    int step_size = do_flow_aggregation ? 1 : frame_gap_of;
    for (int j = 0; j <= frame_gap_of; j += step_size) {
      // get the current frame
      const unsigned char* curr_frame = buffer_rgb +
          (l * sampling_rate_of + j) * height * width * channels_rgb;
      cv::Mat img = cv::Mat::zeros(height, width, CV_8UC3);
      memcpy(
          img.data,
          curr_frame,
          height * width * channels_rgb * sizeof(unsigned char));

      // crop and mirror the frame
      cv::Mat img_cropped = img(rect);
      if (mirror_me) {
        cv::flip(img_cropped, img_cropped, 1);
      }

      cv::Mat gray;
      cv::cvtColor(img_cropped, gray, cv::COLOR_RGB2GRAY);
      grays.push_back(gray);
      rgbs.push_back(img_cropped);
    }

    cv::Mat first_gray, first_rgb;
    cv::Mat flow = cv::Mat::zeros(crop_size, crop_size, CV_32FC2);
    MultiFrameOpticalFlowExtractor(grays, flow_alg_type, flow);

    std::vector<cv::Mat> imgs;
    cv::split(flow, imgs);
    // save the 2-channel optical flow first
    int c = 0;
    for (; c < 2; c++) {
      if (extract_statistics) {
        cv::meanStdDev(imgs[c], mean_img, std_img);
        mean_static[c] += mean_img[0];
        std_static[c] += std_img[0];
      }

      imgs[c] -= mean_of[c];
      imgs[c] *= inv_std_of[c];
      memcpy(
          transformed_clip + c * channel_size_flow + l * frame_size,
          imgs[c].data,
          frame_size * sizeof(float));
    }

    cv::Mat mag;
    std::vector<cv::Mat> chans;
    // augment the optical flow with more channels
    switch (flow_data_type) {
      case FlowDataType::Flow2C:
        // nothing to do if we only need two channels
        break;

      case FlowDataType::Flow3C:
        // use magnitude as the third channel
        mag = cv::abs(imgs[0]) + cv::abs(imgs[1]);
        if (extract_statistics) {
          cv::meanStdDev(mag, mean_img, std_img);
          mean_static[c] += mean_img[0];
          std_static[c] += std_img[0];
        }

        mag -= mean_of[c];
        mag *= inv_std_of[c];
        memcpy(
            transformed_clip + c * channel_size_flow + l * frame_size,
            mag.data,
            frame_size * sizeof(float));
        break;

      case FlowDataType::FlowWithGray:
        // add grayscale image as the third channel
        grays[0].convertTo(first_gray, CV_32FC1);
        if (extract_statistics) {
          cv::meanStdDev(first_gray, mean_img, std_img);
          mean_static[c] += mean_img[0];
          std_static[c] += std_img[0];
        }

        first_gray -= mean_of[c];
        first_gray *= inv_std_of[c];
        memcpy(
            transformed_clip + c * channel_size_flow + l * frame_size,
            first_gray.data,
            frame_size * sizeof(float));
        break;

      case FlowDataType::FlowWithRGB:
        // add all three rgb channels
        rgbs[0].convertTo(first_rgb, CV_32FC3);
        cv::split(first_rgb, chans);
        for (; c < channels_of; c++) {
          if (extract_statistics) {
            cv::meanStdDev(chans[c - 2], mean_img, std_img);
            mean_static[c] += mean_img[0];
            std_static[c] += std_img[0];
          }

          chans[c - 2] -= mean_of[c];
          chans[c - 2] *= inv_std_of[c];
          memcpy(
              transformed_clip + c * channel_size_flow + l * frame_size,
              chans[c - 2].data,
              frame_size * sizeof(float));
        }
        break;

      default:
        LOG(ERROR) << "Unsupported optical flow data type " << flow_data_type;
        break;
    }

    if (extract_statistics) {
      count++;
      if (count % 1000 == 1) {
        for (int i = 0; i < channels_of; i++) {
          LOG(INFO) << i
                    << "-th channel mean: " << mean_static[i] / float(count)
                    << " std: " << std_static[i] / float(count);
        }
      }
    }
  }
}

} // namespace caffe2
