#include <caffe2/video/video_io.h>
#include <caffe2/core/logging.h>
#include <algorithm>
#include <random>
#include <string>

namespace caffe2 {

// assume CLHW order and color channels RGB
void Saturation(
    float* clip,
    const int length,
    const int crop_height,
    const int crop_width,
    const float alpha_rand,
    std::mt19937* randgen) {
  float alpha = 1.0f +
      std::uniform_real_distribution<float>(-alpha_rand, alpha_rand)(*randgen);

  // RGB to Gray scale image: R -> 0.299, G -> 0.587, B -> 0.114
  const int channel_size = length * crop_height * crop_width;
  int p = 0;
  for (int l = 0; l < length; ++l) {
    for (int h = 0; h < crop_height; ++h) {
      for (int w = 0; w < crop_width; ++w) {
        float gray_color = clip[p] * 0.299f + clip[p + channel_size] * 0.587f +
            clip[p + 2 * channel_size] * 0.114f;
        for (int c = 0; c < 3; ++c) {
          clip[c * channel_size + p] =
              clip[c * channel_size + p] * alpha + gray_color * (1.0f - alpha);
        }
        p++;
      }
    }
  }
}

// assume CLHW order and color channels RGB
void Brightness(
    float* clip,
    const int length,
    const int crop_height,
    const int crop_width,
    const float alpha_rand,
    std::mt19937* randgen) {
  float alpha = 1.0f +
      std::uniform_real_distribution<float>(-alpha_rand, alpha_rand)(*randgen);

  int p = 0;
  for (int c = 0; c < 3; ++c) {
    for (int l = 0; l < length; ++l) {
      for (int h = 0; h < crop_height; ++h) {
        for (int w = 0; w < crop_width; ++w) {
          clip[p++] *= alpha;
        }
      }
    }
  }
}

// assume CLHW order and color channels RGB
void Contrast(
    float* clip,
    const int length,
    const int crop_height,
    const int crop_width,
    const float alpha_rand,
    std::mt19937* randgen) {
  const int channel_size = length * crop_height * crop_width;
  float gray_mean = 0;
  int p = 0;
  for (int l = 0; l < length; ++l) {
    for (int h = 0; h < crop_height; ++h) {
      for (int w = 0; w < crop_width; ++w) {
        // RGB to Gray scale image: R -> 0.299, G -> 0.587, B -> 0.114
        gray_mean += clip[p] * 0.299f + clip[p + channel_size] * 0.587f +
            clip[p + 2 * channel_size] * 0.114f;
        p++;
      }
    }
  }
  gray_mean /= (length * crop_height * crop_width);

  float alpha = 1.0f +
      std::uniform_real_distribution<float>(-alpha_rand, alpha_rand)(*randgen);
  p = 0;
  for (int c = 0; c < 3; ++c) {
    for (int l = 0; l < length; ++l) {
      for (int h = 0; h < crop_height; ++h) {
        for (int w = 0; w < crop_width; ++w) {
          clip[p] = clip[p] * alpha + gray_mean * (1.0f - alpha);
          p++;
        }
      }
    }
  }
}

// assume CLHW order and color channels RGB
void ColorJitter(
    float* clip,
    const int length,
    const int crop_height,
    const int crop_width,
    const float saturation,
    const float brightness,
    const float contrast,
    std::mt19937* randgen) {
  std::srand(unsigned(std::time(0)));
  std::vector<int> jitter_order{0, 1, 2};
  // obtain a time-based seed:
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::shuffle(
      jitter_order.begin(),
      jitter_order.end(),
      std::default_random_engine(seed));

  for (int i = 0; i < 3; ++i) {
    if (jitter_order[i] == 0) {
      Saturation(clip, length, crop_height, crop_width, saturation, randgen);
    } else if (jitter_order[i] == 1) {
      Brightness(clip, length, crop_height, crop_width, brightness, randgen);
    } else {
      Contrast(clip, length, crop_height, crop_width, contrast, randgen);
    }
  }
}

// assume CLHW order and color channels RGB
void ColorLighting(
    float* clip,
    const int length,
    const int crop_height,
    const int crop_width,
    const float alpha_std,
    const std::vector<std::vector<float>>& eigvecs,
    const std::vector<float>& eigvals,
    std::mt19937* randgen) {
  std::normal_distribution<float> d(0, alpha_std);
  std::vector<float> alphas(3);
  for (int i = 0; i < 3; ++i) {
    alphas[i] = d(*randgen);
  }

  std::vector<float> delta_rgb(3, 0.0);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      delta_rgb[i] += eigvecs[i][j] * eigvals[j] * alphas[j];
    }
  }

  int p = 0;
  for (int c = 0; c < 3; ++c) {
    for (int l = 0; l < length; ++l) {
      for (int h = 0; h < crop_height; ++h) {
        for (int w = 0; w < crop_width; ++w) {
          clip[p++] += delta_rgb[c];
        }
      }
    }
  }
}

// assume CLHW order and color channels RGB
// mean subtraction and scaling.
void ColorNormalization(
    float* clip,
    const int length,
    const int crop_height,
    const int crop_width,
    const int channels,
    const std::vector<float>& mean,
    const std::vector<float>& inv_std) {
  int p = 0;
  for (int c = 0; c < channels; ++c) {
    for (int l = 0; l < length; ++l) {
      for (int h = 0; h < crop_height; ++h) {
        for (int w = 0; w < crop_width; ++w) {
          clip[p] = (clip[p] - mean[c]) * inv_std[c];
          p++;
        }
      }
    }
  }
}

void ClipTransformRGB(
    const unsigned char* buffer_rgb,
    const int multi_crop_count,
    const int crop_height,
    const int crop_width,
    const int length_rgb,
    const int channels_rgb,
    const int sampling_rate_rgb,
    const int height,
    const int width,
    const int h_off,
    const int w_off,
    const int* multi_crop_h_off,
    const int* multi_crop_w_off,
    const bool mirror_me,
    const bool color_jitter,
    const float saturation,
    const float brightness,
    const float contrast,
    const bool color_lighting,
    const float color_lighting_std,
    const std::vector<std::vector<float>>& color_lighting_eigvecs,
    const std::vector<float>& color_lighting_eigvals,
    const std::vector<float>& mean_rgb,
    const std::vector<float>& inv_std_rgb,
    std::mt19937* randgen,
    float* transformed_clip) {
  CAFFE_ENFORCE_EQ(
      channels_rgb, mean_rgb.size(), "rgb channels must be equal to mean size");
  CAFFE_ENFORCE_EQ(
      mean_rgb.size(),
      inv_std_rgb.size(),
      "mean size must be equal to inv_std size");
  int orig_index, tran_index;
  if (multi_crop_count == 1) {
    // Case 1: Multi_cropping is disabled
    // The order of output dimensions is C, L, H, W
    bool do_color_jitter_lighting =
        (color_jitter || color_lighting) && channels_rgb == 3;
    for (int c = 0; c < channels_rgb; ++c) {
      for (int l = 0; l < length_rgb; ++l) {
        int orig_index_l =
            l * sampling_rate_rgb * height * width * channels_rgb;
        int tran_index_l = (c * length_rgb + l) * crop_height;

        for (int h = 0; h < crop_height; ++h) {
          int orig_index_h = orig_index_l + (h + h_off) * width * channels_rgb;
          int tran_index_h = (tran_index_l + h) * crop_width;

          for (int w = 0; w < crop_width; ++w) {
            orig_index = orig_index_h + (w + w_off) * channels_rgb + c;

            // mirror the frame
            if (mirror_me) {
              tran_index = tran_index_h + (crop_width - 1 - w);
            } else {
              tran_index = tran_index_h + w;
            }

            // normalize and transform the clip
            if (do_color_jitter_lighting) {
              transformed_clip[tran_index] = buffer_rgb[orig_index];
            } else {
              transformed_clip[tran_index] =
                  (buffer_rgb[orig_index] - mean_rgb[c]) * inv_std_rgb[c];
            }
          }
        }
      }
    }
    if (color_jitter && channels_rgb == 3) {
      ColorJitter(
          transformed_clip,
          length_rgb,
          crop_height,
          crop_width,
          saturation,
          brightness,
          contrast,
          randgen);
    }
    if (color_lighting && channels_rgb == 3) {
      ColorLighting(
          transformed_clip,
          length_rgb,
          crop_height,
          crop_width,
          color_lighting_std,
          color_lighting_eigvecs,
          color_lighting_eigvals,
          randgen);
    }
    if (do_color_jitter_lighting) {
      // Color normalization
      // Mean subtraction and division by standard deviation.
      ColorNormalization(
          transformed_clip,
          length_rgb,
          crop_height,
          crop_width,
          channels_rgb,
          mean_rgb,
          inv_std_rgb);
    }
  } else {
    // Case 2: Multi_cropping is enabled. Multi cropping should be only used at
    // testing stage. So color jittering and lighting are not used
    for (int multi_crop_mirror = 0; multi_crop_mirror < 2;
         ++multi_crop_mirror) {
      for (int i = 0; i < multi_crop_count / 2; ++i) {
        for (int c = 0; c < channels_rgb; ++c) {
          for (int l = 0; l < length_rgb; ++l) {
            int orig_index_l =
                l * sampling_rate_rgb * height * width * channels_rgb;
            int tran_index_l = (c * length_rgb + l) * crop_height;

            for (int h = 0; h < crop_height; ++h) {
              int orig_index_h = orig_index_l +
                  (h + multi_crop_h_off[i]) * width * channels_rgb;
              int tran_index_h = (tran_index_l + h) * crop_width;

              for (int w = 0; w < crop_width; ++w) {
                orig_index =
                    orig_index_h + (w + multi_crop_w_off[i]) * channels_rgb + c;

                if (multi_crop_mirror == 1) {
                  tran_index = tran_index_h + (crop_width - 1 - w);
                } else {
                  tran_index = tran_index_h + w;
                }

                transformed_clip[tran_index] =
                    (buffer_rgb[orig_index] - mean_rgb[c]) * inv_std_rgb[c];
              }
            }
          }
        }
        transformed_clip +=
            channels_rgb * length_rgb * crop_height * crop_width;
      }
    }
  }
}

void ClipTransformOpticalFlow(
    const unsigned char* buffer_rgb,
    const int crop_height,
    const int crop_width,
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
  const int frame_size = crop_height * crop_width;
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
    cv::Mat flow = cv::Mat::zeros(crop_height, crop_width, CV_32FC2);
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

void FreeDecodedData(
    std::vector<std::unique_ptr<DecodedFrame>>& sampledFrames) {
  // free the sampledFrames
  for (int i = 0; i < sampledFrames.size(); i++) {
    DecodedFrame* p = sampledFrames[i].release();
    delete p;
  }
  sampledFrames.clear();
}

bool DecodeMultipleClipsFromVideo(
    const char* video_buffer,
    const std::string& video_filename,
    const int encoded_size,
    const Params& params,
    const int start_frm,
    const int clip_per_video,
    const bool use_local_file,
    int& height,
    int& width,
    std::vector<unsigned char*>& buffer_rgb) {
  std::vector<std::unique_ptr<DecodedFrame>> sampledFrames;
  VideoDecoder decoder;

  // decoding from buffer or file
  if (!use_local_file) {
    decoder.decodeMemory(
        video_buffer, encoded_size, params, start_frm, sampledFrames);
  } else {
    decoder.decodeFile(video_filename, params, start_frm, sampledFrames);
  }

  for (int i = 0; i < buffer_rgb.size(); i++) {
    unsigned char* buff = buffer_rgb[i];
    delete[] buff;
  }
  buffer_rgb.clear();

  if (sampledFrames.size() < params.num_of_required_frame_) {
    LOG(ERROR)
        << "The video seems faulty and we could not decode enough frames: "
        << sampledFrames.size() << " VS " << params.num_of_required_frame_;
    FreeDecodedData(sampledFrames);
    return true;
  }

  height = sampledFrames[0]->height_;
  width = sampledFrames[0]->width_;
  float sample_stepsz = 1.0;
  if (clip_per_video > 1) {
    sample_stepsz =
        float(sampledFrames.size() - params.num_of_required_frame_) /
        (clip_per_video - 1.0);
  }
  int image_size = 3 * height * width;
  int clip_size = params.num_of_required_frame_ * image_size;
  // get the RGB frames for each clip
  for (int i = 0; i < clip_per_video; i++) {
    unsigned char* buffer_rgb_ptr = new unsigned char[clip_size];
    int clip_start = floor(i * sample_stepsz);
    for (int j = 0; j < params.num_of_required_frame_; j++) {
      memcpy(
          buffer_rgb_ptr + j * image_size,
          (unsigned char*)sampledFrames[j + clip_start]->data_.get(),
          image_size * sizeof(unsigned char));
    }
    buffer_rgb.push_back(buffer_rgb_ptr);
  }
  FreeDecodedData(sampledFrames);

  return true;
}

} // namespace caffe2
