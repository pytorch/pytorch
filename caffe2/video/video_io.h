#ifndef CAFFE2_VIDEO_VIDEO_IO_H_
#define CAFFE2_VIDEO_VIDEO_IO_H_

#include <caffe2/core/common.h>
#include <caffe2/video/optical_flow.h>
#include <caffe2/video/video_decoder.h>
#include <opencv2/opencv.hpp>
#include <random>

#include <istream>
#include <ostream>

namespace caffe2 {

CAFFE2_API void ClipTransformRGB(
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
    float* transformed_clip);

CAFFE2_API void ClipTransformOpticalFlow(
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
    float* transformed_clip);

CAFFE2_API void FreeDecodedData(
    std::vector<std::unique_ptr<DecodedFrame>>& sampledFrames);

CAFFE2_API bool DecodeMultipleClipsFromVideo(
    const char* video_buffer,
    const std::string& video_filename,
    const int encoded_size,
    const Params& params,
    const int start_frm,
    const int clip_per_video,
    const bool use_local_file,
    int& height,
    int& width,
    std::vector<unsigned char*>& buffer_rgb);

} // namespace caffe2

#endif // CAFFE2_VIDEO_VIDEO_IO_H_
