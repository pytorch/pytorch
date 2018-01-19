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

#ifndef CAFFE2_VIDEO_VIDEO_IO_H_
#define CAFFE2_VIDEO_VIDEO_IO_H_

#include <caffe/proto/caffe.pb.h>
#include <caffe2/video/optical_flow.h>
#include <caffe2/video/video_decoder.h>
#include <opencv2/opencv.hpp>
#include <random>

#include <istream>
#include <ostream>

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
    float* transformed_clip);

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
    float* transformed_clip);

void FreeDecodedData(std::vector<std::unique_ptr<DecodedFrame>>& sampledFrames);

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
    std::vector<unsigned char*>& buffer_rgb);

} // namespace caffe2

#endif // CAFFE2_VIDEO_VIDEO_IO_H_
