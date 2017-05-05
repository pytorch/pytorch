#ifndef CAFFE2_VIDEO_VIDEO_IO_H_
#define CAFFE2_VIDEO_VIDEO_IO_H_

#include <opencv2/opencv.hpp>
#include <random>
#include "caffe/proto/caffe.pb.h"

#include <iostream>

namespace caffe2 {

void ImageChannelToBuffer(const cv::Mat* img, float* buffer, int c);

void ImageDataToBuffer(
    unsigned char* data_buffer,
    int height,
    int width,
    float* buffer,
    int c);

int GetNumberOfFrames(std::string filename);

double GetVideoFPS(std::string filename);

void GetVideoMeta(std::string filename, int& number_of_frames, double& fps);

void ClipTransform(
    const float* clip_data,
    const int channels,
    const int length,
    const int height,
    const int width,
    const int crop,
    const bool mirror,
    float mean,
    float std,
    float* transformed_clip,
    std::mt19937* randgen,
    std::bernoulli_distribution* mirror_this_clip,
    const bool use_center_crop);

bool ReadClipFromFrames(
    std::string input_dir,
    const int start_frm,
    std::string file_extension,
    const int length,
    const int height,
    const int width,
    const int sampling_rate,
    float*& buffer);

bool ReadClipFromVideoLazzy(
    std::string filename,
    const int start_frm,
    const int length,
    const int height,
    const int width,
    const int sampling_rate,
    float*& buffer);

bool ReadClipFromVideoSequential(
    std::string filename,
    const int start_frm,
    const int length,
    const int height,
    const int width,
    const int sampling_rate,
    float*& buffer);

bool ReadClipFromVideo(
    std::string filename,
    const int start_frm,
    const int length,
    const int height,
    const int width,
    const int sampling_rate,
    float*& buffer);

bool DecodeClipFromVideoFile(
    std::string filename,
    const int start_frm,
    const int length,
    const int height,
    const int width,
    const int sampling_rate,
    float*& buffer);

bool DecodeClipFromMemoryBuffer(
    const char* video_buffer,
    const int size,
    const int start_frm,
    const int length,
    const int height,
    const int width,
    const int sampling_rate,
    float*& buffer,
    std::mt19937* randgen);
}

#endif // CAFFE2_VIDEO_VIDEO_IO_H_
