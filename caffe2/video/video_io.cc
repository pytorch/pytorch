#include "caffe2/video/video_io.h"
#include <random>
#include <string>
#include "caffe2/core/logging.h"
#include "caffe2/video/video_decoder.h"

namespace caffe2 {

void ImageChannelToBuffer(const cv::Mat* img, float* buffer, int c) {
  int idx = 0;
  for (int h = 0; h < img->rows; ++h) {
    for (int w = 0; w < img->cols; ++w) {
      buffer[idx++] = static_cast<float>(img->at<cv::Vec3b>(h, w)[c]);
    }
  }
}

void ImageDataToBuffer(
    unsigned char* data_buffer,
    int height,
    int width,
    float* buffer,
    int c) {
  int idx = 0;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      buffer[idx++] =
          static_cast<float>(data_buffer[h * width * 3 + w * 3 + c]);
    }
  }
}

void ClipTransform(
    const float* clip_data,
    const int channels,
    const int length,
    const int height,
    const int width,
    const int crop_size,
    const bool mirror,
    float mean,
    float std,
    float* transformed_clip,
    std::mt19937* randgen,
    std::bernoulli_distribution* mirror_this_clip,
    const bool use_center_crop) {
  int h_off = 0;
  int w_off = 0;

  if (use_center_crop) {
    h_off = (height - crop_size) / 2;
    w_off = (width - crop_size) / 2;
  } else {
    h_off = std::uniform_int_distribution<>(0, height - crop_size)(*randgen);
    w_off = std::uniform_int_distribution<>(0, width - crop_size)(*randgen);
  }

  float inv_std = 1.f / std;
  int top_index, data_index;
  bool mirror_me = mirror && (*mirror_this_clip)(*randgen);

  for (int c = 0; c < channels; ++c) {
    for (int l = 0; l < length; ++l) {
      for (int h = 0; h < crop_size; ++h) {
        for (int w = 0; w < crop_size; ++w) {
          data_index =
              ((c * length + l) * height + h_off + h) * width + w_off + w;
          if (mirror_me) {
            top_index = ((c * length + l) * crop_size + h) * crop_size +
                (crop_size - 1 - w);
          } else {
            top_index = ((c * length + l) * crop_size + h) * crop_size + w;
          }
          transformed_clip[top_index] =
              (clip_data[data_index] - mean) * inv_std;
        }
      }
    }
  }
}

bool ReadClipFromFrames(
    std::string img_dir,
    const int start_frm,
    std::string im_extension,
    const int length,
    const int height,
    const int width,
    const int sampling_rate,
    float*& buffer) {
  char fn_im[512];
  cv::Mat img, img_origin;
  buffer = nullptr;
  int offset = 0;
  int channel_size = 0;
  int image_size = 0;
  int data_size = 0;

  int end_frm = start_frm + length * sampling_rate;
  for (int i = start_frm; i < end_frm; i += sampling_rate) {
    snprintf(fn_im, 512, "%s/%06d%s", img_dir.c_str(), i, im_extension.c_str());
    if (height > 0 && width > 0) {
      img_origin = cv::imread(fn_im, CV_LOAD_IMAGE_COLOR);
      if (!img_origin.data) {
        LOG(ERROR) << "Could not open or find file " << fn_im;
        if (buffer != nullptr) {
          delete[] buffer;
        }
        return false;
      }
      cv::resize(img_origin, img, cv::Size(width, height));
      img_origin.release();
    } else {
      img = cv::imread(fn_im, CV_LOAD_IMAGE_COLOR);
      if (!img.data) {
        LOG(ERROR) << "Could not open or find file " << fn_im;
        if (buffer != nullptr) {
          delete[] buffer;
        }
        return false;
      }
    }

    // If this is the first frame, allocate memory for the buffer
    if (i == start_frm) {
      image_size = img.rows * img.cols;
      channel_size = image_size * length;
      data_size = channel_size * 3;
      buffer = new float[data_size];
    }

    for (int c = 0; c < 3; c++) {
      ImageChannelToBuffer(&img, buffer + c * channel_size + offset, c);
    }
    offset += image_size;
  }
  CAFFE_ENFORCE(offset == channel_size, "Wrong offset size");
  return true;
}

int GetNumberOfFrames(std::string filename) {
  cv::VideoCapture cap;
  cap.open(filename);
  if (!cap.isOpened()) {
    LOG(ERROR) << "Cannot open " << filename;
    return 0;
  }
  int num_of_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
  cap.release();
  return num_of_frames;
}

double GetVideoFPS(std::string filename) {
  cv::VideoCapture cap;
  cap.open(filename);
  if (!cap.isOpened()) {
    LOG(ERROR) << "Cannot open " << filename;
    return 0;
  }
  double fps = cap.get(CV_CAP_PROP_FPS);
  cap.release();
  return fps;
}

void GetVideoMeta(std::string filename, int& number_of_frames, double& fps) {
  cv::VideoCapture cap;
  cap.open(filename);
  if (cap.isOpened()) {
    number_of_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
    fps = cap.get(CV_CAP_PROP_FPS);
    cap.release();
  } else {
    LOG(ERROR) << "Cannot open " << filename;
    number_of_frames = -1;
    fps = 0;
  }
}

bool ReadClipFromVideoLazzy(
    std::string filename,
    const int start_frm,
    const int length,
    const int height,
    const int width,
    const int sampling_rate,
    float*& buffer) {
  cv::VideoCapture cap;
  cv::Mat img, img_origin;
  buffer = nullptr;
  int offset = 0;
  int channel_size = 0;
  int image_size = 0;
  int data_size = 0;
  int end_frm = 0;

  cap.open(filename);
  if (!cap.isOpened()) {
    LOG(ERROR) << "Cannot open " << filename;
    return false;
  }

  int num_of_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
  if (num_of_frames < length * sampling_rate) {
    LOG(INFO) << filename << " does not have enough frames; having "
              << num_of_frames;
    return false;
  }

  CAFFE_ENFORCE_GE(start_frm, 0, "start frame must be greater or equal to 0");

  if (start_frm) {
    cap.set(CV_CAP_PROP_POS_FRAMES, start_frm);
  }
  end_frm = start_frm + length * sampling_rate;
  CAFFE_ENFORCE_LE(
      end_frm,
      num_of_frames,
      "end frame must be less or equal to num of frames");

  for (int i = start_frm; i < end_frm; i += sampling_rate) {
    if (sampling_rate > 1) {
      cap.set(CV_CAP_PROP_POS_FRAMES, i);
    }
    if (height > 0 && width > 0) {
      cap.read(img_origin);
      if (!img_origin.data) {
        LOG(INFO) << filename << " has no data at frame " << i;
        if (buffer != nullptr) {
          delete[] buffer;
        }
        return false;
      }
      cv::resize(img_origin, img, cv::Size(width, height));
    } else {
      cap.read(img);
      if (!img.data) {
        LOG(ERROR) << "Could not open or find file " << filename;
        if (buffer != nullptr) {
          delete[] buffer;
        }
        return false;
      }
    }

    // If this is the fisrt frame, allocate memory for the buffer
    if (i == start_frm) {
      image_size = img.rows * img.cols;
      channel_size = image_size * length;
      data_size = channel_size * 3;
      buffer = new float[data_size];
    }

    for (int c = 0; c < 3; c++) {
      ImageChannelToBuffer(&img, buffer + c * channel_size + offset, c);
    }

    offset += image_size;
  }

  CAFFE_ENFORCE(offset == channel_size, "wrong offset size");
  cap.release();
  return true;
}

bool ReadClipFromVideoSequential(
    std::string filename,
    const int start_frm,
    const int length,
    const int height,
    const int width,
    const int sampling_rate,
    float*& buffer) {
  cv::VideoCapture cap;
  cv::Mat img, img_origin;
  buffer = nullptr;
  int offset = 0;
  int channel_size = 0;
  int image_size = 0;
  int data_size = 0;

  cap.open(filename);
  if (!cap.isOpened()) {
    LOG(ERROR) << "Cannot open " << filename;
    return false;
  }

  int num_of_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
  if (num_of_frames < length * sampling_rate) {
    LOG(INFO) << filename << " does not have enough frames; having "
              << num_of_frames;
    return false;
  }

  CAFFE_ENFORCE_GE(start_frm, 0, "start frame must be greater or equal to 0");

  // Instead of random access, do sequentically access (avoid key-frame issue)
  // This will keep start_frm frames
  int sequential_counter = 0;
  while (sequential_counter < start_frm) {
    cap.read(img_origin);
    sequential_counter++;
  }

  int end_frm = start_frm + length * sampling_rate;
  CAFFE_ENFORCE_LE(
      end_frm,
      num_of_frames,
      "end frame must be less or equal to num of frames");

  for (int i = start_frm; i < end_frm; i++) {
    if (sampling_rate > 1) {
      // If sampling_rate > 1, purposely keep some frames
      if ((i - start_frm) % sampling_rate != 0) {
        cap.read(img_origin);
        continue;
      }
    }
    if (height > 0 && width > 0) {
      cap.read(img_origin);
      if (!img_origin.data) {
        LOG(INFO) << filename << " has no data at frame " << i;
        if (buffer != nullptr) {
          delete[] buffer;
        }
        return false;
      }
      cv::resize(img_origin, img, cv::Size(width, height));
    } else {
      cap.read(img);
      if (!img.data) {
        LOG(ERROR) << "Could not open or find file " << filename;
        if (buffer != nullptr) {
          delete[] buffer;
        }
        return false;
      }
    }

    // If this is the first frame, then we allocate memory for the buffer
    if (i == start_frm) {
      image_size = img.rows * img.cols;
      channel_size = image_size * length;
      data_size = channel_size * 3;
      buffer = new float[data_size];
    }

    for (int c = 0; c < 3; c++) {
      ImageChannelToBuffer(&img, buffer + c * channel_size + offset, c);
    }

    offset += image_size;
  }
  CAFFE_ENFORCE(offset == channel_size, "wrong offset size");
  cap.release();

  return true;
}

bool ReadClipFromVideo(
    std::string filename,
    const int start_frm,
    const int length,
    const int height,
    const int width,
    const int sampling_rate,
    float*& buffer) {
  bool read_status = ReadClipFromVideoLazzy(
      filename, start_frm, length, height, width, sampling_rate, buffer);
  if (!read_status) {
    read_status = ReadClipFromVideoSequential(
        filename, start_frm, length, height, width, sampling_rate, buffer);
  }
  return read_status;
}

bool DecodeClipFromVideoFile(
    std::string filename,
    const int start_frm,
    const int length,
    const int height,
    const int width,
    const int sampling_rate,
    float*& buffer) {
  Params params;
  std::vector<std::unique_ptr<DecodedFrame>> sampledFrames;
  VideoDecoder decoder;

  params.outputHeight_ = height ? height : -1;
  params.outputWidth_ = width ? width : -1;
  params.maximumOutputFrames_ = MAX_DECODING_FRAMES;

  // decode all frames with defaul sampling rate
  decoder.decodeFile(filename, params, sampledFrames);

  buffer = nullptr;
  int offset = 0;
  int channel_size = 0;
  int image_size = 0;
  int data_size = 0;

  int end_frm = start_frm + length * sampling_rate;
  for (int i = start_frm; i < end_frm; i += sampling_rate) {
    if (i == start_frm) {
      image_size = sampledFrames[i]->height_ * sampledFrames[i]->width_;
      channel_size = image_size * length;
      data_size = channel_size * 3;
      buffer = new float[data_size];
    }

    for (int c = 0; c < 3; c++) {
      ImageDataToBuffer(
          (unsigned char*)sampledFrames[i]->data_.get(),
          sampledFrames[i]->height_,
          sampledFrames[i]->width_,
          buffer + c * channel_size + offset,
          c);
    }
    offset += image_size;
  }
  CAFFE_ENFORCE(offset == channel_size, "Wrong offset size");

  // free the sampledFrames
  for (int i = 0; i < sampledFrames.size(); i++) {
    DecodedFrame* p = sampledFrames[i].release();
    delete p;
  }
  sampledFrames.clear();

  return true;
}

bool DecodeClipFromMemoryBuffer(
    const char* video_buffer,
    const int size,
    const int start_frm,
    const int length,
    const int height,
    const int width,
    const int sampling_rate,
    float*& buffer,
    std::mt19937* randgen) {
  Params params;
  std::vector<std::unique_ptr<DecodedFrame>> sampledFrames;
  VideoDecoder decoder;

  params.outputHeight_ = height ? height : -1;
  params.outputWidth_ = width ? width : -1;
  params.maximumOutputFrames_ = MAX_DECODING_FRAMES;

  decoder.decodeMemory(video_buffer, size, params, sampledFrames);

  buffer = nullptr;
  int offset = 0;
  int channel_size = 0;
  int image_size = 0;
  int data_size = 0;

  int use_start_frm = start_frm;
  if (start_frm < 0) { // perform temporal jittering
    if ((int)(sampledFrames.size() - length * sampling_rate) > 0) {
      use_start_frm = std::uniform_int_distribution<>(
          0, (int)(sampledFrames.size() - length * sampling_rate))(*randgen);
    } else {
      use_start_frm = 0;
    }
  }

  CAFFE_ENFORCE_LT(
    use_start_frm,
    sampledFrames.size(),
    "Starting frame must less than total number of video frames");

  int end_frm = use_start_frm + length * sampling_rate;

  CAFFE_ENFORCE_LE(
      end_frm,
      sampledFrames.size(),
      "Ending frame must less than or equal total number of video frames");

  for (int i = use_start_frm; i < end_frm; i += sampling_rate) {
    if (i == use_start_frm) {
      image_size = sampledFrames[i]->height_ * sampledFrames[i]->width_;
      channel_size = image_size * length;
      data_size = channel_size * 3;
      buffer = new float[data_size];
    }

    for (int c = 0; c < 3; c++) {
      ImageDataToBuffer(
          (unsigned char*)sampledFrames[i]->data_.get(),
          sampledFrames[i]->height_,
          sampledFrames[i]->width_,
          buffer + c * channel_size + offset,
          c);
    }
    offset += image_size;
  }
  CAFFE_ENFORCE(offset == channel_size, "Wrong offset size");

  // free the sampledFrames
  for (int i = 0; i < sampledFrames.size(); i++) {
    DecodedFrame* p = sampledFrames[i].release();
    delete p;
  }
  sampledFrames.clear();

  return true;
}

} // caffe2 namespace
