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

#ifndef CAFFE2_VIDEO_VIDEO_INPUT_OP_H_
#define CAFFE2_VIDEO_VIDEO_INPUT_OP_H_

#include <istream>
#include <ostream>
#include <random>
#include <string>

#include <caffe2/core/db.h>
#include <caffe2/core/logging.h>
#include <caffe2/operators/prefetch_op.h>
#include <caffe2/utils/math.h>
#include <caffe2/utils/thread_pool.h>
#include <caffe2/video/video_io.h>

namespace caffe2 {

template <class Context>
class VideoInputOp final : public PrefetchOperator<Context> {
 public:
  using OperatorBase::OutputSize;
  using PrefetchOperator<Context>::context_;
  using PrefetchOperator<Context>::prefetch_thread_;
  explicit VideoInputOp(const OperatorDef& operator_def, Workspace* ws);
  ~VideoInputOp() {
    PrefetchOperator<Context>::Finalize();
  }

  // override methods
  bool Prefetch() override;
  bool CopyPrefetched() override;

 private:
  void CheckParamsAndPrint();

  bool GetClipsAndLabelsFromDBValue(
      const std::string& value,
      int& height,
      int& width,
      std::vector<unsigned char*>& buffer_rgb,
      int* label_data,
      int* video_id_data);

  void DecodeAndTransform(
      const std::string& value,
      float* clip_rgb_data,
      float* clip_of_data,
      int* label_data,
      int* video_id_data,
      std::mt19937* randgen,
      std::bernoulli_distribution* mirror_this_clip);

  const db::DBReader* reader_;
  CPUContext cpu_context_;
  TensorCPU prefetched_clip_rgb_;
  TensorCPU prefetched_clip_of_;
  TensorCPU prefetched_label_;
  TensorCPU prefetched_video_id_;
  Tensor<Context> prefetched_clip_rgb_on_device_;
  Tensor<Context> prefetched_clip_of_on_device_;
  Tensor<Context> prefetched_label_on_device_;
  Tensor<Context> prefetched_video_id_on_device_;
  int batch_size_;
  int clip_per_video_;
  std::vector<float> mean_rgb_;
  std::vector<float> inv_std_rgb_;
  std::vector<float> mean_of_;
  std::vector<float> inv_std_of_;
  int channels_rgb_;
  int channels_of_;
  int crop_size_;
  int scale_h_;
  int scale_w_;
  int short_edge_;
  int length_rgb_;
  int sampling_rate_rgb_;
  int num_of_required_frame_;
  int length_of_;
  int sampling_rate_of_;
  int frame_gap_of_;
  bool random_mirror_;
  int num_of_class_;
  bool use_local_file_;
  bool random_crop_;
  int flow_data_type_;
  int flow_alg_type_;
  int decode_type_;
  int video_res_type_;
  bool do_flow_aggregation_;
  bool get_rgb_;
  bool get_optical_flow_;
  bool get_video_id_;
  bool do_multi_label_;

  // thread pool for parse + decode
  int num_decode_threads_;
  std::shared_ptr<TaskThreadPool> thread_pool_;
};

template <class Context>
void VideoInputOp<Context>::CheckParamsAndPrint() {
  // check whether the input parameters are valid or not
  CAFFE_ENFORCE_GT(batch_size_, 0, "Batch size should be positive.");
  CAFFE_ENFORCE_GT(
      clip_per_video_, 0, "Number of clips per video should be positive.");
  CAFFE_ENFORCE_GT(crop_size_, 0, "Must provide the cropping value.");
  CAFFE_ENFORCE_GT(
      num_of_required_frame_, 0, "Required number of frames must be positive.");

  if (video_res_type_ == VideoResType::USE_SHORT_EDGE) {
    CAFFE_ENFORCE_GT(short_edge_, 0, "Must provide the short edge value.");
    CAFFE_ENFORCE_GE(
        short_edge_,
        crop_size_,
        "The short edge must be no smaller than the crop value.");
  } else if (video_res_type_ == VideoResType::USE_WIDTH_HEIGHT) {
    CAFFE_ENFORCE_GT(scale_h_, 0, "Must provide the scale height value.");
    CAFFE_ENFORCE_GT(scale_w_, 0, "Must provide the scale width value.");
    CAFFE_ENFORCE_GE(
        scale_h_,
        crop_size_,
        "The scaled height must be no smaller than the crop value.");
    CAFFE_ENFORCE_GE(
        scale_w_,
        crop_size_,
        "The scaled width must be no smaller than the crop value.");
  }

  if (get_rgb_) {
    CAFFE_ENFORCE_GT(length_rgb_, 0, "Must provide rgb clip length.");
    CAFFE_ENFORCE_GT(
        sampling_rate_rgb_, 0, "4 frames for mc2; 2 frames for res3d.");
    CAFFE_ENFORCE_EQ(
        channels_rgb_, mean_rgb_.size(), "Number rgb channels is wrong!");
    CAFFE_ENFORCE_EQ(
        channels_rgb_, inv_std_rgb_.size(), "Number rgb channels is wrong!");
  }

  if (get_optical_flow_) {
    CAFFE_ENFORCE_GT(length_of_, 0, "Must provide optical flow clip length.");
    CAFFE_ENFORCE_GT(
        sampling_rate_of_, 0, "4 frames for mc2; 2 frames for res3d.");
    CAFFE_ENFORCE_EQ(
        channels_of_,
        mean_of_.size(),
        "Number of optical flow channels is wrong!");
    CAFFE_ENFORCE_EQ(
        channels_of_,
        inv_std_of_.size(),
        "Number of optical flow channels is wrong!");
  }

  if (clip_per_video_ > 1) {
    CAFFE_ENFORCE_EQ(
        decode_type_,
        DecodeType::DO_UNIFORM_SMP,
        "Only uniformly sampling is supported when sampling multiple clips!");
  }

  if (do_multi_label_) {
    CAFFE_ENFORCE_GT(
        num_of_class_,
        0,
        "Number of classes must be set when using multiple labels.");
  }

  // print out the parameter settings
  LOG(INFO) << "Creating a clip input op with the following setting: ";
  LOG(INFO) << "    Using " << num_decode_threads_ << " CPU threads;";
  LOG(INFO) << "    Outputting in batches of " << batch_size_ << " videos;";
  LOG(INFO) << "    Each video has " << clip_per_video_ << " clips;";
  LOG(INFO) << "    Scaling image to " << scale_h_ << "x" << scale_w_;
  LOG(INFO) << "    Cropping video frame to " << crop_size_
            << (random_mirror_ ? " with " : " without ") << "random mirroring;";
  LOG(INFO) << "    Using " << (random_crop_ ? "random" : "center") << " crop";

  if (get_rgb_) {
    LOG(INFO) << "    Using a clip of " << length_rgb_ << " rgb frames "
              << "with " << channels_rgb_ << " channels "
              << "and a sampling rate of 1:" << sampling_rate_rgb_;
    for (int i = 0; i < channels_rgb_; i++) {
      LOG(INFO) << "    RGB " << i << "-th channel mean: " << mean_rgb_[i]
                << " std: " << 1.f / inv_std_rgb_[i];
    }
  }

  if (get_optical_flow_) {
    LOG(INFO) << "    Using a clip of " << length_of_ << " optical flow frames "
              << "with " << channels_of_ << " channels "
              << "and a sampling rate of 1:" << sampling_rate_of_
              << " flow_data_type_: " << flow_data_type_
              << " flow_alg_type_: " << flow_alg_type_;
    for (int i = 0; i < channels_of_; i++) {
      LOG(INFO) << "    Optical flow" << i
                << "-th channel mean: " << mean_of_[i]
                << " std: " << 1.f / inv_std_of_[i];
    }
  }

  if (video_res_type_ == VideoResType::ORIGINAL_RES) {
    LOG(INFO) << "    Use original resolution";
  } else if (video_res_type_ == VideoResType::USE_SHORT_EDGE) {
    LOG(INFO) << "    Resize and keep aspect ratio";
  } else if (video_res_type_ == VideoResType::USE_WIDTH_HEIGHT) {
    LOG(INFO) << "    Resize and ignore aspect ratio";
  } else {
    LOG(ERROR) << "    Unknown video resolution type";
  }

  if (decode_type_ == DecodeType::DO_TMP_JITTER) {
    LOG(INFO) << "    Do temporal jittering";
  } else if (decode_type_ == DecodeType::USE_START_FRM) {
    LOG(INFO) << "    Use start_frm for decoding";
  } else if (decode_type_ == DecodeType::DO_UNIFORM_SMP) {
    LOG(INFO) << "    Do uniformly sampling";
  } else {
    LOG(ERROR) << "    Unknown video decoding type";
  }
}

template <class Context>
VideoInputOp<Context>::VideoInputOp(
    const OperatorDef& operator_def,
    Workspace* ws)
    : PrefetchOperator<Context>(operator_def, ws),
      reader_(nullptr),
      batch_size_(
          OperatorBase::template GetSingleArgument<int>("batch_size", 0)),
      clip_per_video_(
          OperatorBase::template GetSingleArgument<int>("clip_per_video", 1)),
      channels_rgb_(
          OperatorBase::template GetSingleArgument<int>("channels_rgb", 3)),
      channels_of_(
          OperatorBase::template GetSingleArgument<int>("channels_of", 2)),
      crop_size_(OperatorBase::template GetSingleArgument<int>("crop_size", 0)),
      scale_h_(OperatorBase::template GetSingleArgument<int>("scale_h", 0)),
      scale_w_(OperatorBase::template GetSingleArgument<int>("scale_w", 0)),
      short_edge_(
          OperatorBase::template GetSingleArgument<int>("short_edge", 0)),
      length_rgb_(
          OperatorBase::template GetSingleArgument<int>("length_rgb", 0)),
      sampling_rate_rgb_(OperatorBase::template GetSingleArgument<int>(
          "sampling_rate_rgb",
          1)),
      length_of_(OperatorBase::template GetSingleArgument<int>("length_of", 0)),
      sampling_rate_of_(
          OperatorBase::template GetSingleArgument<int>("sampling_rate_of", 1)),
      frame_gap_of_(
          OperatorBase::template GetSingleArgument<int>("frame_gap_of", 1)),
      random_mirror_(OperatorBase::template GetSingleArgument<bool>(
          "random_mirror",
          true)),
      num_of_class_(
          OperatorBase::template GetSingleArgument<int>("num_of_class", 0)),
      use_local_file_(OperatorBase::template GetSingleArgument<bool>(
          "use_local_file",
          false)),
      random_crop_(
          OperatorBase::template GetSingleArgument<bool>("random_crop", true)),
      flow_data_type_(
          OperatorBase::template GetSingleArgument<int>("flow_data_type", 0)),
      flow_alg_type_(
          OperatorBase::template GetSingleArgument<int>("flow_alg_type", 0)),
      decode_type_(
          OperatorBase::template GetSingleArgument<int>("decode_type", 0)),
      video_res_type_(
          OperatorBase::template GetSingleArgument<int>("video_res_type", 0)),
      do_flow_aggregation_(OperatorBase::template GetSingleArgument<bool>(
          "do_flow_aggregation",
          true)),
      get_rgb_(OperatorBase::template GetSingleArgument<bool>("get_rgb", true)),
      get_optical_flow_(OperatorBase::template GetSingleArgument<bool>(
          "get_optical_flow",
          false)),
      get_video_id_(OperatorBase::template GetSingleArgument<bool>(
          "get_video_id",
          false)),
      do_multi_label_(OperatorBase::template GetSingleArgument<bool>(
          "do_multi_label",
          false)),
      num_decode_threads_(OperatorBase::template GetSingleArgument<int>(
          "num_decode_threads",
          4)),

      thread_pool_(std::make_shared<TaskThreadPool>(num_decode_threads_)) {
  num_of_required_frame_ = 0;
  // if we need RGB as an input
  if (get_rgb_) {
    // how many frames we need for RGB
    num_of_required_frame_ = std::max(
        num_of_required_frame_, (length_rgb_ - 1) * sampling_rate_rgb_ + 1);

    channels_rgb_ = 3;
    for (int i = 4; i < 7; i++) {
      mean_rgb_.push_back(InputDataMean[i]);
      inv_std_rgb_.push_back(1.f / InputDataStd[i]);
    }
  }

  // if we need optical flow as an input
  if (get_optical_flow_) {
    // how many frames we need for optical flow
    num_of_required_frame_ = std::max(
        num_of_required_frame_,
        (length_of_ - 1) * sampling_rate_of_ + frame_gap_of_ + 1);

    // set the parameters for different input data types
    switch (flow_data_type_) {
      case FlowDataType::Flow2C:
        channels_of_ = 2;
        for (int i = 0; i < channels_of_; i++) {
          mean_of_.push_back(InputDataMean[i]);
          inv_std_of_.push_back(1.f / InputDataStd[i]);
        }
        break;

      case FlowDataType::Flow3C:
        channels_of_ = 3;
        for (int i = 0; i < channels_of_; i++) {
          mean_of_.push_back(InputDataMean[i]);
          inv_std_of_.push_back(1.f / InputDataStd[i]);
        }
        break;

      // early fusion with gray
      case FlowDataType::FlowWithGray:
        channels_of_ = 3;
        for (int i = 0; i < 2; i++) {
          mean_of_.push_back(InputDataMean[i]);
          inv_std_of_.push_back(1.f / InputDataStd[i]);
        }
        mean_of_.push_back(InputDataMean[3]);
        inv_std_of_.push_back(1.f / InputDataStd[3]);
        break;

      // early fusion with RGB
      case FlowDataType::FlowWithRGB:
        channels_of_ = 5;
        for (int i = 0; i < 2; i++) {
          mean_of_.push_back(InputDataMean[i]);
          inv_std_of_.push_back(1.f / InputDataStd[i]);
        }
        for (int i = 4; i < 7; i++) {
          mean_of_.push_back(InputDataMean[i]);
          inv_std_of_.push_back(1.f / InputDataStd[i]);
        }
        break;

      default:
        LOG(ERROR) << "Unknown optical flow type " << flow_data_type_;
        break;
    }
  }

  CheckParamsAndPrint();
  // Always need a dbreader, even when using local video files
  CAFFE_ENFORCE_GT(
      operator_def.input_size(), 0, "Need to have a DBReader blob input");

  vector<TIndex> data_shape(5);
  vector<TIndex> label_shape(2);

  // for RGB data
  data_shape[0] = batch_size_ * clip_per_video_;
  data_shape[1] = channels_rgb_;
  data_shape[2] = length_rgb_;
  data_shape[3] = crop_size_;
  data_shape[4] = crop_size_;
  prefetched_clip_rgb_.Resize(data_shape);

  // for optical flow data
  data_shape[1] = channels_of_;
  data_shape[2] = length_of_;
  prefetched_clip_of_.Resize(data_shape);

  // If do_multi_label is used, output label is a binary vector
  // of length num_of_class indicating which labels present
  if (do_multi_label_) {
    label_shape[0] = batch_size_ * clip_per_video_;
    label_shape[1] = num_of_class_;
    prefetched_label_.Resize(label_shape);
  } else {
    prefetched_label_.Resize(vector<TIndex>(1, batch_size_ * clip_per_video_));
  }

  prefetched_video_id_.Resize(vector<TIndex>(1, batch_size_ * clip_per_video_));
}

template <class Context>
bool VideoInputOp<Context>::GetClipsAndLabelsFromDBValue(
    const std::string& value,
    int& height,
    int& width,
    std::vector<unsigned char*>& buffer_rgb,
    int* label_data,
    int* video_id_data) {
  TensorProtos protos;
  int curr_proto_idx = 0;
  CAFFE_ENFORCE(protos.ParseFromString(value));
  const TensorProto& video_proto = protos.protos(curr_proto_idx++);
  const TensorProto& label_proto = protos.protos(curr_proto_idx++);

  int start_frm = 0;
  // start_frm is only valid when sampling 1 clip per video without
  // temporal jitterring
  if (decode_type_ == DecodeType::USE_START_FRM) {
    const TensorProto& start_frm_proto = protos.protos(curr_proto_idx++);
    start_frm = start_frm_proto.int32_data(0);
  }

  if (get_video_id_) {
    const TensorProto& video_id_proto = protos.protos(curr_proto_idx);
    for (int i = 0; i < clip_per_video_; i++) {
      video_id_data[i] = video_id_proto.int32_data(0);
    }
  }

  // assign labels
  if (!do_multi_label_) {
    for (int i = 0; i < clip_per_video_; i++) {
      label_data[i] = label_proto.int32_data(0);
    }
  } else {
    // For multiple label case, output label is a binary vector
    // where presented concepts are makred 1
    memset(label_data, 0, sizeof(int) * num_of_class_ * clip_per_video_);
    for (int i = 0; i < clip_per_video_; i++) {
      for (int j = 0; j < label_proto.int32_data_size(); j++) {
        label_data[i * num_of_class_ + label_proto.int32_data(j)] = 1;
      }
    }
  }

  if (use_local_file_) {
    CAFFE_ENFORCE_EQ(
        video_proto.data_type(),
        TensorProto::STRING,
        "Database with a file_list is expected to be string data");
  }

  // initializing the decoding params
  Params params;
  params.maximumOutputFrames_ = MAX_DECODING_FRAMES;
  params.video_res_type_ = video_res_type_;
  params.crop_size_ = crop_size_;
  params.short_edge_ = short_edge_;
  params.scale_w_ = scale_w_;
  params.scale_h_ = scale_h_;
  params.decode_type_ = decode_type_;
  params.num_of_required_frame_ = num_of_required_frame_;

  char* video_buffer = nullptr; // for decoding from buffer
  std::string video_filename; // for decoding from file
  int encoded_size = 0;
  if (video_proto.data_type() == TensorProto::STRING) {
    const string& encoded_video_str = video_proto.string_data(0);
    if (!use_local_file_) {
      encoded_size = encoded_video_str.size();
      video_buffer = const_cast<char*>(encoded_video_str.data());
    } else {
      video_filename = encoded_video_str;
    }
  } else if (video_proto.data_type() == TensorProto::BYTE) {
    if (!use_local_file_) {
      encoded_size = video_proto.byte_data().size();
      video_buffer = const_cast<char*>(video_proto.byte_data().data());
    } else {
      // TODO: does this works?
      video_filename = video_proto.string_data(0);
    }
  } else {
    LOG(FATAL) << "Unknown video data type.";
  }

  DecodeMultipleClipsFromVideo(
      video_buffer,
      video_filename,
      encoded_size,
      params,
      start_frm,
      clip_per_video_,
      use_local_file_,
      height,
      width,
      buffer_rgb);

  return true;
}

template <class Context>
void VideoInputOp<Context>::DecodeAndTransform(
    const std::string& value,
    float* clip_rgb_data,
    float* clip_of_data,
    int* label_data,
    int* video_id_data,
    std::mt19937* randgen,
    std::bernoulli_distribution* mirror_this_clip) {
  std::vector<unsigned char*> buffer_rgb;
  // get the video resolution after decoding
  int height = 0;
  int width = 0;
  // Decode the video from memory or read from a local file
  CHECK(GetClipsAndLabelsFromDBValue(
      value, height, width, buffer_rgb, label_data, video_id_data));

  int clip_offset_rgb = channels_rgb_ * length_rgb_ * crop_size_ * crop_size_;
  int clip_offset_of = channels_of_ * length_of_ * crop_size_ * crop_size_;
  for (int i = 0; i < std::min(clip_per_video_, int(buffer_rgb.size())); i++) {
    // get the rectangle for cropping
    int h_off = 0;
    int w_off = 0;
    if (random_crop_) {
      // using random crop for training
      h_off = std::uniform_int_distribution<>(0, height - crop_size_)(*randgen);
      w_off = std::uniform_int_distribution<>(0, width - crop_size_)(*randgen);
    } else {
      // using center crop for testing
      h_off = (height - crop_size_) / 2;
      w_off = (width - crop_size_) / 2;
    }
    cv::Rect rect(w_off, h_off, crop_size_, crop_size_);

    // randomly mirror the image or not
    bool mirror_me = random_mirror_ && (*mirror_this_clip)(*randgen);

    if (get_rgb_ && clip_rgb_data) {
      ClipTransformRGB(
          buffer_rgb[i],
          crop_size_,
          length_rgb_,
          channels_rgb_,
          sampling_rate_rgb_,
          height,
          width,
          h_off,
          w_off,
          mirror_me,
          mean_rgb_,
          inv_std_rgb_,
          clip_rgb_data + (i * clip_offset_rgb));
    }

    if (get_optical_flow_ && clip_of_data) {
      ClipTransformOpticalFlow(
          buffer_rgb[i],
          crop_size_,
          length_of_,
          channels_of_,
          sampling_rate_of_,
          height,
          width,
          rect,
          channels_rgb_,
          mirror_me,
          flow_alg_type_,
          flow_data_type_,
          frame_gap_of_,
          do_flow_aggregation_,
          mean_of_,
          inv_std_of_,
          clip_of_data + (i * clip_offset_of));
    }
  }

  if (buffer_rgb.size() > 0) {
    for (int i = 0; i < buffer_rgb.size(); i++) {
      unsigned char* buff = buffer_rgb[i];
      delete[] buff;
    }
  }
  buffer_rgb.clear();
}

template <class Context>
bool VideoInputOp<Context>::Prefetch() {
  // We will get the reader pointer from input.
  // If we use local clips, db will store the list
  reader_ = &OperatorBase::Input<db::DBReader>(0);

  // Call mutable_data() once to allocate the underlying memory.
  prefetched_clip_rgb_.mutable_data<float>();
  prefetched_clip_of_.mutable_data<float>();
  prefetched_label_.mutable_data<int>();
  prefetched_video_id_.mutable_data<int>();

  // Prefetching handled with a thread pool of "decode_threads" threads.
  std::mt19937 meta_randgen(time(nullptr));
  std::vector<std::mt19937> randgen_per_thread;
  for (int i = 0; i < num_decode_threads_; ++i) {
    randgen_per_thread.emplace_back(meta_randgen());
  }

  std::bernoulli_distribution mirror_this_clip(0.5);
  for (int item_id = 0; item_id < batch_size_; ++item_id) {
    std::mt19937* randgen = &randgen_per_thread[item_id % num_decode_threads_];

    int frame_size = crop_size_ * crop_size_;
    // get the clip data pointer for the item_id -th example
    float* clip_rgb_data = prefetched_clip_rgb_.mutable_data<float>() +
        frame_size * length_rgb_ * channels_rgb_ * item_id * clip_per_video_;

    // get the optical flow data for the current clip
    float* clip_of_data = prefetched_clip_of_.mutable_data<float>() +
        frame_size * length_of_ * channels_of_ * item_id * clip_per_video_;

    // get the label data pointer for the item_id -th example
    int* label_data = prefetched_label_.mutable_data<int>() +
        (do_multi_label_ ? num_of_class_ : 1) * item_id * clip_per_video_;

    // get the video id data pointer for the item_id -th example
    int* video_id_data =
        prefetched_video_id_.mutable_data<int>() + item_id * clip_per_video_;

    std::string key, value;
    // read data
    reader_->Read(&key, &value);

    thread_pool_->runTask(std::bind(
        &VideoInputOp<Context>::DecodeAndTransform,
        this,
        std::string(value),
        clip_rgb_data,
        clip_of_data,
        label_data,
        video_id_data,
        randgen,
        &mirror_this_clip));
  } // for over the batch
  thread_pool_->waitWorkComplete();

  // If the context is not CPUContext, we will need to do a copy in the
  // prefetch function as well.
  if (!std::is_same<Context, CPUContext>::value) {
    if (get_rgb_) {
      prefetched_clip_rgb_on_device_.CopyFrom(prefetched_clip_rgb_, &context_);
    }
    if (get_optical_flow_) {
      prefetched_clip_of_on_device_.CopyFrom(prefetched_clip_of_, &context_);
    }
    prefetched_label_on_device_.CopyFrom(prefetched_label_, &context_);
    if (get_video_id_) {
      prefetched_video_id_on_device_.CopyFrom(prefetched_video_id_, &context_);
    }
  }
  return true;
}

template <class Context>
bool VideoInputOp<Context>::CopyPrefetched() {
  int index = 0;
  if (get_rgb_) {
    auto* clip_rgb_output = OperatorBase::Output<Tensor<Context>>(index++);
    if (std::is_same<Context, CPUContext>::value) {
      clip_rgb_output->CopyFrom(prefetched_clip_rgb_, &context_);
    } else {
      clip_rgb_output->CopyFrom(prefetched_clip_rgb_on_device_, &context_);
    }
  }

  if (get_optical_flow_) {
    auto* clip_of_output = OperatorBase::Output<Tensor<Context>>(index++);
    if (std::is_same<Context, CPUContext>::value) {
      clip_of_output->CopyFrom(prefetched_clip_of_, &context_);
    } else {
      clip_of_output->CopyFrom(prefetched_clip_of_on_device_, &context_);
    }
  }

  auto* label_output = OperatorBase::Output<Tensor<Context>>(index++);
  if (std::is_same<Context, CPUContext>::value) {
    label_output->CopyFrom(prefetched_label_, &context_);
  } else {
    label_output->CopyFrom(prefetched_label_on_device_, &context_);
  }

  if (get_video_id_) {
    auto* video_id_output = OperatorBase::Output<Tensor<Context>>(index);
    if (std::is_same<Context, CPUContext>::value) {
      video_id_output->CopyFrom(prefetched_video_id_, &context_);
    } else {
      video_id_output->CopyFrom(prefetched_video_id_on_device_, &context_);
    }
  }

  return true;
}

} // namespace caffe2

#endif // CAFFE2_VIDEO_VIDEO_INPUT_OP_H_
