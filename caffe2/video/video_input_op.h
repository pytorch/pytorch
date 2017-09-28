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

#include <iostream>
#include <random>
#include <string>

#include <opencv2/opencv.hpp>

#include "caffe2/core/db.h"
#include "caffe2/core/logging.h"
#include "caffe2/operators/prefetch_op.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/thread_pool.h"
#include "caffe2/video/video_io.h"

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
  bool GetClipAndLabelFromDBValue(
      const std::string& value,
      float*& buffer,
      int* label_data,
      std::mt19937* randgen);

  void DecodeAndTransform(
      const std::string value,
      float* clip_data,
      int* label_data,
      const int crop_size,
      const bool mirror,
      const float mean,
      const float std,
      std::mt19937* randgen,
      std::bernoulli_distribution* mirror_this_clip);

  const db::DBReader* reader_;
  CPUContext cpu_context_;
  TensorCPU prefetched_clip_;
  TensorCPU prefetched_label_;
  Tensor<Context> prefetched_clip_on_device_;
  Tensor<Context> prefetched_label_on_device_;
  int batch_size_;
  float mean_;
  float std_;
  int crop_;
  int scale_h_;
  int scale_w_;
  int length_;
  int sampling_rate_;
  bool mirror_;
  bool temporal_jitter_;
  bool use_image_;
  bool multiple_label_;
  int num_of_labels_;
  bool use_local_file_;
  bool is_test_;
  std::string im_extension_;

  // thread pool for parse + decode
  int num_decode_threads_;
  std::shared_ptr<TaskThreadPool> thread_pool_;
};

template <class Context>
VideoInputOp<Context>::VideoInputOp(
    const OperatorDef& operator_def,
    Workspace* ws)
    : PrefetchOperator<Context>(operator_def, ws),
      reader_(nullptr),
      batch_size_(
          OperatorBase::template GetSingleArgument<int>("batch_size", 0)),
      mean_(OperatorBase::template GetSingleArgument<float>("mean", 0.)),
      std_(OperatorBase::template GetSingleArgument<float>("std", 1.)),
      crop_(OperatorBase::template GetSingleArgument<int>("crop", -1)),
      scale_h_(OperatorBase::template GetSingleArgument<int>("height", 0)),
      scale_w_(OperatorBase::template GetSingleArgument<int>("width", 0)),
      length_(OperatorBase::template GetSingleArgument<int>("length", 0)),
      sampling_rate_(
          OperatorBase::template GetSingleArgument<int>("sampling_rate", 1)),
      mirror_(OperatorBase::template GetSingleArgument<int>("mirror", 0)),
      temporal_jitter_(
          OperatorBase::template GetSingleArgument<int>("temporal_jitter", 1)),
      use_image_(OperatorBase::template GetSingleArgument<int>("use_image", 0)),
      multiple_label_(
          OperatorBase::template GetSingleArgument<int>("multiple_label", 0)),
      num_of_labels_(
          OperatorBase::template GetSingleArgument<int>("num_of_labels", 0)),
      use_local_file_(
          OperatorBase::template GetSingleArgument<int>("use_local_file", 0)),
      is_test_(OperatorBase::template GetSingleArgument<int>(
          OpSchema::Arg_IsTest,
          0)),
      im_extension_(
          OperatorBase::template GetSingleArgument<string>("im_extension", "")),
      num_decode_threads_(
          OperatorBase::template GetSingleArgument<int>("decode_threads", 4)),

      thread_pool_(new TaskThreadPool(num_decode_threads_)) {
  CAFFE_ENFORCE_GT(batch_size_, 0, "Batch size should be nonnegative.");
  CAFFE_ENFORCE_GE(scale_h_, 0, "Must provide the scale value.");
  CAFFE_ENFORCE_GE(scale_w_, 0, "Must provide the cropping value.");
  CAFFE_ENFORCE_GT(length_, 0, "Must provide the clip length value.");
  CAFFE_ENFORCE_GT(crop_, 0, "Must provide the cropping value.");
  CAFFE_ENFORCE_GE(
      scale_h_,
      crop_,
      "The scaled height must be no smaller than the crop value.");
  CAFFE_ENFORCE_GE(
      scale_w_,
      crop_,
      "The scaled width must be no smaller than the crop value.");
  if (multiple_label_) {
    CAFFE_ENFORCE_GT(
        num_of_labels_,
        0,
        "Number of labels must be set for using multiple label output.");
  }

  // Always need a dbreader, even when using local video files
  CAFFE_ENFORCE_GT(
      operator_def.input_size(), 0, "Need to have a DBReader blob input");

  LOG(INFO) << "Creating a clip input op with the following setting: ";
  LOG(INFO) << "    Using " << num_decode_threads_ << " CPU threads;";
  if (temporal_jitter_) {
    LOG(INFO) << "  Using temporal jittering;";
  }
  LOG(INFO) << "    Outputting in batches of " << batch_size_ << " images;";
  LOG(INFO) << "    Scaling image to " << scale_h_ << "x" << scale_w_;

  LOG(INFO) << "    Cropping video frame to " << crop_
            << (mirror_ ? " with " : " without ") << "random mirroring;";
  LOG(INFO) << "    Using " << (is_test_ ? "center" : "random") << " crop";
  LOG(INFO) << "    Using a clip of " << length_ << " frames;";
  LOG(INFO) << "    Using a sampling rate of 1:" << sampling_rate_;
  LOG(INFO) << "    Subtract mean " << mean_ << " and divide by std " << std_
            << ".";
  vector<TIndex> data_shape(5);
  vector<TIndex> label_shape(2);

  data_shape[0] = batch_size_;
  // Assume color videos, will convert to 3 channels, even with black & with
  // input videos
  data_shape[1] = 3;
  data_shape[2] = length_;
  data_shape[3] = crop_;
  data_shape[4] = crop_;
  prefetched_clip_.Resize(data_shape);

  // If multiple label is used, outout label is a binary vector of length
  // number of labels-dim in indicating which labels present
  if (multiple_label_) {
    label_shape[0] = batch_size_;
    label_shape[1] = num_of_labels_;
    prefetched_label_.Resize(label_shape);
  } else {
    prefetched_label_.Resize(vector<TIndex>(1, batch_size_));
  }
}

template <class Context>
bool VideoInputOp<Context>::GetClipAndLabelFromDBValue(
    const string& value,
    float*& buffer,
    int* label_data,
    std::mt19937* randgen) {
  TensorProtos protos;
  CAFFE_ENFORCE(protos.ParseFromString(value));
  const TensorProto& video_proto = protos.protos(0);
  const TensorProto& label_proto = protos.protos(1);

  int start_frm = -1;
  if (!temporal_jitter_) {
    const TensorProto& start_frm_proto = protos.protos(2);
    start_frm = start_frm_proto.int32_data(0);
  }

  // assign labels
  if (!multiple_label_) {
    label_data[0] = label_proto.int32_data(0);
  } else {
    // For multiple label case, output label is a binary vector
    // where presented concepts are makred 1
    memset(label_data, 0, sizeof(int) * num_of_labels_);
    for (int i = 0; i < label_proto.int32_data_size(); i++) {
      label_data[label_proto.int32_data(i)] = 1;
    }
  }

  if (use_local_file_) {
    CAFFE_ENFORCE_EQ(
        video_proto.data_type(),
        TensorProto::STRING,
        "Database with a file_list is expected to be string data");
  }

  if (video_proto.data_type() == TensorProto::STRING) {
    const string& encoded_video_str = video_proto.string_data(0);
    int encoded_size = encoded_video_str.size();
    if (!use_local_file_) {
      DecodeClipFromMemoryBuffer(
          const_cast<char*>(encoded_video_str.data()),
          encoded_size,
          start_frm,
          length_,
          scale_h_,
          scale_w_,
          sampling_rate_,
          buffer,
          randgen);
    } else {
      // encoded string contains an absolute path to a local file or folder
      std::string filename = encoded_video_str;
      if (use_image_) {
        CAFFE_ENFORCE(
          !temporal_jitter_,
          "Temporal jittering is not suported for image sequence input"
        );
        CHECK(ReadClipFromFrames(
            filename,
            start_frm,
            im_extension_,
            length_,
            scale_h_,
            scale_w_,
            sampling_rate_,
            buffer));
      } else {
        if (temporal_jitter_) {
          int num_of_frames = GetNumberOfFrames(filename);
          start_frm = std::uniform_int_distribution<>(
              0, num_of_frames - length_ * sampling_rate_ + 1)(*randgen);
          CHECK(DecodeClipFromVideoFile(
              filename,
              start_frm,
              length_,
              scale_h_,
              scale_w_,
              sampling_rate_,
              buffer));
        } else {
          CHECK(DecodeClipFromVideoFile(
              filename,
              start_frm,
              length_,
              scale_h_,
              scale_w_,
              sampling_rate_,
              buffer));
        }
      }
    }
  } else if (video_proto.data_type() == TensorProto::BYTE) {
    DecodeClipFromMemoryBuffer(
        video_proto.byte_data().data(),
        video_proto.byte_data().size(),
        start_frm,
        length_,
        scale_h_,
        scale_w_,
        sampling_rate_,
        buffer,
        randgen);
  } else {
    LOG(FATAL) << "Unknown video data type.";
  }
  return true;
}

template <class Context>
void VideoInputOp<Context>::DecodeAndTransform(
    const std::string value,
    float* clip_data,
    int* label_data,
    const int crop_size,
    const bool mirror,
    const float mean,
    const float std,
    std::mt19937* randgen,
    std::bernoulli_distribution* mirror_this_clip) {
  float* buffer = nullptr;

  // Decode the video from memory or read from a local file
  CHECK(GetClipAndLabelFromDBValue(value, buffer, label_data, randgen));

  if (buffer) {
    ClipTransform(
        buffer,
        3,
        length_,
        scale_h_,
        scale_w_,
        crop_size,
        mirror,
        mean,
        std,
        clip_data,
        randgen,
        mirror_this_clip,
        is_test_);

    delete[] buffer;
  }
}

template <class Context>
bool VideoInputOp<Context>::Prefetch() {
  // We will get the reader pointer from input.
  // If we use local clips, db will store the list
  reader_ = &OperatorBase::Input<db::DBReader>(0);

  const int channels = 3;

  // Call mutable_data() once to allocate the underlying memory.
  prefetched_clip_.mutable_data<float>();
  prefetched_label_.mutable_data<int>();

  // Prefetching handled with a thread pool of "decode_threads" threads.
  std::mt19937 meta_randgen(time(nullptr));
  std::vector<std::mt19937> randgen_per_thread;
  for (int i = 0; i < num_decode_threads_; ++i) {
    randgen_per_thread.emplace_back(meta_randgen());
  }

  std::bernoulli_distribution mirror_this_clip(0.5);
  for (int item_id = 0; item_id < batch_size_; ++item_id) {
    std::mt19937* randgen = &randgen_per_thread[item_id % num_decode_threads_];

    // get the label data pointer for the item_id -th example
    int* label_data = prefetched_label_.mutable_data<int>() +
        (multiple_label_ ? num_of_labels_ : 1) * item_id;

    // get the clip data pointer for the item_id -th example
    float* clip_data = prefetched_clip_.mutable_data<float>() +
        crop_ * crop_ * length_ * channels * item_id;

    std::string key, value;
    // read data
    reader_->Read(&key, &value);

    thread_pool_->runTask(std::bind(
        &VideoInputOp<Context>::DecodeAndTransform,
        this,
        std::string(value),
        clip_data,
        label_data,
        crop_,
        mirror_,
        mean_,
        std_,
        randgen,
        &mirror_this_clip));
  } // for over the batch
  thread_pool_->waitWorkComplete();

  // If the context is not CPUContext, we will need to do a copy in the
  // prefetch function as well.
  if (!std::is_same<Context, CPUContext>::value) {
    prefetched_clip_on_device_.CopyFrom(prefetched_clip_, &context_);
    prefetched_label_on_device_.CopyFrom(prefetched_label_, &context_);
  }
  return true;
}

template <class Context>
bool VideoInputOp<Context>::CopyPrefetched() {
  auto* clip_output = OperatorBase::Output<Tensor<Context>>(0);
  auto* label_output = OperatorBase::Output<Tensor<Context>>(1);
  if (std::is_same<Context, CPUContext>::value) {
    clip_output->CopyFrom(prefetched_clip_, &context_);
    label_output->CopyFrom(prefetched_label_, &context_);
  } else {
    clip_output->CopyFrom(prefetched_clip_on_device_, &context_);
    label_output->CopyFrom(prefetched_label_on_device_, &context_);
  }
  return true;
}

} // namespace caffe2

#endif // CAFFE2_VIDEO_VIDEO_INPUT_OP_H_
