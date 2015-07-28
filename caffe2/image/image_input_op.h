#ifndef CAFFE2_IMAGE_IMAGE_INPUT_OP_H_
#define CAFFE2_IMAGE_IMAGE_INPUT_OP_H_

#include <opencv2/opencv.hpp>

#include <iostream>

#include "caffe2/core/db.h"
#include "caffe2/operators/prefetch_op.h"

namespace caffe2 {

template <class DeviceContext>
class ImageInputOp final
    : public PrefetchOperator<DeviceContext> {
 public:
  using OperatorBase::OutputSize;
  using PrefetchOperator<DeviceContext>::prefetch_thread_;
  explicit ImageInputOp(const OperatorDef& operator_def,
                                    Workspace* ws);
  ~ImageInputOp() {
    if (prefetch_thread_.get() != nullptr) {
      prefetch_thread_->join();
    }
  }

  bool Prefetch() override;
  bool CopyPrefetched() override;

 private:
  unique_ptr<db::DB> db_;
  unique_ptr<db::Cursor> cursor_;
  CPUContext cpu_context_;
  Tensor<float, CPUContext> prefetched_image_;
  Tensor<int, CPUContext> prefetched_label_;
  int batch_size_;
  string db_name_;
  string db_type_;
  float mean_;
  float std_;
  bool color_;
  int scale_;
  bool warp_;
  int crop_;
  bool mirror_;
  INPUT_OUTPUT_STATS(0, 0, 2, 2);
  DISABLE_COPY_AND_ASSIGN(ImageInputOp);
};

template <class DeviceContext>
ImageInputOp<DeviceContext>::ImageInputOp(
      const OperatorDef& operator_def, Workspace* ws)
      : PrefetchOperator<DeviceContext>(operator_def, ws),
        batch_size_(
            OperatorBase::template GetSingleArgument<int>("batch_size", 0)),
        db_name_(
            OperatorBase::template GetSingleArgument<string>("db", "")),
        db_type_(OperatorBase::template GetSingleArgument<string>(
            "db_type", "leveldb")),
        mean_(OperatorBase::template GetSingleArgument<float>("mean", 0.)),
        std_(OperatorBase::template GetSingleArgument<float>("std", 1.)),
        color_(OperatorBase::template GetSingleArgument<int>("color", 1)),
        scale_(OperatorBase::template GetSingleArgument<int>("scale", -1)),
        warp_(OperatorBase::template GetSingleArgument<int>("warp", 0)),
        crop_(OperatorBase::template GetSingleArgument<int>("crop", -1)),
        mirror_(OperatorBase::template GetSingleArgument<int>("mirror", 0)) {
  CHECK_GT(batch_size_, 0) << "Batch size should be nonnegative.";
  CHECK_GT(db_name_.size(), 0) << "Must provide a leveldb name.";
  CHECK_GT(scale_, 0) << "Must provide the scaling factor.";
  CHECK_GT(crop_, 0) << "Must provide the cropping value.";
  CHECK_GE(scale_, crop_)
      << "The scale value must be no smaller than the crop value.";

  DLOG(INFO) << "Creating an image input op with the following setting: ";
  DLOG(INFO) << "    Outputting in batches of " << batch_size_ << " images;";
  DLOG(INFO) << "    Treating input image as "
             << (color_ ? "color " : "grayscale ") << "image;";
  DLOG(INFO) << "    Scaling image to " << scale_
             << (warp_ ? " with " : " without ") << "warping;";
  DLOG(INFO) << "    Cropping image to " << crop_
             << (mirror_ ? " with " : " without ") << "random mirroring;";
  DLOG(INFO) << "    Subtract mean " << mean_ << " and divide by std " << std_
             << ".";
  db_.reset(db::CreateDB(db_type_, db_name_, db::READ));
  cursor_.reset(db_->NewCursor());
  cursor_->SeekToFirst();
  prefetched_image_.Reshape(
      vector<int>{batch_size_, crop_, crop_, (color_ ? 3 : 1)});
  prefetched_label_.Reshape(vector<int>(1, batch_size_));
}

template <class DeviceContext>
bool ImageInputOp<DeviceContext>::Prefetch() {
  std::bernoulli_distribution mirror_this_image(0.5);
  float* image_data = prefetched_image_.mutable_data();
  int channels = color_ ? 3 : 1;
  for (int item_id = 0; item_id < batch_size_; ++item_id) {
    // LOG(INFO) << "Prefetching item " << item_id;
    // process data
    TensorProtos protos;
    CHECK(protos.ParseFromString(cursor_->value())) << cursor_->value();
    const TensorProto& image = protos.protos(0);
    const TensorProto& label = protos.protos(1);
    cv::Mat final_img;
    if (image.data_type() == TensorProto::STRING) {
      // Do the image manipuiation, and copy the content.
      DCHECK_EQ(image.string_data_size(), 1);

      const string& encoded_image = image.string_data(0);
      int encoded_size = encoded_image.size();
      cv::Mat img = cv::imdecode(
          cv::Mat(1, &encoded_size, CV_8UC1,
          const_cast<char*>(encoded_image.data())),
          color_ ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
      // Do resizing.
      int scaled_width, scaled_height;
      if (warp_) {
        scaled_width = scale_;
        scaled_height = scale_;
      } else if (img.rows > img.cols) {
        scaled_width = scale_;
        scaled_height = static_cast<float>(img.rows) * scale_ / img.cols;
      } else {
        scaled_height = scale_;
        scaled_width = static_cast<float>(img.cols) * scale_ / img.rows;
      }
      cv::resize(img, final_img, cv::Size(scaled_width, scaled_height), 0, 0,
                   cv::INTER_LINEAR);
    } else if (image.data_type() == TensorProto::BYTE) {
      // In this case, we will always just take the bytes as the raw image.
      CHECK_EQ(image.dims_size(), (color_ ? 3 : 2));
      CHECK_GE(image.dims(0), crop_)
          << "Image height must be bigger than crop.";
      CHECK_GE(image.dims(1), crop_) << "Image width must be bigger than crop.";
      CHECK(!color_ || image.dims(2) == 3);
      final_img = cv::Mat(
          image.dims(0), image.dims(1), color_ ? CV_8UC3 : CV_8UC1,
          const_cast<char*>(image.byte_data().data()));
    }
    // find the cropped region, and copy it to the destination matrix with
    // mean subtraction and scaling.
    int width_offset =
        std::uniform_int_distribution<>(0, final_img.cols - crop_)(
            cpu_context_.RandGenerator());
    int height_offset =
        std::uniform_int_distribution<>(0, final_img.rows - crop_)(
            cpu_context_.RandGenerator());
    // DVLOG(1) << "offset: " << height_offset << ", " << width_offset;
    if (mirror_ && mirror_this_image(cpu_context_.RandGenerator())) {
      // Copy mirrored image.
      for (int h = height_offset; h < height_offset + crop_; ++h) {
        for (int w = width_offset + crop_ - 1; w >= width_offset; --w) {
          const cv::Vec3b& cv_data = final_img.at<cv::Vec3b>(h, w);
          for (int c = 0; c < channels; ++c) {
            *(image_data++) =
                (static_cast<uint8_t>(cv_data[c]) - mean_) / std_;
          }
        }
      }
    } else {
      // Copy normally.
      for (int h = height_offset; h < height_offset + crop_; ++h) {
        for (int w = width_offset; w < width_offset + crop_; ++w) {
          const cv::Vec3b& cv_data = final_img.at<cv::Vec3b>(h, w);
          for (int c = 0; c < channels; ++c) {
            *(image_data++) =
                (static_cast<uint8_t>(cv_data[c]) - mean_) / std_;
          }
        }
      }
    }
    // Copy the label
    DCHECK_EQ(label.data_type(), TensorProto::INT32);
    DCHECK_EQ(label.int32_data_size(), 1);
    prefetched_label_.mutable_data()[item_id] = label.int32_data(0);
    // Advance to the next item.
    cursor_->Next();
    if (!cursor_->Valid()) {
      cursor_->SeekToFirst();
    }
  }
  return true;
}

template <class DeviceContext>
bool ImageInputOp<DeviceContext>::CopyPrefetched() {
  // The first output is the image data.
  auto* image_output = OperatorBase::Output<Tensor<float, DeviceContext> >(0);
  image_output->ReshapeLike(prefetched_image_);
  this->device_context_.template Copy<float, CPUContext, DeviceContext>(
      prefetched_image_.size(), prefetched_image_.data(),
      image_output->mutable_data());
  // The second output is the label.
  auto* label_output = OperatorBase::Output<Tensor<int, DeviceContext> >(1);
  label_output->ReshapeLike(prefetched_label_);
  this->device_context_.template Copy<int, CPUContext, DeviceContext>(
      prefetched_label_.size(), prefetched_label_.data(),
      label_output->mutable_data());
  return true;
}

}  // namespace caffe2

#endif  // CAFFE2_IMAGE_IMAGE_INPUT_OP_H_

