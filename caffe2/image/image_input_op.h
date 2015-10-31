#ifndef CAFFE2_IMAGE_IMAGE_INPUT_OP_H_
#define CAFFE2_IMAGE_IMAGE_INPUT_OP_H_

#include <opencv2/opencv.hpp>

#include <iostream>

#include "caffe/proto/caffe.pb.h"
#include "caffe2/core/db.h"
#include "caffe2/operators/prefetch_op.h"

namespace caffe2 {

template <class Context>
class ImageInputOp final
    : public PrefetchOperator<Context> {
 public:
  using OperatorBase::OutputSize;
  using PrefetchOperator<Context>::prefetch_thread_;
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
  bool GetImageAndLabelFromDBValue(
      const string& value, cv::Mat* img, int* label);
  unique_ptr<db::DB> db_;
  unique_ptr<db::Cursor> cursor_;
  CPUContext cpu_context_;
  TensorCPU prefetched_image_;
  TensorCPU prefetched_label_;
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
  bool use_caffe_datum_;
  INPUT_OUTPUT_STATS(0, 0, 2, 2);
  DISABLE_COPY_AND_ASSIGN(ImageInputOp);
};


template <class Context>
ImageInputOp<Context>::ImageInputOp(
      const OperatorDef& operator_def, Workspace* ws)
      : PrefetchOperator<Context>(operator_def, ws),
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
        mirror_(OperatorBase::template GetSingleArgument<int>("mirror", 0)),
        use_caffe_datum_(OperatorBase::template GetSingleArgument<int>(
              "use_caffe_datum", 0)) {
  CAFFE_CHECK_GT(batch_size_, 0) << "Batch size should be nonnegative.";
  CAFFE_CHECK_GT(db_name_.size(), 0) << "Must provide a leveldb name.";
  CAFFE_CHECK_GT(scale_, 0) << "Must provide the scaling factor.";
  CAFFE_CHECK_GT(crop_, 0) << "Must provide the cropping value.";
  CAFFE_CHECK_GE(scale_, crop_)
      << "The scale value must be no smaller than the crop value.";

  CAFFE_LOG_INFO << "Creating an image input op with the following setting: ";
  CAFFE_LOG_INFO << "    Outputting in batches of " << batch_size_ << " images;";
  CAFFE_LOG_INFO << "    Treating input image as "
            << (color_ ? "color " : "grayscale ") << "image;";
  CAFFE_LOG_INFO << "    Scaling image to " << scale_
            << (warp_ ? " with " : " without ") << "warping;";
  CAFFE_LOG_INFO << "    Cropping image to " << crop_
            << (mirror_ ? " with " : " without ") << "random mirroring;";
  CAFFE_LOG_INFO << "    Subtract mean " << mean_ << " and divide by std " << std_
            << ".";
  db_.reset(db::CreateDB(db_type_, db_name_, db::READ));
  cursor_.reset(db_->NewCursor());
  cursor_->SeekToFirst();
  prefetched_image_.Reshape(
      vector<int>{batch_size_, crop_, crop_, (color_ ? 3 : 1)});
  prefetched_label_.Reshape(vector<int>(1, batch_size_));
}

template <class Context>
bool ImageInputOp<Context>::GetImageAndLabelFromDBValue(
      const string& value, cv::Mat* img, int* label) {
  if (use_caffe_datum_) {
    // The input is a caffe datum format.
    caffe::Datum datum;
    CAFFE_CHECK(datum.ParseFromString(value));
    *label = datum.label();
    if (datum.encoded()) {
      // encoded image in datum.
      *img = cv::imdecode(
          cv::Mat(1, datum.data().size(), CV_8UC1,
          const_cast<char*>(datum.data().data())),
          color_ ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
    } else {
      // Raw image in datum.
      *img = cv::Mat(datum.height(), datum.width(),
                     color_ ? CV_8UC3 : CV_8UC1);
      // Note(Yangqing): I believe that the mat should be created continuous.
      CAFFE_CHECK(img->isContinuous());
      CAFFE_CHECK((color_ && datum.channels() == 3) || datum.channels() == 1);
      if (datum.channels() == 1) {
        memcpy(img->ptr<uchar>(0), datum.data().data(), datum.data().size());
      } else {
        // Datum stores things in CHW order, let's do HWC for images to make
        // things more consistent with conventional image storage.
        for (int c = 0; c < 3; ++c) {
          const char* datum_buffer =
              datum.data().data() + datum.height() * datum.width() * c;
          uchar* ptr = img->ptr<uchar>(0) + c;
          for (int h = 0; h < datum.height(); ++h) {
            for (int w = 0; w < datum.width(); ++w) {
              *ptr = *(datum_buffer++);
              ptr += 3;
            }
          }
        }
      }
    }
  } else {
    // The input is a caffe2 format.
    TensorProtos protos;
    CAFFE_CHECK(protos.ParseFromString(value));
    const TensorProto& image_proto = protos.protos(0);
    const TensorProto& label_proto = protos.protos(1);
    if (image_proto.data_type() == TensorProto::STRING) {
      // encoded image string.
      CAFFE_DCHECK_EQ(image_proto.string_data_size(), 1);
      const string& encoded_image_str = image_proto.string_data(0);
      int encoded_size = encoded_image_str.size();
      // We use a cv::Mat to wrap the encoded str so we do not need a copy.
      *img = cv::imdecode(
          cv::Mat(1, &encoded_size, CV_8UC1,
          const_cast<char*>(encoded_image_str.data())),
          color_ ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
    } else if (image_proto.data_type() == TensorProto::BYTE) {
      // raw image content.
      CAFFE_CHECK_EQ(image_proto.dims_size(), (color_ ? 3 : 2));
      CAFFE_CHECK_GE(image_proto.dims(0), crop_)
          << "Image height must be bigger than crop.";
      CAFFE_CHECK_GE(image_proto.dims(1), crop_)
          << "Image width must be bigger than crop.";
      CAFFE_CHECK(!color_ || image_proto.dims(2) == 3);
      *img = cv::Mat(
          image_proto.dims(0), image_proto.dims(1), color_ ? CV_8UC3 : CV_8UC1);
      memcpy(img->ptr<uchar>(0), image_proto.byte_data().data(),
             image_proto.byte_data().size());
    } else {
      CAFFE_LOG_FATAL << "Unknown image data type.";
    }
    CAFFE_DCHECK_EQ(label_proto.data_type(), TensorProto::INT32);
    CAFFE_DCHECK_EQ(label_proto.int32_data_size(), 1);
    *label = label_proto.int32_data(0);
  }
  // TODO(Yangqing): return false if any error happens.
  return true;
}

template <class Context>
bool ImageInputOp<Context>::Prefetch() {
  std::bernoulli_distribution mirror_this_image(0.5);
  float* image_data = prefetched_image_.mutable_data<float>();
  int channels = color_ ? 3 : 1;
  for (int item_id = 0; item_id < batch_size_; ++item_id) {
    // CAFFE_LOG_INFO << "Prefetching item " << item_id;
    // process data
    cv::Mat img;
    int label;
    cv::Mat scaled_img;
    CAFFE_CHECK(GetImageAndLabelFromDBValue(cursor_->value(), &img, &label));
    // deal with scaling.
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
    cv::resize(img, scaled_img, cv::Size(scaled_width, scaled_height),
               0, 0, cv::INTER_LINEAR);
    // find the cropped region, and copy it to the destination matrix with
    // mean subtraction and scaling.
    int width_offset =
        std::uniform_int_distribution<>(0, scaled_img.cols - crop_)(
            cpu_context_.RandGenerator());
    int height_offset =
        std::uniform_int_distribution<>(0, scaled_img.rows - crop_)(
            cpu_context_.RandGenerator());
    // CAFFE_VLOG(1) << "offset: " << height_offset << ", " << width_offset;
    if (mirror_ && mirror_this_image(cpu_context_.RandGenerator())) {
      // Copy mirrored image.
      for (int h = height_offset; h < height_offset + crop_; ++h) {
        for (int w = width_offset + crop_ - 1; w >= width_offset; --w) {
          const cv::Vec3b& cv_data = scaled_img.at<cv::Vec3b>(h, w);
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
          const cv::Vec3b& cv_data = scaled_img.at<cv::Vec3b>(h, w);
          for (int c = 0; c < channels; ++c) {
            *(image_data++) =
                (static_cast<uint8_t>(cv_data[c]) - mean_) / std_;
          }
        }
      }
    }
    // Copy the label
    prefetched_label_.mutable_data<int>()[item_id] = label;
    // Advance to the next item.
    cursor_->Next();
    if (!cursor_->Valid()) {
      cursor_->SeekToFirst();
    }
  }
  return true;
}

template <class Context>
bool ImageInputOp<Context>::CopyPrefetched() {
  // The first output is the image data.
  auto* image_output = OperatorBase::Output<Tensor<Context> >(0);
  image_output->ReshapeLike(prefetched_image_);
  this->device_context_.template Copy<float, CPUContext, Context>(
      prefetched_image_.size(), prefetched_image_.template data<float>(),
      image_output->template mutable_data<float>());
  // The second output is the label.
  auto* label_output = OperatorBase::Output<Tensor<Context> >(1);
  label_output->ReshapeLike(prefetched_label_);
  this->device_context_.template Copy<int, CPUContext, Context>(
      prefetched_label_.size(), prefetched_label_.template data<int>(),
      label_output->template mutable_data<int>());
  return true;
}

}  // namespace caffe2

#endif  // CAFFE2_IMAGE_IMAGE_INPUT_OP_H_

