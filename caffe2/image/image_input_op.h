#ifndef CAFFE2_IMAGE_IMAGE_INPUT_OP_H_
#define CAFFE2_IMAGE_IMAGE_INPUT_OP_H_

#include <omp.h>
#include <opencv2/opencv.hpp>

#include <iostream>

#include "caffe/proto/caffe.pb.h"
#include "caffe2/core/db.h"
#include "caffe2/utils/math.h"
#include "caffe2/operators/prefetch_op.h"

namespace caffe2 {

template <class Context>
class ImageInputOp final
    : public PrefetchOperator<Context> {
 public:
  using OperatorBase::OutputSize;
  using PrefetchOperator<Context>::context_;
  using PrefetchOperator<Context>::prefetch_thread_;
  explicit ImageInputOp(const OperatorDef& operator_def,
                                    Workspace* ws);
  ~ImageInputOp() {
    PrefetchOperator<Context>::Finalize();
  }

  bool Prefetch() override;
  bool CopyPrefetched() override;

 private:
  bool GetImageAndLabelFromDBValue(
      const string& value, cv::Mat* img, int* label);
  unique_ptr<db::DBReader> owned_reader_;
  const db::DBReader* reader_;
  CPUContext cpu_context_;
  TensorCPU prefetched_image_;
  TensorCPU prefetched_label_;
  Tensor<Context> prefetched_image_on_device_;
  Tensor<Context> prefetched_label_on_device_;
  int batch_size_;
  float mean_;
  float std_;
  bool color_;
  int scale_;
  bool warp_;
  int crop_;
  bool mirror_;
  bool use_caffe_datum_;
};


template <class Context>
ImageInputOp<Context>::ImageInputOp(
      const OperatorDef& operator_def, Workspace* ws)
      : PrefetchOperator<Context>(operator_def, ws),
        reader_(nullptr),
        batch_size_(
            OperatorBase::template GetSingleArgument<int>("batch_size", 0)),
        mean_(OperatorBase::template GetSingleArgument<float>("mean", 0.)),
        std_(OperatorBase::template GetSingleArgument<float>("std", 1.)),
        color_(OperatorBase::template GetSingleArgument<int>("color", 1)),
        scale_(OperatorBase::template GetSingleArgument<int>("scale", -1)),
        warp_(OperatorBase::template GetSingleArgument<int>("warp", 0)),
        crop_(OperatorBase::template GetSingleArgument<int>("crop", -1)),
        mirror_(OperatorBase::template GetSingleArgument<int>("mirror", 0)),
        use_caffe_datum_(OperatorBase::template GetSingleArgument<int>(
              "use_caffe_datum", 0)) {
  if (operator_def.input_size() == 0) {
    LOG(ERROR) << "You are using an old ImageInputOp format that creates "
                       "a local db reader. Consider moving to the new style "
                       "that takes in a DBReader blob instead.";
    string db_name =
        OperatorBase::template GetSingleArgument<string>("db", "");
    CAFFE_ENFORCE_GT(db_name.size(), 0, "Must specify a db name.");
    owned_reader_.reset(new db::DBReader(
        OperatorBase::template GetSingleArgument<string>(
            "db_type", "leveldb"),
        db_name));
    reader_ = owned_reader_.get();
  }
  CAFFE_ENFORCE_GT(batch_size_, 0, "Batch size should be nonnegative.");
  CAFFE_ENFORCE_GT(scale_, 0, "Must provide the scaling factor.");
  CAFFE_ENFORCE_GT(crop_, 0, "Must provide the cropping value.");
  CAFFE_ENFORCE_GE(
      scale_, crop_, "The scale value must be no smaller than the crop value.");

  LOG(INFO) << "Creating an image input op with the following setting: ";
  LOG(INFO) << "    Outputting in batches of " << batch_size_ << " images;";
  LOG(INFO) << "    Treating input image as "
            << (color_ ? "color " : "grayscale ") << "image;";
  LOG(INFO) << "    Scaling image to " << scale_
            << (warp_ ? " with " : " without ") << "warping;";
  LOG(INFO) << "    Cropping image to " << crop_
            << (mirror_ ? " with " : " without ") << "random mirroring;";
  LOG(INFO) << "    Subtract mean " << mean_ << " and divide by std " << std_
            << ".";
  prefetched_image_.Resize(
      TIndex(batch_size_),
      TIndex(crop_),
      TIndex(crop_),
      TIndex(color_ ? 3 : 1));
  prefetched_label_.Resize(vector<TIndex>(1, batch_size_));
}

template <class Context>
bool ImageInputOp<Context>::GetImageAndLabelFromDBValue(
      const string& value, cv::Mat* img, int* label) {
  if (use_caffe_datum_) {
    // The input is a caffe datum format.
    caffe::Datum datum;
    CAFFE_ENFORCE(datum.ParseFromString(value));
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
      CAFFE_ENFORCE(img->isContinuous());
      CAFFE_ENFORCE((color_ && datum.channels() == 3) || datum.channels() == 1);
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
    CAFFE_ENFORCE(protos.ParseFromString(value));
    const TensorProto& image_proto = protos.protos(0);
    const TensorProto& label_proto = protos.protos(1);
    if (image_proto.data_type() == TensorProto::STRING) {
      // encoded image string.
      DCHECK_EQ(image_proto.string_data_size(), 1);
      const string& encoded_image_str = image_proto.string_data(0);
      int encoded_size = encoded_image_str.size();
      // We use a cv::Mat to wrap the encoded str so we do not need a copy.
      *img = cv::imdecode(
          cv::Mat(1, &encoded_size, CV_8UC1,
              const_cast<char*>(encoded_image_str.data())),
          color_ ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
    } else if (image_proto.data_type() == TensorProto::BYTE) {
      // raw image content.
      CAFFE_ENFORCE_EQ(image_proto.dims_size(), (color_ ? 3 : 2));
      CAFFE_ENFORCE_GE(
          image_proto.dims(0), crop_, "Image height must be bigger than crop.");
      CAFFE_ENFORCE_GE(
          image_proto.dims(1), crop_, "Image width must be bigger than crop.");
      CAFFE_ENFORCE(!color_ || image_proto.dims(2) == 3);
      *img = cv::Mat(
          image_proto.dims(0), image_proto.dims(1), color_ ? CV_8UC3 : CV_8UC1);
      memcpy(img->ptr<uchar>(0), image_proto.byte_data().data(),
             image_proto.byte_data().size());
    } else {
      LOG(FATAL) << "Unknown image data type.";
    }
    DCHECK_EQ(label_proto.data_type(), TensorProto::INT32);
    DCHECK_EQ(label_proto.int32_data_size(), 1);
    *label = label_proto.int32_data(0);
  }
  // TODO(Yangqing): return false if any error happens.
  return true;
}

template <class Context>
bool ImageInputOp<Context>::Prefetch() {
  if (!owned_reader_.get()) {
    // if we are not owning the reader, we will get the reader pointer from
    // input. Otherwise the constructor should have already set the reader
    // pointer.
    reader_ = &OperatorBase::Input<db::DBReader>(0);
  }
  const int channels = color_ ? 3 : 1;
  // Call mutable_data() once to allocate the underlying memory.
  prefetched_image_.mutable_data<float>();
  prefetched_label_.mutable_data<int>();
  // TODO(jiayq): Handle this prefetching with a real thread pool. Currently,
  // with 4 threads we should be able to get a decent sheed for AlexNet type
  // training already.
  std::mt19937 meta_randgen(time(nullptr));
  std::vector<std::mt19937> randgen_per_thread;
  for (int i = 0; i < 4; ++i) {
    randgen_per_thread.emplace_back(meta_randgen());
  }
  #pragma omp parallel for num_threads(4)
  for (int item_id = 0; item_id < batch_size_; ++item_id) {
    std::bernoulli_distribution mirror_this_image(0.5);
    std::mt19937& randgen = randgen_per_thread[omp_get_thread_num()];
    float* image_data = prefetched_image_.mutable_data<float>()
        + crop_ * crop_ * channels * item_id;
    string key, value;
    cv::Mat img;
    int label;
    cv::Mat scaled_img;
    // process data
    reader_->Read(&key, &value);
    CAFFE_ENFORCE(GetImageAndLabelFromDBValue(value, &img, &label));
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
    if (scaled_height != img.rows || scaled_width != img.cols) {
      cv::resize(img, scaled_img, cv::Size(scaled_width, scaled_height),
                 0, 0, cv::INTER_LINEAR);
    } else {
      // No scaling needs to be done.
      scaled_img = img;
    }
    // find the cropped region, and copy it to the destination matrix with
    // mean subtraction and scaling.
    int width_offset =
        std::uniform_int_distribution<>(0, scaled_img.cols - crop_)(randgen);
    int height_offset =
        std::uniform_int_distribution<>(0, scaled_img.rows - crop_)(randgen);
    if (mirror_ && mirror_this_image(randgen)) {
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
  }

  // If the context is not CPUContext, we will need to do a copy in the
  // prefetch function as well.
  if (!std::is_same<Context, CPUContext>::value) {
    prefetched_image_on_device_.CopyFrom(prefetched_image_, &context_);
    prefetched_label_on_device_.CopyFrom(prefetched_label_, &context_);
  }
  return true;
}

template <class Context>
bool ImageInputOp<Context>::CopyPrefetched() {
  auto* image_output = OperatorBase::Output<Tensor<Context> >(0);
  auto* label_output = OperatorBase::Output<Tensor<Context> >(1);
  // Note(jiayq): The if statement below should be optimized away by the
  // compiler since std::is_same is a constexpr.
  if (std::is_same<Context, CPUContext>::value) {
    image_output->CopyFrom(prefetched_image_, &context_);
    label_output->CopyFrom(prefetched_label_, &context_);
  } else {
    image_output->CopyFrom(prefetched_image_on_device_, &context_);
    label_output->CopyFrom(prefetched_label_on_device_, &context_);
  }
  return true;
}
}  // namespace caffe2

#endif  // CAFFE2_IMAGE_IMAGE_INPUT_OP_H_

