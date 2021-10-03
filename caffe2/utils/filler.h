#ifndef CAFFE2_FILLER_H_
#define CAFFE2_FILLER_H_

#include <sstream>

#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// TODO: replace filler distribution enum with a better abstraction
enum FillerDistribution { FD_UNIFORM, FD_FIXEDSUM, FD_SYNTHETIC };

class TensorFiller {
 public:
  template <class Type, class Context>
  void Fill(Tensor* tensor, Context* context) const {
    CAFFE_ENFORCE(context, "context is null");
    CAFFE_ENFORCE(tensor, "tensor is null");
    auto min = (min_ < std::numeric_limits<Type>::min())
        ? std::numeric_limits<Type>::min()
        : static_cast<Type>(min_);
    // NOLINTNEXTLINE(clang-diagnostic-implicit-const-int-float-conversion)
    auto max = (max_ > std::numeric_limits<Type>::max())
        ? std::numeric_limits<Type>::max()
        : static_cast<Type>(max_);
    CAFFE_ENFORCE_LE(min, max);

    Tensor temp_tensor(shape_, Context::GetDeviceType());
    std::swap(*tensor, temp_tensor);
    Type* data = tensor->template mutable_data<Type>();

    // select distribution
    switch (dist_) {
      case FD_UNIFORM: {
        math::RandUniform<Type, Context>(
            tensor->numel(), min, max, data, context);
        break;
      }
      case FD_FIXEDSUM: {
        auto fixed_sum = static_cast<Type>(fixed_sum_);
        CAFFE_ENFORCE_LE(min * tensor->numel(), fixed_sum);
        CAFFE_ENFORCE_GE(max * tensor->numel(), fixed_sum);
        math::RandFixedSum<Type, Context>(
            tensor->numel(), min, max, fixed_sum_, data, context);
        break;
      }
      case FD_SYNTHETIC: {
        math::RandSyntheticData<Type, Context>(
            tensor->numel(), min, max, data, context);
        break;
      }
    }
  }

  TensorFiller& Dist(FillerDistribution dist) {
    dist_ = dist;
    return *this;
  }

  template <class Type>
  TensorFiller& Min(Type min) {
    min_ = (double)min;
    return *this;
  }

  template <class Type>
  TensorFiller& Max(Type max) {
    max_ = (double)max;
    return *this;
  }

  template <class Type>
  TensorFiller& FixedSum(Type fixed_sum) {
    dist_ = FD_FIXEDSUM;
    fixed_sum_ = (double)fixed_sum;
    return *this;
  }

  // A helper function to construct the lengths vector for sparse features
  // We try to pad least one index per batch unless the total_length is 0
  template <class Type>
  TensorFiller& SparseLengths(Type total_length) {
    return FixedSum(total_length)
        .Min(std::min(static_cast<Type>(1), total_length))
        .Max(total_length);
  }

  // a helper function to construct the segments vector for sparse features
  template <class Type>
  TensorFiller& SparseSegments(Type max_segment) {
    CAFFE_ENFORCE(dist_ != FD_FIXEDSUM);
    return Min(0).Max(max_segment).Dist(FD_SYNTHETIC);
  }

  TensorFiller& Shape(const std::vector<int64_t>& shape) {
    shape_ = shape;
    return *this;
  }

  template <class Type>
  TensorFiller(const std::vector<int64_t>& shape, Type fixed_sum)
      : shape_(shape), dist_(FD_FIXEDSUM), fixed_sum_((double)fixed_sum) {}

  TensorFiller(const std::vector<int64_t>& shape)
      : shape_(shape), dist_(FD_UNIFORM), fixed_sum_(0) {}

  TensorFiller() : TensorFiller(std::vector<int64_t>()) {}

  std::string DebugString() const {
    std::stringstream stream;
    stream << "shape = [" << shape_ << "]; min = " << min_
           << "; max = " << max_;
    switch (dist_) {
      case FD_FIXEDSUM:
        stream << "; dist = FD_FIXEDSUM";
        break;
      case FD_SYNTHETIC:
        stream << "; dist = FD_SYNTHETIC";
        break;
      default:
        stream << "; dist = FD_UNIFORM";
        break;
    }
    return stream.str();
  }

 private:
  std::vector<int64_t> shape_;
  // TODO: type is unknown until a user starts to fill data;
  // cast everything to double for now.
  double min_ = 0.0;
  double max_ = 1.0;
  FillerDistribution dist_;
  double fixed_sum_;
};

} // namespace caffe2

#endif // CAFFE2_FILLER_H_
