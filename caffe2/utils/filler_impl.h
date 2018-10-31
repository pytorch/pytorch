#ifndef CAFFE2_FILLER_IMPL_H_
#define CAFFE2_FILLER_IMPL_H_

#include <sstream>

#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/filler.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Type, class Context>
void TensorFiller::Fill(Tensor* tensor, Context* context) const {
  CAFFE_ENFORCE(context, "context is null");
  CAFFE_ENFORCE(tensor, "tensor is null");
  auto min = (min_ < std::numeric_limits<Type>::min())
      ? std::numeric_limits<Type>::min()
      : static_cast<Type>(min_);
  auto max = (max_ > std::numeric_limits<Type>::max())
      ? std::numeric_limits<Type>::max()
      : static_cast<Type>(max_);
  CAFFE_ENFORCE_LE(min, max);

  Tensor temp_tensor(shape_, Context::GetDeviceType());
  tensor->swap(temp_tensor);
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

} // namespace caffe2

#endif // CAFFE2_FILLER_IMPL_H_
