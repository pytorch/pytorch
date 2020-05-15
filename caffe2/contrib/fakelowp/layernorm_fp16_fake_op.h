#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <string>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

//#include "caffe2/fb/fbgemm/fbgemm_fp16/include/fbgemm/FbgemmFloat16.h"
//#include <fbgemm/FbgemmFloat16.h>
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"
#include "fp16_fma.h"

C10_DECLARE_bool(caffe2_fbgemm_fake_fp16_clamp);

namespace caffe2 {

class LayerNormFakeFp16Op : public Operator<CPUContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  template <class... Args>
  explicit LayerNormFakeFp16Op(Args&&... args)
      : Operator<CPUContext>(std::forward<Args>(args)...) {}
  ~LayerNormFakeFp16Op() noexcept override {}

  bool RunOnDevice() override {
    return true;
    // return DispatchHelper<InputTypes>::call(this, Input(DATA));
  }

  template <typename InputType>
  bool DoRunWithType() {
    // return DispatchHelper<TensorTypes2<int32_t, int64_t>, InputType>::call(
    //    this, Input(INDICES));
    return true;
  }

  template <typename InputType, typename IndexType>
  bool DoRunWithType2() {
    return true;
  }
};
} // namespace caffe2
