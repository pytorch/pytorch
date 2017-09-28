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

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/mkl/mkl_utils.h"

#ifdef CAFFE2_HAS_MKL_DNN

namespace caffe2 {
namespace mkl {

template <typename T>
class MKLSumOp final : public MKLOperator<T> {
 public:
  MKLSumOp(const OperatorDef& operator_def, Workspace* ws)
      : MKLOperator<T>(operator_def, ws) {
    coefficients_.resize(this->InputSize(), 1);
  }

  bool RunOnDevice() override {
    const MKLMemory<float>& X0 = OperatorBase::Input<MKLMemory<float>>(0);
    MKLMemory<float>* Y = OperatorBase::Output<MKLMemory<float>>(0);
    bool dims_changed;
    CHECK_INPUT_DIMS(X0, dims_changed);
    if (dims_changed) {
      primitive_.Reset(
          dnnSumCreate<float>,
          nullptr,
          this->InputSize(),
          X0.layout(),
          coefficients_.data());
      if (Y != &X0) {
        Y->Reset(X0.dims(), primitive_, dnnResourceDst);
      }
      buffer_.Reset(X0.dims(), primitive_, dnnResourceDst, true);
    }
    for (auto i = 0; i < this->InputSize(); ++i) {
      const MKLMemory<float>& Xi = OperatorBase::Input<MKLMemory<float>>(i);
      CAFFE_ENFORCE(dnnLayoutCompare_F32(X0.layout(), Xi.layout()));
      resources_[dnnResourceMultipleSrc + i] = Xi.buffer();
    }
    if (Y != &X0) {
      // TODO: MKLDNN seems broken in the in-place case, so when we specify
      // in-place we will need to use buffer differnt from X0/Y.
      buffer_.ShareFrom(*Y);
    }
    resources_[dnnResourceDst] = buffer_.buffer();
    MKLDNN_SAFE_CALL(mkl::dnnExecute<T>(primitive_, resources_));
    buffer_.CopyTo(Y, primitive_, dnnResourceDst);
    return true;
  }

 private:
  // Input: X, W, b
  // Output: Y
  std::vector<float> coefficients_;
  vector<TIndex> cached_input_dims_;
  PrimitiveWrapper<T> primitive_;
  MKLMemory<T> buffer_;
  void* resources_[dnnResourceNumber] = {0};
  INPUT_TAGS(INPUT, FILTER, BIAS);
};

} // namespace mkl

REGISTER_MKL_OPERATOR(Sum, mkl::MKLSumOp<float>);

} // namespace caffe2

#endif // CAFFE2_HAS_MKL_DNN
