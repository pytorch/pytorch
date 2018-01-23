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
  USE_MKLOPERATOR_FUNCTIONS(T);

  MKLSumOp(const OperatorDef& operator_def, Workspace* ws)
      : MKLOperator<T>(operator_def, ws) {
    coefficients_.resize(this->InputSize(), 1);
    // caffe2::AddOp support broadcast but dnnSumCreate() doesn't.
    bool broadcast = OperatorBase::GetSingleArgument<bool>("broadcast", false);
    OPERATOR_NEEDS_FEATURE(
        !broadcast, "Broadcast is not yet supported with MKLDNN.");
  }

  bool RunOnDevice() override {
    const MKLMemory<T>& X0 = Input(0);
    MKLMemory<T>* Y = Output(0);
    bool dims_changed;
    CHECK_INPUT_DIMS(X0, dims_changed);
    if (dims_changed) {
      primitive_.Reset(
          dnnSumCreate<T>,
          nullptr,
          this->InputSize(),
          X0.layout(),
          coefficients_.data());
      if (Y != &X0) {
        Y->Reset(X0.dims(), primitive_, dnnResourceDst);
      }
      buffer_.Reset(X0.dims(), primitive_, dnnResourceDst, true);
    }
    input_views_.resize(this->InputSize());
    for (auto i = 0; i < this->InputSize(); ++i) {
      const MKLMemory<T>& Xi = Input(i);
      CAFFE_ENFORCE_EQ(X0.dims(), Xi.dims());
      // Input layouts might be different depending on preceding primitives.
      // Create a consistent view as dnnSumCreate expects it that way.
      input_views_[i] = Xi.View(X0.layout());
      resources_[dnnResourceMultipleSrc + i] = input_views_[i].get();
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
  std::vector<float> coefficients_;
  vector<TIndex> cached_input_dims_;
  vector<std::shared_ptr<void>> input_views_;
};

} // namespace mkl

REGISTER_MKL_OPERATOR(Sum, mkl::MKLSumOp<float>);
REGISTER_MKL_OPERATOR(Add, mkl::MKLSumOp<float>);

} // namespace caffe2

#endif // CAFFE2_HAS_MKL_DNN
