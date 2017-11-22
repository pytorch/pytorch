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

#pragma once

#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
struct SwishCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* /*device_context*/) {
    ConstEigenVectorArrayMap<T> xM(x, n);
    EigenVectorArrayMap<T>(y, n) = xM / (1. + (-xM).exp());
  }
};

template <class Context>
class SwishGradientOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(SwishGradientOp)
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <typename T>
  bool DoRunWithType() {
    auto& Xin = Input(X);
    auto& Yin = Input(Y);
    auto& DYin = Input(DY);
    auto* DXout = Output(DX);
    CAFFE_ENFORCE_EQ(Xin.size(), Yin.size());
    CAFFE_ENFORCE_EQ(DYin.size(), Yin.size());
    DXout->ResizeLike(Yin);

    const float* Xdata = Xin.template data<float>();
    const float* Ydata = Yin.template data<float>();
    const float* dYdata = DYin.template data<float>();
    float* dXdata = DXout->template mutable_data<float>();

    EigenVectorArrayMap<float> dXvec(dXdata, DXout->size());
    ConstEigenVectorArrayMap<float> Xvec(Xdata, Xin.size());
    ConstEigenVectorArrayMap<float> Yvec(Ydata, Yin.size());
    ConstEigenVectorArrayMap<float> dYvec(dYdata, DYin.size());

    // dx = dy * (y + sigmoid(x)*(1-y))
    dXvec = dYvec * (Yvec + (1. / (1. + (-Xvec).exp())) * (1. - Yvec));
    return true;
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, double>>::call(this, Input(X));
  }

 protected:
  INPUT_TAGS(X, Y, DY);
  OUTPUT_TAGS(DX);
};

class GetSwishGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SwishGradient",
        "",
        vector<string>{I(0), O(0), GO(0)},
        vector<string>{GI(0)});
  }
};

} // namespace caffe2
