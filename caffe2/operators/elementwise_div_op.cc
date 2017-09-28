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

#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {

// See the operations supported here:
// https://eigen.tuxfamily.org/dox-devel/group__QuickRefPage.html
#define EIGEN_DIV(x, y) ((x) / (y))
EIGEN_FUNCTOR(Div, EIGEN_DIV, NumericTypes, SameTypeAsInput);
#undef EIGEN_DIV

void ElementWiseDivide(
    CPUContext& /* unused context */,
    const int n,
    float* dXdata,
    float* dYdata,
    const float* dZdata,
    const float* Ydata,
    const float* Zdata) {
  ConstEigenVectorArrayMap<float> dZdataVec(dZdata, n);
  ConstEigenVectorArrayMap<float> YdataVec(Ydata, n);
  ConstEigenVectorArrayMap<float> ZdataVec(Zdata, n);
  EigenVectorArrayMap<float>(dXdata, n) = dZdataVec / YdataVec;
  EigenVectorArrayMap<float>(dYdata, n) = - (dZdataVec * ZdataVec) / YdataVec;
}

REGISTER_CPU_OPERATOR(DivGradient, DivGradientOp<CPUContext>);
}
