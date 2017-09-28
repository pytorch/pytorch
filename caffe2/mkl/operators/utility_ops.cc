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

#include "caffe2/operators/utility_ops.h"
#include "caffe2/core/operator.h"
#include "caffe2/mkl/mkl_utils.h"
#ifdef CAFFE2_HAS_MKL_DNN

namespace caffe2 {
namespace mkl {

class CopyCPUToMKLOp final : public MKLOperator<float> {
 public:
  using MKLOperator<float>::MKLOperator;
  bool RunOnDevice() override {
    const auto& X = OperatorBase::Input<TensorCPU>(0);
    auto* Y = OperatorBase::OutputBlob(0);
    if (!Y->template IsType<MKLMemory<float>>() ||
        Y->Get<MKLMemory<float>>().dims() != X.dims()) {
      Y->Reset(new MKLMemory<float>(X.dims()));
    }
    Y->GetMutable<MKLMemory<float>>()->CopyFrom(X);
    return true;
  }
};

class CopyMKLToCPUOp final : public MKLOperator<float> {
 public:
  using MKLOperator<float>::MKLOperator;

  bool RunOnDevice() override {
    const auto& X = OperatorBase::Input<MKLMemory<float>>(0);
    auto* Y = OperatorBase::Output<TensorCPU>(0);
    X.CopyTo(Y);
    return true;
  }
};

} // namespace mkl

REGISTER_MKL_OPERATOR(CopyCPUToMKL, mkl::CopyCPUToMKLOp);
REGISTER_MKL_OPERATOR(CopyMKLToCPU, mkl::CopyMKLToCPUOp);

} // namespace caffe2

#endif // CAFFE2_HAS_MKL_DNN
