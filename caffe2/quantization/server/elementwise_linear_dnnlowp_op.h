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

#include "caffe2/operators/elementwise_linear_op.h"
#include "caffe2/quantization/server/dnnlowp_op.h"

namespace caffe2 {

using ElementwiseLinearFp32Op = ElementwiseLinearOp<float, CPUContext>;

template <typename T>
class ElementwiseLinearDNNLowPOp final
  : public DNNLowPOp<T, ElementwiseLinearFp32Op> {
 public:
  ElementwiseLinearDNNLowPOp(const OperatorDef& operator_def, Workspace* ws);
  bool RunOnDevice() override;

  USE_OPERATOR_FUNCTIONS(CPUContext);
  USE_DNNLOWP_OPERATOR_BASE_FUNCTIONS(T, ElementwiseLinearFp32Op);

 private:
  bool GetQuantizationParameters_();

  int axis_;

  dnnlowp::RequantizationParams requantization_params_;

  std::vector<T> a_quantized_;
};

} // namespace caffe2
