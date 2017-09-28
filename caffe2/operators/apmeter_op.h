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

#ifndef CAFFE2_MAP_OP_H_
#define CAFFE2_MAP_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class APMeterOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  APMeterOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        buffer_size_(
            OperatorBase::GetSingleArgument<int32_t>("buffer_size", 1000)),
        buffer_used_(0) {}

  bool RunOnDevice() override;

 protected:
  using BufferDataType = std::pair<float, int>;
  // Buffer the predictions for each class
  std::vector<std::vector<BufferDataType>> buffers_;
  // Capacity of the buffer
  int buffer_size_;
  // Used buffer
  int buffer_used_;

  INPUT_TAGS(PREDICTION, LABEL);

 protected:
  // Buffer predictions for N sample and D classes
  void
  BufferPredictions(const float* Xdata, const int* labelData, int N, int D);
};

} // namespace caffe2

#endif // CAFFE2_MAP_OP_H_
