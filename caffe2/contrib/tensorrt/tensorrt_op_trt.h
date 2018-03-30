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

#include "caffe2/contrib/tensorrt/trt_utils.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/operator.h"

#include <NvInfer.h>
#include <iostream>
#include <memory>
#include <unordered_map>

namespace caffe2 {

class TensorRTOp final : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);
  TensorRTOp(const OperatorDef& operator_def, Workspace* ws);
  bool RunOnDevice() override;
  virtual ~TensorRTOp() noexcept {}

 private:
  void MaybeAdjustOutputShape(int output_idx, std::vector<TIndex>* dims);

  TrtLogger logger_;
  int max_batch_size_;
  std::vector<std::pair<int, bool>> binding_hints_;
  std::vector<nvinfer1::Dims> nv_dims_;
  std::unordered_map<int, std::vector<TIndex>> output_size_hints_;
  std::shared_ptr<nvinfer1::ICudaEngine> trt_engine_{nullptr};
  std::shared_ptr<nvinfer1::IExecutionContext > trt_executor_{nullptr};
};

} // namespace caffe2

