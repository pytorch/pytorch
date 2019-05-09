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

#include "caffe2/operators/batch_matmul_op.h"
#include "caffe2/quantization/server/dnnlowp_op.h"
#include "fbgemm/Fbgemm.h"

namespace caffe2 {

template <typename T>
class BatchMatMulDNNLowPOp final
    : public DNNLowPOp<T, BatchMatMulOp<CPUContext>> {
 public:
  BatchMatMulDNNLowPOp(const OperatorDef& operator_def, Workspace* ws);
  bool RunOnDevice() override;

  USE_OPERATOR_FUNCTIONS(CPUContext);
  USE_DNNLOWP_OPERATOR_BASE_FUNCTIONS(T, BatchMatMulOp<CPUContext>);

 private:
  bool trans_a_;
  bool trans_b_;
  bool broadcast_{false};
  bool is_B_constant_{false};

  std::vector<std::int8_t> B_quantized_;
  std::vector<std::unique_ptr<fbgemm::PackBMatrix<std::int8_t>>> Bq_packed_;
  std::vector<std::uint8_t> A_pack_buf_;
  std::vector<std::int32_t> row_offsets_, column_offsets_;

  std::vector<dnnlowp::TensorQuantizationParams> B_qparams_;
  std::vector<dnnlowp::RequantizationParams> requantization_params_;

  std::vector<std::int32_t> Y_int32_;
  bool first_invocation_{true};
}; // BatchMatMulDNNLowPOp

} // namespace caffe2
