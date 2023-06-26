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

#include <fbgemm/FbgemmFP16.h>
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <
    class Context,
    class Engine = DefaultEngine,
    bool TransposeWeight = true,
    typename TPacked = fbgemm::float16>
class FbGemmPackOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  FbGemmPackOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axis_(this->template GetSingleArgument<int32_t>("axis_w", 1)),
        no_packing_(
            this->template GetSingleArgument<int32_t>("no_packing", 0)) {}
  ~FbGemmPackOp() override {}

  bool RunOnDevice() override {
    const auto& X = Input(0);
    const auto canonical_axis = X.canonical_axis_index(axis_);
    const auto N = X.size_to_dim(canonical_axis);
    const auto K = X.size_from_dim(canonical_axis);

    fbgemm::PackedGemmMatrixFP16* resultPtr;
    if (TransposeWeight) {
      resultPtr = new fbgemm::PackedGemmMatrixFP16(
          fbgemm::matrix_op_t::Transpose,
          K,
          N,
          1.0f, /*alpha*/
          X.template data<float>());
    } else {
      resultPtr = new fbgemm::PackedGemmMatrixFP16(
          fbgemm::matrix_op_t::NoTranspose,
          N,
          K,
          1.0f, /*alpha*/
          X.template data<float>());
    }

    if (no_packing_) {
      C10_LOG_FIRST_N(WARNING, 10) << "no_packing will be deprecated soon";

      vector<fbgemm::float16> src_mat(resultPtr->matSize());
      fbgemm::float16* pmat = resultPtr->pmat();
      memcpy(
          src_mat.data(), pmat, resultPtr->matSize() * sizeof(fbgemm::float16));
      resultPtr->unpackFromSrc(fbgemm::matrix_op_t::Transpose, src_mat.data());
    }

    auto* Y =
        this->template Output<unique_ptr<fbgemm::PackedGemmMatrixFP16>>(0);
    Y->reset(resultPtr);
    return true;
  }

 protected:
  size_t axis_{1};
  // Do not pack the layout, for testing only
  bool no_packing_;
};

} // namespace caffe2
