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

#ifndef CAFFE2_OPERATORS_COSINE_EMBEDDING_CRITERION_OP_H_
#define CAFFE2_OPERATORS_COSINE_EMBEDDING_CRITERION_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class CosineEmbeddingCriterionOp final : public Operator<Context> {
 public:
  CosineEmbeddingCriterionOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        OP_SINGLE_ARG(float, "margin", margin_, 0.0) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  float margin_;
};

template <class Context>
class CosineEmbeddingCriterionGradientOp final : public Operator<Context> {
 public:
  CosineEmbeddingCriterionGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        OP_SINGLE_ARG(float, "margin", margin_, 0.0) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  float margin_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_COSINE_EMBEDDING_CRITERION_OP_H_
