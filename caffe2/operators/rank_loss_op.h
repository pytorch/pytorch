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

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// support multiple batches of sessions
template <typename T, class Context>
class PairWiseLossOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(PairWiseLossOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 private:
  INPUT_TAGS(XVALUE, LABEL, LENGTHS);
  OUTPUT_TAGS(YVALUE);
};

template <typename T, class Context>
class PairWiseLossGradientOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(PairWiseLossGradientOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 private:
  INPUT_TAGS(XVALUE, LABEL, DYVALUE, LENGTHS);
  OUTPUT_TAGS(DXVALUE);
};

} // namespace caffe2
