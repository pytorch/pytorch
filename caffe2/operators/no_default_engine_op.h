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

#ifndef CAFFE2_OPERATORS_NO_DEFAULT_ENGINE_OP_H_
#define CAFFE2_OPERATORS_NO_DEFAULT_ENGINE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

/**
 * A helper class to denote that an op does not have a default engine.
 *
 * NoDefaultEngineOp is a helper class that one can use to denote that a
 * specific operator is not intended to be called without an explicit engine
 * given. This is the case for e.g. the communication operators where one has
 * to specify a backend (like MPI or ZEROMQ).
 */
template <class Context>
class NoDefaultEngineOp final : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(NoDefaultEngineOp);
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    CAFFE_THROW(
        "The operator ",
        this->debug_def().type(),
        " does not have a default engine implementation. Please "
        "specify an engine explicitly for this operator.");
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_NO_DEFAULT_ENGINE_OP_H_
