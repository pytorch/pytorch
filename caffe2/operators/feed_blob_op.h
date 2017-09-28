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

#ifndef CAFFE2_OPERATORS_FEED_BLOB_OP_H_
#define CAFFE2_OPERATORS_FEED_BLOB_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class FeedBlobOp : public Operator<Context> {
 public:
  FeedBlobOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    CAFFE_ENFORCE(
        OperatorBase::HasSingleArgumentOfType<string>("value"),
        "value argument must exist and be passed as a string");
    value_ = OperatorBase::GetSingleArgument<string>("value", "");
  }

  bool RunOnDevice() override {
    *OperatorBase::Output<std::string>(0) = value_;
    return true;
  }

 private:
  std::string value_;
};

} // namespace caffe2

#endif
