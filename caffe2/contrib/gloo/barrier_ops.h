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

#include "caffe2/contrib/gloo/common.h"
#include "caffe2/core/operator.h"

#include <gloo/algorithm.h>
#include <gloo/barrier_all_to_one.h>
#include <gloo/common/error.h>
#include <gloo/context.h>

namespace caffe2 {
namespace gloo {

template <class Context>
class BarrierOp final : public Operator<Context> {
 public:
  BarrierOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        ws_(ws),
        status_blob_(
            OperatorBase::GetSingleArgument<std::string>("status_blob", "")) {
    if (status_blob_ != "") {
      ws_->CreateBlob(status_blob_);
    }
  }

  virtual ~BarrierOp() {}

  bool RunOnDevice() override {
    auto context = OperatorBase::Input<std::shared_ptr<::gloo::Context>>(0);
    std::call_once(once_, [&] {
      initContext_ = context;
      // Use an all-to-one barrier synchronizing against rank 0
      algorithm_.reset(new ::gloo::BarrierAllToOne(initContext_, 0));
    });

    // If any parameter has changed in between runs, the initialized
    // algorithm is invalid and cannot be used.
    CAFFE_ENFORCE(context == initContext_, "Context has changed");

    try {
      algorithm_->run();
    } catch (::gloo::IoException& ioe) {
      LOG(ERROR) << "Caught gloo IO exception: " << ioe.what();
      if (status_blob_ != "") {
        signalFailure(ws_->GetBlob(status_blob_), ioe);
        return false;
      } else {
        throw ioe;
      }
    }
    return true;
  }

 protected:
  std::once_flag once_;
  std::shared_ptr<::gloo::Context> initContext_;
  std::unique_ptr<::gloo::Algorithm> algorithm_;
  Workspace* ws_;
  std::string status_blob_;
};
} // namespace gloo
} // namespace caffe2
