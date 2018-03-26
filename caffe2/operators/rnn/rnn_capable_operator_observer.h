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

#include "caffe2/core/net.h"
#include "caffe2/core/observer.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

/**
 *  Inherit to make your class observable.
 */
class RNNCapableOperatorObserver : public ObserverBase<OperatorBase> {
 public:
  explicit RNNCapableOperatorObserver(OperatorBase* op)
      : ObserverBase<OperatorBase>(op){};

  virtual std::unique_ptr<ObserverBase<OperatorBase>> rnnCopy(
      OperatorBase* subject,
      int rnn_order) const = 0;
  virtual ~RNNCapableOperatorObserver(){};

 protected:
  int rnn_order_ = OperatorBase::kNoNetPositionSet;
};

} // namespace caffe2
