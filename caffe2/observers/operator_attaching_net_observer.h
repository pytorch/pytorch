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

// Thin class that attaches the observer to all operators in the net
template <typename TOpObserver, typename TNetObserver>
class OperatorAttachingNetObserver : public ObserverBase<NetBase> {
 public:
  explicit OperatorAttachingNetObserver(
      NetBase* subject_,
      TNetObserver* netObserver)
      : ObserverBase<NetBase>(subject_) {
    const auto& operators = subject_->GetOperators();
    for (auto* op : operators) {
      auto observer = caffe2::make_unique<TOpObserver>(op, netObserver);
      const auto* ob = observer.get();
      op->AttachObserver(std::move(observer));
      operator_observers_.push_back(ob);
    }
  }
  virtual ~OperatorAttachingNetObserver(){};

 protected:
  std::vector<const TOpObserver*> operator_observers_;
};

} // namespace caffe2
