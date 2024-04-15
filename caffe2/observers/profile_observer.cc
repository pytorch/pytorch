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

#include "profile_observer.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

void ProfileOperatorObserver::Dump() const {
  static std::mutex loggingMutex;
  std::lock_guard<std::mutex> lock(loggingMutex);

  LOG(INFO) << "--------- Starting operator " << subject_->debug_def().type()
            << " op#" << getId() << " ---------";
  for (int i = 0; i < subject_->InputSize(); ++i) {
    if (subject_->InputIsTensorType(i, CPU)) {
      const auto& tensor = subject_->Input<Tensor>(i, CPU);
      const auto& name = subject_->debug_def().input(i);
      TensorPrinter printer(name);
      LOG(INFO) << "Input " << i << ": " << printer.MetaStr(tensor);
    } else if (subject_->InputIsTensorType(i, CUDA)) {
      const auto& tensor = subject_->Input<Tensor>(i, CUDA);
      const auto& name = subject_->debug_def().input(i);
      TensorPrinter printer(name);
      LOG(INFO) << "Input " << i << ": " << printer.MetaStr(tensor);
    }
  }

  int a = 0;
  for (const auto& arg : subject_->debug_def().arg()) {
    LOG(INFO) << "Argument " << a << ": " << arg.ShortDebugString();
    ++a;
  }

  for (int o = 0; o < subject_->OutputSize(); ++o) {
    if (subject_->OutputIsTensorType(o, CPU)) {
      auto* tensor = subject_->Output<Tensor>(o, CPU);
      const auto& name = subject_->debug_def().output(o);
      TensorPrinter printer(name);
      LOG(INFO) << "Output " << o << ": " << printer.MetaStr(*tensor);
    } else if (subject_->OutputIsTensorType(o, CUDA)) {
      auto* tensor = subject_->Output<Tensor>(o, CUDA);
      const auto& name = subject_->debug_def().output(o);
      TensorPrinter printer(name);
      LOG(INFO) << "Output " << o << ": " << printer.MetaStr(*tensor);
    }
  }

  LOG(INFO) << "--------- Finished operator " << subject_->debug_def().type()
            << " in " << run_time_ << " ms ---------";
}

void ProfileOperatorObserver::Start() {
  start_time_ = timer_.MilliSeconds();
}

void ProfileOperatorObserver::Stop() {
  run_time_ = timer_.MilliSeconds() - start_time_;
  Dump();
}

std::unique_ptr<ObserverBase<OperatorBase>> ProfileOperatorObserver::rnnCopy(
    OperatorBase* subject,
    int rnn_order) const {
  return std::unique_ptr<ObserverBase<OperatorBase>>(
      new ProfileOperatorObserver(
          subject, netObserver_, net_position_, rnn_order));
}
} // namespace caffe2
