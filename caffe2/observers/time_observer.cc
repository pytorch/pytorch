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

#include "time_observer.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

template <>
bool TimeObserverBase<NetBase>::Start() {
  CAFFE_THROW(
      "This function is overridden by TimeObserver<NetBase>.\
              If it was called there is an issue with compilation.");
  return false;
}

template <>
bool TimeObserverBase<NetBase>::Stop() {
  double current_run = timer_.MilliSeconds() - start_time_;
  total_time_ += current_run;
  VLOG(1) << "This net iteration took " << current_run << " ms to complete.\n";
  return true;
}

template <>
bool TimeObserverBase<OperatorBase>::Start() {
  start_time_ = timer_.MilliSeconds();
  ++iterations_;
  return true;
}

template <>
bool TimeObserverBase<OperatorBase>::Stop() {
  double current_run = timer_.MilliSeconds() - start_time_;
  total_time_ += current_run;
  VLOG(1) << "This operator iteration took " << current_run
          << " ms to complete.\n";
  return true;
}

} // namespace caffe2
