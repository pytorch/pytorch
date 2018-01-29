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

void TimeObserver::Start() {
  start_time_ = timer_.MilliSeconds();
  ++iterations_;
}

void TimeObserver::Stop() {
  double current_run = timer_.MilliSeconds() - start_time_;
  total_time_ += current_run;
  VLOG(1) << "This net iteration took " << current_run << " ms to complete.\n";
}

void TimeOperatorObserver::Start() {
  start_time_ = timer_.MilliSeconds();
  ++iterations_;
}

void TimeOperatorObserver::Stop() {
  double current_run = timer_.MilliSeconds() - start_time_;
  total_time_ += current_run;
  VLOG(1) << "This operator iteration took " << current_run
          << " ms to complete.\n";
}

} // namespace caffe2
