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

#include <gtest/gtest.h>
#include "caffe2/core/context.h"
#include "caffe2/core/event.h"

namespace caffe2 {

TEST(EventCPUTest, EventBasics) {
  DeviceOption device_option;
  device_option.set_device_type(CPU);
  Event event(device_option);
  CPUContext context;

  context.Record(&event);
  event.SetFinished();

  context.WaitEvent(event);
  event.Finish();

  event.Reset();
  event.Record(CPU, &context);
  event.SetFinished();
  event.Wait(CPU, &context);
}

} // namespace caffe2
