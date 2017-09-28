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

#ifndef CAFFE2_CORE_EVENT_H_
#define CAFFE2_CORE_EVENT_H_

#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {

constexpr int MaxDeviceTypes = DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES;
class Event;

// For the following functions, void* shall be interpreted as the corresponding
// context object corresponding to the device type associated with the
// functions.
typedef void (*EventCreateFunction)(const DeviceOption& option, Event*);
typedef void (*EventRecordFunction)(const void*, Event*);
typedef void (*EventWaitFunction)(const Event*, void*);
typedef void (*EventFinishFunction)(const Event*);

class Event {
 public:
  explicit Event(const DeviceOption& option)
      : event_(), type_(option.device_type()) {
    CAFFE_ENFORCE_LT(type_, MaxDeviceTypes);
    CAFFE_ENFORCE(event_creator_[type_]);
    event_creator_[type_](option, this);
  }

  // Nothing needs to be done in the destructor, as the event creator should
  // set the proper destruction process for the unique_ptr.
  ~Event() {}

  void Record(int recorder_type, const void* context) {
    CAFFE_ENFORCE_EQ(
        recorder_type,
        type_,
        "You are trying to record with a wrong device type.");
    CAFFE_ENFORCE(event_recorder_[recorder_type]);
    event_recorder_[recorder_type](context, this);
  }

  void Wait(int waiter_type, void* context) const {
    CAFFE_ENFORCE(event_waiter_[waiter_type][type_]);
    event_waiter_[waiter_type][type_](this, context);
  }

  void Finish() const {
    CAFFE_ENFORCE(event_finisher_[type_]);
    event_finisher_[type_](this);
  }

  // event_ is going to be accessed by the EventCreate/Record/Wait/Finish
  // functions, but one should not use it outside the own Event functionalities.
  // In the future we may move it to a private member.
  std::shared_ptr<void> event_;

 private:
  int type_;
  static EventCreateFunction event_creator_[MaxDeviceTypes];
  static EventRecordFunction event_recorder_[MaxDeviceTypes];
  static EventWaitFunction event_waiter_[MaxDeviceTypes][MaxDeviceTypes];
  static EventFinishFunction event_finisher_[MaxDeviceTypes];

  template <int d>
  friend struct EventCreateFunctionRegisterer;
  template <int d>
  friend struct EventRecordFunctionRegisterer;
  template <int w, int d>
  friend struct EventWaitFunctionRegisterer;
  template <int d>
  friend struct EventFinishFunctionRegisterer;
};

template <int d>
struct EventCreateFunctionRegisterer {
  explicit EventCreateFunctionRegisterer(EventCreateFunction f) {
    static_assert(d < MaxDeviceTypes, "");
    Event::event_creator_[d] = f;
  }
};
#define REGISTER_EVENT_CREATE_FUNCTION(d, f)                     \
  namespace {                                                    \
  static EventCreateFunctionRegisterer<d> g_event_create_##d(f); \
  }

template <int d>
struct EventRecordFunctionRegisterer {
  explicit EventRecordFunctionRegisterer(EventRecordFunction f) {
    static_assert(d < MaxDeviceTypes, "");
    Event::event_recorder_[d] = f;
  }
};
#define REGISTER_EVENT_RECORD_FUNCTION(d, f)                     \
  namespace {                                                    \
  static EventRecordFunctionRegisterer<d> g_event_record_##d(f); \
  }

template <int waiter_type, int event_type>
struct EventWaitFunctionRegisterer {
  explicit EventWaitFunctionRegisterer(EventWaitFunction f) {
    static_assert(waiter_type < MaxDeviceTypes, "");
    static_assert(event_type < MaxDeviceTypes, "");
    Event::event_waiter_[waiter_type][event_type] = f;
  }
};
#define REGISTER_EVENT_WAIT_FUNCTION(w, d, f)                           \
  namespace {                                                           \
  static EventWaitFunctionRegisterer<w, d> g_event_record_##w##_##d(f); \
  }

template <int d>
struct EventFinishFunctionRegisterer {
  explicit EventFinishFunctionRegisterer(EventFinishFunction f) {
    static_assert(d < MaxDeviceTypes, "");
    Event::event_finisher_[d] = f;
  }
};
#define REGISTER_EVENT_FINISH_FUNCTION(d, f)                     \
  namespace {                                                    \
  static EventFinishFunctionRegisterer<d> g_event_finish_##d(f); \
  }

} // namespace caffe2

#endif // CAFFE2_CORE_EVENT_H_
