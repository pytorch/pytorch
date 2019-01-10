#ifndef CAFFE2_CORE_EVENT_H_
#define CAFFE2_CORE_EVENT_H_

#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {

constexpr int MaxDeviceTypes = DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES;
class Event;

enum EventStatus {
  EVENT_INITIALIZED = 0,
  EVENT_SCHEDULED = 1,
  EVENT_SUCCESS = 2,
  EVENT_FAILED = 3,
};

// For the following functions, void* shall be interpreted as the corresponding
// context object corresponding to the device type associated with the
// functions.

// Initializes event
typedef void (*EventCreateFunction)(const DeviceOption& option, Event*);

// Called on event to signal that CPU part of operation is finished,
// Optionally accepts error message from CPU part.
// Should be called no more than once per event
typedef void (*EventRecordFunction)(Event*, const void*, const char*);

// Waits and returns as soon as possible in order schedule next operation,
// e.g. for CUDA->CUDA waits only for CPU part of CUDA op,
// for CUDA->CPU waits till the CUDA op is fully completed.
// Prepares context to synchronize device part of operation.
// Can be called concurrently from multiple threads
typedef void (*EventWaitFunction)(const Event*, void*);

// Waits till operation is fully finished,
// can be called concurrently from multiple threads
typedef void (*EventFinishFunction)(const Event*);

// Queries current status of operation,
// can be called concurrently from multiple threads
typedef EventStatus (*EventQueryFunction)(const Event*);
typedef const std::string& (*EventErrorMessageFunction)(const Event*);
typedef void (*EventSetFinishedFunction)(const Event*, const char*);
typedef void (*EventResetFunction)(Event*);

class Event {
 public:
  explicit Event(const DeviceOption& option)
      : event_(), type_(option.device_type()), option_(option) {
    CAFFE_ENFORCE_LT(type_, MaxDeviceTypes);
    CAFFE_ENFORCE(event_creator_[type_]);
    event_creator_[type_](option, this);
  }

  // Nothing needs to be done in the destructor, as the event creator should
  // set the proper destruction process for the unique_ptr.
  ~Event() {}

  void Record(
      int recorder_type,
      const void* context,
      const char* err_msg = nullptr) {
    CAFFE_ENFORCE_EQ(
        recorder_type,
        type_,
        "You are trying to record with a wrong device type.");
    CAFFE_ENFORCE(event_recorder_[recorder_type]);
    event_recorder_[recorder_type](this, context, err_msg);
  }

  void Wait(int waiter_type, void* context) const {
    CAFFE_ENFORCE(event_waiter_[waiter_type][type_]);
    event_waiter_[waiter_type][type_](this, context);
  }

  void Finish() const {
    CAFFE_ENFORCE(event_finisher_[type_]);
    event_finisher_[type_](this);
  }

  EventStatus Query() const {
    CAFFE_ENFORCE(event_querier_[type_]);
    return event_querier_[type_](this);
  }

  const std::string& ErrorMessage() const {
    CAFFE_ENFORCE(event_err_msg_getter_[type_]);
    return event_err_msg_getter_[type_](this);
  }

  void Reset() {
    CAFFE_ENFORCE(event_resetter_[type_]);
    event_resetter_[type_](this);
  }

  const DeviceOption& GetDeviceOption() const {
    return option_;
  }

  bool IsScheduled() const {
    return Query() == EventStatus::EVENT_SCHEDULED;
  }

  bool IsFinished() const {
    auto status = Query();
    return status == EventStatus::EVENT_SUCCESS ||
        status == EventStatus::EVENT_FAILED;
  }

  void SetFinished(const char* err_msg = nullptr) {
    CAFFE_ENFORCE(event_finished_setter_[type_]);
    return event_finished_setter_[type_](this, err_msg);
  }

  // If parent op has succeeded, then we can run any child op;
  // If parent op is in scheduled state, we need to check that:
  //  - child op supports async scheduling
  //  - there's a way to setup synchronization between async parent and
  //    child - both child and parent should use the same type of device,
  //    non-blocking synchronization between different device types is not
  //    supported
  // If parent op is in another state (initialized or failed) then scheduling
  // is not possible
  bool CanSchedule(const Event& child_event, bool supports_async) const {
    return CanSchedule(type_, Query(), child_event.GetType(), supports_async);
  }

  static bool CanSchedule(
      int parent_type,
      EventStatus parent_status,
      int child_type,
      bool child_supports_async) {
    if (parent_status == EventStatus::EVENT_SUCCESS) {
      return true;
    }
    if (parent_status == EventStatus::EVENT_SCHEDULED) {
      return (parent_type == child_type) && child_supports_async;
    }
    return false;
  }

  int GetType() const {
    return type_;
  }

  // event_ is going to be accessed by the EventCreate/Record/Wait/Finish
  // functions, but one should not use it outside the own Event functionalities.
  // In the future we may move it to a private member.
  std::shared_ptr<void> event_;

 private:
  int type_;
  DeviceOption option_;

  CAFFE2_API static EventCreateFunction event_creator_[MaxDeviceTypes];
  CAFFE2_API static EventRecordFunction event_recorder_[MaxDeviceTypes];
  CAFFE2_API static EventWaitFunction event_waiter_[MaxDeviceTypes]
                                                   [MaxDeviceTypes];
  CAFFE2_API static EventFinishFunction event_finisher_[MaxDeviceTypes];

  CAFFE2_API static EventQueryFunction event_querier_[MaxDeviceTypes];
  CAFFE2_API static EventErrorMessageFunction
      event_err_msg_getter_[MaxDeviceTypes];
  CAFFE2_API static EventSetFinishedFunction
      event_finished_setter_[MaxDeviceTypes];
  CAFFE2_API static EventResetFunction event_resetter_[MaxDeviceTypes];

  template <int d>
  friend struct EventCreateFunctionRegisterer;
  template <int d>
  friend struct EventRecordFunctionRegisterer;
  template <int w, int d>
  friend struct EventWaitFunctionRegisterer;
  template <int d>
  friend struct EventFinishFunctionRegisterer;

  template <int d>
  friend struct EventQueryFunctionRegisterer;
  template <int d>
  friend struct EventErrorMessageFunctionRegisterer;
  template <int d>
  friend struct EventSetFinishedFunctionRegisterer;
  template <int d>
  friend struct EventResetFunctionRegisterer;
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
#define REGISTER_EVENT_WAIT_FUNCTION(w, d, f)                         \
  namespace {                                                         \
  static EventWaitFunctionRegisterer<w, d> g_event_wait_##w##_##d(f); \
  }

template <int d>
struct EventQueryFunctionRegisterer {
  explicit EventQueryFunctionRegisterer(EventQueryFunction f) {
    static_assert(d < MaxDeviceTypes, "");
    Event::event_querier_[d] = f;
  }
};
#define REGISTER_EVENT_QUERY_FUNCTION(d, f)                    \
  namespace {                                                  \
  static EventQueryFunctionRegisterer<d> g_event_query_##d(f); \
  }

template <int d>
struct EventErrorMessageFunctionRegisterer {
  explicit EventErrorMessageFunctionRegisterer(EventErrorMessageFunction f) {
    static_assert(d < MaxDeviceTypes, "");
    Event::event_err_msg_getter_[d] = f;
  }
};
#define REGISTER_EVENT_ERROR_MESSAGE_FUNCTION(d, f)                     \
  namespace {                                                           \
  static EventErrorMessageFunctionRegisterer<d> g_event_err_msg_##d(f); \
  }

template <int d>
struct EventSetFinishedFunctionRegisterer {
  explicit EventSetFinishedFunctionRegisterer(EventSetFinishedFunction f) {
    static_assert(d < MaxDeviceTypes, "");
    Event::event_finished_setter_[d] = f;
  }
};
#define REGISTER_EVENT_SET_FINISHED_FUNCTION(d, f)                          \
  namespace {                                                               \
  static EventSetFinishedFunctionRegisterer<d> g_event_set_finished_##d(f); \
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

template <int d>
struct EventResetFunctionRegisterer {
  explicit EventResetFunctionRegisterer(EventResetFunction f) {
    static_assert(d < MaxDeviceTypes, "");
    Event::event_resetter_[d] = f;
  }
};
#define REGISTER_EVENT_RESET_FUNCTION(d, f)                    \
  namespace {                                                  \
  static EventResetFunctionRegisterer<d> g_event_reset_##d(f); \
  }

} // namespace caffe2

#endif // CAFFE2_CORE_EVENT_H_
