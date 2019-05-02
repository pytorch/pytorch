#ifndef CAFFE2_CORE_EVENT_H_
#define CAFFE2_CORE_EVENT_H_

#include <chrono>

#include <c10/core/DeviceType.h>
#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/proto/caffe2_pb.h"

namespace caffe2 {

constexpr int MaxDeviceTypes =
    DeviceTypeProto::PROTO_COMPILE_TIME_MAX_DEVICE_TYPES;
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

// Sets callback that is called when event is finished
typedef std::function<void()> EventCallbackFunction;
typedef void (*EventSetCallbackFunction)(Event*, EventCallbackFunction);

class CAFFE2_API Event {
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
      DeviceType recorder_type,
      const void* context,
      const char* err_msg = nullptr) {
    auto recorder_index = TypeToProto(recorder_type);
    CAFFE_ENFORCE_EQ(
        recorder_index,
        type_,
        "You are trying to record with a wrong device type.");
    CAFFE_ENFORCE(event_recorder_[recorder_index]);
    event_recorder_[recorder_index](this, context, err_msg);
  }

  void Wait(DeviceType waiter_type, void* context) const {
    auto waiter_index = TypeToProto(waiter_type);
    CAFFE_ENFORCE(event_waiter_[waiter_index][type_]);
    event_waiter_[waiter_index][type_](this, context);
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
#ifdef CAFFE2_USE_EXCEPTION_PTR
    caught_exception_ = nullptr;
    exception_timestamp_ = 0;
#endif // CAFFE2_USE_EXCEPTION_PTR
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

  bool SupportsCallback() const {
    return event_callback_setter_[type_] != nullptr;
  }

  void SetCallback(EventCallbackFunction callback) {
    CAFFE_ENFORCE(
        event_callback_setter_[type_], "Event does not support callbacks");
    event_callback_setter_[type_](this, callback);
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

  void SetFinishedWithException(const char* err_msg = nullptr) {
#ifdef CAFFE2_USE_EXCEPTION_PTR
    if (!caught_exception_) {
      caught_exception_ = std::current_exception();
      typedef std::chrono::high_resolution_clock clock;
      exception_timestamp_ =
          clock::now().time_since_epoch() / std::chrono::milliseconds(1);
    }
    CAFFE_ENFORCE(caught_exception_, "No exception found");
#else
    VLOG(1) << "No support for exceptions in Event";
#endif // CAFFE2_USE_EXCEPTION_PTR
    if (err_msg) {
      SetFinished(err_msg);
    } else {
      SetFinished("Error happened during an operator run");
    }
  }

  bool HasException() const {
#ifdef CAFFE2_USE_EXCEPTION_PTR
    return (bool)caught_exception_;
#else
    VLOG(1) << "No support for exceptions in Event";
    return false;
#endif // CAFFE2_USE_EXCEPTION_PTR
  }

  int64_t ExceptionTimestamp() const {
#ifdef CAFFE2_USE_EXCEPTION_PTR
    return exception_timestamp_;
#else
    VLOG(1) << "No support for exceptions in Event";
    return 0;
#endif // CAFFE2_USE_EXCEPTION_PTR
  }

  void RethrowException() const {
#ifdef CAFFE2_USE_EXCEPTION_PTR
    if (caught_exception_) {
      std::rethrow_exception(caught_exception_);
    }
#else
    VLOG(1) << "No support for exceptions in Event";
#endif // CAFFE2_USE_EXCEPTION_PTR
  }

  // event_ is going to be accessed by the EventCreate/Record/Wait/Finish
  // functions, but one should not use it outside the own Event functionalities.
  // In the future we may move it to a private member.
  std::shared_ptr<void> event_;

 private:
  int type_;
  DeviceOption option_;

#ifdef CAFFE2_USE_EXCEPTION_PTR
  std::exception_ptr caught_exception_;
  int64_t exception_timestamp_{};
#endif // CAFFE2_USE_EXCEPTION_PTR

  static EventCreateFunction event_creator_[MaxDeviceTypes];
  static EventRecordFunction event_recorder_[MaxDeviceTypes];
  static EventWaitFunction event_waiter_[MaxDeviceTypes]
                                        [MaxDeviceTypes];
  static EventFinishFunction event_finisher_[MaxDeviceTypes];

  static EventQueryFunction event_querier_[MaxDeviceTypes];
  static EventErrorMessageFunction
      event_err_msg_getter_[MaxDeviceTypes];
  static EventSetFinishedFunction
      event_finished_setter_[MaxDeviceTypes];
  static EventResetFunction event_resetter_[MaxDeviceTypes];

  static EventSetCallbackFunction event_callback_setter_[MaxDeviceTypes];

  template <DeviceType t>
  friend struct EventCreateFunctionRegisterer;
  template <DeviceType t>
  friend struct EventRecordFunctionRegisterer;
  template <DeviceType w, DeviceType d>
  friend struct EventWaitFunctionRegisterer;
  template <DeviceType t>
  friend struct EventFinishFunctionRegisterer;

  template <DeviceType t>
  friend struct EventQueryFunctionRegisterer;
  template <DeviceType t>
  friend struct EventErrorMessageFunctionRegisterer;
  template <DeviceType t>
  friend struct EventSetFinishedFunctionRegisterer;
  template <DeviceType t>
  friend struct EventSetCallbackFunctionRegisterer;
  template <DeviceType t>
  friend struct EventResetFunctionRegisterer;
};

template <DeviceType t>
struct EventCreateFunctionRegisterer {
  explicit EventCreateFunctionRegisterer(EventCreateFunction f) {
    auto d = TypeToProto(t);
    Event::event_creator_[d] = f;
  }
};
#define REGISTER_EVENT_CREATE_FUNCTION(t, f)                     \
  namespace {                                                    \
  static EventCreateFunctionRegisterer<t> g_event_create_##d(f); \
  }

template <DeviceType t>
struct EventRecordFunctionRegisterer {
  explicit EventRecordFunctionRegisterer(EventRecordFunction f) {
    auto d = TypeToProto(t);
    Event::event_recorder_[d] = f;
  }
};
#define REGISTER_EVENT_RECORD_FUNCTION(t, f)                     \
  namespace {                                                    \
  static EventRecordFunctionRegisterer<t> g_event_record_##d(f); \
  }

template <DeviceType waiter_type, DeviceType event_type>
struct EventWaitFunctionRegisterer {
  explicit EventWaitFunctionRegisterer(EventWaitFunction f) {
    auto waiter_index = TypeToProto(waiter_type);
    auto event_index = TypeToProto(event_type);
    Event::event_waiter_[waiter_index][event_index] = f;
  }
};
#define REGISTER_EVENT_WAIT_FUNCTION(w, d, f)                         \
  namespace {                                                         \
  static EventWaitFunctionRegisterer<w, d> g_event_wait_##w##_##d(f); \
  }

template <DeviceType t>
struct EventQueryFunctionRegisterer {
  explicit EventQueryFunctionRegisterer(EventQueryFunction f) {
    auto d = TypeToProto(t);
    Event::event_querier_[d] = f;
  }
};
#define REGISTER_EVENT_QUERY_FUNCTION(t, f)                    \
  namespace {                                                  \
  static EventQueryFunctionRegisterer<t> g_event_query_##d(f); \
  }

template <DeviceType t>
struct EventErrorMessageFunctionRegisterer {
  explicit EventErrorMessageFunctionRegisterer(EventErrorMessageFunction f) {
    auto d = TypeToProto(t);
    Event::event_err_msg_getter_[d] = f;
  }
};
#define REGISTER_EVENT_ERROR_MESSAGE_FUNCTION(t, f)                     \
  namespace {                                                           \
  static EventErrorMessageFunctionRegisterer<t> g_event_err_msg_##d(f); \
  }

template <DeviceType t>
struct EventSetFinishedFunctionRegisterer {
  explicit EventSetFinishedFunctionRegisterer(EventSetFinishedFunction f) {
    auto d = TypeToProto(t);
    Event::event_finished_setter_[d] = f;
  }
};
#define REGISTER_EVENT_SET_FINISHED_FUNCTION(t, f)                          \
  namespace {                                                               \
  static EventSetFinishedFunctionRegisterer<t> g_event_set_finished_##d(f); \
  }

template <DeviceType t>
struct EventSetCallbackFunctionRegisterer {
  explicit EventSetCallbackFunctionRegisterer(EventSetCallbackFunction f) {
    auto d = TypeToProto(t);
    Event::event_callback_setter_[d] = f;
  }
};
#define REGISTER_EVENT_SET_CALLBACK_FUNCTION(t, f)                          \
  namespace {                                                               \
  static EventSetCallbackFunctionRegisterer<t> g_event_set_callback_##d(f); \
  }

template <DeviceType t>
struct EventFinishFunctionRegisterer {
  explicit EventFinishFunctionRegisterer(EventFinishFunction f) {
    auto d = TypeToProto(t);
    Event::event_finisher_[d] = f;
  }
};
#define REGISTER_EVENT_FINISH_FUNCTION(t, f)                     \
  namespace {                                                    \
  static EventFinishFunctionRegisterer<t> g_event_finish_##d(f); \
  }

template <DeviceType t>
struct EventResetFunctionRegisterer {
  explicit EventResetFunctionRegisterer(EventResetFunction f) {
    auto d = TypeToProto(t);
    Event::event_resetter_[d] = f;
  }
};
#define REGISTER_EVENT_RESET_FUNCTION(t, f)                    \
  namespace {                                                  \
  static EventResetFunctionRegisterer<t> g_event_reset_##d(f); \
  }

} // namespace caffe2

#endif // CAFFE2_CORE_EVENT_H_
