#include "caffe2/core/hip/context_hip.h"
#include "caffe2/core/event_cpu.h"
#include "caffe2/core/operator.h"

#include <atomic>

namespace caffe2 {

struct HipEventWrapper
{
    explicit HipEventWrapper(const DeviceOption& option)
        : hip_stream_(nullptr),
          hip_gpu_id_(option.hip_gpu_id()),
          status_(EventStatus::EVENT_INITIALIZED)
    {
        CAFFE_ENFORCE(option.device_type(), HIP);
        DeviceGuard g(hip_gpu_id_);
        HIP_ENFORCE(hipEventCreate(&hip_event_ /*, hipEventDefault | hipEventDisableTiming*/));
    }
    ~HipEventWrapper()
    {
        DeviceGuard g(hip_gpu_id_);
        HIP_CHECK(hipEventDestroy(hip_event_));
    }

    hipEvent_t hip_event_;
    hipStream_t hip_stream_;
    int hip_gpu_id_;

    std::atomic<int> status_;
    std::mutex mutex_recorded_;
    std::condition_variable cv_recorded_;
    std::string err_msg_;
};

namespace {
const std::string kNoError = "No error";
}

void EventCreateHIP(const DeviceOption& option, Event* event)
{
    event->event_ = std::make_shared<HipEventWrapper>(option);
}

void EventRecordHIP(Event* event, const void* context, const char* err_msg)
{
    auto* wrapper = static_cast<HipEventWrapper*>(event->event_.get());
    {
        std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);

        // Possible state changes:
        //  INITIALIZED -> SCHEDULED/FAILED
        //  SCHEDULED -> SUCCESS/FAILED
        //  SUCCESS/FAILED - terminal
        //
        // No further changes to cuda_event_ and cuda_stream_ after transitioning
        // from INITIALIZED
        // No further changes to err_msg_ after transitioning into FAILED

        CAFFE_ENFORCE_EQ(
            wrapper->status_, EventStatus::EVENT_INITIALIZED, "Calling Record multiple times");

        if(!err_msg)
        {
            // When recording, one needs to make sure that the current gpu id is
            // correct.
            // TODO(jiayq): move the enforce logic to the caller?
            const auto& current_device = CaffeHipGetDevice();
            CAFFE_ENFORCE_EQ(current_device,
                             wrapper->hip_gpu_id_,
                             "When you call EventRecordHIP, your current device should be the same "
                             "as the device specified by the event.");
            CAFFE_ENFORCE_EQ(current_device, static_cast<const HIPContext*>(context)->hip_gpu_id());
            HIP_ENFORCE(hipEventRecord(wrapper->hip_event_,
                                       static_cast<const HIPContext*>(context)->hip_stream()));
            wrapper->hip_stream_ = static_cast<const HIPContext*>(context)->hip_stream();
            wrapper->status_     = EventStatus::EVENT_SCHEDULED;
        }
        else
        {
            wrapper->err_msg_ = err_msg;
            wrapper->status_  = EventStatus::EVENT_FAILED;
        }
    }
    wrapper->cv_recorded_.notify_all();
}

void EventFinishHIP(const Event* event)
{
    auto* wrapper = static_cast<HipEventWrapper*>(event->event_.get());
    {
        std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);
        while(wrapper->status_ == EventStatus::EVENT_INITIALIZED)
        {
            wrapper->cv_recorded_.wait(lock);
        }
    }

    if(wrapper->status_ == EventStatus::EVENT_SCHEDULED)
    {
        // ok, even if event is already completed and status was not yet updated
        DeviceGuard g(wrapper->hip_gpu_id_);
        auto hipResult = hipEventSynchronize(wrapper->hip_event_);
        if(hipResult == hipSuccess)
        {
            wrapper->status_ = EventStatus::EVENT_SUCCESS;
        }
        else
        {
            const auto& err_msg = hipGetErrorString(hipResult);

            std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);
            wrapper->err_msg_ = err_msg;
            wrapper->status_  = EventStatus::EVENT_FAILED;
        }
    }
}

// Both waiter and event are HIP. Non-blocking
void EventWaitHIPHIP(const Event* event, void* context)
{
    auto* wrapper = static_cast<HipEventWrapper*>(event->event_.get());
    {
        std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);
        while(wrapper->status_ == EventStatus::EVENT_INITIALIZED)
        {
            wrapper->cv_recorded_.wait(lock);
        }
    }

    if(wrapper->status_ == EventStatus::EVENT_SCHEDULED)
    {
        // ok, even if event is already completed and status was not yet updated
        auto context_stream = static_cast<HIPContext*>(context)->hip_stream();
        auto event_stream   = wrapper->hip_stream_;
        if(context_stream != event_stream)
        {
            // CAFFE_ENFORCE_EQ(
            //    CaffeCudaGetDevice(),
            //    static_cast<const CUDAContext*>(context)->cuda_gpu_id());
            HIP_CHECK(hipStreamWaitEvent(context_stream, wrapper->hip_event_, 0));
        }
    }
}

// Waiter is CPU, event is HIP
void EventWaitCPUHIP(const Event* event, void* context) { EventFinishHIP(event); }

// Waiter is HIP, event is CPU
void EventWaitHIPCPU(const Event* event, void* context)
{
    event->Finish(); // calls EventFinishCPU
}

EventStatus EventQueryHIP(const Event* event)
{
    auto* wrapper = static_cast<HipEventWrapper*>(event->event_.get());
    if(wrapper->status_ == EventStatus::EVENT_SCHEDULED)
    {
        auto hipResult = hipEventQuery(wrapper->hip_event_);
        if(hipResult == hipSuccess)
        {
            wrapper->status_ = EventStatus::EVENT_SUCCESS;
        }
        else if(hipResult != hipErrorNotReady)
        {
            const auto& err_msg = hipGetErrorString(hipResult);

            std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);
            wrapper->err_msg_ = err_msg;
            wrapper->status_  = EventStatus::EVENT_FAILED;
        }
    }
    return static_cast<EventStatus>(wrapper->status_.load());
}

const std::string& EventErrorMessageHIP(const Event* event)
{
    auto* wrapper = static_cast<HipEventWrapper*>(event->event_.get());
    // supposed to be called after EventQueryCUDA to update status first
    if(wrapper->status_ == EventStatus::EVENT_FAILED)
    {
        return wrapper->err_msg_;
    }
    else
    {
        return kNoError;
    }
}

void EventSetFinishedHIP(const Event* event, const char* err_msg)
{
    auto* wrapper = static_cast<HipEventWrapper*>(event->event_.get());
    {
        std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);

        CAFFE_ENFORCE_EQ(wrapper->status_,
                         EventStatus::EVENT_INITIALIZED,
                         "Calling SetFinished on recorded HIP event");

        if(!err_msg)
        {
            wrapper->status_ = EventStatus::EVENT_SUCCESS;
        }
        else
        {
            wrapper->err_msg_ = err_msg;
            wrapper->status_  = EventStatus::EVENT_FAILED;
        }
    }
    wrapper->cv_recorded_.notify_all();
}

void EventResetHIP(Event* event)
{
    auto* wrapper = static_cast<HipEventWrapper*>(event->event_.get());
    std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);
    wrapper->status_     = EventStatus::EVENT_INITIALIZED;
    wrapper->err_msg_    = "";
    wrapper->hip_stream_ = nullptr;
}

REGISTER_EVENT_CREATE_FUNCTION(HIP, EventCreateHIP);
REGISTER_EVENT_RECORD_FUNCTION(HIP, EventRecordHIP);
REGISTER_EVENT_WAIT_FUNCTION(HIP, HIP, EventWaitHIPHIP);
REGISTER_EVENT_WAIT_FUNCTION(CPU, HIP, EventWaitCPUHIP);
REGISTER_EVENT_WAIT_FUNCTION(HIP, CPU, EventWaitHIPCPU);
REGISTER_EVENT_FINISH_FUNCTION(HIP, EventFinishHIP);

REGISTER_EVENT_QUERY_FUNCTION(HIP, EventQueryHIP);
REGISTER_EVENT_ERROR_MESSAGE_FUNCTION(HIP, EventErrorMessageHIP);
REGISTER_EVENT_SET_FINISHED_FUNCTION(HIP, EventSetFinishedHIP);
REGISTER_EVENT_RESET_FUNCTION(HIP, EventResetHIP);

REGISTER_EVENT_WAIT_FUNCTION(MKLDNN, HIP, EventWaitCPUHIP);
REGISTER_EVENT_WAIT_FUNCTION(HIP, MKLDNN, EventWaitHIPCPU);

} // namespace caffe2
