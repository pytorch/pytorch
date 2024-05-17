#include "caffe2/core/context_gpu.h"
#include "caffe2/core/event_cpu.h"
#include "caffe2/core/operator.h"

#include <atomic>
#include <iostream>

namespace caffe2 {

struct CudaEventWrapper {
  explicit CudaEventWrapper(const DeviceOption& option)
      : cuda_stream_(nullptr),
        device_id_(option.device_id()),
        status_(EventStatus::EVENT_INITIALIZED) {
    CAFFE_ENFORCE(option.device_type(), PROTO_CUDA);
    CUDAGuard g(device_id_);
    try {
      CUDA_ENFORCE(cudaEventCreateWithFlags(
          &cuda_event_, cudaEventDefault | cudaEventDisableTiming));
    } catch (const Error&) {
      std::cerr << "ERROR: Failed to load CUDA.\n"
                << "HINT: Check that this binary contains GPU code."
                << std::endl;
      throw;
    }
  }
  ~CudaEventWrapper() {
    CUDAGuard g(device_id_);
    CUDA_CHECK(cudaEventDestroy(cuda_event_));
  }

  cudaEvent_t cuda_event_;
  cudaStream_t cuda_stream_;
  int device_id_;

  std::atomic<int> status_;
  std::mutex mutex_recorded_;
  std::condition_variable cv_recorded_;
  std::string err_msg_;
};

namespace {
const std::string kNoError = "No error";
}

void EventCreateCUDA(const DeviceOption& option, Event* event) {
  event->event_ = std::make_shared<CudaEventWrapper>(option);
}

void EventRecordCUDA(Event* event, const void* context, const char* err_msg) {
  auto* wrapper = static_cast<CudaEventWrapper*>(event->event_.get());
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
        wrapper->status_,
        EventStatus::EVENT_INITIALIZED,
        "Calling Record multiple times");

    if (!err_msg) {
      // When recording, one needs to make sure that the current gpu id is
      // correct.
      // TODO(jiayq): move the enforce logic to the caller?
      const auto& current_device = CaffeCudaGetDevice();
      CAFFE_ENFORCE_EQ(
          current_device,
          wrapper->device_id_,
          "When you call EventRecordCUDA, your current device should be the same "
          "as the device specified by the event.");
      CAFFE_ENFORCE_EQ(
          current_device,
          static_cast<const CUDAContext*>(context)->device_id());
      CUDA_ENFORCE(cudaEventRecord(
          wrapper->cuda_event_,
          static_cast<const CUDAContext*>(context)->cuda_stream()));
      wrapper->cuda_stream_ =
          static_cast<const CUDAContext*>(context)->cuda_stream();
      wrapper->status_ = EventStatus::EVENT_SCHEDULED;
    } else {
      wrapper->err_msg_ = err_msg;
      wrapper->status_ = EventStatus::EVENT_FAILED;
    }
  }
  wrapper->cv_recorded_.notify_all();
}

void EventFinishCUDA(const Event* event) {
  auto* wrapper = static_cast<CudaEventWrapper*>(event->event_.get());
  {
    std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);
    while (wrapper->status_ == EventStatus::EVENT_INITIALIZED) {
      wrapper->cv_recorded_.wait(lock);
    }
  }

  if (wrapper->status_ == EventStatus::EVENT_SCHEDULED) {
    // ok, even if event is already completed and status was not yet updated
    CUDAGuard g(wrapper->device_id_);
    auto cudaResult = cudaEventSynchronize(wrapper->cuda_event_);
    if (cudaResult == cudaSuccess) {
      wrapper->status_ = EventStatus::EVENT_SUCCESS;
    } else {
      const auto& err_msg = cudaGetErrorString(cudaResult);

      std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);
      wrapper->err_msg_ = err_msg;
      wrapper->status_ = EventStatus::EVENT_FAILED;
    }
  }
}

// Both waiter and event are CUDA. Non-blocking
void EventWaitCUDACUDA(const Event* event, void* context) {
  auto* wrapper = static_cast<CudaEventWrapper*>(event->event_.get());
  {
    std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);
    while (wrapper->status_ == EventStatus::EVENT_INITIALIZED) {
      wrapper->cv_recorded_.wait(lock);
    }
  }

  if (wrapper->status_ == EventStatus::EVENT_SCHEDULED) {
    // ok, even if event is already completed and status was not yet updated
    auto context_stream = static_cast<CUDAContext*>(context)->cuda_stream();
    auto event_stream = wrapper->cuda_stream_;
    if (context_stream != event_stream) {
      // CAFFE_ENFORCE_EQ(
      //    CaffeCudaGetDevice(),
      //    static_cast<const CUDAContext*>(context)->device_id());
      CUDA_CHECK(cudaStreamWaitEvent(context_stream, wrapper->cuda_event_, 0));
    }
  }
}

// Waiter is CPU, event is CUDA
void EventWaitCPUCUDA(const Event* event, void* context) {
  EventFinishCUDA(event);
}

// Waiter is CUDA, event is CPU
void EventWaitCUDACPU(const Event* event, void* context) {
  event->Finish(); // calls EventFinishCPU
}

EventStatus EventQueryCUDA(const Event* event) {
  auto* wrapper = static_cast<CudaEventWrapper*>(event->event_.get());
  if (wrapper->status_ == EventStatus::EVENT_SCHEDULED) {
    auto cudaResult = cudaEventQuery(wrapper->cuda_event_);
    if (cudaResult == cudaSuccess) {
      wrapper->status_ = EventStatus::EVENT_SUCCESS;
    } else if (cudaResult != cudaErrorNotReady) {
      const auto& err_msg = cudaGetErrorString(cudaResult);

      std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);
      wrapper->err_msg_ = err_msg;
      wrapper->status_ = EventStatus::EVENT_FAILED;
    } else {
      // ignore and clear the error if not ready
      (void)cudaGetLastError();
    }
  }
  return static_cast<EventStatus>(wrapper->status_.load());
}

const std::string& EventErrorMessageCUDA(const Event* event) {
  auto* wrapper = static_cast<CudaEventWrapper*>(event->event_.get());
  // supposed to be called after EventQueryCUDA to update status first
  if (wrapper->status_ == EventStatus::EVENT_FAILED) {
    return wrapper->err_msg_;
  } else {
    return kNoError;
  }
}

void EventSetFinishedCUDA(const Event* event, const char* err_msg) {
  auto* wrapper = static_cast<CudaEventWrapper*>(event->event_.get());
  {
    std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);

    CAFFE_ENFORCE_EQ(
        wrapper->status_,
        EventStatus::EVENT_INITIALIZED,
        "Calling SetFinished on recorded CUDA event");

    if (!err_msg) {
      wrapper->status_ = EventStatus::EVENT_SUCCESS;
    } else {
      wrapper->err_msg_ = err_msg;
      wrapper->status_ = EventStatus::EVENT_FAILED;
    }
  }
  wrapper->cv_recorded_.notify_all();
}

void EventResetCUDA(Event* event) {
  auto* wrapper = static_cast<CudaEventWrapper*>(event->event_.get());
  std::unique_lock<std::mutex> lock(wrapper->mutex_recorded_);
  wrapper->status_ = EventStatus::EVENT_INITIALIZED;
  wrapper->err_msg_ = "";
  wrapper->cuda_stream_ = nullptr;
}

REGISTER_EVENT_CREATE_FUNCTION(CUDA, EventCreateCUDA);
REGISTER_EVENT_RECORD_FUNCTION(CUDA, EventRecordCUDA);
REGISTER_EVENT_WAIT_FUNCTION(CUDA, CUDA, EventWaitCUDACUDA);
REGISTER_EVENT_WAIT_FUNCTION(CPU, CUDA, EventWaitCPUCUDA);
REGISTER_EVENT_WAIT_FUNCTION(CUDA, CPU, EventWaitCUDACPU);
REGISTER_EVENT_FINISH_FUNCTION(CUDA, EventFinishCUDA);

REGISTER_EVENT_QUERY_FUNCTION(CUDA, EventQueryCUDA);
REGISTER_EVENT_ERROR_MESSAGE_FUNCTION(CUDA, EventErrorMessageCUDA);
REGISTER_EVENT_SET_FINISHED_FUNCTION(CUDA, EventSetFinishedCUDA);
REGISTER_EVENT_RESET_FUNCTION(CUDA, EventResetCUDA);

REGISTER_EVENT_WAIT_FUNCTION(MKLDNN, CUDA, EventWaitCPUCUDA);
REGISTER_EVENT_WAIT_FUNCTION(CUDA, MKLDNN, EventWaitCUDACPU);

} // namespace caffe2
