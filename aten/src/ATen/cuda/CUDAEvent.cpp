#include "ATen/cuda/CUDAEvent.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/CUDAStream.h"
#include "ATen/cuda/Exceptions.h"
#include "ATen/core/Error.h"
#include "ATen/DeviceGuard.h"

#include <mutex>
#include <atomic>

// Internal implementation is entirely hidden
struct CUDAEventInternals {
  std::atomic<int> refcount;
  int64_t device; // Note: cudaGetDevice works with int32_t, not int64_t
  std::atomic<bool> was_recorded;
  cudaEvent_t event;
};

namespace at {
namespace cuda {

namespace detail {

/*
* Pointer-based event API
*/
CUDAEventInternals* CUDAEvent_create(unsigned int flags) {
  std::unique_ptr<CUDAEventInternals> internals { new CUDAEventInternals() };
  internals->refcount = 1;
  internals->device = current_device();
  AT_CUDA_CHECK(cudaEventCreateWithFlags(&internals->event, flags));
  return internals.release();
}

void CUDAEvent_retain(CUDAEventInternals* internals) {
  AT_CHECK(internals);
  const auto prior_refcount = internals->refcount++;
  if (prior_refcount == 0) {
    AT_ERROR("CUDAEvent_retain called on destroyed event");
  }
}

// If the refcount goes to zero, switches to the event's device and destroys it.
// Note: switching to the device is necessary to prevent the creation of a 
// CUDAContext on a different GPU when destroying the event.
void CUDAEvent_uncheckedFree(CUDAEventInternals* internals) {
  if (!internals) return;
  if (--internals->refcount == 0) {
    at::DeviceGuard device_guard{internals->device};
    cudaEventDestroy(internals->event);
  }
}

cudaEvent_t CUDAEvent_event(CUDAEventInternals* internals) {
  AT_CHECK(internals);
  return internals->event;
}

int64_t CUDAEvent_device(CUDAEventInternals* internals) {
  AT_CHECK(internals);
  return internals->device;
}

void CUDAEvent_record(CUDAEventInternals* internals, const CUDAStream& stream) {
  AT_CHECK(internals)
  internals->was_recorded = true;
  AT_CUDA_CHECK(cudaEventRecord(internals->event, stream));
}

// Records the event if it has never been recorded to, does nothing otherwise.
void CUDAEvent_recordOnce(CUDAEventInternals* internals, const CUDAStream& stream) {
  AT_CHECK(internals)
  const auto was_recorded = internals->was_recorded.exchange(true);
  if (!was_recorded) {
    AT_CUDA_CHECK(cudaEventRecord(internals->event, stream));
  }
}

bool CUDAEvent_happened(CUDAEventInternals* internals) {
  AT_CHECK(internals);
  return (internals->was_recorded
  && cudaEventQuery(internals->event) == cudaSuccess);
}


} // namespace detail

void CUDAEvent::record() {
  record(getCurrentCUDAStream());
}

void CUDAEvent::record(const CUDAStream& stream) {
  detail::CUDAEvent_record(internals_, stream);
}

void CUDAEvent::recordOnce(const CUDAStream& stream) { 
  detail::CUDAEvent_recordOnce(internals_, stream);
}


} // namespace cuda
} // namespace at
