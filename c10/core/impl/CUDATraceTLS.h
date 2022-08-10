#pragma once

#include <c10/core/impl/PyInterpreter.h>

namespace c10 {
namespace impl {

constexpr char trace_cuda_event_creation_fn_name[] =
  "CUDAEventCreationCallbacks";
constexpr char trace_cuda_event_record_fn_name[] =
  "CUDAEventRecordCallbacks";
constexpr char trace_cuda_event_wait_fn_name[] =
  "CUDAEventWaitCallbacks";
constexpr char trace_cuda_memory_allocation_fn_name[] =
  "CUDAMemoryAllocationCallbacks";

struct C10_API CUDATraceTLS {
  static void set_trace(const PyInterpreter*);
  static const PyInterpreter* get_trace();
};

struct C10_API CUDATraceFunctionWrapper {
  using event_creation_sig = void(const PyInterpreter*, uintptr_t event);
  using event_record_sig =
      void(const PyInterpreter*, uintptr_t event, uintptr_t stream);
  using event_wait_sig =
      void(const PyInterpreter*, uintptr_t event, uintptr_t stream);
  using memory_allocation_sig = void(const PyInterpreter*, uintptr_t pointer);

  event_creation_sig* event_creation_fn_;
  event_record_sig* event_record_fn_;
  event_wait_sig* event_wait_fn_;
  memory_allocation_sig* memory_allocation_fn_;

  CUDATraceFunctionWrapper(
    event_creation_sig* event_creation_fn,
    event_record_sig* event_record_fn,
    event_wait_sig* event_wait_fn,
    memory_allocation_sig* memory_allocation_fn) 
    : event_creation_fn_(event_creation_fn),
      event_record_fn_(event_record_fn),
      event_wait_fn_(event_wait_fn),
      memory_allocation_fn_(memory_allocation_fn) {}

  void disarm();
};

} // namespace impl
} // namespace c10
