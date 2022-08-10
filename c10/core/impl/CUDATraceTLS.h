#pragma once

#include <c10/core/impl/PyInterpreter.h>

namespace c10 {
namespace impl {

constexpr char trace_cuda_event_record_fn_name[] =
  "_fire_callbacks_for_cuda_event_record";
constexpr char trace_cuda_event_wait_fn_name[] =
  "_fire_callbacks_for_cuda_event_wait";

struct C10_API CUDATraceTLS {
  static void set_trace(const PyInterpreter*);
  static const PyInterpreter* get_trace();
};

} // namespace impl
} // namespace c10
