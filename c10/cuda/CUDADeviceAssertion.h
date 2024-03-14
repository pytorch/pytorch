#pragma once

#include <c10/cuda/CUDAException.h>
#include <c10/macros/Macros.h>

namespace c10::cuda {

#ifdef TORCH_USE_CUDA_DSA
// Copy string from `src` to `dst`
static __device__ void dstrcpy(char* dst, const char* src) {
  int i = 0;
  // Copy string from source to destination, ensuring that it
  // isn't longer than `C10_CUDA_DSA_MAX_STR_LEN-1`
  while (*src != '\0' && i++ < C10_CUDA_DSA_MAX_STR_LEN - 1) {
    *dst++ = *src++;
  }
  *dst = '\0';
}

static __device__ void dsa_add_new_assertion_failure(
    DeviceAssertionsData* assertions_data,
    const char* assertion_msg,
    const char* filename,
    const char* function_name,
    const int line_number,
    const uint32_t caller,
    const dim3 block_id,
    const dim3 thread_id) {
  // `assertions_data` may be nullptr if device-side assertion checking
  // is disabled at run-time. If it is disabled at compile time this
  // function will never be called
  if (!assertions_data) {
    return;
  }

  // Atomically increment so other threads can fail at the same time
  // Note that incrementing this means that the CPU can observe that
  // a failure has happened and can begin to respond before we've
  // written information about that failure out to the buffer.
  const auto nid = atomicAdd(&(assertions_data->assertion_count), 1);

  if (nid >= C10_CUDA_DSA_ASSERTION_COUNT) {
    // At this point we're ran out of assertion buffer space.
    // We could print a message about this, but that'd get
    // spammy if a lot of threads did it, so we just silently
    // ignore any other assertion failures. In most cases the
    // failures will all probably be analogous anyway.
    return;
  }

  // Write information about the assertion failure to memory.
  // Note that this occurs only after the `assertion_count`
  // increment broadcasts that there's been a problem.
  auto& self = assertions_data->assertions[nid];
  dstrcpy(self.assertion_msg, assertion_msg);
  dstrcpy(self.filename, filename);
  dstrcpy(self.function_name, function_name);
  self.line_number = line_number;
  self.caller = caller;
  self.block_id[0] = block_id.x;
  self.block_id[1] = block_id.y;
  self.block_id[2] = block_id.z;
  self.thread_id[0] = thread_id.x;
  self.thread_id[1] = thread_id.y;
  self.thread_id[2] = thread_id.z;
}

// Emulates a kernel assertion. The assertion won't stop the kernel's progress,
// so you should assume everything the kernel produces is garbage if there's an
// assertion failure.
// NOTE: This assumes that `assertions_data` and  `assertion_caller_id` are
//       arguments of the kernel and therefore accessible.
#define CUDA_KERNEL_ASSERT2(condition)                                   \
  do {                                                                   \
    if (C10_UNLIKELY(!(condition))) {                                    \
      /* Has an atomic element so threads can fail at the same time */   \
      c10::cuda::dsa_add_new_assertion_failure(                          \
          assertions_data,                                               \
          C10_STRINGIZE(condition),                                      \
          __FILE__,                                                      \
          __FUNCTION__,                                                  \
          __LINE__,                                                      \
          assertion_caller_id,                                           \
          blockIdx,                                                      \
          threadIdx);                                                    \
      /* Now that the kernel has failed we early exit the kernel, but */ \
      /* otherwise keep going and rely on the host to check UVM and */   \
      /* determine we've had a problem */                                \
      return;                                                            \
    }                                                                    \
  } while (false)
#else
#define CUDA_KERNEL_ASSERT2(condition) assert(condition)
#endif

} // namespace c10::cuda
