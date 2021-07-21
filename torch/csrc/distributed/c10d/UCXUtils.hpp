#pragma once

#ifdef USE_C10D_UCC

#include <memory>
#include <stdexcept>
#include <ucp/api/ucp.h>
#include <c10/util/Exception.h>
#include <c10/core/DeviceType.h>

namespace c10 {
class UCXError : public c10::Error {
  using Error::Error;
};

} // namespace c10

#define TORCH_UCX_CHECK(st, ...) TORCH_CHECK_WITH(UCXError, (st) == UCS_OK, __VA_ARGS__, " Error code: ", ucs_status_string(st))

namespace c10d {

// Singleton object holding UCP objects
class UCPContext {
  static std::unique_ptr<UCPContext> instance;
  UCPContext();
public:
  ucp_context_h context;
  ucp_worker_h worker;
  static UCPContext *get();
  ~UCPContext();
};

inline ucs_memory_type getUCSMemoryType(c10::DeviceType type) {
  switch(type) {
  case c10::kCPU:
    return UCS_MEMORY_TYPE_HOST;
  case c10::kCUDA:
    return UCS_MEMORY_TYPE_CUDA;
  case c10::kHIP:
    return UCS_MEMORY_TYPE_ROCM;
  default:
    return UCS_MEMORY_TYPE_UNKNOWN;
  }
}

} // namespace c10d

#endif
