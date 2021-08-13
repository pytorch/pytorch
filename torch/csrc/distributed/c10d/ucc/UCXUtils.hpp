#pragma once

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

#define TORCH_UCX_CHECK(st, ...) TORCH_CHECK_WITH(UCXError, (st) == UCS_OK, __VA_ARGS__, " Error: ", ucs_status_string(st))

namespace c10d {

class UCPEndpoint;

class UCPWorker: public std::enable_shared_from_this<UCPWorker> {
  ucp_worker_h worker;
public:
  UCPWorker();
  ucp_worker_h get() const { return worker; }
  ~UCPWorker();

  // non-copyable
  UCPWorker(const UCPWorker&) = delete;
  UCPWorker& operator=(const UCPWorker &) = delete;

  using Address = std::vector<uint8_t>;
  Address address() const;
  std::shared_ptr<UCPEndpoint> connect(const Address &address) const;
  unsigned progress();
};

class UCPEndpoint {
  ucp_ep_h endpoint;
  std::shared_ptr<const UCPWorker> worker;

  // UCPEndpoint should be created by UCPWorker::connect
  UCPEndpoint(const std::shared_ptr<const UCPWorker> &worker, const UCPWorker::Address &address);
  friend UCPWorker;
public:
  ~UCPEndpoint();
  ucp_ep_h get() const { return endpoint; }

  // non-copyable
  UCPEndpoint(const UCPEndpoint&) = delete;
  UCPEndpoint& operator=(const UCPEndpoint &) = delete;
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
