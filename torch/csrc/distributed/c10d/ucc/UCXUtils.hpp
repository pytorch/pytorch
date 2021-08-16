#pragma once

#include <functional>
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

// When calling UCP async operations like `ucp_tag_send_nbx`, `ucp_tag_recv_nbx`,
// etc., UCP will create a request object and return its pointer to the user.
// This request object is used to track the status of async operations. It is
// the user's responsibility to free these objects with `ucp_request_free`. Here
// we use RAII to implement this create-by-ucp-and-destroy-by-user logic. Some UCP
// operations finishes immediately. If this is the case, then no request object
// will be created.
class UCPRequest {
public:
  struct Data {
    bool completed;
  };

  bool is_completed() const {
    if (data == nullptr) {
      return true;
    }
    return data->completed;
  }

  ~UCPRequest() { if (data != nullptr) { ucp_request_free(data); } }

private:
  // Pointer towards the real underlying request object created by UCP.
  // A nullptr represents that a request is finished immediately.
  Data *data;

  // `UCPRequest` objects should only be created by `UCPEndpoint`
  // (for send/recv with an endpoint) or `UCPWorker` (for recv from any source).
  // `UCPRequest` objects are non-copyable: The underlying data should only be
  // allocated by UCP, and it should only be deallocated once.
  friend class UCPWorker;
  friend class UCPEndpoint;
  UCPRequest(Data *data): data(data) {}
  UCPRequest(const UCPRequest&) = delete;
  UCPRequest& operator=(const UCPRequest &) = delete;
};

class UCPEndpoint;

class UCPWorker: public std::enable_shared_from_this<UCPWorker> {
  ucp_worker_h worker;
public:
  UCPWorker();
  ucp_worker_h get() const { return worker; }
  ~UCPWorker() { ucp_worker_destroy(worker); }

  // Non-copyable
  UCPWorker(const UCPWorker&) = delete;
  UCPWorker& operator=(const UCPWorker &) = delete;

  using Address = std::vector<uint8_t>;
  Address address() const;
  std::shared_ptr<UCPEndpoint> connect(const Address &address) const;
  unsigned progress() const { return ucp_worker_progress(worker); }

  std::shared_ptr<UCPRequest> submit_p2p_request(c10::DeviceType device, const std::function<ucs_status_ptr_t(const ucp_request_param_t *)> &work) const;
  std::shared_ptr<UCPRequest> recv_with_tag_and_mask(void *data, size_t size, ucp_tag_t tag, ucp_tag_t tag_mask, c10::DeviceType device) const;

  // Receive from any source. See [Receive from an endpoint]
  std::shared_ptr<UCPRequest> recv_any_with_tag(void *data, size_t size, int tag, c10::DeviceType device) const;
};

class UCPEndpoint {
  ucp_ep_h endpoint;
  std::shared_ptr<const UCPWorker> worker;

  // UCPEndpoint should be created by UCPWorker::connect
  UCPEndpoint(const std::shared_ptr<const UCPWorker> &worker, const UCPWorker::Address &address);
  friend UCPWorker;
public:
  ~UCPEndpoint();

  // Non-copyable
  UCPEndpoint(const UCPEndpoint&) = delete;
  UCPEndpoint& operator=(const UCPEndpoint &) = delete;

  // Send data to this endpoint
  std::shared_ptr<UCPRequest> send_with_tag(void *data, size_t size, int tag, c10::DeviceType device) const;

  // Receive data from this endpoint
  //
  // Note [Receive from an endpoint]:
  // UCP does not support receiving from a specific endpoint. So we use tag
  // matching to simulate this behavior. We use higher bits of a tag to store
  // rank, and use lower bits to store the real tag. When receiving from any
  // source, tag mask is used to disable higher bits.
  //
  // TODO: unit test should be modified so that recv from different endpoint
  // with the same tag are covered.
  std::shared_ptr<UCPRequest> recv_with_tag(void *data, size_t size, int tag, c10::DeviceType device) const;
};

} // namespace c10d
