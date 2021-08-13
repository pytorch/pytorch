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

// When calling UCP operations like `ucp_tag_send_nbx`, `ucp_tag_recv_nbx`, etc.,
// UCP will create a request object and return its pointer to the user. It is
// the user's responsibility to free these objects by `ucp_request_free`. Here
// we use RAII to implement this logic.
// Some UCP operations finishes immediately. If this is the case, then no request
// object will be created.
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

  // non-copyable
  UCPRequest(const UCPRequest&) = delete;
  UCPRequest& operator=(const UCPRequest &) = delete;
private:
  Data *data;

  // Only `UCPWorker` and `UCPEndpoint` can create `UCPRequest` objects.
  UCPRequest(Data *data): data(data) {}
  friend class UCPWorker;
  friend class UCPEndpoint;
};

class UCPEndpoint;

class UCPWorker: public std::enable_shared_from_this<UCPWorker> {
  ucp_worker_h worker;
public:
  UCPWorker();
  ucp_worker_h get() const { return worker; }
  ~UCPWorker() { ucp_worker_destroy(worker); }

  // non-copyable
  UCPWorker(const UCPWorker&) = delete;
  UCPWorker& operator=(const UCPWorker &) = delete;

  using Address = std::vector<uint8_t>;
  Address address() const;
  std::shared_ptr<UCPEndpoint> connect(const Address &address) const;
  unsigned progress() const { return ucp_worker_progress(worker); }

  std::shared_ptr<UCPRequest> submit_p2p_request(size_t size, c10::DeviceType device, const std::function<ucs_status_ptr_t(const ucp_request_param_t *)> &work) const;
  std::shared_ptr<UCPRequest> recv_with_tag(void *data, size_t size, int tag, c10::DeviceType device) const; // receive from any source
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

  std::shared_ptr<UCPRequest> send_with_tag(void *data, size_t size, int tag, c10::DeviceType device) const;
  std::shared_ptr<UCPRequest> recv_with_tag(void *data, size_t size, int tag, c10::DeviceType device) const;
};

} // namespace c10d
