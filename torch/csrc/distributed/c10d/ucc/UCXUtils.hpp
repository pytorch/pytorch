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
#define TORCH_UCX_CHECK_MAYBE_INPROGRESS(st, ...)                                          \
  do {                                                                                     \
    auto _s_t_a_t_u_s = (st);                                                              \
    auto _i_s_o_k = (_s_t_a_t_u_s == UCS_OK || _s_t_a_t_u_s == UCS_INPROGRESS);            \
    TORCH_CHECK_WITH(UCXError, _i_s_o_k, __VA_ARGS__, " Error: ", ucs_status_string(st));  \
  } while(0)

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
    ucs_status_t status;
    ucp_tag_recv_info_t info;
    void reset() {
      status = UCS_INPROGRESS;
      info = {};
    }
  };

  ucs_status_t status() const {
    if (data == nullptr) {
      return UCS_OK;
    }
    return data->status;
  }

  const ucp_tag_recv_info_t &info() const {
    TORCH_INTERNAL_ASSERT(data != nullptr);
    return data->info;
  }

  ~UCPRequest() {
    if (data != nullptr) {
      // Requests may be reused, and when reused, the `request_init`
      // callback function specified in the context creation will not
      // be invoked. So we have to manually reset a request before
      // freeing it.
      data->reset();
      ucp_request_free(data);
    }
  }

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
  std::shared_ptr<UCPRequest> send_with_tag(void *data, size_t size, ucp_tag_t tag, c10::DeviceType device) const;
};

// Note [Receive from an endpoint]:
// UCP does not support receiving from a specific endpoint. So we use tag
// matching to simulate this behavior. In PyTorch, the world_size is int,
// the tag is also int, and in UCP, the ucp_tag_t is uint64_t. So we use
// the higher 32 bits of ucp_tag_t for rank, and use lower 32 bits for the
// real tag. When receiving from a specified endpoint, the entire ucp_tag_t
// should match. And when receiving from any source, tag mask is used to
// disable the matching of the higher bits.

// TODO: add test for INT_MAX tag

using world_size_type = int;
using tag_type = int;

union tag_union {
  ucp_tag_t raw;
  struct fields_t {
    tag_type tag;
    world_size_type rank;
  } fields;
};

static_assert(
  sizeof(tag_union) == sizeof(ucp_tag_t) &&
  sizeof(tag_union) == sizeof(tag_union::fields_t),
  "The implementation of UCP tag matching has unsatisfied assumptions.");

constexpr ucp_tag_t wrap_tag(world_size_type rank, tag_type tag) {
  tag_union u = {
    .fields = { .tag = tag, .rank = rank }
  };
  return u.raw;
}

constexpr world_size_type get_rank_from_tag(ucp_tag_t tag) {
  tag_union u = { .raw = tag };
  return u.fields.rank;
}

constexpr ucp_tag_t any_source_mask() {
  return wrap_tag(0, ~tag_type(0));
}

constexpr ucp_tag_t complete_tag_mask() {
  return wrap_tag(~world_size_type(0), ~tag_type(0));
}

} // namespace c10d
