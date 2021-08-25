#include <c10d/ucc/UCXUtils.hpp>
#include <string>
#include <iostream>
namespace c10d {

class UCPContext {
  ucp_context_h context;
public:
  UCPContext();
#if false
  // Intentionally leak the context
  ~UCPContext()  { ucp_cleanup(context); }
#endif
  ucp_context_h get() const { return context; }
};

UCPContext::UCPContext() {
  ucp_params_t params = {};
  ucp_config_t* config = {};
  ucs_status_t st;

  // get config
  st = ucp_config_read("TORCH", nullptr, &config);
  TORCH_UCX_CHECK(st, "Failed to read UCP config.");

  // initialize context
  params.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_REQUEST_SIZE |
      UCP_PARAM_FIELD_REQUEST_INIT | UCP_PARAM_FIELD_REQUEST_CLEANUP;
  params.request_size = sizeof(UCPRequest::Data);
  params.features = UCP_FEATURE_TAG;
  params.request_init = [](void* request) {
    auto r = reinterpret_cast<UCPRequest::Data *>(request);
    r->status = UCS_INPROGRESS;
  };
  params.request_cleanup = [](void*) {};
  st = ucp_init(&params, config, &context);
  ucp_config_release(config);
  TORCH_UCX_CHECK(st, "Failed to init UCP context.");
}

ucp_context_h getUCPContext() {
  static UCPContext context_wrapper;
  return context_wrapper.get();
}

UCPWorker::UCPWorker() {
  ucp_worker_params_t worker_params = {};
  worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
  ucs_status_t st = ucp_worker_create(getUCPContext(), &worker_params, &worker);
  TORCH_UCX_CHECK(st, "Failed to create UCP worker.");
}

UCPWorker::Address UCPWorker::address() const {
  ucp_address_t* local_addr;
  size_t local_addr_len;

  ucs_status_t st = ucp_worker_get_address(worker, &local_addr, &local_addr_len);
  TORCH_UCX_CHECK(st, "Failed to get worker address.");
  Address addr = std::vector<uint8_t>(
      reinterpret_cast<uint8_t*>(local_addr),
      reinterpret_cast<uint8_t*>(local_addr) + local_addr_len);
  ucp_worker_release_address(worker, local_addr);
  return addr;
}

std::shared_ptr<UCPEndpoint> UCPWorker::connect(const UCPWorker::Address &address) const {
  return std::shared_ptr<UCPEndpoint>(new UCPEndpoint(shared_from_this(), address));
}

UCPEndpoint::UCPEndpoint(const std::shared_ptr<const UCPWorker> &worker, const UCPWorker::Address &address): worker(worker) {
  ucp_ep_params_t ep_params;
  ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
  ep_params.address = reinterpret_cast<const ucp_address_t*>(address.data());
  ucs_status_t st = ucp_ep_create(worker->get(), &ep_params, &endpoint);
  TORCH_UCX_CHECK(st, "Failed to create endpoint.");
}

UCPEndpoint::~UCPEndpoint() {
  ucs_status_t st;

  ucs_status_ptr_t request = ucp_ep_close_nb(endpoint, UCP_EP_CLOSE_MODE_FLUSH);
  if (UCS_PTR_IS_ERR(request)) {
    // It is generally not a good idea to throw in a destructor. So we raise a warning instead.
    TORCH_WARN("Will leak endpoint because it fails to close. Error: ", ucs_status_string(UCS_PTR_STATUS(request)));
  }
  if (UCS_PTR_IS_PTR(request)) {
    do {
      worker->progress();
      st = ucp_request_check_status(request);
    } while (st != UCS_OK);
    ucp_request_free(request);
  }
}

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

std::shared_ptr<UCPRequest> UCPWorker::submit_p2p_request(
  c10::DeviceType device,
  const std::function<ucs_status_ptr_t(const ucp_request_param_t *)> &work
) const {
  ucp_request_param_t params;
  params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_MEMORY_TYPE | UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
  params.memory_type = getUCSMemoryType(device);
  params.cb.recv = [](void* request,
                      ucs_status_t status,
                      const ucp_tag_recv_info_t* info,
                      void* user_data) {
    // http://openucx.github.io/ucx/api/latest/html/group___u_c_p___c_o_m_m.html#ga70e110cf7c85ed5f281bd52438488d75
    auto r = reinterpret_cast<UCPRequest::Data *>(request);
    r->status = status;
    if (status == UCS_OK) {
      r->info = *info;
    }
  };
  ucs_status_ptr_t request = work(&params);
  TORCH_UCX_CHECK_MAYBE_INPROGRESS(UCS_PTR_STATUS(request), "Failed to start p2p operation.");
  progress();
  return std::shared_ptr<UCPRequest>(
    new UCPRequest(reinterpret_cast<UCPRequest::Data *>(request)));
}

std::shared_ptr<UCPRequest> UCPWorker::recv_with_tag_and_mask(void *data, size_t size, ucp_tag_t tag, ucp_tag_t tag_mask, c10::DeviceType device) const {
  return submit_p2p_request(device, [&](const ucp_request_param_t *params) {
    return ucp_tag_recv_nbx(worker, data, size, tag, tag_mask, params);
  });
}

std::shared_ptr<UCPRequest> UCPEndpoint::send_with_tag(void *data, size_t size, ucp_tag_t tag, c10::DeviceType device) const {
  return worker->submit_p2p_request(device, [&](const ucp_request_param_t *params) {
    return ucp_tag_send_nbx(endpoint, data, size, tag, params);
  });
}

} // namespace c10d
