#include <c10d/ucc/UCXUtils.hpp>
#include <string>

namespace c10d {

class UCPContext {
  ucp_context_h context;
public:
  UCPContext();
  ~UCPContext();
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
  params.request_size = sizeof(bool);
  params.features = UCP_FEATURE_TAG;
  params.request_init = [](void* request) {
    *static_cast<bool *>(request) = false;
  };
  params.request_cleanup = [](void*) {};
  st = ucp_init(&params, config, &context);
  ucp_config_release(config);
  TORCH_UCX_CHECK(st, "Failed to init UCP context.");
}

UCPContext::~UCPContext() {
  ucp_cleanup(context);
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

UCPWorker::~UCPWorker() {
  ucp_worker_destroy(worker);
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

unsigned UCPWorker::progress() {
  return ucp_worker_progress(worker);
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
      ucp_worker_progress(worker->get());
      st = ucp_request_check_status(request);
    } while (st != UCS_OK);
    ucp_request_free(request);
  }
}

} // namespace c10d
