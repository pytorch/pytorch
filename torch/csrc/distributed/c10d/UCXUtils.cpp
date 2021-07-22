#ifdef USE_C10D_UCC

#include <c10d/UCXUtils.hpp>
#include <string>

namespace c10d {

UCPContext::UCPContext() {
  ucp_params_t params = {};
  ucp_config_t* config = {};
  ucs_status_t st;
  ucp_worker_params_t worker_params = {};

  // get config
  st = ucp_config_read("TORCH", nullptr, &config);
  TORCH_UCX_CHECK(st, "Failed to read UCP config.");

  // initialize context
  params.field_mask = UCP_PARAM_FIELD_FEATURES;
  params.features = UCP_FEATURE_TAG;
  st = ucp_init(&params, config, &context);
  ucp_config_release(config);
  TORCH_UCX_CHECK(st, "Failed to init UCP context.");

  // initialize worker
  worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
  st = ucp_worker_create(context, &worker_params, &worker);
  if (st != UCS_OK) {
    ucp_cleanup(context);
    TORCH_UCX_CHECK(st, "Failed to create UCP worker.");
  }
}

UCPContext::~UCPContext() {
  ucp_worker_destroy(worker);
  ucp_cleanup(context);
}

UCPContext *UCPContext::get() {
  if (instance == nullptr) {
    instance = std::unique_ptr<UCPContext>(new UCPContext());
  }
  return instance.get();
}

std::unique_ptr<UCPContext> UCPContext::instance;

UCPEndpoint::UCPEndpoint(ucp_address_t* address) {
  ucp_ep_params_t ep_params;
  ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
  ep_params.address = address;
  ucs_status_t st = ucp_ep_create(UCPContext::get()->worker, &ep_params, &endpoint);
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
      ucp_worker_progress(UCPContext::get()->worker);
      st = ucp_request_check_status(request);
    } while (st != UCS_OK);
    ucp_request_free(request);
  }
}

} // namespace c10d

#endif
