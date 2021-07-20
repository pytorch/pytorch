#ifdef USE_C10D_UCC

#include <c10d/UCXUtils.hpp>
#include <string>

UCPContext::UCPContext() {
  ucp_params_t params = {};
  ucp_config_t* config = {};
  ucs_status_t st;
  ucp_worker_params_t worker_params = {};

  // get config
  st = ucp_config_read("TORCH", nullptr, &config);
  if (st != UCS_OK) {
    throw UCXError(std::string("Failed to read UCP config: ") + ucs_status_string(st));
  }

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
  if (st != UCS_OK) {
    throw UCXError(std::string("Failed to init UCP context: ") + ucs_status_string(st));
  }

  // initialize worker
  worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
  st = ucp_worker_create(context, &worker_params, &worker);
  if (st != UCS_OK) {
    ucp_cleanup(context);
    throw UCXError(std::string("Failed to create UCP worker: ") + ucs_status_string(st));
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

#endif
