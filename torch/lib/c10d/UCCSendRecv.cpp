#include <c10d/UCCSendRecv.hpp>

namespace c10d {

static void torch_ucx_req_init(void* request) {
  torch_ucx_request_t* req = static_cast<torch_ucx_request_t*>(request);
  req->status = TORCH_UCX_REQUEST_ACTIVE;
}

static void torch_ucx_req_cleanup(void* request) {}

torch_ucx_status_t torch_ucx_comm_init(
    torch_ucx_comm_t** ucx_comm,
    int size,
    int rank,
    const c10::intrusive_ptr<Store>& store) {
  ucp_params_t params;
  ucp_config_t* config;
  ucs_status_t st;
  torch_ucx_comm_t* comm;
  ucp_worker_params_t worker_params;
  ucp_address_t* local_addr;
  size_t local_addr_len;
  std::string key;
  std::vector<uint8_t> val;
  ucp_worker_attr_t worker_attr;

  comm = new torch_ucx_comm_t;
  comm->rank = rank;
  comm->size = size;

  st = ucp_config_read("TORCH", nullptr, &config);
  if (st != UCS_OK) {
    fprintf(
        stderr,
        "TorchUCC: failed to read ucp config %s\n",
        ucs_status_string(st));
    goto free_comm;
  }

  memset(&params, 0, sizeof(ucp_params_t));
  params.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_REQUEST_SIZE |
      UCP_PARAM_FIELD_ESTIMATED_NUM_EPS | UCP_PARAM_FIELD_TAG_SENDER_MASK |
      UCP_PARAM_FIELD_REQUEST_INIT | UCP_PARAM_FIELD_REQUEST_CLEANUP;
  params.request_size = sizeof(torch_ucx_request_t);
  params.features = UCP_FEATURE_TAG;
  params.estimated_num_eps = size;
  params.request_init = torch_ucx_req_init;
  params.request_cleanup = torch_ucx_req_cleanup;
  params.tag_sender_mask = TORCH_UCX_RANK_MASK;
  st = ucp_init(&params, config, &comm->ctx);
  ucp_config_release(config);
  if (st != UCS_OK) {
    fprintf(
        stderr,
        "TorchUCC: failed to init ucp context %s\n",
        ucs_status_string(st));
    goto free_comm;
  }

  memset(&worker_params, 0, sizeof(ucp_worker_params_t));
  worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
  st = ucp_worker_create(comm->ctx, &worker_params, &comm->worker);
  if (st != UCS_OK) {
    fprintf(
        stderr,
        "TorchUCC: failed to init ucp worker %s\n",
        ucs_status_string(st));
    goto close_ctx;
  }

  worker_attr.field_mask = UCP_WORKER_ATTR_FIELD_THREAD_MODE;
  ucp_worker_query(comm->worker, &worker_attr);
  if (worker_attr.thread_mode != UCS_THREAD_MODE_MULTI) {
    fprintf(stderr, "TorchUCC: Thread mode multi is not supported\n");
  }

  st = ucp_worker_get_address(comm->worker, &local_addr, &local_addr_len);
  if (st != UCS_OK) {
    fprintf(
        stderr,
        "TorchUCC: failed to get ucp worker address %s\n",
        ucs_status_string(st));
    goto close_worker;
  }

  key = "wa" + std::to_string(rank);
  val = std::vector<uint8_t>(
      reinterpret_cast<uint8_t*>(local_addr),
      reinterpret_cast<uint8_t*>(local_addr) + local_addr_len);
  store->set(key, val);
  ucp_worker_release_address(comm->worker, local_addr);
  comm->eps = new ucp_ep_h[size];
  for (int i = 0; i < size; i++) {
    std::vector<uint8_t> peer_addr = store->get("wa" + std::to_string(i));
    ucp_ep_params_t ep_params;

    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address = reinterpret_cast<ucp_address_t*>(peer_addr.data());
    st = ucp_ep_create(comm->worker, &ep_params, &(comm->eps[i]));
    if (st != UCS_OK) {
      fprintf(
          stderr,
          "TorchUCC: failed to create ucp ep %s\n",
          ucs_status_string(st));
      goto close_ep;
    }
  }

  *ucx_comm = comm;
  return TORCH_UCX_OK;

close_ep:
  delete[] comm->eps;
close_worker:
  ucp_worker_destroy(comm->worker);
close_ctx:
  ucp_cleanup(comm->ctx);
free_comm:
  delete comm;
  *ucx_comm = nullptr;
  return TORCH_UCX_ERROR;
}

void torch_ucx_comm_close(
    torch_ucx_comm_t* comm,
    const c10::intrusive_ptr<Store>& store) {
  ucs_status_ptr_t close_req;
  ucs_status_t st;

  if (!comm) {
    return;
  }

  for (int i = 0; i < comm->size; i++) {
    close_req = ucp_ep_close_nb(comm->eps[i], UCP_EP_CLOSE_MODE_FLUSH);
    if (UCS_PTR_IS_ERR(close_req)) {
      return;
    }
    if (UCS_PTR_IS_PTR(close_req)) {
      do {
        ucp_worker_progress(comm->worker);
        st = ucp_request_check_status(close_req);
      } while (st != UCS_OK);
      ucp_request_free(close_req);
    }
  }

  auto key_ep_closed = "epclosed";
  auto num_closed_ep = store->add(key_ep_closed, 1);

  std::vector<std::string> key_finished{"finished"};
  if (num_closed_ep == comm->size) {
    store->add(key_finished[0], 1);
  } else {
    store->wait(key_finished);
  }
  delete[] comm->eps;
  ucp_worker_destroy(comm->worker);
  ucp_cleanup(comm->ctx);
  delete comm;
}

void torch_ucx_send_cmpl_cb(
    void* request,
    ucs_status_t status,
    void* user_data) {
  torch_ucx_request_t* req = static_cast<torch_ucx_request_t*>(request);
  req->status = TORCH_UCX_REQUEST_DONE;
}

void torch_ucx_recv_cmpl_cb(
    void* request,
    ucs_status_t status,
    const ucp_tag_recv_info_t* info,
    void* user_data) {
  torch_ucx_request_t* req = static_cast<torch_ucx_request_t*>(request);
  req->status = TORCH_UCX_REQUEST_DONE;
}

} // namespace c10d
