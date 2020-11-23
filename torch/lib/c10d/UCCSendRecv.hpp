#pragma once

#include <cinttypes>
#include <cstring>
#include <map>
#include <memory>

#include <c10d/Store.hpp>
#include <ucp/api/ucp.h>

namespace c10d {

#define TORCH_UCX_COMM_BITS 15
#define TORCH_UCX_RANK_BITS 16
#define TORCH_UCX_TAG_BITS 32
#define TORCH_UCX_OOB_BITS 1

#define TORCH_UCX_COMM_BITS_OFFSET 0
#define TORCH_UCX_RANK_BITS_OFFSET TORCH_UCX_COMM_BITS
#define TORCH_UCX_TAG_BITS_OFFSET (TORCH_UCX_COMM_BITS + TORCH_UCX_RANK_BITS)
#define TORCH_UCX_OOB_BITS_OFFSET \
  (TORCH_UCX_COMM_BITS + TORCH_UCX_RANK_BITS + TORCH_UCX_TAG_BITS)

#define TORCH_UCX_MAX_COMM ((((uint64_t)1) << TORCH_UCX_COMM_BITS) - 1)
#define TORCH_UCX_MAX_RANK ((((uint64_t)1) << TORCH_UCX_RANK_BITS) - 1)
#define TORCH_UCX_MAX_TAG ((((uint64_t)1) << TORCH_UCX_TAG_BITS) - 1)
#define TORCH_UCX_MAX_OOB ((((uint64_t)1) << TORCH_UCX_OOB_BITS) - 1)

#define TORCH_UCX_COMM_MASK (TORCH_UCX_MAX_COMM << TORCH_UCX_COMM_BITS_OFFSET)
#define TORCH_UCX_RANK_MASK (TORCH_UCX_MAX_RANK << TORCH_UCX_RANK_BITS_OFFSET)
#define TORCH_UCX_TAG_MASK (TORCH_UCX_MAX_TAG << TORCH_UCX_TAG_BITS_OFFSET)
#define TORCH_UCX_OOB_MASK (TORCH_UCX_MAX_OOB << TORCH_UCX_OOB_BITS_OFFSET)

#define TORCH_UCX_MAKE_P2P_TAG(_tag, _rank, _comm)       \
  ((((uint64_t)(_tag)) << TORCH_UCX_TAG_BITS_OFFSET) |   \
   (((uint64_t)(_rank)) << TORCH_UCX_RANK_BITS_OFFSET) | \
   (((uint64_t)(_comm)) << TORCH_UCX_COMM_BITS_OFFSET))

#define TORCH_UCX_MAKE_OOB_TAG(_tag, _rank, _comm)       \
  ((((uint64_t)(_tag)) << TORCH_UCX_OOB_BITS_OFFSET) |   \
   (((uint64_t)(_rank)) << TORCH_UCX_RANK_BITS_OFFSET) | \
   (((uint64_t)(_rank)) << TORCH_UCX_COMM_BITS_OFFSET))

#define TORCH_UCX_MAKE_SEND_TAG(_ucp_tag, _tag, _rank, _comm)      \
  do {                                                             \
    (_ucp_tag) = TORCH_UCX_MAKE_P2P_TAG((_tag), (_rank), (_comm)); \
  } while (0)

#define TORCH_UCX_ANY_SOURCE (TORCH_UCX_MAX_RANK - 1)
#define TORCH_UCX_ANY_SOURCE_MASK (~TORCH_UCX_RANK_MASK)
#define TORCH_UCX_SPECIFIC_SOURCE_MASK ((uint64_t)-1)

#define TORCH_UCX_MAKE_RECV_TAG(_ucp_tag, _ucp_tag_mask, _tag, _rank, _comm) \
  do {                                                                       \
    (_ucp_tag) = TORCH_UCX_MAKE_P2P_TAG((_tag), (_rank), (_comm));           \
    if ((_rank) == TORCH_UCX_ANY_SOURCE) {                                   \
      (_ucp_tag_mask) = TORCH_UCX_ANY_SOURCE_MASK;                           \
    } else {                                                                 \
      (_ucp_tag_mask) = TORCH_UCX_SPECIFIC_SOURCE_MASK;                      \
    }                                                                        \
  } while (0)

#define TORCH_UCX_MAKE_OOB_SEND_TAG(_ucp_tag, _tag, _rank, _comm)  \
  do {                                                             \
    (_ucp_tag) = TORCH_UCX_MAKE_OOB_TAG((_tag), (_rank), (_comm)); \
  } while (0)

#define TORCH_UCX_MAKE_OOB_RECV_TAG(                               \
    _ucp_tag, _ucp_tag_mask, _tag, _rank, _comm)                   \
  do {                                                             \
    (_ucp_tag) = TORCH_UCX_MAKE_OOB_TAG((_tag), (_rank), (_comm)); \
    (_ucp_tag_mask) = (uint64_t)-1;                                \
  } while (0)

enum torch_ucx_status_t {
  TORCH_UCX_OK = 0,
  TORCH_UCX_INPROGRESS = 1,
  TORCH_UCX_ERROR = -1,
};

enum torch_ucx_tag_type_t {
  TORCH_UCX_P2P_TAG,
  TORCH_UCX_OOB_TAG
};

enum torch_ucx_request_status_t {
  TORCH_UCX_REQUEST_ACTIVE,
  TORCH_UCX_REQUEST_DONE,
};

struct torch_ucx_request_t {
  torch_ucx_request_status_t status;
};

const std::map<c10::DeviceType, ucs_memory_type_t> ucs_mtype_map = {
    {c10::kCPU, UCS_MEMORY_TYPE_HOST},
    {c10::kCUDA, UCS_MEMORY_TYPE_CUDA},
    {c10::kHIP, UCS_MEMORY_TYPE_ROCM},
    {c10::kFPGA, UCS_MEMORY_TYPE_UNKNOWN},
    {c10::kMSNPU, UCS_MEMORY_TYPE_UNKNOWN},
    {c10::kXLA, UCS_MEMORY_TYPE_UNKNOWN},
    {c10::kVulkan, UCS_MEMORY_TYPE_UNKNOWN},
    {c10::kMetal, UCS_MEMORY_TYPE_UNKNOWN},
};

struct torch_ucx_comm_t {
  int size;
  int rank;
  ucp_context_h ctx;
  ucp_ep_h* eps;
  ucp_worker_h worker;
};

static inline void torch_ucx_request_free(torch_ucx_request_t* request) {
  request->status = TORCH_UCX_REQUEST_ACTIVE;
  ucp_request_free(request);
}

static inline torch_ucx_status_t torch_ucx_check_req(ucs_status_ptr_t st) {
  if (UCS_PTR_IS_ERR(st)) {
    fprintf(
        stderr, "ProcessGroupUCC: %s\n", ucs_status_string(UCS_PTR_STATUS(st)));
    return TORCH_UCX_ERROR;
  }

  return TORCH_UCX_OK;
}

void torch_ucx_send_cmpl_cb(
    void* request,
    ucs_status_t status,
    void* user_data);

void torch_ucx_recv_cmpl_cb(
    void* request,
    ucs_status_t status,
    const ucp_tag_recv_info_t* info,
    void* user_data);

torch_ucx_status_t torch_ucx_comm_init(
    torch_ucx_comm_t** comm,
    int size,
    int rank,
    const c10::intrusive_ptr<Store>& store);
void torch_ucx_comm_close(
    torch_ucx_comm_t* comm,
    const c10::intrusive_ptr<Store>& store);

static inline torch_ucx_status_t torch_ucx_send_nb(
    torch_ucx_comm_t* comm,
    void* data,
    ucs_memory_type_t mtype,
    size_t size,
    int dst_rank,
    uint32_t tag,
    torch_ucx_request_t** req,
    torch_ucx_tag_type_t type) {
  ucp_tag_t ucp_tag;
  ucs_status_ptr_t st;
  ucp_request_param_t params;

  switch (type) {
    case TORCH_UCX_P2P_TAG:
      TORCH_UCX_MAKE_SEND_TAG(ucp_tag, tag, comm->rank, 0);
      break;
    case TORCH_UCX_OOB_TAG:
      TORCH_UCX_MAKE_OOB_SEND_TAG(ucp_tag, tag, comm->rank, 0);
      break;
    default:
      return TORCH_UCX_ERROR;
  };
  // fprintf(stderr, "rank %d send tag %" PRIu64 "(%d) shift %d\n", comm->rank,
  // ucp_tag, tag, TORCH_UCX_OOB_TAG_BITS_OFFSET);

  params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
      UCP_OP_ATTR_FIELD_DATATYPE | UCP_OP_ATTR_FIELD_MEMORY_TYPE;
  params.datatype = ucp_dt_make_contig(size);
  params.memory_type = mtype;
  params.cb.send = torch_ucx_send_cmpl_cb;
  st = ucp_tag_send_nbx(comm->eps[dst_rank], data, 1, ucp_tag, &params);
  if (torch_ucx_check_req(st) != TORCH_UCX_OK) {
    return TORCH_UCX_ERROR;
  };
  *req = reinterpret_cast<torch_ucx_request_t*>(st);

  return TORCH_UCX_OK;
}

static inline torch_ucx_status_t torch_ucx_recv_nb(
    torch_ucx_comm_t* comm,
    void* data,
    ucs_memory_type_t mtype,
    size_t size,
    int src_rank,
    uint32_t tag,
    torch_ucx_request_t** req,
    torch_ucx_tag_type_t type) {
  ucp_tag_t ucp_tag, ucp_tag_mask;
  ucs_status_ptr_t st;
  ucp_request_param_t params;

  switch (type) {
    case TORCH_UCX_P2P_TAG:
      TORCH_UCX_MAKE_RECV_TAG(ucp_tag, ucp_tag_mask, tag, src_rank, 0);
      break;
    case TORCH_UCX_OOB_TAG:
      TORCH_UCX_MAKE_OOB_RECV_TAG(ucp_tag, ucp_tag_mask, tag, src_rank, 0);
      break;
    default:
      return TORCH_UCX_ERROR;
  };

  // fprintf(stderr, "rank %d recv tag %" PRIu64 " (%d) mask %" PRIu64 "\n",
  // comm->rank, ucp_tag, tag, ucp_tag_mask );
  params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
      UCP_OP_ATTR_FIELD_DATATYPE | UCP_OP_ATTR_FIELD_MEMORY_TYPE;
  params.datatype = ucp_dt_make_contig(size);
  params.cb.recv = torch_ucx_recv_cmpl_cb;
  params.memory_type = mtype;
  st = ucp_tag_recv_nbx(comm->worker, data, 1, ucp_tag, ucp_tag_mask, &params);
  if (torch_ucx_check_req(st) != TORCH_UCX_OK) {
    return TORCH_UCX_ERROR;
  };
  *req = reinterpret_cast<torch_ucx_request_t*>(st);
  /*TODO: check request*/

  return TORCH_UCX_OK;
}

static inline unsigned torch_ucx_comm_progress(torch_ucx_comm_t* comm) {
  return ucp_worker_progress(comm->worker);
}

static inline torch_ucx_status_t torch_ucx_req_test(
    torch_ucx_comm_t* comm,
    torch_ucx_request_t** reqs,
    int n_reqs,
    int* completed_idx,
    int poll_count,
    int n_completions_required) {
  int n_polls = 0;
  int n_completed;

  if (n_completions_required == 0) {
    return TORCH_UCX_OK;
  }

  while (poll_count < 0 || n_polls++ < poll_count) {
    n_completed = 0;
    for (int i = 0; i < n_reqs; i++) {
      if (reqs[i] == nullptr) {
        if (completed_idx) {
          *completed_idx = i;
        }
        n_completed++;
      } else {
        if (reqs[i]->status != TORCH_UCX_REQUEST_DONE) {
          torch_ucx_comm_progress(comm);
        } else {
          torch_ucx_request_free(reqs[i]);
          reqs[i] = nullptr;
          if (completed_idx) {
            *completed_idx = i;
          }
          n_completed++;
        }
      }
      if (n_completed == n_completions_required) {
        return TORCH_UCX_OK;
      }
    }
  }
  return TORCH_UCX_INPROGRESS;
}

} // namespace c10d
