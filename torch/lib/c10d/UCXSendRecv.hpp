/**
 * * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * *
 * * See file LICENSE for terms.
 * */

#pragma once

#include <memory>
#include <string.h>
#include <inttypes.h>

#include <ucp/api/ucp.h>
#include <c10d/Store.hpp>


namespace c10d {

#define TORCH_UCX_RANK_BITS    18
#define TORCH_UCX_COL_TAG_BITS 13
#define TORCH_UCX_P2P_TAG_BITS 32
#define TORCH_UCX_OOB_TAG_BITS 1

#define TORCH_UCX_RANK_BITS_OFFSET    0
#define TORCH_UCX_COL_TAG_BITS_OFFSET (TORCH_UCX_RANK_BITS)
#define TORCH_UCX_P2P_TAG_BITS_OFFSET (TORCH_UCX_RANK_BITS + \
                                       TORCH_UCX_COL_TAG_BITS)
#define TORCH_UCX_OOB_TAG_BITS_OFFSET (TORCH_UCX_RANK_BITS + \
                                       TORCH_UCX_COL_TAG_BITS + \
                                       TORCH_UCX_P2P_TAG_BITS)

#define TORCH_UCX_MAX_RANK    ((((uint64_t)1) << TORCH_UCX_RANK_BITS   ) - 1)
#define TORCH_UCX_MAX_COL_TAG ((((uint64_t)1) << TORCH_UCX_COL_TAG_BITS) - 1)
#define TORCH_UCX_MAX_P2P_TAG ((((uint64_t)1) << TORCH_UCX_P2P_TAG_BITS) - 1)
#define TORCH_UCX_MAX_OOB_TAG ((((uint64_t)1) << TORCH_UCX_OOB_TAG_BITS) - 1)

#define TORCH_UCX_RANK_MASK    (TORCH_UCX_MAX_RANK    << TORCH_UCX_RANK_BITS_OFFSET)
#define TORCH_UCX_COL_TAG_MASK (TORCH_UCX_MAX_COL_TAG << TORCH_UCX_COL_TAG_BITS_OFFSET)
#define TORCH_UCX_P2P_TAG_MASK (TORCH_UCX_MAX_P2P_TAG << TORCH_UCX_P2P_TAG_BITS_OFFSET)
#define TORCH_UCX_OOB_TAG_MASK (TORCH_UCX_MAX_OOB_TAG << TORCH_UCX_OOB_TAG_BITS_OFFSET)


#define TORCH_UCX_MAKE_P2P_TAG(_tag, _rank)                    \
    ((((uint64_t) (_tag))  << TORCH_UCX_P2P_TAG_BITS_OFFSET) | \
     (((uint64_t) (_rank)) << TORCH_UCX_RANK_BITS_OFFSET))

#define TORCH_UCX_MAKE_COLL_TAG(_tag, _rank)                   \
    ((((uint64_t) (_tag))  << TORCH_UCX_COL_TAG_BITS_OFFSET) | \
     (((uint64_t) (_rank)) << TORCH_UCX_RANK_BITS_OFFSET))

#define TORCH_UCX_MAKE_OOB_TAG(_tag, _rank)                    \
    ((((uint64_t) (_tag))  << TORCH_UCX_OOB_TAG_BITS_OFFSET) | \
     (((uint64_t) (_rank)) << TORCH_UCX_RANK_BITS_OFFSET))

#define TORCH_UCX_MAKE_COLL_SEND_TAG(_ucp_tag, _tag, _rank) do { \
        (_ucp_tag) = TORCH_UCX_MAKE_COLL_TAG((_tag), (_rank));   \
    } while(0)

#define TORCH_UCX_MAKE_COLL_RECV_TAG(_ucp_tag, _ucp_tag_mask, _tag, _rank) do { \
        (_ucp_tag)      = TORCH_UCX_MAKE_COLL_TAG((_tag), (_rank));             \
        (_ucp_tag_mask) = (uint64_t)-1;                                         \
    } while(0)

#define TORCH_UCX_MAKE_P2P_SEND_TAG(_ucp_tag, _tag, _rank) do { \
        (_ucp_tag) = TORCH_UCX_MAKE_P2P_TAG((_tag), (_rank));   \
    } while(0)

#define TORCH_UCX_MAKE_P2P_RECV_TAG(_ucp_tag, _ucp_tag_mask, _tag, _rank) do { \
        (_ucp_tag)      = TORCH_UCX_MAKE_P2P_TAG((_tag), (_rank));             \
        (_ucp_tag_mask) = (uint64_t)-1;                                        \
    } while(0)

#define TORCH_UCX_MAKE_OOB_SEND_TAG(_ucp_tag, _tag, _rank) do { \
        (_ucp_tag) = TORCH_UCX_MAKE_OOB_TAG((_tag), (_rank));   \
    } while(0)

#define TORCH_UCX_MAKE_OOB_RECV_TAG(_ucp_tag, _ucp_tag_mask, _tag, _rank) do { \
        (_ucp_tag)      = TORCH_UCX_MAKE_OOB_TAG((_tag), (_rank));             \
        (_ucp_tag_mask) = (uint64_t)-1;                                        \
    } while(0)

enum torch_ucx_status_t {
    TORCH_UCX_OK          =  0,
    TORCH_UCX_INPROGRESS  =  1,
    TORCH_UCX_ERROR       = -1,
};

enum torch_ucx_tag_type_t {
    TORCH_UCX_COLL_TAG,
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

struct torch_ucx_comm_t {
    int           size;
    int           rank;
    ucp_context_h ctx;
    ucp_ep_h      *eps;
    ucp_worker_h  worker;
    uint32_t      tag;
};


static inline void torch_ucx_request_free(torch_ucx_request_t *request)
{
    request->status = TORCH_UCX_REQUEST_ACTIVE;
    ucp_request_free(request);
}

void torch_ucx_send_cmpl_cb(void* request, ucs_status_t status);
void torch_ucx_recv_cmpl_cb(void* request, ucs_status_t status,
                            ucp_tag_recv_info_t *info);

torch_ucx_status_t
torch_ucx_comm_init(torch_ucx_comm_t **comm,
                    int size, int rank,
                    const std::shared_ptr<Store>& store);
void 
torch_ucx_comm_close(torch_ucx_comm_t *comm,
                     const std::shared_ptr<Store>& store);

static inline torch_ucx_status_t
torch_ucx_send_nb(torch_ucx_comm_t *comm,
                  void *data, size_t size, int dst_rank,
                  uint32_t tag, torch_ucx_request_t **req,
                  torch_ucx_tag_type_t type)
{
    ucp_tag_t        ucp_tag;
    ucp_datatype_t   dt;
    ucp_ep_h         ep;
    ucs_status_ptr_t st;

    ep = comm->eps[dst_rank];
    dt = ucp_dt_make_contig(size);
    switch(type) {
        case TORCH_UCX_COLL_TAG:
            TORCH_UCX_MAKE_COLL_SEND_TAG(ucp_tag, tag, comm->rank);
            break;
        case TORCH_UCX_P2P_TAG:
            TORCH_UCX_MAKE_P2P_SEND_TAG(ucp_tag, tag, comm->rank);
            break;
        case TORCH_UCX_OOB_TAG:
            TORCH_UCX_MAKE_OOB_SEND_TAG(ucp_tag, tag, comm->rank);
            break;
        default:
            return TORCH_UCX_ERROR;
    };
    //fprintf(stderr, "rank %d send tag %" PRIu64 "\n", comm->rank, ucp_tag);    
    st = ucp_tag_send_nb(ep, data, 1, dt, ucp_tag, torch_ucx_send_cmpl_cb);
    *req = reinterpret_cast<torch_ucx_request_t*>(st);
    /*TODO: check request*/

    return TORCH_UCX_OK;
}

static inline torch_ucx_status_t
torch_ucx_recv_nb(torch_ucx_comm_t *comm,
                  void *data, size_t size, int src_rank,
                  uint32_t tag, torch_ucx_request_t **req,
                  torch_ucx_tag_type_t type)
{
    ucp_tag_t      ucp_tag, ucp_tag_mask;
    ucp_datatype_t dt;
    ucs_status_ptr_t st;

    dt = ucp_dt_make_contig(size);
    switch(type) {
        case TORCH_UCX_COLL_TAG:
            TORCH_UCX_MAKE_COLL_RECV_TAG(ucp_tag, ucp_tag_mask, tag, src_rank);
            break;
        case TORCH_UCX_P2P_TAG:
            TORCH_UCX_MAKE_P2P_RECV_TAG(ucp_tag, ucp_tag_mask, tag, src_rank);
            break;
        case TORCH_UCX_OOB_TAG:
            TORCH_UCX_MAKE_OOB_RECV_TAG(ucp_tag, ucp_tag_mask, tag, src_rank);
            break;
        default:
            return TORCH_UCX_ERROR;
    };

    //fprintf(stderr, "rank %d recv tag %" PRIu64 " mask %" PRIu64 "\n", comm->rank, ucp_tag, ucp_tag_mask );
    st = ucp_tag_recv_nb(comm->worker, data, 1, dt, ucp_tag, ucp_tag_mask,
                         torch_ucx_recv_cmpl_cb);
    *req = reinterpret_cast<torch_ucx_request_t*>(st);
    /*TODO: check request*/

    return TORCH_UCX_OK;
}

static inline unsigned
torch_ucx_comm_progress(torch_ucx_comm_t *comm)
{
    return ucp_worker_progress(comm->worker);
}

static inline torch_ucx_status_t
torch_ucx_req_test(torch_ucx_comm_t *comm, torch_ucx_request_t **reqs,
                   int n_reqs, int *completed_idx, int poll_count,
                   int n_completions_required)
{
    int n_polls = 0;
    int n_completed;

    while (poll_count < 0 || n_polls++ < poll_count) {
        n_completed = 0;
        for (int i = 0; i < n_reqs; i++) {
            if (NULL == reqs[i]) {
                if (completed_idx) {
                    *completed_idx = i;
                }
                n_completed++;
            } else {
                if (reqs[i]->status != TORCH_UCX_REQUEST_DONE) {
                    torch_ucx_comm_progress(comm);
                } else {
                    torch_ucx_request_free(reqs[i]);
                    reqs[i] = NULL;
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

}
