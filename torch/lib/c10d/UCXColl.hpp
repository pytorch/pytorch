/**
 * * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * *
 * * See file LICENSE for terms.
 * */

#pragma once

#include <c10d/UCXSendRecv.hpp>

namespace c10d {

struct torch_ucx_coll_request_t;
typedef torch_ucx_status_t (*torch_ucx_progress_p)(torch_ucx_coll_request_t *request);

enum torch_ucx_memtype_t {
    TORCH_UCX_HOST,
    TORCH_UCX_CUDA
};

struct torch_ucx_coll_config_t {
    int  chunk;
    bool reverse;
    int  max_polls;
};

struct torch_ucx_coll_comm_t {
    torch_ucx_comm_t        *p2p_comm;
    torch_ucx_coll_config_t config;
    uint32_t                last_tag;
};

struct torch_ucx_coll_request_t {
    torch_ucx_coll_comm_t   *comm;
    uint32_t                tag;
    torch_ucx_progress_p    progress;
    torch_ucx_status_t      status;
    torch_ucx_memtype_t     src_buf_mtype;
    void                    *src_buffer;
    torch_ucx_memtype_t     dst_buf_mtype;
    void                    *dst_buffer;
    size_t                  len;
    std::vector<int>        send_lengths;
    std::vector<int>        send_offsets;
    std::vector<int>        recv_lengths;
    std::vector<int>        recv_offsets;
    torch_ucx_request_t     **reqs;
    int                     n_sreqs;
    int                     n_rreqs;
};

torch_ucx_status_t torch_ucx_coll_comm_init(torch_ucx_comm_t *p2p_comm,
                                            torch_ucx_coll_comm_t **comm);

torch_ucx_status_t torch_ucx_coll_test(torch_ucx_coll_request_t *request);

torch_ucx_status_t torch_ucx_alltoall_start(torch_ucx_coll_comm_t *comm,
                                            torch_ucx_coll_request_t *request);

torch_ucx_status_t torch_ucx_alltoall_progress(torch_ucx_coll_request_t *request);

torch_ucx_status_t torch_ucx_alltoallv_start(torch_ucx_coll_comm_t *comm,
                                             torch_ucx_coll_request_t *request);

torch_ucx_status_t torch_ucx_alltoallv_progress(torch_ucx_coll_request_t *request);

void torch_ucx_coll_comm_close(torch_ucx_coll_comm_t *comm);

}
