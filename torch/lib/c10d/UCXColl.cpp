/**
 * * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * *
 * * See file LICENSE for terms.
 * */

#include <cstdlib>
#include <c10d/UCXColl.hpp>

namespace c10d {

static void torch_ucx_get_coll_config(torch_ucx_coll_config_t *config)
{
    char *env;

    config->chunk     = 1;
    config->reverse   = 0;
    config->max_polls = 10;
 
    env = std::getenv("TORCH_UCX_CHUNK");
    if (env) {
        config->chunk = std::atoi(env);
    }
    env = std::getenv("TORCH_UCX_REVERSE");
    if (env) {
        config->reverse = std::atoi(env);
    }
    env = std::getenv("TORCH_UCX_MAX_POLLS");
    if (env) {
        config->max_polls = std::atoi(env);
    }
}

torch_ucx_status_t torch_ucx_coll_comm_init(torch_ucx_comm_t *p2p_comm,
                                            torch_ucx_coll_comm_t **comm)
{
    torch_ucx_coll_comm_t *coll_comm;

    coll_comm = new torch_ucx_coll_comm_t;
    torch_ucx_get_coll_config(&coll_comm->config);
    coll_comm->p2p_comm = p2p_comm;
    coll_comm->last_tag = 0;

    *comm = coll_comm;
    return TORCH_UCX_OK;
}

torch_ucx_status_t torch_ucx_coll_test(torch_ucx_coll_request_t *request)
{
    if (request->status == TORCH_UCX_INPROGRESS) {
        request->progress(request);
    }
    return request->status;
}

void torch_ucx_coll_comm_close(torch_ucx_coll_comm_t *comm)
{
    delete comm;
}

}
