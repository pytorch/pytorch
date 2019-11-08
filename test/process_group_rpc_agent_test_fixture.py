#!/usr/bin/env python3

import torch.distributed.rpc as rpc


class ProcessGroupRpcAgentTestFixture(object):
    @property
    def rpc_backend_name(self):
        return "PROCESS_GROUP"

    @property
    def rpc_agent_options(self):
        return rpc.backend_registry.construct_rpc_agent_options(
            self.rpc_backend,
            # Use enough 'num_send_recv_threads' until we fix https://github.com/pytorch/pytorch/issues/26359
            num_send_recv_threads=16,
        )
