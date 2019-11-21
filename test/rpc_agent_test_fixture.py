#!/usr/bin/env python3

import dist_utils
import torch.distributed.rpc as rpc


class RpcAgentTestFixture(object):
    @property
    def world_size(self):
        return 4

    @property
    def init_method(self):
        return dist_utils.INIT_METHOD_TEMPLATE.format(file_name=self.file_name)

    @property
    def rpc_backend(self):
        return rpc.backend_registry.BackendType[dist_utils.TEST_CONFIG.rpc_backend_name]

    @property
    def rpc_backend_options(self):
        return dist_utils.TEST_CONFIG.build_rpc_backend_options(self)
