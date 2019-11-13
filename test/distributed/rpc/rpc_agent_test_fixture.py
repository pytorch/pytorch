#!/usr/bin/env python3

import distributed.rpc.dist_utils as dist_utils
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
    def rpc_backend_name(self):
        raise NotImplementedError(
            "self.rpc_backend_name property is required to be implemented."
        )

    def test_backend_selected(self):
        # Make sure the correct backend is selected from the registry.
        self.assertEqual(self.rpc_backend.name, self.rpc_backend_name)
