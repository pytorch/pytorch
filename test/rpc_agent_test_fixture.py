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
    def rpc_backend_name(self):
        raise NotImplementedError("self.rpc_backend_name property is required to be implemented.")

    @property
    def rpc_agent_options(self):
        # This is an abstract test suite class,
        # different RpcAgent has different types of RpcAgentOptions.
        raise NotImplementedError(
            (
                "self.rpc_agent_options property not implemented for {rpc_backend}."
            ).format(rpc_backend=self.rpc_backend)
        )

    def test_backend_selected(self):
        # Make sure the correct backend is selected from the registry.
        self.assertEqual(self.rpc_backend.name, self.rpc_backend_name)
