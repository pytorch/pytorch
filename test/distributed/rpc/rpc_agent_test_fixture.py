#!/usr/bin/env python3

import distributed.rpc.dist_utils as dist_utils


class RpcAgentTestFixture(object):
    @property
    def world_size(self):
        return 4

    @property
    def init_method(self):
        return dist_utils.INIT_METHOD_TEMPLATE.format(file_name=self.file_name)

    @property
    def rpc_backend(self):
        raise NotImplementedError(
            "self.rpc_backend property is required to be implemented."
        )
