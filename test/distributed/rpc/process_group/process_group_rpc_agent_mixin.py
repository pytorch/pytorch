#!/usr/bin/env python3


class ProcessGroupRpcAgentMixin(object):
    @property
    def rpc_backend(self):
        return rpc.backend_registry.BackendType["PROCESS_GROUP"]
