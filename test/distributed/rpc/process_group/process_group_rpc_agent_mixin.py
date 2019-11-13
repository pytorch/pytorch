#!/usr/bin/env python3


class ProcessGroupRpcAgentMixin(object):
    @property
    def rpc_backend_name(self):
        return "PROCESS_GROUP"
