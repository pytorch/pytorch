from __future__ import absolute_import, division, print_function, unicode_literals

from distributed.rpc.dist_autograd_test import DistAutogradTest
from distributed.rpc.process_group.process_group_rpc_agent_mixin import (
    ProcessGroupRpcAgentMixin,
)


class ProcessGroupDistAutogradTest(ProcessGroupRpcAgentMixin, DistAutogradTest):
    """
        Allowing customization on top of `DistAutogradTest`,
        including skipping and adding test methods
        specifically for `ProcessGroupAgent`.
    """

    pass
