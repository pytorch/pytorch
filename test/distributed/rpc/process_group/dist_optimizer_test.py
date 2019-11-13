from __future__ import absolute_import, division, print_function, unicode_literals

from distributed.rpc.dist_optimizer_test import DistOptimizerTest
from distributed.rpc.process_group.process_group_rpc_agent_mixin import (
    ProcessGroupRpcAgentMixin,
)


class ProcessGroupDistOptimizerTest(ProcessGroupRpcAgentMixin, DistOptimizerTest):
    """
        Allowing customization on top of `DistOptimizerTest`,
        including skipping and adding test methods
        specifically for `ProcessGroupAgent`.
    """

    pass
