from __future__ import absolute_import, division, print_function, unicode_literals

import dist_autograd_test
import process_group_rpc_agent_test_fixture


class ProcessGroupDistAutogradTest(
    process_group_rpc_agent_test_fixture.ProcessGroupRpcAgentTestFixture,
    dist_autograd_test.DistAutogradTest,
):
    """
        Allowing customization on top of `DistAutogradTest`,
        including skipping and adding test methods
        specifically for `ProcessGroupAgent`.
    """
