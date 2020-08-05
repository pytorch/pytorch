from abc import ABC, abstractmethod

import torch.testing._internal.dist_utils


class RpcAgentTestFixture(ABC):
    @property
    def world_size(self):
        return 4

    @property
    def init_method(self):
        return torch.testing._internal.dist_utils.INIT_METHOD_TEMPLATE.format(
            file_name=self.file_name
        )

    @property
    @abstractmethod
    def rpc_backend(self):
        pass

    @property
    @abstractmethod
    def rpc_backend_options(self):
        pass

    def setup_fault_injection(self, faulty_messages, messages_to_delay):
        """Method used by dist_init to prepare the faulty agent.

        Does nothing for other agents.
        """
        pass
