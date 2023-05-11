import os
from abc import ABC, abstractmethod

import torch.testing._internal.dist_utils


class RpcAgentTestFixture(ABC):
    @property
    def world_size(self) -> int:
        return 4

    @property
    def init_method(self):
        use_tcp_init = os.environ.get("RPC_INIT_WITH_TCP", None)
        if use_tcp_init == "1":
            master_addr = os.environ["MASTER_ADDR"]
            master_port = os.environ["MASTER_PORT"]
            return f"tcp://{master_addr}:{master_port}"
        else:
            return self.file_init_method

    @property
    def file_init_method(self):
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

    # Shutdown sequence is not well defined, so we may see any of the following
    # errors when running tests that simulate errors via a shutdown on the
    # remote end.
    @abstractmethod
    def get_shutdown_error_regex(self):
        """
        Return various error message we may see from RPC agents while running
        tests that check for failures. This function is used to match against
        possible errors to ensure failures were raised properly.
        """
        pass

    @abstractmethod
    def get_timeout_error_regex(self):
        """
        Returns a partial string indicating the error we should receive when an
        RPC has timed out. Useful for use with assertRaisesRegex() to ensure we
        have the right errors during timeout.
        """
        pass
