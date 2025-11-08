from __future__ import annotations

from torch._inductor.scheduler import BaseScheduling


class PallasScheduling(BaseScheduling):
    """
    The scheduler for the XLA backend. It is responsible for converting
    Inductor's IR into a representation that XLA can compile (e.g., HLO).

    It will need to implement fusion logic and kernel code generation.
    """

    def __init__(self, scheduler) -> None:
        super().__init__(scheduler)

    def can_fuse_vertical(self, node1, node2) -> bool:
        # TODO: Implement fusion logic for vertical fusion
        return False

    def can_fuse_horizontal(self, node1, node2) -> bool:
        # TODO: Implement fusion logic for horizontal fusion
        return False

    def codegen_node(self, node) -> None:
        """
        This is the main entry point for code generation.
        This method will take a SchedulerNode and generate the corresponding
        XLA kernel (e.g., by creating an HLO graph and calling the XLA compiler).
        """
        # TODO: Implement the logic to convert an Inductor IR node
        # into an XLA-compatible representation and generate the kernel.
        pass