from __future__ import annotations

from typing import Any, Optional

import torch
from torch._inductor import ir


class KernelInputs:
    """
    Class to store and provide access to input nodes for kernels.
    This class takes in a tuple of input nodes and provides methods to access
    information about these nodes, such as their device type and device.
    """

    def __init__(self, input_nodes: tuple[Any, ...]):
        """
        Initialize with a tuple of input nodes.

        Args:
            input_nodes: A tuple of input nodes to store
        """
        self._input_nodes = input_nodes
        self._device_name: Optional[str] = None

    def nodes(self) -> tuple[Any, ...]:
        """
        Return the stored input nodes.

        Returns:
            The tuple of input nodes
        """
        return self._input_nodes

    @property
    def device_type(self) -> Optional[str]:
        """
        Get the device type of the first node.

        Returns:
            The device type (e.g., 'cuda', 'cpu')
        """

        return ir.get_device_type(self._input_nodes[0])

    def device(self) -> torch.device:
        """
        Get the device of the first node.

        Returns:
            The device of the first node
        """
        return self._input_nodes[0].get_device()

    def device_name(self) -> Optional[str]:
        """
        Get the device name information.

        Returns:
            A tuple of (gpu_name, vendor, model)
        """
        if self._device_name is None:
            device = self.device()
            if self.device_type == "cuda":
                device_properties = torch.cuda.get_device_properties(device)
                self._device_name = device_properties.gcnArchName
        return self._device_name

    def shapes_symbolic(self) -> tuple[tuple[Any, ...], ...]:
        """
        Get the symbolic shapes of all input nodes.

        Returns:
            A tuple of shape tuples for each input node
        """
        return tuple(node.get_size() for node in self._input_nodes)

    def shapes_hinted(self) -> tuple[tuple[int, ...], ...]:
        """
        Get the size hints for shapes of all input nodes.

        Returns:
            A tuple of shape tuples with integer hints for each input node
        """
        import torch._inductor.config
        from torch._inductor.virtualized import V

        return tuple(
            V.graph.sizevars.size_hints(
                node.get_size(),
                fallback=torch._inductor.config.unbacked_symint_fallback,
            )
            for node in self._input_nodes
        )

    def strides_symbolic(self) -> tuple[tuple[Any, ...], ...]:
        """
        Get the symbolic strides of all input nodes.

        Returns:
            A tuple of stride tuples for each input node
        """
        return tuple(node.get_stride() for node in self._input_nodes)

    def strides_hinted(self) -> tuple[tuple[int, ...], ...]:
        """
        Get the size hints for strides of all input nodes.

        Returns:
            A tuple of stride tuples with integer hints for each input node
        """
        import torch._inductor.config
        from torch._inductor.virtualized import V

        return tuple(
            V.graph.sizevars.size_hints(
                node.get_stride(),
                fallback=torch._inductor.config.unbacked_symint_fallback,
            )
            for node in self._input_nodes
        )

    def dtypes(self) -> tuple[torch.dtype, ...]:
        """
        Get the dtypes of all input nodes.

        Returns:
            A tuple of dtypes for each input node
        """
        return tuple(node.get_dtype() for node in self._input_nodes)

    def dtype(self, idx: int = 0) -> torch.dtype:
        """
        Get the dtype of a specific input node.

        Args:
            idx: Index of the node to get the dtype from (default: 0)

        Returns:
            The dtype of the specified input node
        """
        return self._input_nodes[idx].get_dtype()


class MMKernelInputs(KernelInputs):
    """
    Specialized KernelInputs for matrix multiplication operations.
    Provides additional methods to access M, N, K dimensions.
    """

    def mnk_symbolic(self) -> tuple[Any, Any, Any]:
        """
        Get the symbolic M, N, K dimensions for matrix multiplication.

        M is extracted from the first dimension of the first operand (2nd to last node).
        N is extracted from the second dimension of the second operand (last node).
        K is extracted from the second dimension of the first operand (2nd to last node).

        Returns:
            A tuple of (M, N, K) dimensions
        """
        operands = self.nodes()[-2:]  # Get the last two nodes as operands
        m = operands[0].get_size()[
            0
        ]  # M from first dimension of first operand (2nd to last)
        k = operands[0].get_size()[
            1
        ]  # K from second dimension of first operand (2nd to last)
        n = operands[1].get_size()[
            1
        ]  # N from second dimension of second operand (last)
        return (m, n, k)

    def mnk_hinted(self) -> tuple[int, int, int]:
        """
        Get the hinted M, N, K dimensions for matrix multiplication.

        Uses shapes_hinted from the base class to get integer hints for dimensions.

        Returns:
            A tuple of (M, N, K) dimensions as integers
        """
        # Get the hinted shapes for the last two nodes (operands)
        hinted_shapes = self.shapes_hinted()[-2:]

        # Extract M, N, K from the hinted shapes
        m = hinted_shapes[0][0]  # M from first dimension of first operand (2nd to last)
        k = hinted_shapes[0][
            1
        ]  # K from second dimension of first operand (2nd to last)
        n = hinted_shapes[1][1]  # N from second dimension of second operand (last)

        # Ensure K dimensions match between operands
        k_check = hinted_shapes[1][0]
        assert k == k_check, f"K dimensions don't match: {k} vs {k_check}"

        return (m, n, k)
