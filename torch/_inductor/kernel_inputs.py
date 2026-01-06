from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, TYPE_CHECKING, Union

import torch
import torch._inductor.config
from torch._inductor import ir
from torch._inductor.virtualized import V

from .ir import FixedLayout, FlexibleLayout, Layout


if TYPE_CHECKING:
    from collections.abc import Sequence

    import sympy


class KernelInputs(ABC):
    """
    Class to store and provide access to input nodes for kernels.
    This class takes in a tuple of input nodes and provides methods to access
    information about these nodes, such as their device type and device.
    """

    def __init__(
        self,
        input_nodes: list[Any],
        scalars: Optional[dict[str, Union[float, int]]] = None,
        out_dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize with a tuple of input nodes.

        Args:
            input_nodes: A tuple of input nodes to store
            out_dtype: Optional output dtype to store
        """
        self._input_nodes = input_nodes
        self._device_name: Optional[str] = None
        self._scalars = scalars if scalars is not None else {}
        self._out_dtype = out_dtype
        assert len(input_nodes) > 0, "Expected at least one input node"

    def nodes(self, reorder: Optional[Sequence[int]] = None) -> list[Any]:
        """
        Return the stored input nodes, optionally reordered.

        Args:
            reorder: Optional sequence of indices to reorder the nodes.
                    For example, (2, 0, 1) would return nodes in that order.

        Returns:
            The tuple of input nodes, optionally reordered
        """
        if reorder is None:
            return self._input_nodes
        assert len(self._input_nodes) == len(reorder), (
            f"Reorder length mismatch: {len(self._input_nodes)} vs {len(reorder)}"
        )
        return [self._input_nodes[i] for i in reorder]

    @property
    def count(self) -> int:
        """
        Get the number of input nodes.

        Returns:
            The number of input nodes
        """
        return len(self._input_nodes)

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
        return tuple(
            V.graph.sizevars.size_hints(
                node.get_size(),
                fallback=torch._inductor.config.unbacked_symint_fallback,
            )
            for node in self._input_nodes
        )

    def strides_symbolic(self) -> tuple[tuple[sympy.Integer, ...], ...]:
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

    @abstractmethod
    def out_dtype(self) -> torch.dtype:
        """
        Get the output dtype, whether passed in or inferred from the nodes

        Returns:
            The output dtype
        """

    def get_scalar(self, name: str) -> Union[float, int]:
        """
        Get the scalar value for a given name.

        Args:
            name: Name of the scalar to get

        Returns:
            The scalar value
        """
        assert name in self._scalars, f"Scalar {name} not found, but required"
        return self._scalars[name]

    @abstractmethod
    def output_layout(self, flexible: bool = True) -> Layout:
        """
        Abstract method to handle output layout generation.

        Args:
            out_dtype: Optional output dtype. If not provided, infer from inputs
            flexible: If True, return FlexibleLayout, otherwise FixedLayout
        """


class MMKernelInputs(KernelInputs):
    """
    Specialized KernelInputs for matrix multiplication operations.
    Provides additional methods to access M, N, K dimensions.
    """

    def __init__(
        self,
        input_nodes: list[Any],
        scalars: Optional[dict[str, Union[float, int]]] = None,
        out_dtype: Optional[torch.dtype] = None,
        mat1_idx: int = -2,
        mat2_idx: int = -1,
    ):
        """
        Initialize with a tuple of input nodes.

        By default, we assume the last 2 input nodes are mat1 and mat2, but
        the caller can adjust when necessary
        """
        super().__init__(input_nodes, scalars, out_dtype)
        # for mm, we need at least 2 nodes, and we need to know which nodes
        # are the main matrixes e.g. addmm is (bias, mat1, mat2) whereas others
        # might be (mat1, mat2, scale), etc.
        assert len(self._input_nodes) >= 2, "Expected at least 2 input nodes"

        # Adjust assertions to handle negative indices
        m1_idx, m2_idx = mat1_idx, mat2_idx
        if mat1_idx < 0:
            m1_idx += len(input_nodes)
        if mat2_idx < 0:
            m2_idx += len(input_nodes)

        assert 0 <= m1_idx < len(input_nodes), f"Invalid mat1_idx: {mat1_idx}"
        assert 0 <= m2_idx < len(input_nodes), f"Invalid mat2_idx: {mat2_idx}"

        self._mat1_idx = mat1_idx
        self._mat2_idx = mat2_idx

    def mnk_symbolic(
        self,
    ) -> tuple[sympy.Integer, sympy.Integer, sympy.Integer]:
        """
        Get the symbolic M, N, K dimensions for matrix multiplication.
        Handles both 2D (MM) and 3D (BMM) tensors.

        M is extracted from the second-to-last dimension of the first operand (mat1).
        N is extracted from the last dimension of the second operand (mat2).
        K is extracted from the last dimension of the first operand (mat1).

        Returns:
            A tuple of (M, N, K) dimensions
        """
        mat1 = self.nodes()[self._mat1_idx]
        mat2 = self.nodes()[self._mat2_idx]

        m = mat1.get_size()[-2]  # M from second-to-last dimension of mat1
        k = mat1.get_size()[-1]  # K from last dimension of mat1
        n = mat2.get_size()[-1]  # N from last dimension of mat2

        # Ensure K dimensions match between operands
        k0 = mat2.get_size()[-2]  # K from second-to-last dimension of mat2
        V.graph.sizevars.check_equals(k, k0)
        return (m, n, k)

    def out_dtype(self) -> torch.dtype:
        """
        Get the output dtype, whether passed in or inferred from the nodes

        Returns:
            The output dtype
        """
        if self._out_dtype is not None:
            return self._out_dtype
        return self.mat1mat2()[0].get_dtype()

    def output_layout(self, flexible: bool = True) -> Layout:
        """
        Handle output layout generation for matrix multiplication.

        Args:
            out_dtype: Optional output dtype. If not provided, infer from inputs
            flexible: If True, return FlexibleLayout, otherwise FixedLayout
        """
        mat1, mat2 = self.mat1mat2()
        out_dtype = self.out_dtype()
        # NOTE: taken from mm_common.mm_args
        *b1, m, k1 = mat1.get_size()
        *b2, k2, n = mat2.get_size()
        b = [V.graph.sizevars.check_equals_and_simplify(a, b) for a, b in zip(b1, b2)]
        size = [*b, m, n]
        if flexible:
            return FlexibleLayout(self.device(), out_dtype, size)
        else:
            return FixedLayout(self.device(), out_dtype, size)

    def mat1mat2(self) -> tuple[Any, Any]:
        """
        Get the mat1 and mat2 nodes.

        Returns:
            A tuple of (mat1, mat2) nodes
        """
        nodes = self.nodes()
        return nodes[self._mat1_idx], nodes[self._mat2_idx]

    def mnk_hinted(self) -> tuple[int, int, int]:
        """
        Get the hinted M, N, K dimensions for matrix multiplication.
        Handles both 2D (MM) and 3D (BMM) tensors.

        Uses shapes_hinted from the base class to get integer hints for dimensions.

        Returns:
            A tuple of (M, N, K) dimensions as integers
        """
        hinted_shapes = self.shapes_hinted()
        mat1_shape = hinted_shapes[self._mat1_idx]
        mat2_shape = hinted_shapes[self._mat2_idx]

        m = mat1_shape[-2]  # M from second-to-last dimension of mat1
        k = mat1_shape[-1]  # K from last dimension of mat1
        n = mat2_shape[-1]  # N from last dimension of mat2

        # Ensure K dimensions match between operands
        k_check = mat2_shape[-2]  # K from second-to-last dimension of mat2
        assert k == k_check, f"K dimensions don't match: {k} vs {k_check}"

        return (m, n, k)
