# mypy: allow-untyped-defs
"""
MultiDimKernelDispatcher for dispatching to the closest kernel based on runtime values.

This utility allows generating multiple specialized kernels for different ranges of symbolic
integers and dispatching to the closest one at runtime based on Euclidean distance.
"""

from typing import Callable, List, Tuple, Sequence, TypeVar, Any
from typing_extensions import ParamSpec
import math

_P = ParamSpec("_P")
_T = TypeVar("_T")

# Type alias for kernels with .run(...) method
KernelType = Any  # Will be specific kernel types like TritonKernel


class MultiDimKernelDispatcher:
    """
    Dispatcher that selects the closest kernel based on runtime parameter values.
    
    During compilation, we split symbolic integer ranges and generate specialized kernels
    for different points in that range. At runtime, we dispatch to the kernel that was
    specialized for the point closest to the actual runtime values.
    
    Args:
        kernels: List of (kernel, specialization_point) tuples where:
            - kernel: The specialized kernel object with a .run() method
            - specialization_point: Tuple of values this kernel was specialized for
    """
    
    def __init__(self, kernels: List[Tuple[KernelType, Sequence[float]]]) -> None:
        # Store as (specialization_point, kernel) for easier lookup
        self.specializations: List[Tuple[Tuple[float, ...], KernelType]] = [
            (tuple(specialization), fn) for fn, specialization in kernels
        ]
    
    def _distance(self, a: Sequence[float], b: Sequence[float]) -> float:
        """Compute Euclidean distance between two points."""
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))
    
    def dispatch(self, point: Sequence[float]) -> Tuple[KernelType, Tuple[int, int, int]]:
        """
        Dispatch to the closest kernel and compute grid dimensions.
        
        Args:
            point: Runtime values to dispatch on (e.g., actual batch size)
            
        Returns:
            Tuple of (kernel, grid_dimensions)
        """
        if not self.specializations:
            raise RuntimeError("No kernel specializations available")
            
        # Find the kernel with the closest specialization point
        best_kernel = min(
            self.specializations,
            key=lambda pair: self._distance(point, pair[0])
        )[1]
        
        # Generic grid computation based on point[0] (e.g., batch size)
        # This follows the pattern: 64 * ((127 + batch_size) // 128)
        grid_x = 64 * ((127 + int(point[0])) // 128)
        return best_kernel, (grid_x, 1, 1)


def create_symint_specialization_points(
    symint_min: int = 0, 
    symint_max: int = 4096, 
    num_points: int = 4
) -> List[float]:
    """
    Create evenly spaced specialization points for a symbolic integer range.
    
    Args:
        symint_min: Minimum value for the symbolic integer
        symint_max: Maximum value for the symbolic integer  
        num_points: Number of specialization points to create
        
    Returns:
        List of specialization points (midpoints of evenly divided ranges)
    """
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    
    if num_points == 1:
        return [(symint_min + symint_max) / 2]
    
    # Split range into num_points evenly spaced ranges
    range_size = (symint_max - symint_min) / num_points
    
    specialization_points = []
    for i in range(num_points):
        range_start = symint_min + i * range_size
        range_end = symint_min + (i + 1) * range_size
        midpoint = (range_start + range_end) / 2
        specialization_points.append(midpoint)
    
    return specialization_points


def get_symint_range_bounds(symint_expr, default_min: int = 0, default_max: int = 4096) -> Tuple[int, int]:
    """
    Get the range bounds for a symbolic integer expression.
    
    Args:
        symint_expr: The symbolic integer expression
        default_min: Default minimum if bounds cannot be determined
        default_max: Default maximum if bounds cannot be determined
        
    Returns:
        Tuple of (min_value, max_value)
    """
    # TODO: Integrate with PyTorch's symbolic shape analysis to get actual bounds
    # For now, use defaults extended to ensure we cover the full range
    actual_min = min(default_min, 0)
    actual_max = max(default_max, 4096)
    
    return actual_min, actual_max