from collections.abc import Callable
from typing import Annotated

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir


Scalar = cutlass.Numeric
StorageScalar = Annotated[Scalar, "storage"]
ComputeScalar = Annotated[Scalar, "compute"]
ScalarResult = Scalar | ir.Value
MathResult = ScalarResult | tuple[ScalarResult, ...]
MathFn = Callable[..., MathResult]
NumericType = type[Scalar]
Index = ScalarResult | int | cute.SymInt
