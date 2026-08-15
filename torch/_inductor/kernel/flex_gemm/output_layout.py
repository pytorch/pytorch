# mypy: allow-untyped-defs
"""Cutlass-free descriptors for caller-owned FlexGEMM output layouts.

CuTeDSL callbacks live in ``output_layout_cutedsl`` so analysis and fake-tensor
allocation can import this module without loading Cutlass. QuACK fingerprints
and executes those callbacks without interpreting the physical layout.
"""

import dataclasses
import importlib
from collections.abc import Callable
from typing import Any


@dataclasses.dataclass(frozen=True)
class FlexGemmOutputLayout:
    """Carry one physical output-layout contract across analysis and runtime."""

    symbol: str
    name: str
    carrier_shape_fn: Callable[[Any, Any, Any], tuple[Any, ...]]
    cutedsl_callbacks_fn: Callable[[], tuple[Callable[..., Any], Callable[..., Any]]]
    carrier_ndim: int
    validate_geometry_fn: Callable[[Any], None]
    supports_config_fn: Callable[[Any, int, int], bool] | None = None
    validate_carrier_fn: Callable[[Any], None] | None = None

    def validate_geometry(self, geometry: Any) -> None:
        """Validate layout-specific logical reduction geometry."""
        self.validate_geometry_fn(geometry)

    def runtime_view(self, tensor: Any, batch: Any, rows: Any, cols: Any) -> Any:
        """Attach the descriptor's physical carrier shape without moving data."""
        return tensor.view(self.carrier_shape_fn(batch, rows, cols))

    def quack_layout(self, grouped_reduce: Any) -> Any:
        """Build QuACK's generic descriptor from caller-owned callbacks."""
        tensor_fn, fake_shape_fn = self.cutedsl_callbacks_fn()
        return grouped_reduce.GroupedLocalReduceOutputLayout(
            name=self.name,
            tensor_fn=tensor_fn,
            carrier_shape_fn=self.carrier_shape_fn,
            fake_shape_fn=fake_shape_fn,
            carrier_ndim=self.carrier_ndim,
            supports_config_fn=self.supports_config_fn,
            validate_carrier_fn=self.validate_carrier_fn,
        )

    def codegen_reference(self) -> str:
        """Return this module constant's generated-runtime expression."""
        module = importlib.import_module(__name__)
        if getattr(module, self.symbol, None) is not self:
            raise ValueError(
                f"FlexGEMM output layout {self.name!r} must be bound as {self.symbol}"
            )
        return f"flex_gemm_output_layout.{self.symbol}"


def output_layout_cutedsl() -> Any:
    """Import physical layout callbacks only when building a QuACK EpiOp."""
    return importlib.import_module(
        "torch._inductor.kernel.flex_gemm.output_layout_cutedsl"
    )


def blocked_128x4_cutedsl_callbacks() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Return blocked-layout tensor and fake-shape callbacks."""
    cutedsl = output_layout_cutedsl()
    return cutedsl.blocked_128x4_output_tensor, cutedsl.blocked_128x4_fake_shape


def blocked_128x4_carrier_shape(batch: Any, rows: Any, cols: Any) -> tuple[Any, ...]:
    """Return the physical carrier shape for blocked 128x4 matrices."""
    return (batch, (rows + 127) // 128, (cols + 3) // 4, 512)


def blocked_128x4_validate_geometry(geometry: Any) -> None:
    """Require blocked scales to contract groups along logical N."""
    if geometry.axis != 1:
        raise NotImplementedError(
            "blocked local-reduce outputs currently support only axis 1"
        )


def blocked_128x4_validate_carrier(tensor: Any) -> None:
    """Require the physical 128x4 carrier expected by the swizzle callback."""
    if not tensor.is_contiguous():
        raise ValueError("blocked_128x4 carrier must be contiguous")


def blocked_128x4_supports_config(config: Any, axis: int, group: int) -> bool:
    """Return whether a GEMM tile composes with the blocked layout atoms."""
    if axis != 1 or config.device_capacity != 10 or config.tile_m % config.cluster_m:
        return False
    cta_tile_m = config.tile_m // config.cluster_m
    if config.swap_ab:
        if cta_tile_m % group:
            return False
        row_tile = config.tile_n
        column_tile = cta_tile_m // group
    else:
        if config.tile_n % group:
            return False
        row_tile = cta_tile_m
        column_tile = config.tile_n // group
    return (128 % row_tile == 0 or row_tile % 128 == 0) and (
        4 % column_tile == 0 or column_tile % 4 == 0
    )


def validate_any_geometry(_geometry: Any) -> None:
    """Accept any grouped-reduction geometry."""


def transposed_cutedsl_callbacks() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Return transposed-layout tensor and fake-shape callbacks."""
    cutedsl = output_layout_cutedsl()
    return cutedsl.transposed_output_tensor, cutedsl.transposed_fake_shape


def transposed_carrier_shape(batch: Any, rows: Any, cols: Any) -> tuple[Any, ...]:
    """Return contiguous transposed storage with an explicit batch dimension."""
    return (batch, cols, rows)


def transposed_supports_config(config: Any, _axis: int, _group: int) -> bool:
    """Reject swap because the storage transform already transposes the output."""
    return not config.swap_ab


def transposed_validate_carrier(tensor: Any) -> None:
    """Require the dense carrier represented by a contiguous output transpose."""
    if not tensor.is_contiguous():
        raise ValueError("transposed carrier must be contiguous")


BLOCKED_128X4 = FlexGemmOutputLayout(
    symbol="BLOCKED_128X4",
    name="blocked_128x4",
    carrier_shape_fn=blocked_128x4_carrier_shape,
    cutedsl_callbacks_fn=blocked_128x4_cutedsl_callbacks,
    carrier_ndim=4,
    validate_geometry_fn=blocked_128x4_validate_geometry,
    supports_config_fn=blocked_128x4_supports_config,
    validate_carrier_fn=blocked_128x4_validate_carrier,
)

TRANSPOSED = FlexGemmOutputLayout(
    symbol="TRANSPOSED",
    name="transposed",
    carrier_shape_fn=transposed_carrier_shape,
    cutedsl_callbacks_fn=transposed_cutedsl_callbacks,
    carrier_ndim=3,
    validate_geometry_fn=validate_any_geometry,
    supports_config_fn=transposed_supports_config,
    validate_carrier_fn=transposed_validate_carrier,
)
