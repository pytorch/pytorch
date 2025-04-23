# mypy: allow-untyped-defs
import importlib.util
from collections.abc import Sequence

import numpy as np
from torch._prims_common import ShapeType
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor.placement_types import Placement, Shard

__all__ = ["visualize_sharding"]

Color = tuple[float, float, float] | str


def _device_coords_in_mesh(mesh: DeviceMesh) -> dict[int, tuple[int, ...]]:
    """Given a device mesh, returns a dict from device index to coordinate."""
    return {
        device_index: coord for coord, device_index in np.ndenumerate(np.array(mesh.mesh.tolist()))
    }


def _shard_info(
    global_shape: ShapeType,
    mesh: DeviceMesh,
    placements: tuple[Placement],
    device_coords: dict[int, tuple[int, ...]],
    device_index: int,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Return shard shape, offset in the DTensor, and device name."""
    coord = device_coords[device_index]
    local_shape = list(global_shape)
    global_offset = [0] * len(global_shape)

    for idx, placement in enumerate(placements):
        mesh_dim_size = mesh.size(idx)
        if isinstance(placement, Shard):
            shard_dim = placement.dim
            local_offset = [0] * len(global_shape)
            assert shard_dim < len(
                local_shape
            ), f"Sharding dim {shard_dim} greater than tensor ndim {len(local_shape)}"
            shard_size, shard_offset = placement._local_shard_size_on_dim(
                local_shape[shard_dim], mesh_dim_size, coord[idx], return_offset=True
            )
            local_shape[shard_dim] = shard_size
            local_offset[shard_dim] = shard_offset
            # On a given dimension, if the local_offset[shard_dim] is smaller
            # than global_offset[shard_dim], it means that this dimension has
            # been already sharded in previous placement.  Therefore, we cannot
            # simply replace the global_offset[shard_dim] with
            # local_offset[shard_dim].  Instead, for the given shard_dim, we
            # need to add local_offset[shard_dim] to existing
            # global_offset[shard_dim].
            if global_offset[shard_dim] <= local_offset[shard_dim]:
                global_offset[shard_dim] = local_offset[shard_dim]
            else:
                global_offset[shard_dim] += local_offset[shard_dim]

    # TODO(wyi): global_offset[1] may not exist!
    row_range = (global_offset[0], global_offset[0] + local_shape[0] - 1)
    column_range = (global_offset[1], global_offset[1] + local_shape[1] - 1)
    return row_range, column_range, device_index


def _create_table(shards: list[tuple[tuple[int, ...], tuple[int, ...], int]]):
    """
    Creates a tabulate table given row and column ranges with device name
    """
    from tabulate import tabulate

    # Extract unique row and column ranges
    row_ranges = sorted({block[0] for block in shards})
    col_ranges = sorted({block[1] for block in shards})

    # Create a matrix initialized with empty strings
    matrix = [["" for _ in col_ranges] for _ in row_ranges]

    # Fill the matrix with values
    for block in shards:
        row_index = row_ranges.index(block[0])
        col_index = col_ranges.index(block[1])
        if matrix[row_index][col_index] == "":
            matrix[row_index][col_index] = str(block[2])
        else:
            matrix[row_index][col_index] += "," + str(block[2])

    # Prepare headers
    row_headers = [f"Row {r[0]}-{r[1]}" for r in row_ranges]
    col_headers = [f"Col {c[0]}-{c[1]}" for c in col_ranges]
    return tabulate(matrix, headers=col_headers, showindex=row_headers)


def make_color_iter(color_map, num_rows, num_cols):
    num_colors = num_rows * num_cols
    idx = 0
    for _ in range(num_colors):
        yield color_map(idx)
        idx += 1


def _canonicalize_color(color: Color) -> str:
    if isinstance(color, str):
        return color
    r, g, b = (int(a * 255) for a in color)
    return f"#{r:02X}{g:02X}{b:02X}"


def _get_text_color(color: str) -> str:
    r, g, b = map(lambda x: int(x, 16), (color[1:3], color[3:5], color[5:7]))
    if (r * 0.299 + g * 0.587 + b * 0.114) > 186:
        return "#000000"
    return "#ffffff"


def _create_rich_table(
    shape: ShapeType,
    shards: list[tuple[tuple[int, ...], tuple[int, ...], str]],
    device_kind: str = "",
    scale: float = 1.0,
    min_width: int = 9,
    max_width: int = 80,
):
    import matplotlib
    import rich.align
    import rich.box
    import rich.console
    import rich.padding
    import rich.style
    import rich.table

    dtensor_height = shape[0] if len(shape) > 0 else 1
    dtensor_width = shape[1] if len(shape) > 0 else shape[0]

    row_ranges = sorted({s[0] for s in shards})
    col_ranges = sorted({s[1] for s in shards})
    num_rows, num_cols = len(row_ranges), len(col_ranges)

    console = rich.console.Console(width=max_width)
    use_color = console.color_system
    color_iter = make_color_iter(matplotlib.colormaps["tab20b"], num_rows, num_cols)

    base_height = int(10 * scale)
    aspect_ratio = (shape[1] if len(shape) == 2 else 1) / shape[0]
    base_width = int(base_height * aspect_ratio)
    height_to_width_ratio = 2.5

    table = rich.table.Table(
        show_header=False,
        show_lines=not use_color,
        padding=0,
        highlight=not use_color,
        pad_edge=False,
        box=rich.box.SQUARE if not use_color else None,
    )
    for row in range(num_rows):
        table_row = []
        for col in range(num_cols):
            entry = (
                device_kind
                + ":"
                + ",".join(
                    [
                        str(device_id)
                        for row_range, col_range, device_id in shards
                        if row_range == row_ranges[row] and col_range == col_ranges[col]
                    ]
                )
            )
            width = (col_ranges[col][1] - col_ranges[col][0]) / dtensor_width
            width = int(width * base_width * height_to_width_ratio)
            height = (row_ranges[row][1] - row_ranges[row][0]) / dtensor_height
            height = int(height * base_height)
            left_padding, remainder = divmod(width - len(entry) - 2, 2)
            right_padding = left_padding + remainder
            top_padding, remainder = divmod(height - 2, 2)
            bottom_padding = top_padding + remainder
            if use_color:
                color = _canonicalize_color(next(color_iter)[:3])
                text_color = _get_text_color(color)
                top_padding += 1
                bottom_padding += 1
                left_padding += 1
                right_padding += 1
            else:
                color = None
                text_color = None
            padding = (
                max(top_padding, 0),
                max(right_padding, 0),
                max(bottom_padding, 0),
                max(left_padding, 0),
            )
            table_row.append(
                rich.padding.Padding(
                    rich.align.Align(entry, "center", vertical="middle"),
                    padding,
                    style=rich.style.Style(bgcolor=color, color=text_color),
                )
            )
        table.add_row(*table_row)
    console.print(table, end="\n\n")


def _has_rich_and_matplotlib() -> bool:
    return importlib.util.find_spec("rich") and importlib.util.find_spec("matplotlib")


def _has_tabulate() -> bool:
    return importlib.util.find_spec("tabulate")


def visualize_sharding(dtensor, header="", use_rich: bool = False):
    """Visualizes sharding in the terminal for :class:`DTensor` that are 1D or 2D."""
    if dtensor.numel() == 0:  # Do not print empty dtensors.
        return

    if len(dtensor.shape) >= 3:
        raise RuntimeError("visualize sharding supports only 1D or 2D DTensor")

    if dtensor.device_mesh.get_coordinate() is None:  # current rank is not in the mesh
        return

    # Only display the visualization once for each DTensor, on the rank whose
    # coordinate is 0 on all dimensions. For example, if the mesh is a full mesh,
    # we will only print on rank 0.
    local_rank_zero_on_all_dim = all(
        dtensor.device_mesh.get_local_rank(mesh_dim=dim) == 0
        for dim in range(dtensor.device_mesh.ndim)
    )
    if not local_rank_zero_on_all_dim:
        return

    device_coords = _device_coords_in_mesh(dtensor.device_mesh)  # {dev_id:mesh_coord}
    shards = [
        _shard_info(
            dtensor.shape,
            dtensor.device_mesh,
            dtensor.placements,
            device_coords,
            device_index,
        )
        for device_index in device_coords  # [rows, cols, dev_id]
    ]

    if _has_rich_and_matplotlib() and use_rich:
        _create_rich_table(dtensor.shape, shards, device_kind=dtensor.device_mesh.device_type)
    elif _has_tabulate():
        print(header)
        print(_create_table(shards))
    else:
        raise ValueError("`visualize_sharding` requires either `rich` or `tabulate`.")
