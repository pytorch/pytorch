from typing import List, Sequence, Tuple
from torch._prims_common import ShapeType
from torch.distributed._tensor import DeviceMesh, Shard
from torch.distributed._tensor.placement_types import (
    Placement,
    Shard,
)
import numpy as np

def _mesh_to_coordinate(mesh, device_type):
    '''
    Given a n-dimensional list of device mesh, this function creates a map of
    device and its coordinate
    '''
    # Convert the n-dimensional list to a NumPy array
    np_mesh = np.array(mesh.mesh.tolist())

    # Create a dictionary to map each value to its coordinate
    device_to_coordinate_map = {}
    for coord, value in np.ndenumerate(np_mesh):
        # device is unique in device_mesh
        device_to_coordinate_map[f"{device_type}:{str(value)}"] = list(coord)

    return device_to_coordinate_map

def _convert_offset_to_ranges(all_offsets):
    '''
    Using tabulate package to create a table is easier when we specify row and col ranges
    This function converts offsets to ranges.
    '''
    converted_blocks = []

    for offset in all_offsets:
        shape, offset, value = offset

        # Calculate row_range and column_range
        row_range = (offset[0], offset[0] + shape[0] - 1)
        column_range = (offset[1], offset[1] + shape[1] - 1)

        # Convert value to string to match your desired format
        converted_block = {
            'row_range': row_range, 
            'column_range': column_range, 
            'value': str(value)
        }
        converted_blocks.append(converted_block)

    return converted_blocks

def _create_table(blocks):
    '''
    Creates a tabulate table given row and column ranges with device name
    '''
    try:
        from tabulate import tabulate
    except ImportError:
        raise ImportError("tabulate package is required to visualize sharding")

    # Extract unique row and column ranges
    row_ranges = sorted(set([block['row_range'] for block in blocks]))
    col_ranges = sorted(set([block['column_range'] for block in blocks]))

    # Create a matrix initialized with empty strings
    matrix = [['' for _ in col_ranges] for _ in row_ranges]

    # Fill the matrix with values
    for block in blocks:
        row_index = row_ranges.index(block['row_range'])
        col_index = col_ranges.index(block['column_range'])
        if matrix[row_index][col_index] == '':
            matrix[row_index][col_index] = block['value']
        else:
            matrix[row_index][col_index] += ', ' + block['value']

    # Prepare headers
    row_headers = [f"Row {r[0]}-{r[1]}" for r in row_ranges]
    col_headers = [f"Col {c[0]}-{c[1]}" for c in col_ranges]

    return tabulate(matrix, headers=col_headers, showindex=row_headers)

def compute_local_shape_and_global_offset(
    global_shape: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement], my_coordinate: List[int]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Same as torch.distributed._tensor._utils.compute_local_shape_and_global_offset but
    with custom my_coordinate input

    Compute the local tensor shape and the global offsets into the original tensor
    of a DTensor on its current global rank. This is useful for checkpointing purpose.

    Example (2 host with 4GPUs each):
    # Below is a DeviceMesh with mesh_shape of (2, 4)
    mesh = DeviceMesh(device_type="cuda",
                        mesh=[
                        [0, 1, 2, 3],
                        [4, 5, 6, 7]
                        ],
    )

    Let's say we distribute a global_tensor of shape (8,4) over the above DeviceMesh
    with a placements of [Shard(0), Shard(0)].
    The local shape and global offset will be as follows:
    rank0 -- local_shape:[1, 4], global_offset:[0, 0]
    rank1 -- local_shape:[1, 4], global_offset:[1, 0]
    rank2 -- local_shape:[1, 4], global_offset:[2, 0]
    rank5 -- local_shape:[1, 4], global_offset:[5, 0]
    rank3 -- local_shape:[1, 4], global_offset:[3, 0]
    rank4 -- local_shape:[1, 4], global_offset:[4, 0]
    rank6 -- local_shape:[1, 4], global_offset:[6, 0]
    rank7 -- local_shape:[1, 4], global_offset:[7, 0]
    """

    if my_coordinate is None:
        # if rank not in the mesh, return empty offset
        return ((), ())
    else:
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
                    local_shape[shard_dim],
                    mesh_dim_size,
                    my_coordinate[idx],
                    return_offset=True,
                )

                local_shape[shard_dim] = shard_size
                local_offset[shard_dim] = shard_offset

                # On a given dimension, if the local_offset[shard_dim] is smaller than global_offset[shard_dim],
                # it means that this dimension has been already sharded in previous placement.
                # Therefore, we cannot simply replace the global_offset[shard_dim] with local_offset[shard_dim].
                # Instead, for the given shard_dim, we need to add local_offset[shard_dim] to existing global_offset[shard_dim].
                if global_offset[shard_dim] <= local_offset[shard_dim]:
                    global_offset[shard_dim] = local_offset[shard_dim]
                else:
                    global_offset[shard_dim] += local_offset[shard_dim]

        return tuple(local_shape), tuple(global_offset)

def visualize_sharding(dtensor):
    '''
    Visualizes sharding in 1D-2D dtensors
    Requires tabulate, install with `pip install tabulate`
    '''
    if len(dtensor.shape) >= 3:
        raise RuntimeError("visualize sharding is only implemented for 1D or 2D dtensor")
    placements = dtensor.placements
    device_mesh = dtensor.device_mesh
    device_type = dtensor.device_mesh.device_type

    device_map = _mesh_to_coordinate(device_mesh, device_type)
    all_offsets = []
    for device in device_map:
        local_shape, global_offset = compute_local_shape_and_global_offset(dtensor.shape, device_mesh, placements, device_map[device])
        all_offsets.append([local_shape, global_offset, device])
    
    # Convert offsets to blocks with row_ranges for tabulate
    blocks = _convert_offset_to_ranges(all_offsets)
    if device_mesh.get_rank() == 0:
        print(_create_table(blocks))