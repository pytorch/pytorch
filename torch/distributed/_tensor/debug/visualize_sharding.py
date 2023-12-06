import torch
from torch.distributed._tensor import DeviceMesh, Shard, Replicate, distribute_tensor
import os

def shard_or_replicate(curr_ranges, placements, num_splits, idx, grid, device_mesh):
    if len(placements) == idx:
        assert not isinstance(device_mesh, list)
        for i in curr_ranges['row']:
            for j in curr_ranges['col']:
                grid[i][j] = f"Device:{str(device_mesh)}" if grid[i][j] == "" else grid[i][j] + f", Device:{str(device_mesh)}"
        return curr_ranges

    placement = placements[idx]

    next_ranges = []
    if placement.is_shard():
        curr_dim = 'row' if placement.dim == 0 else 'col'
        other_dim = 'row' if placement.dim == 1 else 'col'

        chunk_size, extra = divmod(len(curr_ranges[curr_dim]), num_splits[idx])
        start_entry_index = 0
        for part_idx in range(num_splits[idx]):
            curr_chunk = chunk_size + (1 if part_idx < extra else 0)
            new_range = {curr_dim: curr_ranges[curr_dim][start_entry_index:start_entry_index + curr_chunk], other_dim: curr_ranges[other_dim]}
            next_ranges.append(shard_or_replicate(new_range, placements, num_splits, idx+1, grid, device_mesh[part_idx]))
            start_entry_index += curr_chunk
    elif placement.is_replicate():
        for part_idx in range(num_splits[idx]):
            next_ranges.append(shard_or_replicate(curr_ranges, placements, num_splits, idx+1, grid, device_mesh[part_idx]))
    else:
        raise RuntimeError
    return next_ranges


def find_unique_blocks(grid):
    rows = len(grid)
    columns = len(grid[0]) if rows > 0 else 0
    compact_blocks = []

    for i in range(rows):
        for j in range(columns):
            # Check if cell is the top-left corner of a new block
            if (i == 0 or grid[i][j] != grid[i-1][j]) and (j == 0 or grid[i][j] != grid[i][j-1]):
                value = grid[i][j]
                block_rows, block_cols = 1, 1

                # Determine the size of the block
                while i + block_rows < rows and grid[i + block_rows][j] == value:
                    block_rows += 1
                while j + block_cols < columns and grid[i][j + block_cols] == value:
                    block_cols += 1

                # Add block information to the list
                compact_blocks.append({
                    'row_range': (i, i + block_rows - 1),
                    'column_range': (j, j + block_cols - 1),
                    'value': value
                })

    return compact_blocks

def create_table(blocks):
    from tabulate import tabulate

    # Extract unique row and column ranges
    row_ranges = sorted(set([block['row_range'] for block in blocks]))
    col_ranges = sorted(set([block['column_range'] for block in blocks]))

    # Create a matrix initialized with empty strings
    matrix = [['' for _ in col_ranges] for _ in row_ranges]

    # Fill the matrix with values
    for block in blocks:
        row_index = row_ranges.index(block['row_range'])
        col_index = col_ranges.index(block['column_range'])
        matrix[row_index][col_index] = block['value']

    # Prepare headers
    row_headers = [f"Row {r[0]}-{r[1]}" for r in row_ranges]
    col_headers = [f"Col {c[0]}-{c[1]}" for c in col_ranges]

    return tabulate(matrix, headers=col_headers, showindex=row_headers)

def visualize_sharding(dtensor):
    placements = dtensor.placements
    device_mesh = dtensor.device_mesh.mesh.tolist()

    # Find number of devices per level for nested device_mesh
    num_splits = []
    curr_devices = device_mesh.copy()
    while isinstance(curr_devices, list):
        num_splits.append(len(curr_devices))
        curr_devices = curr_devices[0]

    # We decide splits based on range
    start_range = {'row': list(range(dtensor.shape[0])), 'col': list(range(dtensor.shape[1]))}

    # Initialize the visualization grid assuming 2D tensor
    grid = [["" for _ in range(dtensor.shape[1])] for _ in range(dtensor.shape[0])]
    shard_or_replicate(start_range, placements, num_splits, 0, grid, device_mesh)
    # The grid is now 2D array containing device names, we need to make it compact to remove redundancy
    blocks = find_unique_blocks(grid)
    # Finally create the desired table
    print(create_table(blocks))

if __name__ == "__main__":
    world_size = int(os.environ['WORLD_SIZE'])
    
    # Example 1
    tensor = torch.randn(4, 4)
    mesh = DeviceMesh("cuda", list(range(world_size)))
    dtensor = distribute_tensor(tensor, mesh, [Shard(dim=1)])
    if int(os.environ['LOCAL_RANK']) == 0:
        visualize_sharding(dtensor)
        '''
                 Col 0-0    Col 1-1    Col 2-2    Col 3-3
        -------  ---------  ---------  ---------  ---------
        Row 0-3  Device:0   Device:1   Device:2   Device:3
        '''
    
    # Example 2
    tensor = torch.randn(4, 4)
    mesh = DeviceMesh("cuda", list(range(world_size)))
    dtensor = distribute_tensor(tensor, mesh, [Shard(dim=0)])
    if int(os.environ['LOCAL_RANK']) == 0:
        visualize_sharding(dtensor)
        '''
                 Col 0-3
        -------  ---------
        Row 0-0  Device:0
        Row 1-1  Device:1
        Row 2-2  Device:2
        Row 3-3  Device:3
        '''
    
    # Example 3
    tensor = torch.randn(4, 4)
    mesh = DeviceMesh("cuda", [[0, 1], [2, 3]])
    dtensor = distribute_tensor(tensor, mesh, [Shard(dim=0), Replicate()])
    if int(os.environ['LOCAL_RANK']) == 0:
        visualize_sharding(dtensor)
        '''
                 Col 0-3
        -------  ------------------
        Row 0-1  Device:0, Device:1
        Row 2-3  Device:2, Device:3
        '''

    # Example 4
    tensor = torch.randn(4, 4)
    mesh = DeviceMesh("cuda", [[0, 1], [2, 3]])
    dtensor = distribute_tensor(tensor, mesh, [Replicate(), Shard(dim=0)])
    if int(os.environ['LOCAL_RANK']) == 0:
        visualize_sharding(dtensor)
        '''
                 Col 0-3
        -------  ------------------
        Row 0-1  Device:0, Device:2
        Row 2-3  Device:1, Device:3
        '''