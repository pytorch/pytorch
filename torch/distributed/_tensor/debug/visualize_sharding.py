

def create_visualization(dtensor):
    """
    Tool of visualize sharding of 1D or 2D torch tensors
    """

    from tabulate import tabulate

    device_mesh = dtensor.device_mesh.mesh.tolist()
    # WIP: What happens when there are more than 1 placements in the tuple?
    placement_type = dtensor.placements[0].dim
    tensor_size = dtensor.shape
    num_devices = len(device_mesh)
    rows, cols = tensor_size

    table = []

    if placement_type == 0:
        split_size = rows // num_devices
        remainder = rows % num_devices
        row_start = 0
        for i, device in enumerate(device_mesh):
            row_end = row_start + split_size + (1 if i < remainder else 0)
            table.append([f"Row {row_start}-{row_end}", f"Device: {device}"])   
            row_start = row_end     
        print(tabulate(table, headers=["Row Range", "Device"], tablefmt="grid"))

    else:
        split_size = cols // num_devices
        remainder = cols % num_devices
        header = []
        row = []
        header.append(f"Col Range")
        row.append(f"Device")
        col_start = 0
        for i, device in enumerate(device_mesh):
            col_end = col_start + split_size + (1 if i < remainder else 0)
            header.append(f"Col {col_start}-{col_end}")
            row.append(f"Device: {device}")
            col_start = col_end
        
        table.append(row)
        print(tabulate([row], headers=header, tablefmt="grid"))