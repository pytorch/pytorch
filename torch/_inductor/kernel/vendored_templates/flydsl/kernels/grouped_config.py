"""Host-side tile counts for grouped GEMMs with device-resident row offsets."""


def grouped_row_tiles_upper_bound(total_m: int, group_count: int, block_m: int) -> int:
    if total_m <= 0 or group_count <= 0:
        return 0
    nonempty = min(group_count, total_m)
    return nonempty + (total_m - nonempty) // block_m
