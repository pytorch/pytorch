# mypy: allow-untyped-defs
# flake8: noqa: TOR901
"""
Registration of symmetric memory arguments for symm_mem operations.

This module registers which arguments require symmetric memory allocation
for all torch.ops.symm_mem operations, enabling automatic lowering in Inductor.
"""

from torch.library import Library  # noqa: TOR901


def register_symm_mem_operations():
    """Register symm_mem args for all symmetric memory operations."""
    lib = Library("symm_mem", "FRAGMENT")

    # One-shot all_reduce operations
    lib.register_symm_mem_args("one_shot_all_reduce", ["input"])
    lib.register_symm_mem_args("one_shot_all_reduce_out", ["input", "out"])
    lib.register_symm_mem_args(
        "one_shot_all_reduce_copy", ["symm_buffer", "local_input"]
    )
    lib.register_symm_mem_args(
        "one_shot_all_reduce_copy_out", ["symm_buffer", "local_input", "out"]
    )

    # Two-shot all_reduce operations
    lib.register_symm_mem_args("two_shot_all_reduce_", ["input"])
    lib.register_symm_mem_args("two_shot_all_reduce_out", ["input", "output"])

    # Multimem operations
    lib.register_symm_mem_args("multimem_all_reduce_", ["input"])
    lib.register_symm_mem_args("multimem_one_shot_all_reduce", ["input"])
    lib.register_symm_mem_args("multimem_one_shot_all_reduce_out", ["input", "out"])
    lib.register_symm_mem_args("multimem_one_shot_reduce_out", ["input", "out"])
    lib.register_symm_mem_args("multimem_all_gather_out", ["input", "out"])

    # Other operations
    lib.register_symm_mem_args("reduce_scatter_out", ["input", "output"])
    lib.register_symm_mem_args("all_to_all_vdev", ["input", "out"])
    lib.register_symm_mem_args("all_to_all_vdev_2d", ["input", "out"])
    lib.register_symm_mem_args("all_to_all_vdev_2d_offset", ["input", "out"])
    lib.register_symm_mem_args("tile_reduce", ["in_tile", "out_tile"])
    lib.register_symm_mem_args("multi_root_tile_reduce", ["in_tiles", "out_tile"])


# Register on module import
register_symm_mem_operations()
