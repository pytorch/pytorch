"""
Auto-generated heuristic for kernel: cross_entropy
Backend: decision_tree
"""

import torch


def key_cross_entropy(*args) -> int:
    is_tensor = len(args) > 0 and isinstance(args[0], torch.Tensor)
    _arg0_dim0 = int(args[0].shape[0]) if is_tensor and args[0].ndim > 0 else 0
    _arg0_dim1 = int(args[0].shape[1]) if is_tensor and args[0].ndim > 1 else 0
    _arg0_numel = int(args[0].numel()) if is_tensor else 0
    if _arg0_numel <= 262144000.0:
        if _arg0_dim0 <= 1024.0:
            return 6
        else:
            return 2
    else:
        if _arg0_dim1 <= 151936.0:
            if _arg0_dim1 <= 128256.0:
                if _arg0_dim0 <= 2048.0:
                    return 1
                else:
                    if _arg0_numel <= 2097152000.0:
                        if _arg0_dim1 <= 128000.0:
                            return 0
                        else:
                            return 4
                    else:
                        return 0
            else:
                if _arg0_dim1 <= 129280.0:
                    if _arg0_dim0 <= 2048.0:
                        return 1
                    else:
                        if _arg0_dim0 <= 4096.0:
                            return 0
                        else:
                            return 1
                else:
                    return 1
        else:
            if _arg0_dim0 <= 4096.0:
                if _arg0_dim1 <= 152064.0:
                    return 3
                else:
                    return 5
            else:
                return 0


def autotune_cross_entropy(*args) -> dict:
    _C = [
        {
            "block_sizes": [1],
            "reduction_loops": [256],
            "range_unroll_factors": [1],
            "range_warp_specializes": [False],
            "range_num_stages": [0],
            "range_multi_buffers": [False],
            "range_flattens": [None],
            "load_eviction_policies": ["", "", "last", "", "first"],
            "num_warps": 4,
            "num_stages": 3,
            "indexing": [
                "pointer",
                "pointer",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "tensor_descriptor",
            ],
            "atomic_indexing": [],
            "pid_type": "persistent_interleaved",
            "num_sm_multiplier": 128,
        },
        {
            "block_sizes": [1],
            "reduction_loops": [8192],
            "range_unroll_factors": [0],
            "range_warp_specializes": [None],
            "range_num_stages": [0],
            "range_multi_buffers": [None],
            "range_flattens": [None],
            "load_eviction_policies": ["", "first", "last", "first", "first"],
            "num_warps": 8,
            "num_stages": 2,
            "indexing": [
                "pointer",
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
        },
        {
            "block_sizes": [1],
            "reduction_loops": [None],
            "range_unroll_factors": [0],
            "range_warp_specializes": [None],
            "range_num_stages": [0],
            "range_multi_buffers": [None],
            "range_flattens": [None],
            "load_eviction_policies": ["", "", "first", "", ""],
            "num_warps": 16,
            "num_stages": 8,
            "indexing": [
                "pointer",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
        },
        {
            "block_sizes": [1],
            "reduction_loops": [512],
            "range_unroll_factors": [2],
            "range_warp_specializes": [False],
            "range_num_stages": [4],
            "range_multi_buffers": [True],
            "range_flattens": [False],
            "load_eviction_policies": ["", "", "", "first", "last"],
            "num_warps": 8,
            "num_stages": 1,
            "indexing": [
                "pointer",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
            ],
            "atomic_indexing": [],
            "pid_type": "persistent_blocked",
            "num_sm_multiplier": 32,
            "maxnreg": 64,
        },
        {
            "block_sizes": [1],
            "reduction_loops": [256],
            "range_unroll_factors": [4],
            "range_warp_specializes": [False],
            "range_num_stages": [0],
            "range_multi_buffers": [True],
            "range_flattens": [True],
            "load_eviction_policies": ["first", "", "", "first", "last"],
            "num_warps": 4,
            "num_stages": 2,
            "indexing": [
                "pointer",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
            ],
            "atomic_indexing": [],
            "pid_type": "persistent_interleaved",
            "num_sm_multiplier": 128,
            "maxnreg": 32,
        },
        {
            "block_sizes": [1],
            "reduction_loops": [1024],
            "range_unroll_factors": [4],
            "range_warp_specializes": [False],
            "range_num_stages": [2],
            "range_multi_buffers": [False],
            "range_flattens": [None],
            "load_eviction_policies": ["first", "", "last", "", ""],
            "num_warps": 16,
            "num_stages": 6,
            "indexing": [
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
            ],
            "atomic_indexing": [],
            "pid_type": "persistent_interleaved",
            "num_sm_multiplier": 16,
            "maxnreg": 32,
        },
        {
            "block_sizes": [1],
            "reduction_loops": [2048],
            "range_unroll_factors": [4],
            "range_warp_specializes": [False],
            "range_num_stages": [3],
            "range_multi_buffers": [False],
            "range_flattens": [False],
            "load_eviction_policies": ["first", "first", "first", "", "first"],
            "num_warps": 32,
            "num_stages": 7,
            "indexing": [
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
                "pointer",
            ],
            "atomic_indexing": [],
            "pid_type": "persistent_interleaved",
            "num_sm_multiplier": 32,
        },
    ]
    return _C[key_cross_entropy(*args)]
