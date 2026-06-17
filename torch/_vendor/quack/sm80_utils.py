# Copyright (c) 2025-2026, Tri Dao.
import cutlass.cute as cute
from cutlass import Float32, const_expr


def partition_fragment_ABC(
    thr_mma: cute.ThrMma,
    shape_mnk: cute.Shape,
    sA: cute.Tensor,
    sB: cute.Tensor,
    swap_AB: bool = False,
):
    if const_expr(not swap_AB):
        acc = cute.make_rmem_tensor(thr_mma.partition_shape_C(shape_mnk[:2]), Float32)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = thr_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = thr_mma.make_fragment_B(tCsB[None, None, None, 0])
    else:
        acc = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((shape_mnk[1], shape_mnk[0])), Float32
        )
        tCsB = thr_mma.partition_A(sB)
        tCsA = thr_mma.partition_B(sA)
        tCrB = thr_mma.make_fragment_A(tCsB[None, None, None, 0])
        tCrA = thr_mma.make_fragment_B(tCsA[None, None, None, 0])
    return acc, tCsA, tCsB, tCrA, tCrB
