# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Generates combination of kernels - implementations and registry

# Kernels are ordered (see `sort_index`), and when dispatching,
# we select the first kernel in the list that supports the inputs

import collections
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypeVar
import argparse

DTYPES = {
    "f32": "float",
    "f16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
}

SM = [50, 70, 75, 80, 100]  # Sm80 kernels support up to Sm100

KERNEL_IMPL_TEMPLATE = """__global__ void __launch_bounds__(
    {CPP_CLASS}::kNumThreads,
    {CPP_CLASS}::kMinBlocksPerSm)
{NAME}(typename {CPP_CLASS}::Params p) {{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= {SM}0
#if __CUDA_ARCH__ < {SM_MAX}0
  if (!p.advance_to_block()) {{
    return;
  }}
  {CPP_CLASS}::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `{NAME}` is for sm{SM}-sm{SM_MAX}, but was built for sm%d\\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}}
"""


@dataclass(order=True)
class FwdKernel:
    sort_index: Tuple[int, ...] = field(init=False, repr=False)
    aligned: bool
    dtype: str
    sm_range: Tuple[int, int]
    q: int
    k: int
    max_k: int
    supports_dropout: bool = True
    supports_bias: bool = True
    dispatch_cond: Optional[str] = None

    def __post_init__(self) -> None:
        # Set kernel selection priority
        # The lowest value that matches inputs
        # will be selected
        self.sort_index = (
            # First select aligned kernel
            0 if self.aligned else 1,
            # Then keep output in RF
            self.max_k,
            self.k,
            # Prefer kernels without dropout/bias if available
            1 if self.supports_dropout else 0,
            1 if self.supports_bias else 0,
        )

    @property
    def _aligned_suffix(self) -> str:
        return "aligned" if self.aligned else "notaligned"

    @property
    def name(self) -> str:
        acc = "rf" if self.max_k <= self.k else "gmem"
        return f"fmha_cutlassF_{self.dtype}_{self._aligned_suffix}_{self.q}x{self.k}_{acc}_sm{self.sm_range[0]}"

    @property
    def cpp_class(self) -> str:
        template_args = ", ".join(
            [
                DTYPES[self.dtype],
                f"cutlass::arch::Sm{self.sm_range[0]}",
                "true" if self.aligned else "false",
                str(self.q),
                str(self.k),
                str(self.max_k),
                "true" if self.supports_dropout else "false",
                "true" if self.supports_bias else "false",
            ]
        )
        return f"AttentionKernel<{template_args}>"

    @property
    def impl_group(self) -> str:
        # Maps to file which will contain the implementation
        return f"{self.dtype}_{self._aligned_suffix}"

    @property
    def cpp_impl(self) -> str:
        return KERNEL_IMPL_TEMPLATE.format(
            CPP_CLASS=self.cpp_class,
            NAME=self.name,
            SM=self.sm_range[0],
            SM_MAX=self.sm_range[1],
        )

    @classmethod
    def get_all(cls) -> List["FwdKernel"]:
        kernels: List[FwdKernel] = []
        for aligned, dtype, (sm, sm_max) in itertools.product(
            [True, False], DTYPES.keys(), zip(SM, SM[1:])
        ):
            # Remove some kernels we don't use
            if dtype == "bf16" and sm < 80:
                continue
            if not aligned and sm >= 80:
                continue
            for q, k, max_k in [
                (64, 64, 64),
                # We get better perf with 64x128 on A100
                (64 if sm > 75 else 32, 128, 128),
                (32, 128, 2**16),
            ]:
                kernels.append(
                    cls(
                        aligned=aligned,
                        dtype=dtype,
                        sm_range=(sm, sm_max),
                        q=q,
                        k=k,
                        max_k=max_k,
                    )
                )
        return kernels


@dataclass(order=True)
class BwdKernel:
    sort_index: Tuple[int, ...] = field(init=False, repr=False)
    sm_range: Tuple[int, int]
    dtype: str
    aligned: bool
    apply_dropout: bool
    preload_mmas: bool
    block_i: int
    block_j: int
    max_k: int
    dispatch_cond: Optional[str] = None
    keys_queries_aligned_to_blocksizes: bool = False

    def __post_init__(self) -> None:
        # Set kernel selection priority
        # The lowest value that matches inputs
        # will be selected
        self.sort_index = (
            # First select aligned kernel
            0 if self.aligned else 1,
            # Take a kernel without dropout if possible
            1 if self.apply_dropout else 0,
            # Then take the smallest maxK
            self.max_k,
            # .. and the highest block_i
            -self.block_i,
            # and finally avoid bounds-checks if possible
            0 if self.keys_queries_aligned_to_blocksizes else 1,
        )

    @property
    def _aligned_suffix(self) -> str:
        return "aligned" if self.aligned else "notaligned"

    @property
    def name(self) -> str:
        dropout_suffix = "_dropout" if self.apply_dropout else ""
        seqlen_aligned_suffix = (
            "_seqaligned" if self.keys_queries_aligned_to_blocksizes else ""
        )
        return (
            f"fmha_cutlassB_{self.dtype}_{self._aligned_suffix}"
            f"_{self.block_i}x{self.block_j}_k{self.max_k}{dropout_suffix}{seqlen_aligned_suffix}_sm{self.sm_range[0]}"
        )

    @property
    def cpp_class(self) -> str:
        template_args = ", ".join(
            [
                f"cutlass::arch::Sm{self.sm_range[0]}",
                DTYPES[self.dtype],
                "true" if self.aligned else "false",
                "true" if self.apply_dropout else "false",
                "true" if self.preload_mmas else "false",
                str(self.block_i),
                str(self.block_j),
                str(self.max_k),
            ]
        )
        if self.keys_queries_aligned_to_blocksizes:
            template_args += ", true"
        return f"AttentionBackwardKernel<{template_args}>"

    @property
    def impl_group(self) -> str:
        # Maps to file which will contain the implementation
        dropout_suffix = "_dropout" if self.apply_dropout else ""
        return f"{self.dtype}_{self._aligned_suffix}_k{self.max_k}{dropout_suffix}"

    @property
    def cpp_impl(self) -> str:
        return KERNEL_IMPL_TEMPLATE.format(
            CPP_CLASS=self.cpp_class,
            NAME=self.name,
            SM=self.sm_range[0],
            SM_MAX=self.sm_range[1],
        )

    @classmethod
    def get_all(cls) -> List["BwdKernel"]:
        kernels: List[BwdKernel] = []
        for aligned, dtype, (sm, sm_max), apply_dropout, max_k in itertools.product(
            [True, False],
            DTYPES.keys(),
            zip(SM, SM[1:]),
            [True, False],
            [32, 64, 128, 2**16],
        ):
            if dtype == "bf16" and sm < 80:
                continue
            if not aligned and sm >= 80:
                continue
            is_half = dtype in ["bf16", "f16"]

            bi_values = [64]
            # Some architectures have more shmem and can use 128
            # We still need fallback to 64 for GPUs with less shmem
            # (Sm75, Sm86 ...)
            if sm >= 80 or (sm >= 70 and is_half):
                if max_k > 64:
                    bi_values.append(128)
            for bi in bi_values:
                output_in_rf = is_half and max_k <= bi
                preload_mmas = is_half and sm >= 80 and output_in_rf
                bj = 128 if (preload_mmas and max_k > 64) else 64
                kernels.append(
                    cls(
                        aligned=aligned,
                        dtype=dtype,
                        sm_range=(sm, sm_max),
                        apply_dropout=apply_dropout,
                        preload_mmas=preload_mmas,
                        block_i=bi,
                        block_j=bj,
                        max_k=max_k,
                    )
                )
                # A few specialized kernels that are faster
                if apply_dropout or max_k > 128 or not is_half or not aligned:
                    continue
                if sm not in [70, 80]:
                    continue
                kernels.append(
                    cls(
                        aligned=aligned,
                        dtype=dtype,
                        sm_range=(sm, sm_max),
                        apply_dropout=apply_dropout,
                        preload_mmas=preload_mmas,
                        block_i=bi,
                        block_j=bj,
                        max_k=max_k,
                        keys_queries_aligned_to_blocksizes=True,
                    )
                )
        # Add some specialized kernels for stable diffusion BW (K=80)
        # This is the only kernel that can keep the outputs on RF on
        # Sm86/Sm89, so it's much faster than the 64x64 one
        for dtype in ["f16", "bf16"]:
            kernels.append(
                cls(
                    aligned=True,
                    dtype=dtype,
                    sm_range=(80, SM[SM.index(80) + 1]),
                    apply_dropout=False,
                    preload_mmas=True,
                    block_i=128,
                    block_j=64,
                    max_k=96,
                    # Sm80 has a faster kernel for this case
                    dispatch_cond="cc == 86 || cc == 89",
                )
            )
        return kernels


T = TypeVar("T", FwdKernel, BwdKernel)


def write_decl_impl(
    kernels: List[T], family_name: str, impl_file: str, autogen_dir: Path, disable_def: str = None
) -> None:
    cpp_file_header = """/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// This file is auto-generated. See "generate_kernels.py"
"""

    kernels.sort()

    implfile_to_kernels: Dict[str, List[T]] = collections.defaultdict(list)
    cat_to_kernels: Dict[Tuple[str, int, int], List[T]] = collections.defaultdict(list)

    dispatch_all = ""
    declarations = cpp_file_header + "#pragma once\n"
    # declarations += f"#ifndef {disable_def}\n"
    declarations += f"""#include {impl_file}\n"""

    # Declaration of kernel functions
    for k in kernels:
        implfile_to_kernels[k.impl_group].append(k)
        cat_to_kernels[(k.dtype, k.sm_range[0], k.sm_range[1])].append(k)

    for (cat_dt, cat_sm, cat_sm_max), kernels in cat_to_kernels.items():
        declarations += f"// ======== {cat_dt} / sm{cat_sm} ========\n"
        declarations += "\n".join(
            k.cpp_impl.split("{")[0].rstrip() + ";" for k in kernels
        )
        dispatch_category_fn = f"dispatch_{family_name}_{cat_dt}_sm{cat_sm}"
        declarations += (
            f"\n\ntemplate <typename T> void {dispatch_category_fn}(T cb, int cc) {{\n"
        )
        for k in kernels:
            _call = f"cb({k.cpp_class}(), {k.name});\n"
            if k.dispatch_cond is not None:
                _call = f"if ({k.dispatch_cond}) {_call}"
            declarations += f"    {_call}"
        declarations += "}\n\n"
        dispatch_all += f"""
    if (std::is_same<DT, {DTYPES[cat_dt]}>::value && {cat_sm} <= cc && cc < {cat_sm_max}) {{
        {dispatch_category_fn}(cb, cc);
    }}"""

    declarations += f"""
template <typename DT, typename T>
void dispatch_{family_name}(T cb, int cc = 0) {{
{dispatch_all}
}}
"""
    # declarations += f"#endif // {disable_def}\n"

    # Write declarations to family header
    (autogen_dir / f"{family_name}.h").write_text(declarations)

    for f, f_kernels in implfile_to_kernels.items():
        impl_cu = cpp_file_header
        # impl_cu += f"#ifndef {disable_def}\n"
        impl_cu += f"""#include {impl_file}\n"""
        for k in f_kernels:
            impl_cu += k.cpp_impl
        # impl_cu += f"#endif // {disable_def}\n"
        (autogen_dir / f"{family_name}_{f}.cu").write_text(impl_cu)


def main(output_dir: Optional[str]) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)
    write_decl_impl(
        FwdKernel.get_all(),
        "cutlassF",
        impl_file="<ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>",
        autogen_dir=output_dir
    )
    write_decl_impl(
        BwdKernel.get_all(),
        "cutlassB",
        impl_file="<ATen/native/transformers/cuda/mem_eff_attention/kernel_backward.h>",
        autogen_dir=output_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='generate_kernels',
        description='Generate the mem-eff kernels template instantiations')
    # Set an optional output directory
    parser.add_argument('-o', '--output_dir', required=False, help="Where to generate the kernels "
                        " will default to <ATen/native/transformers/cuda/mem_eff_attention/kernels/> ")
    args = parser.parse_args()
    main(args.output_dir)
