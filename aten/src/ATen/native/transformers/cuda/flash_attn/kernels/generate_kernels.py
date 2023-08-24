# This file is run to generate the kernel instantiations for the flash_attn kernels
# They are written to several files in order to speed up compilation

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

DTYPE_MAP = {
    "fp16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
}

SM = [80]  # Sm80 kernels support up to
HEAD_DIMENSIONS = [32, 64, 96, 128, 160, 192, 224, 256]
KERNEL_IMPL_TEMPLATE_FWD = """
template<>
void run_mha_fwd_<{DTYPE}, {HEAD_DIM}>(Flash_fwd_params &params, cudaStream_t stream) {{
    run_mha_fwd_hdim{HEAD_DIM}<{DTYPE}>(params, stream);
}}
"""

KERNEL_IMPL_TEMPLATE_BWD = """
template<>
void run_mha_bwd_<{DTYPE}, {HEAD_DIM}>(Flash_bwd_params &params, cudaStream_t stream, const bool configure) {{
    run_mha_bwd_hdim{HEAD_DIM}<{DTYPE}>(params, stream, configure);
}}
"""


@dataclass
class Kernel:
    sm: int
    dtype: str
    head_dim: int
    direction: str

    @property
    def template(self) -> str:
        if self.direction == "fwd":
            return KERNEL_IMPL_TEMPLATE_FWD.format(
                DTYPE=DTYPE_MAP[self.dtype], HEAD_DIM=self.head_dim
            )
        else:
            return KERNEL_IMPL_TEMPLATE_BWD.format(
                DTYPE=DTYPE_MAP[self.dtype], HEAD_DIM=self.head_dim
            )

    @property
    def filename(self) -> str:
        return f"flash_{self.direction}_hdim{self.head_dim}_{self.dtype}_sm{self.sm}.cu"


def get_all_kernels() -> List[Kernel]:
    for dtype, head_dim, sm in itertools.product(DTYPE_MAP.keys(), HEAD_DIMENSIONS, SM):
        for direction in ["fwd", "bwd"]:
            yield Kernel(sm=sm, dtype=dtype, head_dim=head_dim, direction=direction)


def write_kernel(kernel: Kernel, autogen_dir: Path) -> None:
    prelude = """
// Copyright (c) 2023, Tri Dao.

// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"\n
"""
    include = f"#include <ATen/native/transformers/cuda/flash_attn/flash_{kernel.direction}_launch_template.h>\n"
    namespace = "namespace pytorch_flash{\n"
    namespace_end = "} // namespace pytorch_flash\n"
    (autogen_dir / kernel.filename).write_text(
        prelude + include + namespace + kernel.template + namespace_end
    )


def main(output_dir: Optional[str]) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    for kernel in get_all_kernels():
        write_kernel(kernel, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate_kernels",
        description="Generate the flash_attention kernels template instantiations",
    )
    # Set an optional output directory
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="Where to generate the kernels "
        " will default to <ATen/native/transformers/cuda/flash_attn/kernels/> ",
    )
    args = parser.parse_args()
    main(args.output_dir)
