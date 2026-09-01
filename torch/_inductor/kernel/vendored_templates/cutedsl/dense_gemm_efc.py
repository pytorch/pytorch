# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: S101

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import types
import typing

import cuda.bindings.driver as cuda  # pyrefly: ignore [missing-import]

import cutlass
import cutlass.cute as cute


try:
    # `nvidia-cutlass-operators>=0.2.0` has `efc` not `common_efc`
    import cutlass.operators.providers.cutedsl.evt.efc as common_efc
except ModuleNotFoundError:
    import cutlass.operators.providers.cutedsl.evt.common_efc as common_efc

import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05


log = common_efc.log

from torch._inductor.kernel.vendored_templates.cutedsl.reduction_utils import (
    get_lane_warp_layouts,
    partition_for_epilogue,
)
from torch.utils._ordered_set import OrderedSet


"""
Common base infrastructure for high-performance persistent batched dense GEMM with custom epilogue fusion
for the NVIDIA Blackwell SM100 architecture using CUTE DSL and Epilogue Fusion Configuration (EFC).

This module provides the PersistentDenseGemmEFCKernel base class that implements the core GEMM
functionality with support for custom epilogue operations. Subclasses define specific epilogue
configurations by providing an epilogue function that operates on the accumulator and supplemental
tensors.

Key Features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Blackwell's tcgen05.mma for matrix multiply-accumulate (MMA) operations (including 2CTA MMA instructions)
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Supports persistent tile scheduling to better overlap memory load/store with MMA across tiles
    - Supports warp specialization to avoid explicit pipelining between mainloop load and MMA
    - Uses Epilogue Fusion Configuration (EFC) to define custom epilogue operations

GEMM Execution Flow:
1. DMA warp: Load A and B matrices from global memory (GMEM) to shared memory (SMEM) using TMA operations.
2. MMA warp: Perform matrix multiply-accumulate (MMA) operations using tcgen05.mma instruction.
3. EPILOGUE warp (customizable via EFC):
    - Load completed accumulator from tensor memory (TMEM) to registers (RMEM) using tcgen05.ld.
    - Load supplemental input tensors from GMEM to SMEM using TMA, then to RMEM.
    - Execute custom epilogue function (defined by subclass) that:
      * Accesses the accumulator via efc_config.accum()
      * Reads from supplemental input tensors via tensor.load()
      * Writes to supplemental output tensors via tensor.store()
      * Can apply arbitrary element-wise operations, scaling, and fusion
    - Store result tensors from RMEM to SMEM to GMEM with TMA operations.

SM100 tcgen05.mma instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Write accumulator to TMEM
The accumulator in TMEM must then be loaded to registers before writing back to GMEM.

Base Tensor Dimensions:
- Matrix A is MxKxL, L is batch dimension, A can be row-major("K") or column-major("M")
- Matrix B is NxKxL, L is batch dimension, B can be row-major("N") or column-major("K")
- Supplemental tensors are MxNxL with layout matching the epilogue configuration

Common Constraints:
* Supported input data types: fp16, bf16, tf32, int8, uint8, fp8 (e4m3fn, e5m2)
* A/B tensors must have the same data type
* MMA tiler M must be 64/128 (use_2cta_instrs=False) or 128/256 (use_2cta_instrs=True)
* MMA tiler N must be 32-256, step 32
* Cluster shape M/N must be positive and power of 2, total cluster size <= 16
* Cluster shape M must be multiple of 2 if use_2cta_instrs=True
* The contiguous dimension of all tensors must be at least 16 bytes aligned,
  i.e., number of elements is a multiple of 4, 8, and 16 for TFloat32,
  Float16/BFloat16, and Int8/Uint8/Float8, respectively.
* OOB tiles are not allowed when TMA store is disabled

Subclass Examples:
- custom_epilogue_dense_gemm.py: Custom fused epilogue with multiple read/write tensors
- broadcast_custom_epilogue_dense_gemm.py: Broadcasting and mode remapping with transposed tensors
- activation_custom_epilogue_dense_gemm.py: Activation functions with Ada FP8 GEMM epilogue pattern
- synthetic_custom_epilogue_dense_gemm.py: Synthetic epilogue for testing with configurable tensor counts

Writing an Epilogue Function (EFC User Guide)
==============================================

The Epilogue Fusion Configuration (EFC) lets you define custom epilogue
operations as a plain Python function. The framework analyzes the
function signature, JIT-compiles it for the GPU kernel, and also
executes it on the CPU with PyTorch for automatic verification.

Function Signature
------------------

The epilogue function must follow this contract:

    def epilogue(efc_config, <parameter_1>, <parameter_2>, ...):
        ...

- The first parameter **must** be named ``efc_config``. It is an
  ``EFC.Configuration`` instance that provides access to the GEMM
  accumulator, activation functions, and mode remapping.
- The remaining parameters are user-defined. Their names must match
  the supplemental arguments passed to the GEMM ``compile()`` call.
- **Tensor** parameters correspond to ``cutlass.cute.Tensor`` values
  and support ``.load()`` / ``.store()`` operations.
- **Scalar** parameters (``float``, ``int``, etc.) are used directly
  in arithmetic expressions without ``.load()``.
- The function does **not** return a value. Results are written via
  ``.store()`` on tensor parameters.  At least one ``.store()`` call
  is required for a useful computation.

Accessing the Accumulator
~~~~~~~~~~~~~~~~~~~~~~~~~

``efc_config.accum()`` returns the GEMM accumulator (the result of
A * B). It can be called multiple times within the same epilogue::

    def epilogue(efc_config, D):
        D.store(efc_config.accum())

Loading and Storing Tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor parameters expose two operations:

- ``tensor.load()`` — reads the tensor value (marks it as an input).
- ``tensor.store(value)`` — writes a computed value (marks it as an output).

A tensor can be both loaded and stored (read-modify-write). When
using multiple tensors, all ``.load()`` calls should be issued before
any ``.store()`` calls::

    def epilogue(efc_config, C, D, alpha, beta):
        D.store(efc_config.accum() * alpha + C.load() * beta)

Activation Functions
~~~~~~~~~~~~~~~~~~~~

The ``efc_config`` object provides built-in activation functions:

- ``efc_config.identity(x)`` — f(x) = x
- ``efc_config.relu(x)`` — f(x) = max(0, x)
- ``efc_config.leaky_relu(x, negative_slope=0.01)``
- ``efc_config.tanh(x)``
- ``efc_config.sigmoid(x)`` — f(x) = 1 / (1 + exp(-x))
- ``efc_config.silu(x)`` — f(x) = x * sigmoid(x) (Swish)
- ``efc_config.hardswish(x)`` — f(x) = x * relu6(x + 3) / 6
- ``efc_config.gelu(x)``

Example::

    def epilogue(efc_config, C, D, alpha, beta):
        D.store(efc_config.relu(efc_config.accum() * alpha + C.load() * beta))

Element-wise Utilities
~~~~~~~~~~~~~~~~~~~~~~

- ``efc_config.maximum(x, y)`` — element-wise maximum
- ``efc_config.minimum(x, y)`` — element-wise minimum

Additionally, any method called on ``efc_config`` that is not
explicitly defined is dispatched to ``cutlass.cute.<name>()`` on GPU
and ``torch.<name>()`` during PyTorch verification. This gives access
to functions like ``efc_config.abs(x)``, ``efc_config.exp(x)``,
``efc_config.sqrt(x)``, ``efc_config.clamp(x, min, max)``,
``efc_config.full_like(x, value)``, etc.

Broadcasting and Mode Remapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When supplemental tensors have fewer dimensions or a different mode
ordering than the output tensor, use ``tensor.remap_modes[...]`` to
broadcast or transpose them::

    tensor.remap_modes[<subscript>]

The target shape is implicitly the output shape ``(m, n, l)`` derived
from the GEMM input matrices ``A(m, k, l)`` and ``B(n, k, l)``.

The subscript uses Python's ``__getitem__`` syntax where each element
is either:

- An **integer** — the index of the source mode to place at this
  position.
- ``:`` — broadcast the source along this dimension (stride 0).

The returned remapped tensor proxy supports ``.load()`` and
``.store()`` just like regular tensor parameters.

Examples (output is (M, N, L)):

- **Broadcast along M** — ``C`` is (N, L)::

    C.remap_modes[:, 0, 1].load()

- **Broadcast along N** — ``X`` is (M, L)::

    X.remap_modes[0, :, 1].load()

- **Scalar broadcast** — ``x_factor`` is a 0-d tensor::

    x_factor.remap_modes[:, :, :].load()

- **Transpose** — ``Y`` is (N, M, L)::

    Y.remap_modes[1, 0, 2].store(value)

A remapped tensor can also be both loaded and stored for
read-modify-write on a transposed layout::

    remapped_Y = Y.remap_modes[1, 0, 2]
    remapped_Y.store(efc_config.accum() + remapped_Y.load())

Complete Examples
~~~~~~~~~~~~~~~~~

**Alpha-beta scaling** (simplest useful epilogue)::

    def epilogue(efc_config, C, D, alpha, beta):
        D.store(efc_config.accum() * alpha + C.load() * beta)

**Multi-tensor with activation**::

    def epilogue(efc_config, C, D, alpha, beta, X, x_factor, Y):
        Y.store(efc_config.accum())
        result = (
            efc_config.relu(efc_config.accum() * alpha + C.load() * beta)
            + X.load() * x_factor
        )
        D.store(result)

**Ada FP8 GEMM pattern with dynamic activation**::

    def epilogue(efc_config, C, Aux, alpha, beta, bias,
                 scale_a, scale_b, scale_c, D, leaky_relu_alpha):
        aux_val = (
            (alpha * scale_a * scale_b) * efc_config.accum()
            + (beta * scale_c) * C.load()
            + bias
        )
        Aux.store(aux_val)
        D.store(efc_config.relu(aux_val))

**Broadcasting with mode remapping**::

    def epilogue(efc_config, C, D, alpha, beta, X, x_factor, Y):
        # Y is (N, M, L), output is (M, N, L): transpose via [1, 0, 2]
        Y.remap_modes[1, 0, 2].store(efc_config.accum())
        # C is (N, L): broadcast along M via [:, 0, 1]
        # X is (M, L): broadcast along N via [0, :, 1]
        result = (
            efc_config.relu(
                efc_config.accum() * alpha
                + C.remap_modes[:, 0, 1].load() * beta
            )
            + X.remap_modes[0, :, 1].load()
            * x_factor.remap_modes[:, :, :].load()
        )
        D.store(result)
"""


class PersistentDenseGemmEFCKernel:
    """Base class for batched GEMM with custom epilogue fusion using EFC.

    This class provides the core infrastructure for persistent batched GEMM operations
    with customizable epilogue fusion. Subclasses define specific epilogue behaviors
    by providing an epilogue configuration function that describes operations on the
    accumulator and supplemental tensors.

    The class handles:
    - GEMM mainloop (A * B computation)
    - TMA-based memory operations
    - Warp specialization
    - Persistent tile scheduling
    - EFC (Epilogue Fusion Configuration) integration
    - Tensor creation and validation

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param epi_dtype: Data type for epilogue operation
    :type epi_dtype: type[cutlass.Numeric]
    :param use_2cta_instrs: Whether to use CTA group 2 for advanced thread cooperation
    :type use_2cta_instrs: bool
    :param mma_tiler_mn: Shape of the Matrix Multiply-Accumulate (MMA) tile (M,N)
    :type mma_tiler_mn: tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: tuple[int, int]
    :param epilogue_function_configuration: Function defining the epilogue behavior via EFC
    :type epilogue_function_configuration: Callable

    :note: Supported A/B data types:
        - TFloat32
        - Float16/BFloat16
        - Int8/Uint8
        - Float8E4M3FN/Float8E5M2
        (A and B must have the same data type)

    :note: Supported accumulator data types:
        - Float32 (for all floating point A/B data types)
        - Float16 (only for fp16 and fp8 A/B data types)
        - Int32 (only for uint8/int8 A/B data types)

    :note: Supported supplemental tensor data types (epilogue-dependent):
        - Float32 (for float32 and int32 accumulator data types)
        - Int32 (for float32 and int32 accumulator data types)
        - Float16/BFloat16 (for fp16 and fp8 accumulator data types)
        - Int8/Uint8 (for uint8/int8 accumulator data types)
        - Float8E4M3FN/Float8E5M2 (for float32 accumulator data types)

    :note: Constraints:
        - MMA tiler M must be 64/128 (use_2cta_instrs=False) or 128/256 (use_2cta_instrs=True)
        - MMA tiler N must be 32-256, step 32
        - Cluster shape M must be multiple of 2 if use_2cta_instrs=True
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 16
    """

    def __init__(
        self,
        acc_dtype: type[cutlass.Numeric],
        epi_dtype: type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        epilogue_function_configuration: typing.Callable,
    ):
        """Initializes the configuration for a Blackwell dense GEMM kernel with EFC.

        This configuration includes several key aspects:

        1.  MMA Instruction Settings (tcgen05):
            - acc_dtype: Data type for MMA accumulator.
            - mma_tiler_mn: The (M, N) shape of the MMA instruction tiler.
            - use_2cta_instrs: Boolean indicating if the tcgen05 MMA variant
              with cta_group=2 should be used.

        2.  Cluster Shape:
            - cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster.

        3.  Epilogue Configuration:
            - epilogue_function_configuration: Defines custom epilogue behavior
              that operates on accumulator and supplemental tensors.

        :param acc_dtype: Data type of the accumulator.
        :type acc_dtype: type[cutlass.Numeric]
        :param epi_dtype: Data type of the epilogue.
        :type epi_dtype: type[cutlass.Numeric]
        :param use_2cta_instrs: Boolean, True to use cta_group=2 MMA variant.
        :type use_2cta_instrs: bool
        :param mma_tiler_mn: tuple (M, N) shape of the MMA instruction.
        :type mma_tiler_mn: tuple[int, int]
        :param cluster_shape_mn: tuple (ClusterM, ClusterN) shape of the cluster.
        :type cluster_shape_mn: tuple[int, int]
        :param epilogue_function_configuration: Function defining epilogue behavior via EFC.
        :type epilogue_function_configuration: Callable
        """
        self.acc_dtype: type[cutlass.Numeric] = acc_dtype
        self.epi_dtype: type[cutlass.Numeric] = epi_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler_mn = mma_tiler_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.arch = "sm_100"

        self.c_dtype = self.epi_dtype

        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.occupancy = 1
        # Set specialized warp ids:

        # The warps responsible for computing the epilogue function and storing
        # the results.
        self.epilogue_warp_id = (0, 1, 2, 3)
        # The warp responsible for computing the matrix multiplication.
        self.mma_warp_id = 4
        # The warp responsible for loading the tensors A & B to feed the MMA.
        self.tma_warp_id = 5
        # The warp responsible for loading the auxiliary tensors used in the epilogue.
        self.epilogue_load_warp_id = 6
        self.threads_per_cta = 32 * len(
            (
                self.mma_warp_id,
                self.tma_warp_id,
                *self.epilogue_warp_id,
                self.epilogue_load_warp_id,
            )
        )
        # Barrier ids for cta sync, epilogue sync and tmem ptr sync.
        self.epilogue_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3
        # Amount of available shared memory.
        self.smem_capacity = utils.get_smem_capacity_in_bytes(self.arch)

        # Setup the EFC from the given function representing the epilogue
        # configuration.
        self.efc = common_efc.EFC(self, epilogue_function_configuration)

    def _create_tiled_mma(self):
        """Make a tiled MMA atom with given data type, leading dimension, CTA
        group and MMA tile shape. Use SMEM operand source for A.
        """
        return utils.sm100.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C/D stage counts in shared memory
        - Computing A/B/C/D shared memory layout
        - Computing tensor memory allocation columns
        """
        # Get the right tiled MMA.
        self._tiled_mma = self._create_tiled_mma()
        log(f"{self._tiled_mma = !s}")

        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(self._tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        # Extend mma_tiler with k-dimension (MMA_M, MMA_N, MMA_K)
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        log(f"{self.mma_tiler = !s}")
        # CTA tiler with the 2CTA instruction correction.
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(self._tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        log(f"{self.cta_tile_shape_mnk = !s}")
        # Compute cluster layout, V for the 2CTA instructions.
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (self._tiled_mma.thr_id.shape,),
        )
        log(f"{cute.make_layout((*self.cluster_shape_mn, 1)) = !s}")
        log(f"{self.cluster_layout_vmnk = !s}")
        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        log(f"{self.num_mcast_ctas_a = }, {self.num_mcast_ctas_b = }")
        log(f"{self.is_a_mcast = }, {self.is_b_mcast = }")

        # Compute epilogue (EPI_TILE_M, EPI_TILE_N) subtile of cta_tile_shape_mnk
        # according to some heuristics.
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            layout_d=self.d_layout,
            elem_ty_d=self.d_dtype,
            layout_c=self.c_layout,
            elem_ty_c=self.c_dtype,
        )
        self.epi_tile_m = cute.size(self.epi_tile[0])
        self.epi_tile_n = cute.size(self.epi_tile[1])
        self.has_cross_warp_local_reduce = cutlass.const_expr(
            (self.has_local_reduce_tensor or self.local_reduce_feeds_main)
            and (
                (
                    self.local_reduce_axis == 0
                    and (self.local_reduce_group > 4 or self.local_reduce_feeds_main)
                )
                or (
                    self.local_reduce_axis == 1
                    and self.local_reduce_group > self.epi_tile_n
                )
            )
        )
        self.local_reduce_smem_rows = (
            self.cta_tile_shape_mnk[1]
            if self.local_reduce_axis == 0
            else self.cta_tile_shape_mnk[0]
        )
        self.local_reduce_smem_cols = (
            3 + self.cta_tile_shape_mnk[0] // self.local_reduce_group
            if self.local_reduce_feeds_main and self.local_reduce_axis == 0
            else max(
                3,
                self.local_reduce_group // self.epi_tile_n
                if self.local_reduce_axis == 1
                else 3,
            )
        )
        self.local_reduce_smem_bytes = (
            self.local_reduce_smem_rows * self.local_reduce_smem_cols * 4
            if self.has_cross_warp_local_reduce
            else 0
        )
        log(f"{self.epi_tile = !s}")

        # Setup A/B/C/D pipeline stage count in shared memory and ACC stage
        # count in tensor memory.
        self.compute_stages()
        log(f"{self.num_acc_stage = }, {self.num_ab_stage = }, {self.num_c_stage = }")
        # Compute A/B shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            self._tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        log(f"{self.a_smem_layout_staged = !s}")
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            self._tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        log(f"{self.b_smem_layout_staged = !s}")
        # Get the smem_layout for the tensors used in the EFC.
        self.efc.jit.smem_layout()

        # Compute the number of tensor memory allocation columns
        self.compute_num_tmem_alloc_cols()

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        local_reduce_tensor: cute.Tensor,
        local_reduce_feed_tensor: cute.Tensor,
        local_reduce_secondary_feed_tensor: cute.Tensor,
        local_reduce_group: cutlass.Constexpr,
        local_reduce_axis: cutlass.Constexpr,
        local_reduce_type: cutlass.Constexpr,
        local_reduce_algorithm: cutlass.Constexpr,
        local_reduce_feeds_main: cutlass.Constexpr,
        local_reduce_op: cutlass.Constexpr,
        local_reduce_init: cutlass.Constexpr,
        local_reduce_combine: cutlass.Constexpr,
        local_reduce_source_op: cutlass.Constexpr,
        local_reduce_finalize: cutlass.Constexpr,
        local_reduce_consumer: cutlass.Constexpr,
        local_reduce_secondary_consumer: cutlass.Constexpr,
        *supplemental_parameters,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a: Input tensor A in (L, M, K) or (M, K) layout.
        :type a: cute.Tensor
        :param b: Input tensor B in (L, K, N) or (K, N) layout.
        :type b: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution.
        :type stream: cuda.CUstream
        :param supplemental_parameters: Variadic epilogue parameters (tensors and scalars).
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        :raises AssertionError: If OOB (Out-Of-Bounds) tiles are present when TMA store is disabled.

        """

        def add_batch_mode(tensor: cute.Tensor) -> cute.Tensor:
            return cute.make_tensor(
                tensor.iterator,
                cute.prepend(tensor.layout, cute.make_layout(1), up_to_rank=3),
            )

        a = add_batch_mode(a)
        b = add_batch_mode(b)
        if cutlass.const_expr(local_reduce_tensor is not None):
            local_reduce_tensor = add_batch_mode(local_reduce_tensor)
        if cutlass.const_expr(local_reduce_feed_tensor is not None):
            local_reduce_feed_tensor = add_batch_mode(local_reduce_feed_tensor)
            local_reduce_feed_tensor = cute.make_tensor(
                local_reduce_feed_tensor.iterator,
                cute.select(local_reduce_feed_tensor.layout, [1, 2, 0]),
            )
        if cutlass.const_expr(local_reduce_secondary_feed_tensor is not None):
            local_reduce_secondary_feed_tensor = add_batch_mode(
                local_reduce_secondary_feed_tensor
            )
            local_reduce_secondary_feed_tensor = cute.make_tensor(
                local_reduce_secondary_feed_tensor.iterator,
                cute.select(local_reduce_secondary_feed_tensor.layout, [1, 2, 0]),
            )

        # Permute tensor modes from torch to cute convention
        # A: (L, M, K) -> (M, K, L)
        a = cute.make_tensor(a.iterator, cute.select(a.layout, [1, 2, 0]))
        # B: (L, K, N) -> (N, K, L)
        b = cute.make_tensor(b.iterator, cute.select(b.layout, [2, 1, 0]))
        self.local_reduce_group = local_reduce_group
        self.local_reduce_axis = local_reduce_axis
        self.local_reduce_type = local_reduce_type
        self.local_reduce_algorithm = local_reduce_algorithm
        self.local_reduce_feeds_main = local_reduce_feeds_main
        self.local_reduce_op = local_reduce_op
        self.local_reduce_init = local_reduce_init
        self.local_reduce_combine = local_reduce_combine
        self.local_reduce_source_op = local_reduce_source_op
        self.local_reduce_finalize = local_reduce_finalize
        self.local_reduce_consumer = local_reduce_consumer
        self.local_reduce_secondary_consumer = local_reduce_secondary_consumer
        self.has_local_reduce_tensor = local_reduce_tensor is not None
        # Identify parameters that are remap_modes sources (broadcast tensors).
        # These skip add_batch_mode + permute entirely — they are passed as-is
        # to the EFC and remap_modes inside the kernel handles all the layout
        # work (broadcast expansion, dimension reordering).
        remap_sources = OrderedSet(
            [
                attrs.mapped_source
                for attrs in self.efc.parameter_attributes.values()
                if attrs.mapped_source
            ]
        )

        # Add batch mode and permute epilogue parameters, except for
        # broadcast remap sources which are passed untouched.
        param_names = self.efc.epilogue_parameter_names
        supplemental_parameters = (
            # Broadcast source: pass through untouched.
            t
            if name in remap_sources
            # Regular tensor: (L, M, N) -> rank 3 -> (M, N, L).
            else cute.make_tensor(
                add_batch_mode(t).iterator,
                cute.select(add_batch_mode(t).layout, [1, 2, 0]),
            )
            if isinstance(t, cute.Tensor)
            # Scalar: pass through.
            else t
            for name, t in zip(param_names, supplemental_parameters)
        )

        # Process the variadic parameters.
        self.efc.jit.unpack_parameters(supplemental_parameters)

        # - Matrix A is MxKxL, L is batch dimension, A can be row-major("K") or column-major("M")
        # - Matrix B is NxKxL, L is batch dimension, B can be row-major("N") or column-major("K")
        # - Supplemental tensors are MxNxL with layout matching the epilogue configuration

        # The output shape (m, n, l) is derived from A(m, k, l) and B(n, k, l).
        self.efc.output_shape = (a.shape[0], b.shape[0], a.shape[2])
        self.efc.jit.handle_remapping()

        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: type[cutlass.Numeric] = a.element_type
        self.b_dtype: type[cutlass.Numeric] = b.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()

        # Gather all the auxiliary tensor element data types.
        self.efc.jit.record_tensor_dtypes()

        # There is no D tensor to be used as a returned tensor. In the
        # following, D is used more like a "store" concept. So use the
        # written tensor with the biggest element_type to set up all the tiling
        # heuristics and epilogue store pipeline.
        self.d_name_bigger = self.efc.jit.written_tensor_name_with_bigger_element_type()
        d = self.efc.jit.parameter[self.d_name_bigger]
        self.d_dtype: type[cutlass.Numeric] = d.element_type
        self.d_layout = utils.LayoutEnum.from_tensor(d)
        log(f"d{self.d_name_bigger} = {d!s}")

        # C is the read tensor with the biggest element_type, if any, used by
        # some heuristics for tiling.
        self.c_dtype = typing.cast(type[cutlass.Numeric], None)
        self.c_layout = None
        self.c_name_bigger = self.efc.jit.read_tensor_name_with_bigger_element_type()
        if cutlass.const_expr(self.c_name_bigger):
            # The tensor with the biggest data type might be a broadcast vector.
            c = self.efc.jit.get_remapped_tensor_or_itself(self.c_name_bigger)
            log(f"{self.c_name_bigger = } -> {c = !s}")
            self.c_dtype = c.element_type
            # If only broadcast tensors are read, there is no leading dimension
            # defined, so use the same as the output.
            try:
                self.c_layout = utils.LayoutEnum.from_tensor(c)
            except ValueError:
                self.c_layout = self.d_layout

        # Check if input data types are compatible with MMA instruction
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Types must match: {self.a_dtype} != {self.b_dtype}")

        # Setup attributes that depend on gemm inputs
        self._setup_attributes()

        atom_thr_size = cute.size(self._tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = utils.sm100.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, self._tiled_mma.thr_id
        )
        log(f"{a_op = !s}")
        # Get rid of the pipeline dimension.
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        log(f"{a_smem_layout = !s}")
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            self._tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if a.element_type is cutlass.Float32 else None
            ),
        )
        log(f"{tma_atom_a = !s}")
        log(f"{tma_tensor_a = !s}")
        # Setup TMA load for B
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, self._tiled_mma.thr_id
        )
        log(f"{b_op = !s}")
        # Get rid of the pipeline dimension.
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        log(f"{b_smem_layout = !s}")
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            self._tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if b.element_type is cutlass.Float32 else None
            ),
        )
        log(f"{tma_atom_b = !s}")
        log(f"{tma_tensor_b = !s}")
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size
        log(f"{self.num_tma_load_bytes = }")

        # Set the TMA related arguments for the tensors used in the EFC.
        self.efc.jit.create_tma_arguments()

        # Compute grid size
        self.tile_sched_params, grid = self._compute_grid(
            d, self.cta_tile_shape_mnk, self.cluster_shape_mn, max_active_clusters
        )

        self.efc.jit.create_supplemental_arguments_for_kernel()

        # Launch the kernel synchronously
        self.kernel(
            self._tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            local_reduce_tensor,
            local_reduce_feed_tensor,
            local_reduce_secondary_feed_tensor,
            self.efc.kernel.pack_arguments(),
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        local_reduce_tensor: cute.Tensor,
        local_reduce_feed_tensor: cute.Tensor,
        local_reduce_secondary_feed_tensor: cute.Tensor,
        supplemental_parameters: tuple,
    ):
        """GPU device kernel performing the Persistent batched GEMM computation."""
        # Process the variadic parameters.
        self.efc.kernel.unpack_parameters(supplemental_parameters)

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch TMA descriptors
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            # Prefetch the TMA descriptors for all the supplemental tensors.
            self.efc.kernel.prefetch_tma_descriptors()

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        self.buffer_align_bytes = 1024

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            # Barriers used by the supplemental load tensor pipeline.
            c_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_c_stage]
            c_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_c_stage]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sLocalReduce: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    self.local_reduce_smem_rows * self.local_reduce_smem_cols
                    if self.has_cross_warp_local_reduce
                    else 1,
                ],
                16,
            ]

        self.shared_storage = SharedStorage

        self.smem = utils.SmemAllocator()
        storage = self.smem.allocate(self.shared_storage)
        sLocalReduce = storage.sLocalReduce.get_tensor(
            cute.make_layout((self.local_reduce_smem_rows, self.local_reduce_smem_cols))
            if self.has_cross_warp_local_reduce
            else cute.make_layout(1)
        )

        # Allocate the shared memory for all the supplemental tensors.
        self.efc.kernel.allocate_smem()

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id) * (
            2 if self.use_2cta_instrs else 1
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Load pipeline, used to load all the supplemental tensors of the
        # epilogue.
        c_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        c_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len(self.epilogue_warp_id),
        )
        c_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.c_full_mbar_ptr.data_ptr(),
            num_stages=self.num_c_stage,
            producer_group=c_producer_group,
            consumer_group=c_consumer_group,
            # Unlock the barrier when all the tensor bytes have been loaded.
            tx_count=self.efc.jit.total_tma_load_bytes,
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # Cluster arrive after barrier init
        pipeline.pipeline_init_arrive(
            cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True
        )

        #
        # Setup smem tensor A/B
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )

        #
        # Compute multicast mask for A/B buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(
            self.is_a_mcast or self.is_b_mcast or self.use_2cta_instrs
        ):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, loopM, loopK, loopL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, loopN, loopK, loopL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )

        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        #
        # Partition global tensor for TiledMMA_A/B/D
        #
        self.thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, loopM, loopK, loopL)
        tCgA = self.thr_mma.partition_A(gA_mkl)
        log(f"{tCgA = !s}")
        # (MMA, MMA_N, MMA_K, loopN, loopK, loopL)
        tCgB = self.thr_mma.partition_B(gB_nkl)
        log(f"{tCgB = !s}")
        # Create the local_tile gX_mnl for all the EFC supplemental tensors.
        self.efc.kernel.partition_global_tensors_for_tiled_mma()

        #
        # Partition global/shared tensor for TMA load A/B
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), tiles_m, tiles_k, tiles_l)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), tiles_n, tiles_k, tiles_l)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C/D
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        log(f"{tCrA = !s}")
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        log(f"{tCrB = !s}")
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        log(f"{acc_shape = !s}")
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )
        log(f"{tCtAcc_fake = !s}")

        # Named barriers
        #
        epilogue_sync_barrier = pipeline.NamedBarrier(
            self.epilogue_sync_bar_id, 32 * len(self.epilogue_warp_id)
        )

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline.pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        #
        # Specialized TMA load warp
        #

        if warp_idx == self.tma_warp_id:
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), loopK)
                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                # ((atom_v, rest_v), loopK)
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                ab_producer.reset()
                peek_ab_empty_status = ab_producer.try_acquire()

                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Conditionally wait for AB buffer empty
                    handle = ab_producer.acquire_and_advance(peek_ab_empty_status)

                    # TMA load A/B
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, handle.count)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, handle.count)],
                        tBsB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=b_full_mcast_mask,
                    )

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_ab_empty_status = ab_producer.try_acquire()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait A/B buffer empty
            #
            ab_producer.tail()

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Bar sync to retrieve tensor memory ptr from shared mem
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Set tensor memory buffer for current tile
                # (MMA, MMA_M, MMA_N)
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer.reset()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()

                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                #
                # Reset the ACCUMULATE field for each tile
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                #
                # Mma mainloop
                #
                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)

                        # tCtAcc += tCrA * tCrB
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                        tile_crd = (None, None, None, handle.index)
                        cute.gemm(
                            tiled_mma, tCtAcc, tCrA[tile_crd], tCrB[tile_crd], tCtAcc
                        )

                        # Async arrive AB buffer empty
                        handle.release()

                        # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                        peek_ab_full_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()

                #
                # Async arrive accumulator buffer full
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)
        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # Bar sync to retrieve tensor memory ptr from shared memory
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            log(f"tmem_ptr = {tmem_ptr!s}")
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            log(f"tCtAcc_base = {tCtAcc_base!s}")
            #
            # Partition for epilogue
            #
            epi_tidx = tidx
            tCgD = self.efc.kernel.tCgD_written[self.d_name_bigger]
            log(f"tCgD (aka tCgD_written[{self.d_name_bigger}])= {tCgD!s}")

            (
                tiled_copy_t2r,  # (EPI_TILE_M, EPI_TILE_N)
                tTR_tAcc_base,  # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
                tTR_rAcc,  # (T2R, T2R_M, T2R_N)
            ) = self.epilogue_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, tCgD, epi_tile
            )
            log(f"{tiled_copy_t2r = !s}")
            log(f"{tTR_tAcc_base = !s}")
            log(f"{tTR_rAcc = !s}")
            # Copy and partition for the supplemental EFC tensors.
            self.efc.kernel.copy_and_partition_supplemental_rmem_tensors(
                tiled_copy_t2r, tTR_rAcc, epi_tidx, epi_tile
            )

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            # Store D pipeline used for all the written tensors in the epilogue.
            d_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilogue_warp_id),
            )
            d_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_d_stage,
                producer_group=d_producer_group,
            )

            c_pipeline_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_c_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Slice the supplemental written tensors per MMA tile index.
                self.efc.kernel.slice_written_tensors_per_mma_tile_index(
                    mma_tile_coord_mnl
                )

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_consumer_state.index)
                ]
                log(f"tTR_tAcc = {tTR_tAcc!s}")
                #
                # Wait for accumulator buffer full
                #
                acc_pipeline.consumer_wait(acc_consumer_state)

                # Group together the EPI_M, EPI_M which are starting at group 3.
                # (T2R, T2R_M, T2R_N, (EPI_M, EPI_M))
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                log(f"group_modes tTR_tAcc = {tTR_tAcc!s}")
                #
                # Store accumulator to global memory in subtiles
                #
                # Use EPI_M*EPI_M to iterate using the 1-D coordinate.
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
                epi_tile_n = cutlass.const_expr(self.epi_tile_n)
                local_reduce_partial: typing.Any = None
                local_reduce_nan_partial: typing.Any = None
                if cutlass.const_expr(
                    (local_reduce_tensor is not None or self.local_reduce_feeds_main)
                    and self.local_reduce_axis == 1
                    and self.local_reduce_group > self.epi_tile_n
                ):
                    local_reduce_partial = cute.make_rmem_tensor(
                        ((1, self.epi_tile_n, 1), 1, 1), self.acc_dtype
                    )
                    if cutlass.const_expr(self.local_reduce_type in ("max", "min")):
                        local_reduce_nan_partial = cute.make_rmem_tensor(
                            ((1, self.epi_tile_n, 1), 1, 1), self.acc_dtype
                        )
                for subtile_idx in cutlass.range(subtile_cnt):
                    #
                    # Load accumulator from tensor memory buffer to register
                    #
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
                    log(f"cute.copy tiled_copy_t2r = {tiled_copy_t2r!s}")
                    log(f"cute.copy tTR_tAcc_mn = {tTR_tAcc_mn!s}")
                    log(f"cute.copy tTR_rAcc = {tTR_rAcc!s}")

                    # Wait for the EFC tensor loads to complete.  The wait is
                    # blocking even if the tx_count is 0 when there is no tensor
                    # to load. So, only wait if there is something to read,
                    # i.e. smem_read is not empty.
                    if cutlass.const_expr(self.efc.kernel.smem_read):
                        c_pipeline.consumer_wait(c_pipeline_consumer_state)

                    # Load supplemental tensors from shared memory to register.
                    self.efc.kernel.load_tensors_from_smem_to_register(
                        c_pipeline_consumer_state.index
                    )

                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    c_pipeline.consumer_release(c_pipeline_consumer_state)

                    # Advance pipeline states
                    c_pipeline_consumer_state.advance()

                    #
                    # Perform epilogue op on accumulator.
                    #
                    tiled_copy_r2s = self.efc.kernel.tiled_copy_r2s[self.d_name_bigger]
                    log(f"tiled_copy_r2s = {tiled_copy_r2s!s}")
                    # Use a SimpleNamespace to pass easily some local content as
                    # an extensible class compatible with CuTe DSL
                    # implementation.
                    epilogue_context = types.SimpleNamespace()
                    # Load the accumulator cast to the epi_dtype used to do all
                    # the computations in the epilogue.
                    # Retile the accumulator subtile to fit the destination
                    # subtile vector TV layout.
                    epilogue_context.acc_vec = (
                        tiled_copy_r2s.retile(tTR_rAcc).load().to(self.epi_dtype)
                    )
                    log(f"before .retile tTR_rAcc = {tTR_rAcc!s}")
                    log(
                        f"tiled_copy_r2s.retile(tTR_rAcc) = {tiled_copy_r2s.retile(tTR_rAcc)!s}"
                    )
                    log(
                        f"tiled_copy_r2s.retile(tTR_rAcc).load() = {tiled_copy_r2s.retile(tTR_rAcc).load()!s}"
                    )
                    log(f"epilogue_context.acc_vec = {epilogue_context.acc_vec!s}")

                    if cutlass.const_expr(
                        local_reduce_secondary_feed_tensor is not None
                        and local_reduce_tensor is None
                        and not self.local_reduce_feeds_main
                    ):
                        assert self.local_reduce_secondary_consumer is not None
                        direct_fragment = cute.make_rmem_tensor_like(
                            epilogue_context.acc_vec, self.acc_dtype
                        )
                        direct_fragment.store(
                            epilogue_context.acc_vec.to(self.acc_dtype)
                        )
                        direct_flt = cute.filter_zeros(direct_fragment)
                        direct_coordinates = partition_for_epilogue(
                            cute.make_identity_tensor(self.cta_tile_shape_mnk[:2]),
                            epi_tile=epi_tile,
                            tiled_copy=tiled_copy_t2r,
                            tidx=epi_tidx,
                            reference_src=False,
                        )[(None, None, None, 0, subtile_idx)]
                        direct_coord_flt = cute.filter_zeros(direct_coordinates)
                        output_m = cute.size(mA_mkl, mode=[0])
                        output_n = cute.size(mB_nkl, mode=[0])
                        for i in cutlass.range(cute.size(direct_flt), unroll_full=True):
                            global_m = (
                                mma_tile_coord_mnl[0] * self.cta_tile_shape_mnk[0]
                                + direct_coord_flt[i][0]
                            )
                            global_n = (
                                mma_tile_coord_mnl[1] * self.cta_tile_shape_mnk[1]
                                + direct_coord_flt[i][1]
                            )
                            if global_m < output_m and global_n < output_n:
                                local_reduce_secondary_feed_tensor[
                                    global_m,
                                    global_n,
                                    mma_tile_coord_mnl[2],
                                ] = self.local_reduce_secondary_consumer(
                                    direct_flt[i], 0.0, 0.0
                                ).to(local_reduce_secondary_feed_tensor.element_type)

                    if cutlass.const_expr(
                        (
                            local_reduce_tensor is not None
                            or self.local_reduce_feeds_main
                        )
                        and self.local_reduce_axis == 0
                    ):
                        assert self.local_reduce_type in (
                            "sum",
                            "mean",
                            "prod",
                            "max",
                            "min",
                        )
                        group = cutlass.const_expr(self.local_reduce_group)
                        acc_fragment = cute.make_rmem_tensor_like(
                            epilogue_context.acc_vec, self.acc_dtype
                        )
                        acc_fragment.store(epilogue_context.acc_vec.to(self.acc_dtype))
                        reduced_flt = cute.filter_zeros(acc_fragment)
                        coordinates = partition_for_epilogue(
                            cute.make_identity_tensor(self.cta_tile_shape_mnk[:2]),
                            epi_tile=epi_tile,
                            tiled_copy=tiled_copy_t2r,
                            tidx=epi_tidx,
                            reference_src=False,
                        )
                        coordinates = coordinates[(None, None, None, 0, subtile_idx)]
                        coord_flt = cute.filter_zeros(
                            tiled_copy_r2s.retile(coordinates)
                        )
                        lane_layout_mn, warp_layout_mn = get_lane_warp_layouts(
                            tiled_copy_t2r, reference_src=False
                        )
                        lanes_in_m = cutlass.const_expr(
                            cute.size(lane_layout_mn, mode=[0])
                        )
                        assert group > 1 and group <= self.cta_tile_shape_mnk[0]
                        reduction_group = cutlass.const_expr(min(group, lanes_in_m))
                        assert group % reduction_group == 0
                        assert lanes_in_m % reduction_group == 0
                        for i in cutlass.range(
                            cute.size(reduced_flt), unroll_full=True
                        ):
                            reduced_flt[i] = self.local_reduce_source_op(reduced_flt[i])
                            rows = reduction_group // 2
                            while rows > 0:
                                other = cute.arch.shuffle_sync_bfly(
                                    reduced_flt[i],
                                    offset=cute.crd2idx((rows, 0), lane_layout_mn),
                                )
                                reduced_flt[i] = self.local_reduce_combine(
                                    reduced_flt[i], other
                                )
                                rows //= 2
                        groups_per_cta = cutlass.const_expr(
                            self.cta_tile_shape_mnk[0] // group
                        )
                        g_reduce: typing.Any = None
                        if cutlass.const_expr(local_reduce_tensor is not None):
                            m_reduce = local_reduce_tensor[
                                mma_tile_coord_mnl[2], None, None
                            ]
                            g_reduce = cute.local_tile(
                                m_reduce,
                                (groups_per_cta, self.cta_tile_shape_mnk[1]),
                                mma_tile_coord_mnl[:2],
                            )
                        limit_n = (
                            min(
                                cute.size(local_reduce_tensor, mode=[2])
                                - mma_tile_coord_mnl[1] * self.cta_tile_shape_mnk[1],
                                self.cta_tile_shape_mnk[1],
                            )
                            if cutlass.const_expr(local_reduce_tensor is not None)
                            else self.cta_tile_shape_mnk[1]
                        )
                        limit_groups = (
                            cute.size(local_reduce_tensor, mode=[1])
                            if cutlass.const_expr(local_reduce_tensor is not None)
                            else cute.size(mA_mkl, mode=[0]) // group
                        )
                        group_warps = cutlass.const_expr(group // lanes_in_m)
                        warp_m = warp_layout_mn[0]
                        warps_in_m = cutlass.const_expr(cute.size(warp_m))
                        assert group_warps <= warps_in_m
                        warp_idx = cute.arch.make_warp_uniform(
                            epi_tidx // cute.arch.WARP_SIZE
                        )
                        warp_m_idx = warp_layout_mn.get_hier_coord(warp_idx)[0]
                        feed_fragment = cute.make_rmem_tensor_like(
                            acc_fragment, self.acc_dtype
                        )
                        feed_fragment.store(epilogue_context.acc_vec.to(self.acc_dtype))
                        feed_flt = cute.filter_zeros(feed_fragment)
                        if cutlass.const_expr(group > lanes_in_m):
                            for i in cutlass.range(
                                cute.size(reduced_flt), unroll_full=True
                            ):
                                row_idx = coord_flt[i][0]
                                n_idx = coord_flt[i][1]
                                group_idx = row_idx // group
                                group_warp_start = group_idx * group_warps
                                group_warp_idx = warp_m_idx - group_warp_start
                                if (
                                    row_idx % lanes_in_m == 0
                                    and group_warp_idx > 0
                                    and group_warp_idx < group_warps
                                ):
                                    sLocalReduce[n_idx, warp_m_idx - group_idx - 1] = (
                                        reduced_flt[i]
                                    )
                            epilogue_sync_barrier.arrive_and_wait()
                        for i in cutlass.range(
                            cute.size(reduced_flt), unroll_full=True
                        ):
                            row_idx = coord_flt[i][0]
                            n_idx = coord_flt[i][1]
                            group_idx = row_idx // group
                            global_group_idx = (
                                mma_tile_coord_mnl[0] * groups_per_cta + group_idx
                            )
                            group_value = reduced_flt[i]
                            group_warp_start = group_idx * group_warps
                            if cutlass.const_expr(group > lanes_in_m):
                                if (
                                    row_idx % group == 0
                                    and warp_m_idx == group_warp_start
                                ):
                                    for warp_offset in cutlass.range_constexpr(
                                        1, group_warps
                                    ):
                                        other = sLocalReduce[
                                            n_idx,
                                            group_warp_start
                                            + warp_offset
                                            - group_idx
                                            - 1,
                                        ]
                                        group_value = self.local_reduce_combine(
                                            group_value, other
                                        )
                            consumer_value = group_value
                            group_value = self.local_reduce_finalize(group_value, group)
                            if cutlass.const_expr(self.local_reduce_feeds_main):
                                if row_idx % group == 0 and (
                                    group <= lanes_in_m
                                    or warp_m_idx == group_warp_start
                                ):
                                    feed_flt[i] = (
                                        consumer_value
                                        if self.local_reduce_consumer is not None
                                        else group_value
                                    )
                            if cutlass.const_expr(local_reduce_tensor is not None):
                                if (
                                    row_idx % group == 0
                                    and n_idx < limit_n
                                    and global_group_idx < limit_groups
                                ):
                                    g_reduce[group_idx, n_idx] = group_value
                        if cutlass.const_expr(group > lanes_in_m):
                            epilogue_sync_barrier.arrive_and_wait()
                        if cutlass.const_expr(self.local_reduce_feeds_main):
                            primary_acc_vec = epilogue_context.acc_vec
                            for i in cutlass.range(
                                cute.size(reduced_flt), unroll_full=True
                            ):
                                row_idx = coord_flt[i][0]
                                n_idx = coord_flt[i][1]
                                group_idx = row_idx // group
                                group_warp_start = group_idx * group_warps
                                if row_idx % group == 0 and (
                                    group <= lanes_in_m
                                    or warp_m_idx == group_warp_start
                                ):
                                    sLocalReduce[n_idx, group_idx + 3] = feed_flt[i]
                            epilogue_sync_barrier.arrive_and_wait()
                            for i in cutlass.range(
                                cute.size(reduced_flt), unroll_full=True
                            ):
                                row_idx = coord_flt[i][0]
                                n_idx = coord_flt[i][1]
                                reduced_flt[i] = sLocalReduce[
                                    n_idx, row_idx // group + 3
                                ]
                            epilogue_sync_barrier.arrive_and_wait()
                            normalized_fragment = cute.make_rmem_tensor_like(
                                acc_fragment, self.acc_dtype
                            )
                            normalized_fragment.store(
                                epilogue_context.acc_vec.to(self.acc_dtype)
                            )
                            normalized_flt = cute.filter_zeros(normalized_fragment)
                            for i in cutlass.range(
                                cute.size(normalized_flt), unroll_full=True
                            ):
                                if cutlass.const_expr(
                                    self.local_reduce_consumer is not None
                                ):
                                    normalized_flt[i] = self.local_reduce_consumer(
                                        normalized_flt[i], reduced_flt[i], 0.0
                                    )
                                else:
                                    normalized_flt[i] = reduced_flt[i]
                            if cutlass.const_expr(local_reduce_feed_tensor is not None):
                                output_m = cute.size(mA_mkl, mode=[0])
                                output_n = cute.size(mB_nkl, mode=[0])
                                for i in cutlass.range(
                                    cute.size(normalized_flt), unroll_full=True
                                ):
                                    global_m = (
                                        mma_tile_coord_mnl[0]
                                        * self.cta_tile_shape_mnk[0]
                                        + coord_flt[i][0]
                                    )
                                    global_n = (
                                        mma_tile_coord_mnl[1]
                                        * self.cta_tile_shape_mnk[1]
                                        + coord_flt[i][1]
                                    )
                                    if global_m < output_m and global_n < output_n:
                                        local_reduce_feed_tensor[
                                            global_m,
                                            global_n,
                                            mma_tile_coord_mnl[2],
                                        ] = normalized_flt[i].to(
                                            local_reduce_feed_tensor.element_type
                                        )
                            if cutlass.const_expr(
                                local_reduce_secondary_feed_tensor is not None
                            ):
                                output_m = cute.size(mA_mkl, mode=[0])
                                output_n = cute.size(mB_nkl, mode=[0])
                                if cutlass.const_expr(
                                    self.local_reduce_secondary_consumer is not None
                                ):
                                    secondary_fragment = cute.make_rmem_tensor_like(
                                        acc_fragment, self.acc_dtype
                                    )
                                    secondary_fragment.store(
                                        primary_acc_vec.to(self.acc_dtype)
                                    )
                                    secondary_flt = cute.filter_zeros(
                                        secondary_fragment
                                    )
                                    for i in cutlass.range(
                                        cute.size(secondary_flt), unroll_full=True
                                    ):
                                        secondary_flt[i] = (
                                            self.local_reduce_secondary_consumer(
                                                secondary_flt[i],
                                                reduced_flt[i],
                                                0.0,
                                            )
                                        )
                                else:
                                    secondary_flt = normalized_flt
                                for i in cutlass.range(
                                    cute.size(secondary_flt), unroll_full=True
                                ):
                                    global_m = (
                                        mma_tile_coord_mnl[0]
                                        * self.cta_tile_shape_mnk[0]
                                        + coord_flt[i][0]
                                    )
                                    global_n = (
                                        mma_tile_coord_mnl[1]
                                        * self.cta_tile_shape_mnk[1]
                                        + coord_flt[i][1]
                                    )
                                    if global_m < output_m and global_n < output_n:
                                        local_reduce_secondary_feed_tensor[
                                            global_m,
                                            global_n,
                                            mma_tile_coord_mnl[2],
                                        ] = secondary_flt[i].to(
                                            local_reduce_secondary_feed_tensor.element_type
                                        )
                            if cutlass.const_expr(local_reduce_feed_tensor is not None):
                                epilogue_context.acc_vec = primary_acc_vec
                            else:
                                epilogue_context.acc_vec = normalized_fragment.load()

                    if cutlass.const_expr(
                        (
                            local_reduce_tensor is not None
                            and self.local_reduce_axis == 1
                        )
                        or (
                            self.local_reduce_feeds_main and self.local_reduce_axis == 1
                        )
                    ):
                        assert self.local_reduce_type in (
                            "sum",
                            "mean",
                            "prod",
                            "max",
                            "min",
                        )
                        group = cutlass.const_expr(self.local_reduce_group)
                        fragment_n = epi_tile_n
                        n_subtiles = cutlass.const_expr(
                            self.cta_tile_shape_mnk[1] // fragment_n
                        )
                        fragments_per_group = cutlass.const_expr(
                            max(1, group // fragment_n)
                        )
                        subtile_n_idx = subtile_idx % n_subtiles
                        assert group > 1
                        acc_fragment = cute.make_rmem_tensor_like(
                            epilogue_context.acc_vec, self.acc_dtype
                        )
                        acc_fragment.store(epilogue_context.acc_vec.to(self.acc_dtype))
                        acc_flt = cute.filter_zeros(acc_fragment)
                        wide_reduced = None
                        wide_nan = None
                        if cutlass.const_expr(group > fragment_n):
                            local_reduce_vec = epilogue_context.acc_vec.to(
                                self.acc_dtype
                            )
                            if cutlass.const_expr(
                                self.local_reduce_type in ("max", "min")
                            ):
                                nan_grouped = (
                                    (local_reduce_vec != local_reduce_vec)
                                    .to(self.acc_dtype)
                                    .reshape(((1, fragment_n, 1), 1, 1))
                                )
                                wide_nan = nan_grouped.reduce(
                                    cute.ReductionOp.ADD,
                                    init_val=0.0,
                                    reduction_profile=((None, 1, None), 1, 1),
                                )
                                wide_nan = wide_nan.reshape(
                                    ((1, 1, 1), 1, 1)
                                ).broadcast_to(nan_grouped.shape)
                            local_reduce_vec = self.local_reduce_source_op(
                                local_reduce_vec
                            )
                            fragment_width = cutlass.const_expr(
                                cute.size(local_reduce_vec.shape, mode=[0])
                            )
                            grouped = local_reduce_vec.reshape(
                                ((1, fragment_width, 1), 1, 1)
                            )
                            wide_reduced = grouped.reduce(
                                self.local_reduce_op,
                                init_val=self.local_reduce_init,
                                reduction_profile=((None, 1, None), 1, 1),
                            )
                            wide_reduced = wide_reduced.reshape(
                                ((1, 1, 1), 1, 1)
                            ).broadcast_to(grouped.shape)
                        reduced_fragment = cute.make_rmem_tensor_like(
                            acc_fragment, cutlass.Float32
                        )
                        reduced_flt = cute.filter_zeros(reduced_fragment)
                        wide_partial_fragment = cute.make_rmem_tensor_like(
                            acc_fragment, cutlass.Float32
                        )
                        wide_partial_flt = cute.filter_zeros(wide_partial_fragment)
                        for i in cutlass.range(
                            cute.size(wide_partial_flt), unroll_full=True
                        ):
                            wide_partial_flt[i] = self.local_reduce_init
                        coordinates = partition_for_epilogue(
                            cute.make_identity_tensor(self.cta_tile_shape_mnk[:2]),
                            epi_tile=epi_tile,
                            tiled_copy=tiled_copy_t2r,
                            tidx=epi_tidx,
                            reference_src=False,
                        )
                        coordinates = coordinates[(None, None, None, 0, subtile_idx)]
                        coord_flt = cute.filter_zeros(coordinates)
                        acc_coord_flt = cute.filter_zeros(
                            tiled_copy_r2s.retile(coordinates)
                        )
                        groups_per_cta = cutlass.const_expr(
                            self.cta_tile_shape_mnk[1] // group
                        )
                        g_reduce: typing.Any = None
                        if cutlass.const_expr(local_reduce_tensor is not None):
                            m_reduce = local_reduce_tensor[
                                mma_tile_coord_mnl[2], None, None
                            ]
                            g_reduce = cute.local_tile(
                                m_reduce,
                                (self.cta_tile_shape_mnk[0], groups_per_cta),
                                mma_tile_coord_mnl[:2],
                            )
                        limit_m = min(
                            cute.size(mA_mkl, mode=[0])
                            - mma_tile_coord_mnl[0] * self.cta_tile_shape_mnk[0],
                            self.cta_tile_shape_mnk[0],
                        )
                        limit_groups = (
                            cute.size(local_reduce_tensor, mode=[2])
                            if cutlass.const_expr(local_reduce_tensor is not None)
                            else cute.size(mB_nkl, mode=[0]) // group
                        )
                        for i in cutlass.range(cute.size(acc_flt), unroll_full=True):
                            row_idx = coord_flt[i][0]
                            n_idx = coord_flt[i][1]
                            logical_row = row_idx
                            group_idx = (
                                subtile_n_idx // fragments_per_group
                                if cutlass.const_expr(group > fragment_n)
                                else n_idx // group
                            )
                            global_group_idx = (
                                mma_tile_coord_mnl[1] * groups_per_cta + group_idx
                            )
                            if (
                                (
                                    self.local_reduce_feeds_main
                                    or n_idx
                                    % (
                                        fragment_n
                                        if cutlass.const_expr(group > fragment_n)
                                        else group
                                    )
                                    == 0
                                )
                                and logical_row < limit_m
                                and global_group_idx < limit_groups
                            ):
                                reduced = self.local_reduce_init
                                for j in cutlass.range(
                                    cute.size(acc_flt), unroll_full=True
                                ):
                                    if coord_flt[j][0] == row_idx and (
                                        cutlass.const_expr(group > fragment_n)
                                        or coord_flt[j][1] // group == group_idx
                                    ):
                                        value = acc_flt[j]
                                        value = self.local_reduce_source_op(value)
                                        reduced = self.local_reduce_combine(
                                            reduced, value
                                        )
                                if cutlass.const_expr(
                                    self.local_reduce_algorithm == "logsumexp"
                                    and group <= fragment_n
                                ):
                                    shift = reduced
                                    if cute.math.abs(shift) == cutlass.Float32.inf:
                                        shift = cutlass.Float32(0.0)
                                    exp_sum = cutlass.Float32(0.0)
                                    for j in cutlass.range(
                                        cute.size(acc_flt), unroll_full=True
                                    ):
                                        if (
                                            coord_flt[j][0] == row_idx
                                            and coord_flt[j][1] // group == group_idx
                                        ):
                                            exp_sum += cute.math.exp(acc_flt[j] - shift)
                                    reduced = cute.math.log(exp_sum) + shift
                                elif cutlass.const_expr(
                                    self.local_reduce_algorithm == "variance"
                                    and group <= fragment_n
                                ):
                                    mean = reduced / group
                                    reduced = cutlass.Float32(0.0)
                                    for j in cutlass.range(
                                        cute.size(acc_flt), unroll_full=True
                                    ):
                                        if (
                                            coord_flt[j][0] == row_idx
                                            and coord_flt[j][1] // group == group_idx
                                        ):
                                            delta = acc_flt[j] - mean
                                            reduced += delta * delta
                                    reduced = reduced / group
                                if cutlass.const_expr(group <= fragment_n):
                                    if cutlass.const_expr(
                                        self.local_reduce_consumer is not None
                                        or self.local_reduce_secondary_consumer
                                        is not None
                                    ):
                                        reduced_flt[i] = reduced
                                    reduced = self.local_reduce_finalize(reduced, group)
                                if cutlass.const_expr(
                                    self.local_reduce_feeds_main
                                    and self.local_reduce_consumer is None
                                ):
                                    reduced_flt[i] = reduced
                                if cutlass.const_expr(group > fragment_n):
                                    wide_partial_flt[i] = reduced
                                if cutlass.const_expr(local_reduce_tensor is not None):
                                    if cutlass.const_expr(group > fragment_n):
                                        if n_idx % fragment_n == 0:
                                            fragment_idx = (
                                                subtile_n_idx % fragments_per_group
                                            )
                                            sLocalReduce[logical_row, fragment_idx] = (
                                                reduced
                                            )
                                    elif n_idx % group == 0:
                                        g_reduce[logical_row, group_idx] = reduced
                        if cutlass.const_expr(
                            group > fragment_n
                            and local_reduce_tensor is not None
                            and self.local_reduce_type == "sum"
                        ):
                            sum_fragment_idx = subtile_n_idx % fragments_per_group
                            sum_group_idx = subtile_n_idx // fragments_per_group
                            sum_global_group_idx = (
                                mma_tile_coord_mnl[1] * groups_per_cta + sum_group_idx
                            )
                            sum_row_idx = epi_tidx
                            if sum_fragment_idx == 0:
                                if (
                                    sum_row_idx < limit_m
                                    and sum_global_group_idx < limit_groups
                                ):
                                    shared_element = cute.make_tensor(
                                        sLocalReduce.iterator + sum_row_idx,
                                        cute.make_layout(1),
                                    )
                                    shared_element[0] = cutlass.Float32(0.0)
                            epilogue_sync_barrier.arrive_and_wait()
                            for i in cutlass.range(
                                cute.size(acc_flt), unroll_full=True
                            ):
                                if (
                                    sum_row_idx < limit_m
                                    and sum_global_group_idx < limit_groups
                                ):
                                    cute.arch.atomic_add(
                                        sLocalReduce.iterator + sum_row_idx,
                                        acc_flt[i],
                                        scope="cta",
                                    )
                            epilogue_sync_barrier.arrive_and_wait()
                            if sum_fragment_idx == fragments_per_group - 1:
                                if (
                                    sum_row_idx < limit_m
                                    and sum_global_group_idx < limit_groups
                                ):
                                    shared_element = cute.make_tensor(
                                        sLocalReduce.iterator + sum_row_idx,
                                        cute.make_layout(1),
                                    )
                                    output_element = cute.make_tensor(
                                        g_reduce.iterator  # pyrefly: ignore[unbound-name]
                                        + sum_row_idx * limit_groups,
                                        cute.make_layout(1),
                                    )
                                    output_element[0] = self.local_reduce_finalize(
                                        shared_element[0], group
                                    )
                        if cutlass.const_expr(
                            group > fragment_n and self.local_reduce_type != "sum"
                        ):
                            fragment_idx = subtile_n_idx % fragments_per_group
                            partial = local_reduce_partial.load()
                            current = wide_reduced
                            if fragment_idx == 0:
                                partial = current
                            else:
                                partial = self.local_reduce_combine(partial, current)
                            local_reduce_partial.store(partial)
                            if cutlass.const_expr(
                                self.local_reduce_type in ("max", "min")
                            ):
                                nan_partial = local_reduce_nan_partial.load()
                                if fragment_idx == 0:
                                    nan_partial = wide_nan
                                else:
                                    nan_partial += wide_nan
                                local_reduce_nan_partial.store(nan_partial)
                            if fragment_idx == fragments_per_group - 1:
                                final_fragment = cute.make_rmem_tensor_like(
                                    local_reduce_partial, cutlass.Float32
                                )
                                final_fragment.store(partial)
                                final_flt = cute.filter_zeros(final_fragment)
                                nan_flt: typing.Any = None
                                if cutlass.const_expr(
                                    self.local_reduce_type in ("max", "min")
                                ):
                                    nan_fragment = cute.make_rmem_tensor_like(
                                        local_reduce_nan_partial, cutlass.Float32
                                    )
                                    nan_fragment.store(local_reduce_nan_partial.load())
                                    nan_flt = cute.filter_zeros(nan_fragment)
                                store_coordinates = partition_for_epilogue(
                                    cute.make_identity_tensor(
                                        self.cta_tile_shape_mnk[:2]
                                    ),
                                    epi_tile=epi_tile,
                                    tiled_copy=tiled_copy_t2r,
                                    tidx=epi_tidx,
                                    reference_src=False,
                                )
                                store_coordinates = store_coordinates[
                                    (
                                        None,
                                        None,
                                        None,
                                        0,
                                        subtile_idx - fragment_idx,
                                    )
                                ]
                                store_coord_flt = cute.filter_zeros(store_coordinates)
                                for i in cutlass.range(
                                    cute.size(final_flt), unroll_full=True
                                ):
                                    row_idx = store_coord_flt[i][0]
                                    n_idx = store_coord_flt[i][1]
                                    group_idx = subtile_n_idx // fragments_per_group
                                    global_group_idx = (
                                        mma_tile_coord_mnl[1] * groups_per_cta
                                        + group_idx
                                    )
                                    if (
                                        n_idx % fragment_n == 0
                                        and row_idx < limit_m
                                        and global_group_idx < limit_groups
                                    ):
                                        value = final_flt[i]
                                        if cutlass.const_expr(
                                            self.local_reduce_type in ("max", "min")
                                        ):
                                            if nan_flt[i] > 0.0:
                                                value = float("nan")
                                        value = self.local_reduce_finalize(value, group)
                                        g_reduce[row_idx, group_idx] = value
                        if cutlass.const_expr(
                            local_reduce_secondary_feed_tensor is not None
                            and self.local_reduce_secondary_consumer is not None
                        ):
                            output_m = cute.size(mA_mkl, mode=[0])
                            output_n = cute.size(mB_nkl, mode=[0])
                            for i in cutlass.range(
                                cute.size(acc_flt), unroll_full=True
                            ):
                                global_m = (
                                    mma_tile_coord_mnl[0] * self.cta_tile_shape_mnk[0]
                                    + acc_coord_flt[i][0]
                                )
                                global_n = (
                                    mma_tile_coord_mnl[1] * self.cta_tile_shape_mnk[1]
                                    + acc_coord_flt[i][1]
                                )
                                if global_m < output_m and global_n < output_n:
                                    local_reduce_secondary_feed_tensor[
                                        global_m,
                                        global_n,
                                        mma_tile_coord_mnl[2],
                                    ] = self.local_reduce_secondary_consumer(
                                        acc_flt[i], reduced_flt[i], 0.0
                                    ).to(
                                        local_reduce_secondary_feed_tensor.element_type
                                    )
                        if cutlass.const_expr(self.local_reduce_feeds_main):
                            primary_acc_vec = epilogue_context.acc_vec
                            for i in cutlass.range(
                                cute.size(acc_flt), unroll_full=True
                            ):
                                if cutlass.const_expr(
                                    self.local_reduce_consumer is not None
                                ):
                                    value = self.local_reduce_consumer(
                                        acc_flt[i], reduced_flt[i], 0.0
                                    )
                                else:
                                    value = reduced_flt[i]
                                acc_flt[i] = value
                            if cutlass.const_expr(local_reduce_feed_tensor is not None):
                                output_m = cute.size(mA_mkl, mode=[0])
                                output_n = cute.size(mB_nkl, mode=[0])
                                for i in cutlass.range(
                                    cute.size(acc_flt), unroll_full=True
                                ):
                                    global_m = (
                                        mma_tile_coord_mnl[0]
                                        * self.cta_tile_shape_mnk[0]
                                        + acc_coord_flt[i][0]
                                    )
                                    global_n = (
                                        mma_tile_coord_mnl[1]
                                        * self.cta_tile_shape_mnk[1]
                                        + acc_coord_flt[i][1]
                                    )
                                    if global_m < output_m and global_n < output_n:
                                        local_reduce_feed_tensor[
                                            global_m,
                                            global_n,
                                            mma_tile_coord_mnl[2],
                                        ] = acc_flt[i].to(
                                            local_reduce_feed_tensor.element_type
                                        )
                            if cutlass.const_expr(local_reduce_feed_tensor is not None):
                                epilogue_context.acc_vec = primary_acc_vec
                            else:
                                epilogue_context.acc_vec = acc_fragment.load().to(
                                    self.epi_dtype
                                )

                    # Execute the EFC epilogue.
                    self.efc.kernel.epilogue_computation(epilogue_context)
                    d_buffer = (num_prev_subtiles + subtile_idx) % self.num_d_stage

                    # Store the EFC written tensors to shared memory.
                    self.efc.kernel.store_written_tensors_to_smem(d_buffer)

                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    epilogue_sync_barrier.arrive_and_wait()

                    #
                    # TMA store D to global memory
                    #
                    if warp_idx == self.epilogue_warp_id[0]:
                        # Store with TMA the written EFC tensors to global memory.
                        self.efc.kernel.tma_store_written_tensors_to_gmem(
                            d_buffer, subtile_idx
                        )
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        d_pipeline.producer_commit()
                        d_pipeline.producer_acquire()

                    epilogue_sync_barrier.arrive_and_wait()

                #
                # Async arrive accumulator buffer empty
                #
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            epilogue_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
            #
            # Wait for D store complete
            #
            d_pipeline.producer_tail()

        #
        # Specialized epilogue load warp
        #
        if warp_idx == self.epilogue_load_warp_id:
            # Create the tiled tensors to be loaded in the epilogue.
            self.efc.kernel.create_epilogue_subtile_tensors(tidx, epi_tile)

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            c_pipeline_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_c_stage
            )

            # Setup the pipelines reading the EFC supplemental tensors.

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                # Prepare the EFC tensors to be loaded by the subtiles.
                subtile_cnt = self.efc.kernel.prepare_tensor_load_for_subtiles(
                    mma_tile_coord_mnl,
                )

                # Assume the pipeline can work even in the case there is no
                # tensor to load and so subtile_cnt is 0.
                for subtile_idx in cutlass.range(subtile_cnt):
                    # Load C from global memory to shared memory.
                    c_pipeline.producer_acquire(c_pipeline_producer_state)

                    # Load the subtiles of EFC tensors.
                    self.efc.kernel.load_tensor_subtiles(
                        subtile_idx, c_pipeline, c_pipeline_producer_state
                    )

                    c_pipeline_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait for the load buffer to be empty.
            #
            c_pipeline.producer_tail(c_pipeline_producer_state)

    def epilogue_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        tCgC: cute.Tensor,
        epi_tile: cute.Tile,
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination)."""
        # Make tiledCopy for tensor memory load
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.d_layout,  # Take this as the reference layout for the epilogue tile.
            self.epi_dtype,  # But we get the accumulator as epi_dtype in the epilogue.
            self.acc_dtype,
            epi_tile,
            self.use_2cta_instrs,
        )

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tCgC_epi = cute.flat_divide(
            tCgC[((None, None), 0, 0, None, None, None)], epi_tile
        )
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tTR_gC = thr_copy_t2r.partition_D(tCgC_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilogue_smem_copy_and_partition_load(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """Make tiledCopy for shared memory load, then use it to partition register array (destination) and shared memory (source).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rC: The partitioned accumulator tensor
        :type tTR_rC: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor

        :return: A tuple containing (tiled_copy_s2r, tSR_rC, tSR_sC) where:
            - tiled_copy_s2r: The tiled copy operation for smem to register copy(s2r)
            - tSR_rC: The partitioned tensor C (register destination)
            - tSR_sC: The partitioned tensor C (smem source)
        :rtype: tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_s2r = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype)
        tiled_copy_s2r = cute.make_tiled_copy_D(copy_atom_s2r, tiled_copy_t2r)
        # (S2R, S2R_M, S2R_N, PIPE_C)
        thr_copy_s2r = tiled_copy_s2r.get_slice(tidx)
        tSR_sC = thr_copy_s2r.partition_D(sC)
        # (S2R, S2R_M, S2R_N)
        tSR_rC = tiled_copy_s2r.retile(tTR_rC)
        return tiled_copy_s2r, tSR_rC, tSR_sC

    def epilogue_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: cute.CopyAtom | cute.TiledCopy,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
        dtype: type[cutlass.Numeric],
    ) -> tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """Make tiledCopy for global memory store, then use it to:
        - partition register array (source) and global memory (destination) for non-TMA store version;
        - partition shared memory (source) and global memory (destination) for TMA store version.

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param atom: The copy_atom_c to be used for TMA store version, or tiled_copy_t2r for non-TMA store version
        :type atom: cute.CopyAtom or cute.TiledCopy
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor

        :return: A tuple containing either:
            - For TMA store: (tma_atom_c, bSG_sC, bSG_gC) where:
                - tma_atom_c: The TMA copy atom
                - bSG_sC: The partitioned shared memory tensor C
                - bSG_gC: The partitioned global tensor C
            - For non-TMA store: (simt_atom, tTR_rC, tTR_gC) where:
                - simt_atom: The SIMT copy atom
                - tTR_rC: The register tensor C
                - tTR_gC: The partitioned global tensor C
        :rtype: tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, tiles_m, tiles_n, tiles_l)
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )

        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, tiles_m, tiles_n, tiles_l)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    def compute_stages(self) -> None:
        """Compute and set the number of stages for A/B/C/D operands.

        Uses instance attributes to compute and assign:
        `self.num_acc_stage`, `self.num_ab_stage`, `self.num_c_stage`,
        and `self.num_d_stage`.
        """
        # Defaults
        self.num_acc_stage = 2
        # To read the tensors needed for the epilogue:
        self.num_c_stage = 2
        # To write the tensors produced by the epilogue:
        self.num_d_stage = 2

        # Calculate smem layout and size for one stage of A, B, C, and D
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            self._tiled_mma, self.mma_tiler, self.a_dtype, 1
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            self._tiled_mma, self.mma_tiler, self.b_dtype, 1
        )

        # Get the contribution from the tensors used in the EFC.
        self.efc.jit.compute_stage()

        ab_bytes_per_stage = cute.size_in_bytes(
            self.a_dtype, a_smem_layout_stage_one
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout_staged_one)
        mbar_helpers_bytes = 2048 + self.local_reduce_smem_bytes
        # Contribution from the tensors loaded in the EFC.
        c_bytes_per_stage = self.efc.jit.smem_size_in_bytes_of_read_tensors()
        c_bytes = c_bytes_per_stage * self.num_c_stage
        # Contribution from the tensors stored in the EFC. There is at least 1
        # written tensor, so the following is strictly positive.
        d_bytes_per_stage = self.efc.jit.smem_size_in_bytes_of_written_tensors()
        d_bytes = d_bytes_per_stage * self.num_d_stage

        # Calculate A/B stages
        self.num_ab_stage = (
            self.smem_capacity // self.occupancy
            - (mbar_helpers_bytes + c_bytes + d_bytes)
        ) // ab_bytes_per_stage
        log(f"{common_efc.TAB}{self.num_ab_stage = }")

        if self.num_ab_stage <= 0:
            raise MemoryError("Not enough smem capacity to allocate all the tensors.")

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B stages and reserved bytes.
        # Add remaining unused smem to epilogue.
        self.num_d_stage += (
            self.smem_capacity
            - self.occupancy * ab_bytes_per_stage * self.num_ab_stage
            - self.occupancy * (mbar_helpers_bytes + c_bytes + d_bytes)
        ) // (self.occupancy * d_bytes_per_stage)
        log(f"{common_efc.TAB}new {self.num_d_stage = }")

    @staticmethod
    def _compute_grid(
        d: cute.Tensor,
        cta_tile_shape_mnk: tuple[int, int, int],
        cluster_shape_mn: tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]:
        """Use persistent tile scheduler to compute the grid size for the output tensor D.

        :param d: The output tensor D
        :type d: cute.Tensor
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        :rtype: tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]
        """
        log(f"compute_grid: {max_active_clusters = }")
        d_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gd = cute.zipped_divide(d, tiler=d_shape)
        num_ctas_mnl = gd[(0, (None, None, None))].shape
        common_efc.if_debug(
            lambda: cute.printf("compute_grid: num_ctas_mnl = {}", num_ctas_mnl)
        )
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        common_efc.if_debug(lambda: cute.printf("compute_grid: grid = {}", grid))
        return tile_sched_params, grid

    def compute_num_tmem_alloc_cols(self) -> None:
        """Compute and set the number of tensor memory allocation columns.

        This method uses the instance attributes computed during setup to
        determine the number of tensor memory allocation columns and stores
        the result in `self.num_tmem_alloc_cols`.
        """
        acc_shape = self._tiled_mma.partition_shape_C(self.mma_tiler[:2])
        log(f"compute_num_tmem_alloc_cols: {acc_shape = !s}")
        tCtAcc_fake = self._tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )
        log(f"compute_num_tmem_alloc_cols: {tCtAcc_fake = !s}")
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        log(f"compute_num_tmem_alloc_cols: {self.num_tmem_alloc_cols = }")
