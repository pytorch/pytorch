# Copyright (c) 2026 PyTorch Contributors

import math

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils as cutlass_utils
import triton
import triton.language as tl

import torch
from torch._inductor.kernel.vendored_templates.cutedsl.kernels.cutedsl_grouped_gemm import (
    create_tensor_and_stride,
    GroupedGemmKernel,
)


@triton.jit
def _mirror_symmetric_pairs(
    output_ptrs,
    sizes,
    c_ptrs,
    alpha,
    beta,
    HAS_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile = tl.program_id(0)
    group = tl.program_id(1)
    m = tl.load(sizes + group)
    tiles_n = (m + BLOCK_N - 1) // BLOCK_N
    tiles_m = (m + BLOCK_M - 1) // BLOCK_M
    tile_m = tile // tiles_n
    tile_n = tile % tiles_n
    row = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    col = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    output = tl.load(output_ptrs + group).to(tl.pointer_type(tl.bfloat16))
    valid = (tile < tiles_m * tiles_n) & (row < m) & (col < m)
    if HAS_C:
        mask = valid & (row >= col)
        value = tl.load(output + row * m + col, mask=mask)
        c = tl.load(c_ptrs + group).to(tl.pointer_type(tl.bfloat16))
        value = alpha * value + beta * tl.load(c + row * m + col, mask=mask)
        tl.store(output + row * m + col, value, mask=mask)
        tl.store(output + col * m + row, value, mask=mask)
    else:
        mask = valid & (col > row)
        value = tl.load(output + col * m + row, mask=mask)
        tl.store(output + row * m + col, value, mask=mask)


class GroupedSymmetricPlan:
    """A reusable fixed-pointer plan for independent symmetric Gram matrices.

    The plan owns its outputs and retains its inputs. Callers must reuse the
    returned tensors rather than treating each invocation as a fresh functional
    allocation.
    """

    def __init__(
        self,
        inputs: list[torch.Tensor],
        c: list[torch.Tensor] | None = None,
        alpha: float = 1.0,
        beta: float = 0.0,
        outputs: list[torch.Tensor] | None = None,
        *,
        mma_tiler: tuple[int, int] = (256, 256),
        cluster_shape: tuple[int, int] = (2, 1),
        use_2cta: bool = True,
        row_block: int = 256,
        tensormap_update_mode=cutlass_utils.TensorMapUpdateMode.SMEM,
        compiled_plan: "GroupedSymmetricPlan | None" = None,
    ) -> None:
        if not inputs:
            raise ValueError("grouped symmetric GEMM requires at least one input")
        if c is not None and len(c) != len(inputs):
            raise ValueError("grouped symmetric GEMM C list has the wrong length")
        if any(
            x.ndim != 2
            or x.dtype != torch.bfloat16
            or not x.is_cuda
            or not x.is_contiguous()
            for x in inputs
        ):
            raise ValueError(
                "grouped symmetric GEMM requires contiguous CUDA BF16 matrices"
            )
        if any(x.device != inputs[0].device for x in inputs):
            raise ValueError("grouped symmetric GEMM inputs must use one device")
        if c is not None and any(
            tensor.shape != (x.shape[0], x.shape[0])
            or tensor.dtype != x.dtype
            or tensor.device != x.device
            or not tensor.is_contiguous()
            for x, tensor in zip(inputs, c)
        ):
            raise ValueError("grouped symmetric GEMM C tensors have invalid metadata")

        self.inputs = inputs
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.outputs = outputs or [
            torch.empty(x.shape[0], x.shape[0], dtype=x.dtype, device=x.device)
            for x in inputs
        ]
        if len(self.outputs) != len(inputs) or any(
            out.shape != (x.shape[0], x.shape[0])
            or out.dtype != x.dtype
            or out.device != x.device
            or not out.is_contiguous()
            for x, out in zip(inputs, self.outputs)
        ):
            raise ValueError("grouped symmetric GEMM outputs have invalid metadata")
        self.output_ptrs = torch.tensor(
            [out.data_ptr() for out in self.outputs],
            device=inputs[0].device,
            dtype=torch.int64,
        )
        self.output_sizes = torch.tensor(
            [out.shape[0] for out in self.outputs],
            device=inputs[0].device,
            dtype=torch.int32,
        )
        self.c_ptrs = torch.tensor(
            [tensor.data_ptr() for tensor in c] if c is not None else [0],
            device=inputs[0].device,
            dtype=torch.int64,
        )

        problems = []
        strides = []
        pointers = []
        for x, out in zip(inputs, self.outputs):
            m, k = x.shape
            for row in range(0, m, row_block):
                rows = min(row_block, m - row)
                end = row + rows
                problems.append((rows, end, k, 1))
                strides.append(((k, 1), (k, 1), (m, 1)))
                pointers.append(
                    (
                        x.data_ptr() + row * k * x.element_size(),
                        x.data_ptr(),
                        out.data_ptr() + row * m * out.element_size(),
                    )
                )

        with torch.cuda.device(inputs[0].device):
            self.problem_sizes, self._problem_sizes_torch = self._metadata(
                problems, torch.int32, cutlass.Int32
            )
            self.strides, self._strides_torch = self._metadata(
                strides, torch.int32, cutlass.Int32
            )
            self.pointers, self._pointers_torch = self._metadata(
                pointers, torch.int64, cutlass.Int64
            )
            hardware = cutlass_utils.HardwareInfo()
            sm_count = hardware.get_max_active_clusters(1)
            tensormap_shape = (
                sm_count,
                GroupedGemmKernel.num_tensormaps,
                GroupedGemmKernel.bytes_per_tensormap // 8,
            )
            self.tensormaps, self._tensormaps_torch = cutlass_torch.cute_tensor_like(
                torch.empty(tensormap_shape, dtype=torch.int64),
                cutlass.Int64,
                is_dynamic_layout=False,
            )
            self.initial = [
                create_tensor_and_stride(1, 8, 8, False, cutlass.BFloat16)[2]
                for _ in range(3)
            ]
            max_active_clusters = hardware.get_max_active_clusters(
                math.prod(cluster_shape)
            )
            cta_m = mma_tiler[0] // (2 if use_2cta else 1)
            cluster_tile = (
                cta_m * cluster_shape[0],
                mma_tiler[1] * cluster_shape[1],
            )
            total_clusters = sum(
                math.ceil(m / cluster_tile[0]) * math.ceil(n / cluster_tile[1])
                for m, n, _, _ in problems
            )
            self._compile_key = (
                len(problems),
                total_clusters,
                inputs[0].device,
                mma_tiler,
                cluster_shape,
                use_2cta,
                tensormap_update_mode,
            )
            if (
                compiled_plan is not None
                and compiled_plan._compile_key != self._compile_key
            ):
                raise ValueError("compiled grouped symmetric GEMM plan is incompatible")
            self.compiled = (
                compiled_plan.compiled if compiled_plan is not None else None
            )
            if self.compiled is None:
                kernel = GroupedGemmKernel(
                    cutlass.Float32,
                    use_2cta,
                    mma_tiler,
                    cluster_shape,
                    tensormap_update_mode,
                )
                self.compiled = cute.compile(
                    kernel,
                    *self.initial,
                    len(problems),
                    self.problem_sizes,
                    self.strides,
                    self.pointers,
                    total_clusters,
                    self.tensormaps,
                    max_active_clusters,
                    cutlass_torch.current_stream(),
                    options="--opt-level 2",
                )

    @staticmethod
    def _metadata(values, torch_dtype, cutlass_dtype):
        return cutlass_torch.cute_tensor_like(
            torch.tensor(values, dtype=torch_dtype),
            cutlass_dtype,
            is_dynamic_layout=False,
            assumed_align=16,
        )

    def __call__(self) -> list[torch.Tensor]:
        with torch.cuda.device(self.inputs[0].device):
            stream = cutlass_torch.current_stream()
            compiled = self.compiled
            if compiled is None:
                raise RuntimeError("grouped symmetric GEMM plan was not compiled")
            compiled(
                *self.initial,
                self.problem_sizes,
                self.strides,
                self.pointers,
                self.tensormaps,
                stream,
            )
            block_m, block_n = 32, 128
            max_tiles = max(
                triton.cdiv(out.shape[0], block_m) * triton.cdiv(out.shape[0], block_n)
                for out in self.outputs
            )
            _mirror_symmetric_pairs[(max_tiles, len(self.outputs))](
                self.output_ptrs,
                self.output_sizes,
                self.c_ptrs,
                self.alpha,
                self.beta,
                HAS_C=self.c is not None,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
            )
        return self.outputs


class SymmetricMuonPlan:
    def __init__(
        self,
        shape: tuple[int, int],
        device: torch.device,
        coefficients: tuple[float, float, float],
        steps: int,
        workspace: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        self.a, self.b, self.c = coefficients
        m, k = min(shape), max(shape)
        self.inputs = torch.empty(m, k, device=device, dtype=torch.bfloat16)
        self.alternate = torch.empty_like(self.inputs)
        if workspace is None:
            self.gram = torch.empty(m, m, device=device, dtype=torch.bfloat16)
            self.update = torch.empty_like(self.gram)
        else:
            self.gram, self.update = workspace
            if any(
                tensor.shape != (m, m)
                or tensor.device != device
                or tensor.dtype != torch.bfloat16
                for tensor in workspace
            ):
                raise ValueError("symmetric Muon workspace has invalid metadata")
        self.num_steps = steps
        self.outputs = self.inputs if steps % 2 == 0 else self.alternate

    def __call__(self, updates: list[torch.Tensor], eps: float) -> list[torch.Tensor]:
        from torch._vendor.quack.gemm_symmetric import gemm_symmetric

        if len(updates) != 1:
            raise ValueError("symmetric Muon plan requires exactly one update")
        source = updates[0]
        source = source.T if source.shape[0] > source.shape[1] else source
        self.inputs.copy_(source)
        self.inputs.div_(self.inputs.norm().clamp_min(eps))
        buffers = (self.inputs, self.alternate)
        for step in range(self.num_steps):
            x = buffers[step % 2]
            out = buffers[(step + 1) % 2]
            gemm_symmetric(x, self.gram)
            gemm_symmetric(
                self.gram,
                self.update,
                C=self.gram,
                alpha=self.c,
                beta=self.b,
            )
            torch.addmm(x, self.update, x, beta=self.a, out=out)
        return [self.outputs]


class SequentialSymmetricMuonPlan:
    def __init__(
        self,
        shapes: list[tuple[int, int]],
        device: torch.device,
        coefficients: tuple[float, float, float],
        steps: int,
    ) -> None:
        if not shapes:
            raise ValueError("sequential symmetric Muon plan requires an input")
        first = SymmetricMuonPlan(shapes[0], device, coefficients, steps)
        workspace = (first.gram, first.update)
        self.plans = [first]
        self.plans.extend(
            SymmetricMuonPlan(shape, device, coefficients, steps, workspace)
            for shape in shapes[1:]
        )

    def __call__(self, updates: list[torch.Tensor], eps: float) -> list[torch.Tensor]:
        if len(updates) != len(self.plans):
            raise ValueError(
                "sequential symmetric Muon plan has the wrong update count"
            )
        return [plan([update], eps)[0] for plan, update in zip(self.plans, updates)]


class PackedSymmetricMuonPlan:
    def __init__(
        self,
        shapes: list[tuple[int, int]],
        device: torch.device,
        coefficients: tuple[float, float, float],
        steps: int,
    ) -> None:
        if not shapes:
            raise ValueError("packed symmetric Muon plan requires an input")
        normalized = (min(shapes[0]), max(shapes[0]))
        if any((min(shape), max(shape)) != normalized for shape in shapes[1:]):
            raise ValueError("packed symmetric Muon plan requires one shape")
        self.a, self.b, self.c = coefficients
        m, k = normalized
        self.inputs = torch.empty(
            len(shapes), m, k, device=device, dtype=torch.bfloat16
        )
        self.alternate = torch.empty_like(self.inputs)
        self.gram = torch.empty(len(shapes), m, m, device=device, dtype=torch.bfloat16)
        self.update = torch.empty_like(self.gram)
        self.input_views = list(self.inputs.unbind())
        self.num_steps = steps
        outputs = self.inputs if steps % 2 == 0 else self.alternate
        self.outputs = list(outputs.unbind())

    def __call__(self, updates: list[torch.Tensor], eps: float) -> list[torch.Tensor]:
        from torch._vendor.quack.gemm_symmetric import gemm_symmetric

        if len(updates) != len(self.input_views):
            raise ValueError("packed symmetric Muon plan has the wrong update count")
        sources = [
            source.T if source.shape[0] > source.shape[1] else source
            for source in updates
        ]
        torch._foreach_copy_(self.input_views, sources)
        norms = torch.linalg.vector_norm(
            self.inputs, dim=(-2, -1), keepdim=True
        ).clamp_min_(eps)
        self.inputs.div_(norms)
        buffers = (self.inputs, self.alternate)
        for step in range(self.num_steps):
            x = buffers[step % 2]
            out = buffers[(step + 1) % 2]
            gemm_symmetric(x, self.gram)
            gemm_symmetric(
                self.gram,
                self.update,
                C=self.gram,
                alpha=self.c,
                beta=self.b,
            )
            torch.bmm(self.update, x, out=out)
            out.add_(x, alpha=self.a)
        return self.outputs


class GroupedMuonPlan:
    r"""Reusable grouped Newton-Schulz plan for a fixed sequence of matrix shapes.

    Matrices with the same normalized shape share contiguous batched storage,
    while the grouped symmetric kernels retain fixed pointers for every matrix.
    """

    def __init__(
        self,
        shapes: list[tuple[int, int]],
        device: torch.device,
        coefficients: tuple[float, float, float],
        steps: int,
        *,
        mma_tiler: tuple[int, int] = (256, 256),
        cluster_shape: tuple[int, int] = (2, 1),
        use_2cta: bool = True,
        row_block: int = 256,
        tensormap_update_mode=cutlass_utils.TensorMapUpdateMode.SMEM,
    ) -> None:
        if not shapes:
            raise ValueError("grouped Muon plan requires an input")
        self.shapes = shapes
        self.a, self.b, self.c = coefficients
        normalized = [(min(shape), max(shape)) for shape in shapes]
        counts: dict[tuple[int, int], int] = {}
        for shape in normalized:
            counts[shape] = counts.get(shape, 0) + 1
        storage_views = {}
        self.storage_groups = []
        for (m, k), count in counts.items():
            input_storage = torch.empty(
                count, m, k, device=device, dtype=torch.bfloat16
            )
            alternate_storage = torch.empty_like(input_storage)
            gram_storage = torch.empty(count, m, m, device=device, dtype=torch.bfloat16)
            update_storage = torch.empty_like(gram_storage)
            storage_views[(m, k)] = tuple(
                list(storage.unbind())
                for storage in (
                    input_storage,
                    alternate_storage,
                    gram_storage,
                    update_storage,
                )
            )
            self.storage_groups.append(
                (m, input_storage, alternate_storage, update_storage)
            )
        offsets = dict.fromkeys(counts, 0)
        self.inputs = []
        self.alternate = []
        gram_outputs = []
        update_outputs = []
        for shape in normalized:
            offset = offsets[shape]
            views = storage_views[shape]
            self.inputs.append(views[0][offset])
            self.alternate.append(views[1][offset])
            gram_outputs.append(views[2][offset])
            update_outputs.append(views[3][offset])
            offsets[shape] += 1
        plan_kwargs = {
            "mma_tiler": mma_tiler,
            "cluster_shape": cluster_shape,
            "use_2cta": use_2cta,
            "row_block": row_block,
            "tensormap_update_mode": tensormap_update_mode,
        }
        first_gram = GroupedSymmetricPlan(
            self.inputs, outputs=gram_outputs, **plan_kwargs
        )
        self.gram_plans = (
            first_gram,
            GroupedSymmetricPlan(
                self.alternate,
                outputs=first_gram.outputs,
                compiled_plan=first_gram,
                **plan_kwargs,
            ),
        )
        self.update_plan = GroupedSymmetricPlan(
            first_gram.outputs,
            c=first_gram.outputs,
            alpha=self.c,
            beta=self.b,
            outputs=update_outputs,
            compiled_plan=first_gram,
            **plan_kwargs,
        )
        self.num_steps = steps
        self.outputs = self.inputs if steps % 2 == 0 else self.alternate

    def __call__(self, updates: list[torch.Tensor], eps: float) -> list[torch.Tensor]:
        sources = [
            source.T if source.shape[0] > source.shape[1] else source
            for source in updates
        ]
        torch._foreach_copy_(self.inputs, sources)
        if len(self.storage_groups) == 1:
            input_storage = self.storage_groups[0][1]
            norms = torch.linalg.vector_norm(
                input_storage, dim=(-2, -1), keepdim=True
            ).clamp_min_(eps)
            input_storage.div_(norms)
        else:
            norms = torch._foreach_norm(self.inputs)
            torch._foreach_clamp_min_(norms, eps)
            torch._foreach_div_(self.inputs, norms)
        for step in range(self.num_steps):
            self.gram_plans[step % 2]()
            self.update_plan()
            for (
                m,
                input_storage,
                alternate_storage,
                update_storage,
            ) in self.storage_groups:
                packed_buffers = (input_storage, alternate_storage)
                x = packed_buffers[step % 2]
                out = packed_buffers[(step + 1) % 2]
                if x.shape[0] == 1:
                    torch.addmm(x[0], update_storage[0], x[0], beta=self.a, out=out[0])
                elif m >= 3200:
                    torch.bmm(update_storage, x, out=out)
                    out.add_(x, alpha=self.a)
                else:
                    torch.baddbmm(x, update_storage, x, beta=self.a, out=out)
        return self.outputs
