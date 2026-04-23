# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import contextlib
import itertools
import random
import unittest
import unittest.mock
from collections.abc import Callable
from typing import Any, ClassVar

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.experimental._attention import (
    _CausalBehavior,
    _context_parallel_shard,
    _ContextParallel,
    _cp_options,
    _disable_context_parallel_dispatcher,
    _enable_context_parallel_dispatcher,
    _HeadTailLoadBalancer,
    _is_causal_behavior,
    _LoadBalancer,
    _PerDocumentHeadTailLoadBalancer,
    _PTRRLoadBalancer,
    _RotateMethod,
    _VarlenPTRRLoadBalancer,
    context_parallel,
    context_parallel_unshard,
    set_rotate_method,
    VarlenMetadata,
)
from torch.distributed.tensor.experimental._context_parallel._cp_custom_ops import (
    flex_cp_allgather,
)
from torch.distributed.tensor.experimental._context_parallel._sharding_rules import (
    register_cp_sharding_rules,
    unregister_cp_sharding_rules,
)
from torch.distributed.tensor.parallel import parallelize_module
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    AuxOutput,
    AuxRequest,
    BlockMask,
    create_block_mask,
    flex_attention,
)
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_CUDNN_ATTENTION,
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_FUSED_ATTENTION,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests, skipIfRocm, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
    DTensorTestBase,
    map_local_tensor_for_rank,
    with_comms,
)


c10d_functional = torch.ops.c10d_functional
backends = []
if PLATFORM_SUPPORTS_FLASH_ATTENTION:
    backends.append(SDPBackend.FLASH_ATTENTION)
if PLATFORM_SUPPORTS_MEM_EFF_ATTENTION:
    backends.append(SDPBackend.EFFICIENT_ATTENTION)
if PLATFORM_SUPPORTS_CUDNN_ATTENTION:
    backends.append(SDPBackend.CUDNN_ATTENTION)

rotater_enum_to_str = {
    _RotateMethod.ALL_GATHER: "allgather",
    _RotateMethod.ALL_TO_ALL: "alltoall",
}  # mapping from _RotateMethod enum to string


class SDPAWrapper(torch.nn.Module):
    def __init__(self, compiled: bool, backend: SDPBackend) -> None:
        super().__init__()
        if compiled:
            self.sdpa = torch.compile(
                F.scaled_dot_product_attention,
                fullgraph=True,
                backend="aot_eager",
            )
        else:
            self.sdpa = F.scaled_dot_product_attention
        self.backend = backend

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        with sdpa_kernel(self.backend):
            return self.sdpa(*args, **kwargs)


class RingAttentionTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return torch.cuda.device_count()

    @property
    def destroy_pg_upon_exit(self) -> bool:
        return False

    @skip_if_lt_x_gpu(2)
    @skipIfRocm  # Missing _c10d_functional_autograd::all_to_all_single
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_ATTENTION,
        "Does not support flash nor efficient attention",
    )
    @with_comms
    def test_ring_attention_sdpa(self) -> None:
        self.run_subtests(
            {
                "is_causal": [True, False],
                "compiled": [True, False],
                "backend": backends,
                "load_balance": [True, False],
                "rotater": [_RotateMethod.ALL_TO_ALL, _RotateMethod.ALL_GATHER],
                "test_forward_only": [True, False],
                "use_context": [True, False],
            },
            self._test_ring_attention_sdpa,
        )

    def _ring_attention_sdpa(
        self,
        cp_q: torch.Tensor,
        cp_k: torch.Tensor,
        cp_v: torch.Tensor,
        *,
        fn_eval: Callable,
        mesh: DeviceMesh,
        seq_dim: int,
        is_causal: bool,
        compiled: bool,
        backend: SDPBackend,
        rotater: _RotateMethod,
        test_forward_only: bool,
        load_balance: bool,
        use_context: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not use_context:
            cp_plan = _ContextParallel(
                seq_dim=seq_dim,
                attention_type=_ContextParallel.AttentionType.SDPA,
            )
            attention = SDPAWrapper(compiled=compiled, backend=backend)
            attention = parallelize_module(attention, mesh, cp_plan)
            if load_balance:
                seq_len = cp_q.size(seq_dim)
                load_balancer = _HeadTailLoadBalancer(seq_len, mesh.size(), cp_q.device)
            else:
                load_balancer = None
            cp_q, cp_k, cp_v = _context_parallel_shard(
                mesh, (cp_q, cp_k, cp_v), (seq_dim,) * 3, load_balancer=load_balancer
            )
            _enable_context_parallel_dispatcher()
        else:
            # Theoretically, context_parallel() should not be used to shard
            # parameters because when require_grad is True, resize_ is not
            # allowed. But requires_grad of cp_q, cp_k, and cp_v are False
            # now. So we can just use context_parallel() to shard q, k, v.
            # In reality, context_parallel() should be used to shard the input.
            # In reality, context_parallel() should only be used to shard
            # the model inputs (batch).

            _cp_options.enable_load_balance = load_balance
            cp_context = context_parallel(
                mesh, buffers=(cp_q, cp_k, cp_v), buffer_seq_dims=(seq_dim,) * 3
            )
            cp_context.__enter__()

            # NOTE: This demonstrates that monkey patching is not fully reliable.
            # If we use SDPAWrapper directly, the monkey patching dispatch mode
            # does not function correctly. To ensure proper behavior,
            # F.scaled_dot_product_attention must be referenced within the
            # context_parallel() scope.
            attention = F.scaled_dot_product_attention
            if compiled:
                attention = torch.compile(
                    attention, fullgraph=True, backend="aot_eager"
                )

        for target in [cp_q, cp_k, cp_v]:
            target.requires_grad = True

        check_comm_counts = not compiled and rotater == _RotateMethod.ALL_TO_ALL
        comm_mode = CommDebugMode() if check_comm_counts else contextlib.nullcontext()
        with comm_mode:
            with sdpa_kernel(backend):
                cp_out = fn_eval(
                    attention,
                    cp_q,
                    cp_k,
                    cp_v,
                    is_causal=is_causal,
                )

            if check_comm_counts:
                expect_all2all_count = (
                    self.world_size - 1
                    if test_forward_only
                    else self.world_size * 3 - 2
                )
                self.assertDictEqual(
                    comm_mode.get_comm_counts(),
                    {c10d_functional.all_to_all_single: expect_all2all_count},
                )
        cp_dq, cp_dk, cp_dv = cp_q.grad, cp_k.grad, cp_v.grad
        for target in [cp_q, cp_k, cp_v]:
            target.requires_grad = False

        if not use_context:
            _disable_context_parallel_dispatcher()
        else:
            cp_context.__exit__(None, None, None)

        return cp_out, cp_dq, cp_dk, cp_dv

    def _test_ring_attention_sdpa(
        self,
        is_causal: bool,
        compiled: bool,
        backend: SDPBackend,
        load_balance: bool,
        rotater: _RotateMethod,
        test_forward_only: bool,
        use_context: bool,
    ) -> None:
        def fn_eval(fn, *args, **kwargs):
            if test_forward_only:
                with torch.no_grad():
                    return fn(*args, **kwargs)
            else:
                out = fn(*args, **kwargs)
                out.sum().backward()
                return out

        if load_balance and not is_causal:
            return

        # Compilation with context_parallel doesn't work yet -- both paths
        # (use_context=True monkey-patch and use_context=False parallelize_module)
        # fail during tracing because DTensor dispatch interferes with sdpa.
        # Previously CommDebugMode was active for all subtests, which caused
        # the frame to be silently skipped, masking this limitation.
        if compiled:
            return

        set_rotate_method(rotater_enum_to_str[rotater])
        self.assertEqual(_cp_options.rotate_method, rotater)
        device_mesh = DeviceMesh(self.device_type, torch.arange(0, self.world_size))
        dtype = torch.bfloat16
        bs = 8
        seq_length = 1024
        seq_dim = 2
        dim = 32
        nheads = 8
        torch.manual_seed(10)
        dtype = (
            torch.bfloat16
            if backend == SDPBackend.FLASH_ATTENTION
            or backend == SDPBackend.CUDNN_ATTENTION
            else torch.float32
        )

        q, k, v = [
            torch.rand(
                (bs, nheads, seq_length * self.world_size, dim),
                device=self.device_type,
                dtype=dtype,
                requires_grad=True,
            )
            for _ in range(3)
        ]

        # Ensure all ranks have the same initialization data.
        with torch.no_grad():
            dist.broadcast(q, src=0)
            dist.broadcast(k, src=0)
            dist.broadcast(v, src=0)

        with sdpa_kernel(backend):
            out = fn_eval(F.scaled_dot_product_attention, q, k, v, is_causal=is_causal)

        cp_q, cp_k, cp_v = [target.detach().clone() for target in [q, k, v]]
        cp_out, cp_dq, cp_dk, cp_dv = self._ring_attention_sdpa(
            cp_q,
            cp_k,
            cp_v,
            fn_eval=fn_eval,
            mesh=device_mesh,
            seq_dim=seq_dim,
            is_causal=is_causal,
            compiled=compiled,
            backend=backend,
            rotater=rotater,
            test_forward_only=test_forward_only,
            load_balance=load_balance,
            use_context=use_context,
        )

        # Due to numerical error, we need to choose different atol for different
        # attention kernels
        (cp_out,) = context_parallel_unshard(device_mesh, [cp_out], [seq_dim])
        atol = (
            2e-06
            if backend == SDPBackend.EFFICIENT_ATTENTION
            else 8e-3 * self.world_size
        )
        rtol = (
            1e-05
            if backend == SDPBackend.EFFICIENT_ATTENTION
            else 1e-3 * self.world_size
        )
        torch.testing.assert_close(out, cp_out, atol=atol, rtol=rtol)

        if test_forward_only:
            return

        cp_dq, cp_dk, cp_dv = context_parallel_unshard(
            device_mesh,
            [cp_dq, cp_dk, cp_dv],
            [seq_dim] * 3,
        )
        torch.testing.assert_close(q.grad, cp_dq, atol=atol, rtol=rtol)
        torch.testing.assert_close(k.grad, cp_dk, atol=atol, rtol=rtol)
        torch.testing.assert_close(v.grad, cp_dv, atol=atol, rtol=rtol)

    def test_is_causal_behavior(self) -> None:
        _cp_options.enable_load_balance = False
        self.assertEqual(
            _is_causal_behavior(rank=0, world_size=4, i=0, is_causal=False),
            _CausalBehavior.NOT_IS_CAUSAL,
        )

        ranks = [
            [_CausalBehavior.IS_CAUSAL, _CausalBehavior.SKIP],
            [_CausalBehavior.IS_CAUSAL, _CausalBehavior.NOT_IS_CAUSAL],
        ]
        for rank, iters in enumerate(ranks):
            for i, behavior in enumerate(iters):
                self.assertEqual(
                    _is_causal_behavior(rank=rank, world_size=2, i=i, is_causal=True),
                    behavior,
                )

        _cp_options.enable_load_balance = True
        ranks = [
            [_CausalBehavior.IS_CAUSAL, _CausalBehavior.NOT_IS_CAUSAL],
            [_CausalBehavior.IS_CAUSAL, _CausalBehavior.NOT_IS_CAUSAL],
        ]
        for rank, iters in enumerate(ranks):
            for i, behavior in enumerate(iters):
                self.assertEqual(
                    _is_causal_behavior(rank=rank, world_size=2, i=i, is_causal=True),
                    behavior,
                )


# Compile the flex_attention function
compiled_flex_attention = torch.compile(flex_attention, dynamic=False, fullgraph=True)
compiled_create_block_mask = torch.compile(
    create_block_mask, dynamic=False, fullgraph=True
)


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


# copied from https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/masks/document_mask.py
def generate_random_lengths(total_length, num_documents) -> list[int]:
    # Initialize all lengths to 1 to ensure each document has at least one token
    lengths = [1] * num_documents
    remaining_length = total_length - num_documents

    # Randomly distribute the remaining length
    for _ in range(remaining_length):
        index = random.randint(0, num_documents - 1)
        lengths[index] += 1

    return lengths


def generate_random_lengths_in_chunks(
    total_length, num_documents, chunk_size
) -> list[int]:
    # Generate a list of random document lengths so that each document contains
    # some number of chunks of size `chunk_size`. This means each document's length
    # must be a multiple of `chunk_size`. Besides, the lengths of all the documents
    # sum up to `total_length`.
    num_chunks = total_length // chunk_size
    if not (total_length % chunk_size == 0 and num_chunks >= num_documents):
        raise AssertionError(
            f"total_length % chunk_size == {total_length % chunk_size} (expected 0), "
            f"num_chunks={num_chunks} vs num_documents={num_documents}"
        )

    num_chunks_per_document = [1] * num_documents
    remaining_chunks = num_chunks - num_documents
    # Randomly distribute the remaining chunks
    for _ in range(remaining_chunks):
        index = random.randint(0, num_documents - 1)  # document_id
        num_chunks_per_document[index] += 1

    return [num_chunks * chunk_size for num_chunks in num_chunks_per_document]


def length_to_offsets(lengths: list[list[int]], device: str | torch.device) -> Tensor:
    """Converts a list of lengths to a list of offsets.

    Args:
        lengths: A list of lengths.

    """
    offsets = [[0] + lengths_in_batch for lengths_in_batch in lengths]
    offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
    offsets = torch.cumsum(offsets, dim=-1)
    return offsets


def _offsets_to_doc_ids_tensor(offsets):
    doc_ids = []
    device = offsets.device
    for batch_idx in range(offsets.size(0)):
        counts = offsets[batch_idx][1:] - offsets[batch_idx][:-1]
        doc_id = torch.repeat_interleave(
            torch.arange(len(counts), device=device, dtype=torch.int32), counts
        )
        doc_ids.append(doc_id)

    return torch.stack(doc_ids)


def generate_doc_mask_mod(
    mask_mod: _mask_mod_signature, offsets: Tensor
) -> _mask_mod_signature:
    """Generates mask mods that apply to inputs to flex attention in the sequence stacked
    format.

    Args:
        mask_mod: The mask mod to apply to the documents
        offsets: This tensor should be of shape(num_documents + 1)
            this should contain the cumulative counts of document tokens.
            e.g. if you have 3 documents of length 2, 4, 3 then
            offsets = [0, 2, 6, 9]

    Note:
        What is the sequence stacked format? When assembling batches of inputs, we
        take multiple sequences and stack them together to form 1 large sequence. We then
        use masking to ensure that the attention scores are only applied to tokens within
        the same document.
    """
    document_id = _offsets_to_doc_ids_tensor(offsets)

    def doc_mask_mod(b, h, q_idx, kv_idx):
        same_doc = document_id[b][q_idx] == document_id[b][kv_idx]
        q_logical = q_idx - offsets[b, document_id[b, q_idx]]
        kv_logical = kv_idx - offsets[b, document_id[b, kv_idx]]
        inner_mask = mask_mod(b, h, q_logical, kv_logical)
        return same_doc & inner_mask

    return doc_mask_mod


class FlexAttentionWrapper(torch.nn.Module):
    _flex_attn: ClassVar[Callable] = torch.compile(flex_attention)

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, *args: object, **kwargs: object
    ) -> [
        torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, AuxOutput],
    ]:
        return FlexAttentionWrapper._flex_attn(*args, **kwargs)


class CPFlexAttentionTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _test_cp_flex_attention(
        self,
        *,
        qkv_size: int,
        B: int = 1,
        block_mask,
        lb_type: str,
        document_lengths: list[list[int]] | None = None,
    ) -> None:
        torch.use_deterministic_algorithms(True)
        torch.cuda.manual_seed(1234)

        dtype = torch.float32
        bs = B if B > 1 else 8
        dim = 32
        nheads = 8
        seq_dim = 2
        lb = self._get_load_balancer(
            lb_type,
            {
                "seq_length": qkv_size,
                "document_lengths": document_lengths,
                "block_mask": block_mask,
            },
        )

        qkv = [
            torch.rand(
                (bs, nheads, qkv_size, dim),
                device=self.device_type,
                dtype=dtype,
                requires_grad=True,
            )
            for _ in range(3)
        ]

        expect_out, expect_aux = compiled_flex_attention(
            *qkv, block_mask=block_mask, return_aux=AuxRequest(lse=True)
        )
        expect_out.sum().backward()

        # Prepare the required global vars for CP+Flex:
        device_mesh = init_device_mesh(
            device_type=self.device_type,
            mesh_shape=(self.world_size,),
            mesh_dim_names=("cp",),
        )

        flex_attention_wrapper_module = FlexAttentionWrapper()
        cp_plan = _ContextParallel(
            seq_dim=seq_dim,
            attention_type=_ContextParallel.AttentionType.FLEX,
        )
        parallelize_module(
            flex_attention_wrapper_module,
            device_mesh,
            cp_plan,
        )

        *cp_qkv, cp_block_mask = _context_parallel_shard(
            device_mesh,
            [t.detach().clone() for t in qkv] + [block_mask],
            [seq_dim] * 4,
            load_balancer=lb,
        )
        for t in cp_qkv:
            t.requires_grad = True

        cp_out, cp_aux = flex_attention_wrapper_module(
            *cp_qkv,
            block_mask=cp_block_mask,
            return_aux=AuxRequest(lse=True),
        )

        # backward run
        cp_out.sum().backward()

        atol = 2e-06
        rtol = 1e-05
        # unshard the output
        cp_out, cp_lse = context_parallel_unshard(
            device_mesh,
            buffers=[cp_out, cp_aux.lse],
            seq_dims=[seq_dim] * 2,
            load_balancer=lb,
        )
        torch.testing.assert_close(cp_out, expect_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(cp_lse, expect_aux.lse, atol=atol, rtol=rtol)

        # unshard the gradient
        cp_qkv_grad = context_parallel_unshard(
            device_mesh,
            buffers=[t.grad for t in cp_qkv],
            seq_dims=[seq_dim] * 3,
            load_balancer=lb,
        )

        qkv_grad = [t.grad for t in qkv]
        for grad, cp_grad in zip(qkv_grad, cp_qkv_grad):
            torch.testing.assert_close(grad, cp_grad, atol=atol, rtol=rtol)

    def _get_load_balancer(
        self, lb_type: str, kwargs: dict[str, Any]
    ) -> _LoadBalancer | None:
        seq_length = kwargs["seq_length"]
        document_lengths = kwargs["document_lengths"]
        block_mask = kwargs["block_mask"]

        # generate load balancer
        if lb_type == "None":
            load_balancer = None  # no load-balance
        elif lb_type == "_HeadTailLoadBalancer":
            if not isinstance(seq_length, int):
                raise AssertionError(f"Expected int, got {type(seq_length)}")
            load_balancer = _HeadTailLoadBalancer(
                seq_length, self.world_size, torch.device(self.device_type)
            )
        elif lb_type == "_PerDocumentHeadTailLoadBalancer":
            if not isinstance(document_lengths, list):
                raise AssertionError(f"Expected list, got {type(document_lengths)}")
            load_balancer = _PerDocumentHeadTailLoadBalancer(
                document_lengths, self.world_size, torch.device(self.device_type)
            )
        elif lb_type == "_PTRRLoadBalancer":
            if not isinstance(block_mask, BlockMask):
                raise AssertionError(f"Expected BlockMask, got {type(block_mask)}")
            load_balancer = _PTRRLoadBalancer(
                block_mask,
                self.world_size,
            )
        else:
            raise ValueError(f"load_balancer type {lb_type} is not supported!")

        return load_balancer

    @skip_if_lt_x_gpu(2)
    @with_comms
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention"
    )
    def test_cp_flex_attention_causal_mask(self) -> None:
        seq_length_list = [256 * self.world_size, 2048]
        load_balance_type_list = [
            "None",
            "_HeadTailLoadBalancer",
            "_PTRRLoadBalancer",
        ]

        # NOTE: Each (seq_len, load_balance_type) tuple introduces 2
        # create_block_mask compilations: 1 for single-rank flex_attention and 1 for
        # CP flex_attention. In order to avoid the "exceeds_recompile_limit" error,
        # we need to increase the cache_size_limit to 2 * num_of_sub_test_runs which
        # will be the total number of compilations in our test case.
        torch._dynamo.config.cache_size_limit = (len(seq_length_list) + 1) * (
            1 + len(load_balance_type_list)
        )

        for qkv_size, lb_type in itertools.product(
            seq_length_list, load_balance_type_list
        ):
            block_mask = compiled_create_block_mask(
                causal_mask,
                B=1,
                H=1,
                Q_LEN=qkv_size,
                KV_LEN=qkv_size,
                device=self.device_type,
            )
            self._test_cp_flex_attention(
                qkv_size=qkv_size, block_mask=block_mask, lb_type=lb_type
            )

        # NOTE: Context Parallel should not be used for small attentions (block_size < 128)
        qkv_size = 64 * self.world_size
        block_mask = compiled_create_block_mask(
            causal_mask,
            B=1,
            H=1,
            Q_LEN=qkv_size,
            KV_LEN=qkv_size,
            device=self.device_type,
        )

        for lb_type in ["None", "_HeadTailLoadBalancer"]:
            with self.assertRaisesRegex(
                NotImplementedError,
                f"Q_LEN {qkv_size} is not divisible",
            ):
                self._test_cp_flex_attention(
                    qkv_size=qkv_size, block_mask=block_mask, lb_type=lb_type
                )

        for lb_type in ["_PTRRLoadBalancer"]:
            with self.assertRaisesRegex(
                NotImplementedError,
                "must be divisible by group_size",
            ):
                self._test_cp_flex_attention(
                    qkv_size=qkv_size, block_mask=block_mask, lb_type=lb_type
                )

    # TODO: merge with the above test
    @skip_if_lt_x_gpu(2)
    @with_comms
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention"
    )
    def test_cp_flex_attention_document_mask(self) -> None:
        random.seed(10)

        # parameters for testing
        doc_count = 28
        batch_size_list = [2, 4, 8]
        max_seq_len_list = [
            256 * self.world_size,
            2048,
            # 128 * self.world_size  # NOTE: Mismatched elements: 8 / 131072 (0.0%),
        ]
        load_balance_type = [
            "None",
            "_HeadTailLoadBalancer",
            "_PerDocumentHeadTailLoadBalancer",
            "_PTRRLoadBalancer",
        ]

        # NOTE: Each (batch_size, seq_len, load_balance_type) tuple introduces 2
        # create_block_mask compilations: 1 for single-rank flex_attention and 1 for
        # CP flex_attention. In order to avoid the "exceeds_recompile_limit" error,
        # we need to increase the cache_size_limit to 2 * num_of_sub_test_runs which
        # will be the total number of compilations in our test case.
        torch._dynamo.config.cache_size_limit = (
            2 * len(batch_size_list) * len(max_seq_len_list) * len(load_balance_type)
        )

        # TODO: change this for-loop to run_subtests
        # Use a for-loop instead of run_subtests because we need to initialize the mask
        # for each subtest. This can be baked into self._test_cp_flex_attention as
        # a str argument denoting mask type.
        for batch_size, max_seq_len, lb_type in itertools.product(
            batch_size_list,
            max_seq_len_list,
            load_balance_type,
        ):
            # initialize document mask
            lengths = [
                (
                    generate_random_lengths_in_chunks(
                        max_seq_len, doc_count, chunk_size=2 * self.world_size
                    )
                    if lb_type == "_PerDocumentHeadTailLoadBalancer"
                    else generate_random_lengths(max_seq_len, doc_count)
                )
                for _ in range(batch_size)
            ]
            offsets = length_to_offsets(lengths, self.device_type)
            document_causal_mask = generate_doc_mask_mod(causal_mask, offsets)
            block_mask = compiled_create_block_mask(
                document_causal_mask,
                B=batch_size,
                H=1,
                Q_LEN=max_seq_len,
                KV_LEN=max_seq_len,
                device=self.device_type,
            )

            self._test_cp_flex_attention(
                qkv_size=max_seq_len,
                B=batch_size,
                lb_type=lb_type,
                block_mask=block_mask,
                document_lengths=lengths,
            )


class TestCPCustomOps(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_flex_cp_custom_op(self) -> None:
        mesh = init_device_mesh(
            device_type=self.device_type,
            mesh_shape=(self.world_size,),
            mesh_dim_names=("cp",),
        )
        examples_k_v = [
            (
                torch.randn(8, 8, 8, 8, device=self.device_type),
                torch.randn(8, 8, 8, 8, device=self.device_type),
                2,
                c10d._get_process_group_name(mesh.get_group()),
            ),
            (
                torch.randn(8, 8, 8, 8, device=self.device_type, requires_grad=True),
                torch.randn(8, 8, 8, 8, device=self.device_type, requires_grad=True),
                2,
                c10d._get_process_group_name(mesh.get_group()),
            ),
        ]
        for example in examples_k_v:
            torch.library.opcheck(flex_cp_allgather, example)


class TestSharding(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_context_parallel_shard(self) -> None:
        B = 4
        seq_len = 32

        device_mesh = init_device_mesh(
            mesh_shape=(2,), mesh_dim_names=("cp",), device_type=self.device_type
        )
        freqs_cis = torch.arange(0, seq_len, device=self.device_type)
        q = torch.ones(B * seq_len, device=self.device_type).reshape(B, seq_len)
        k = torch.ones(B * seq_len, device=self.device_type).reshape(B, seq_len)
        v = torch.ones(B * seq_len, device=self.device_type).reshape(B, seq_len)

        load_balancer = _HeadTailLoadBalancer(
            seq_len, self.world_size, torch.device(self.device_type)
        )
        freqs_cis_shard, q_shard, k_shard, v_shard = _context_parallel_shard(
            device_mesh, [freqs_cis, q, k, v], [0, 1, 1, 1], load_balancer=load_balancer
        )
        self.assertEqual(freqs_cis_shard.size(), (seq_len // 2,))
        chunks = freqs_cis.chunk(self.world_size * 2)
        self.assertEqual(
            freqs_cis_shard,
            map_local_tensor_for_rank(
                chunks,
                self.rank,
                lambda chunks, rank: torch.cat(
                    [chunks[rank], chunks[self.world_size * 2 - rank - 1]],
                    dim=0,
                ),
            ),
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_context_parallel_shard_with_positions(self) -> None:
        """Test context parallel sharding with expanded batch dimensions.

        This test validates the fix for buffer sharding when the batch dimension
        is created through expand() or view() operations. Before the fix, the
        loop-based torch.index_select approach failed on expanded tensors.
        """
        B = 4
        seq_len = 32

        device_mesh = init_device_mesh(
            mesh_shape=(2,), mesh_dim_names=("cp",), device_type=self.device_type
        )

        # Create positions tensor and expand to add batch dimension
        positions = torch.arange(0, seq_len, device=self.device_type)
        positions = positions.expand(B, seq_len)

        q = torch.ones(B * seq_len, device=self.device_type).reshape(B, seq_len)
        k = torch.ones(B * seq_len, device=self.device_type).reshape(B, seq_len)
        v = torch.ones(B * seq_len, device=self.device_type).reshape(B, seq_len)

        load_balancer = _HeadTailLoadBalancer(
            seq_len, self.world_size, torch.device(self.device_type)
        )

        # positions has seq_dim=1 (same as q, k, v) after expansion
        positions_shard, q_shard, k_shard, v_shard = _context_parallel_shard(
            device_mesh, [positions, q, k, v], [1, 1, 1, 1], load_balancer=load_balancer
        )

        # Verify the sharded positions tensor has correct shape
        self.assertEqual(positions_shard.size(), (B, seq_len // 2))

        # Verify the sharded values match expected chunked and concatenated results
        # For each batch, the positions should be chunked and rearranged
        chunks = positions.chunk(self.world_size * 2, dim=1)
        expected_positions = map_local_tensor_for_rank(
            chunks,
            self.rank,
            lambda chunks, rank: torch.cat(
                [chunks[rank], chunks[self.world_size * 2 - rank - 1]],
                dim=1,
            ),
        )
        self.assertEqual(positions_shard, expected_positions)

    @skip_if_lt_x_gpu(2)
    @with_comms
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_ATTENTION,
        "Does not support flash nor efficient attention",
    )
    def test_attention_shard_without_cp(self) -> None:
        """Test that sharding on sequence dimension without CP enabled is not supported."""
        from torch.distributed.tensor import distribute_tensor, Replicate, Shard

        B = 2
        nheads = 4
        seq_len = 256
        dim = 32

        device_mesh = init_device_mesh(
            mesh_shape=(2,), mesh_dim_names=("cp",), device_type=self.device_type
        )

        for backend in backends:
            with sdpa_kernel(backend):
                dtype = torch.bfloat16
                if backend == SDPBackend.EFFICIENT_ATTENTION:
                    dtype = torch.float32
                # Create q, k, v tensors with shape (B, nheads, seq_len, dim)
                q = torch.randn(
                    B, nheads, seq_len, dim, device=self.device_type, dtype=dtype
                )
                k = torch.randn(
                    B, nheads, seq_len, dim, device=self.device_type, dtype=dtype
                )
                v = torch.randn(
                    B, nheads, seq_len, dim, device=self.device_type, dtype=dtype
                )
                q_dt = distribute_tensor(q, device_mesh, [Shard(2)])
                k_dt = distribute_tensor(k, device_mesh, [Shard(2)])
                v_dt = distribute_tensor(v, device_mesh, [Shard(2)])

                register_cp_sharding_rules()
                out = F.scaled_dot_product_attention(q_dt, k_dt, v_dt)
                unregister_cp_sharding_rules(clear_the_cache=True)
                out = F.scaled_dot_product_attention(q_dt, k_dt, v_dt)
                # Run SDPA with sequence-sharded tensors WITHOUT enabling CP
                # Without CP enabled, DTensor should select a different strategy
                # (not sequence-sharded) because Shard(2) strategy is only available with CP

                # Verify the output is NOT sharded on sequence dimension (dim 2)
                # This proves that CP sharding rules were not used
                self.assertNotEqual(
                    out.placements[0], Shard(2), f"Placement {out.placements}"
                )
                # The output should be replicated or sharded on batch head dimensions.
                self.assertIn(out.placements[0], [Replicate(), Shard(0), Shard(1)])


class TestContextParallelStyle(DTensorTestBase):
    """Test suite for _ContextParallel.flex_input_fn argument handling"""

    @property
    def world_size(self) -> int:
        return 2

    def _create_test_tensors(self):
        """Helper to create test query, key, value tensors"""
        query = torch.randn(2, 4, 128, 64, device=self.device_type)
        key = torch.randn(2, 4, 128, 64, device=self.device_type)
        value = torch.randn(2, 4, 128, 64, device=self.device_type)
        return query, key, value

    def _setup_mock_and_context(self, mock_allgather, key, value):
        """Helper to setup mock and create CP instance + device mesh"""
        # Setup mock with transformed tensors
        mock_key = key * 2
        mock_value = value * 3
        mock_allgather.return_value = (mock_key, mock_value)

        # Create CP instance and device mesh
        cp_style = _ContextParallel(
            seq_dim=2, attention_type=_ContextParallel.AttentionType.FLEX
        )
        device_mesh = DeviceMesh(self.device_type, torch.arange(0, self.world_size))

        return cp_style, device_mesh, mock_key, mock_value

    @with_comms
    @unittest.mock.patch(
        "torch.distributed.tensor.experimental._context_parallel._attention.flex_cp_allgather"
    )
    def test_flex_input_fn_all_positional(self, mock_allgather):
        """Test flex_input_fn with all positional arguments"""
        query, key, value = self._create_test_tensors()
        cp_style, device_mesh, mock_key, mock_value = self._setup_mock_and_context(
            mock_allgather, key, value
        )

        # Call with all positional args
        args = (query, key, value)
        kwargs = {}
        out_args, out_kwargs = cp_style.flex_input_fn(None, args, kwargs, device_mesh)

        # Verify mock called and output structure
        mock_allgather.assert_called_once()
        self.assertEqual(len(out_args), 3)
        self.assertEqual(len(out_kwargs), 0)

        # Verify query unchanged, key/value replaced
        torch.testing.assert_close(out_args[0], query)
        torch.testing.assert_close(out_args[1], mock_key)
        torch.testing.assert_close(out_args[2], mock_value)

    @with_comms
    @unittest.mock.patch(
        "torch.distributed.tensor.experimental._context_parallel._attention.flex_cp_allgather"
    )
    def test_flex_input_fn_all_keyword(self, mock_allgather):
        """Test flex_input_fn with all keyword arguments"""
        query, key, value = self._create_test_tensors()
        cp_style, device_mesh, mock_key, mock_value = self._setup_mock_and_context(
            mock_allgather, key, value
        )

        # Call with all keyword args
        args = ()
        kwargs = {"query": query, "key": key, "value": value}
        out_args, out_kwargs = cp_style.flex_input_fn(None, args, kwargs, device_mesh)

        # Verify mock called and output structure
        mock_allgather.assert_called_once()
        self.assertEqual(len(out_args), 0)
        self.assertIn("query", out_kwargs)
        self.assertIn("key", out_kwargs)
        self.assertIn("value", out_kwargs)

        # Verify query unchanged, key/value replaced
        torch.testing.assert_close(out_kwargs["query"], query)
        torch.testing.assert_close(out_kwargs["key"], mock_key)
        torch.testing.assert_close(out_kwargs["value"], mock_value)

    @with_comms
    @unittest.mock.patch(
        "torch.distributed.tensor.experimental._context_parallel._attention.flex_cp_allgather"
    )
    def test_flex_input_fn_query_positional_kv_keyword(self, mock_allgather):
        """Test with query positional, key/value keyword"""
        query, key, value = self._create_test_tensors()
        cp_style, device_mesh, mock_key, mock_value = self._setup_mock_and_context(
            mock_allgather, key, value
        )

        # Query positional, key/value keyword
        args = (query,)
        kwargs = {"key": key, "value": value}
        out_args, out_kwargs = cp_style.flex_input_fn(None, args, kwargs, device_mesh)

        # Verify mock called and output structure
        mock_allgather.assert_called_once()
        self.assertEqual(len(out_args), 1)
        torch.testing.assert_close(out_args[0], query)

        # Verify key/value in kwargs and updated
        self.assertIn("key", out_kwargs)
        self.assertIn("value", out_kwargs)
        torch.testing.assert_close(out_kwargs["key"], mock_key)
        torch.testing.assert_close(out_kwargs["value"], mock_value)

    @with_comms
    @unittest.mock.patch(
        "torch.distributed.tensor.experimental._context_parallel._attention.flex_cp_allgather"
    )
    def test_flex_input_fn_qk_positional_v_keyword(self, mock_allgather):
        """Test with query/key positional, value keyword"""
        query, key, value = self._create_test_tensors()
        cp_style, device_mesh, mock_key, mock_value = self._setup_mock_and_context(
            mock_allgather, key, value
        )

        # Query/key positional, value keyword
        args = (query, key)
        kwargs = {"value": value}
        out_args, out_kwargs = cp_style.flex_input_fn(None, args, kwargs, device_mesh)

        # Verify mock called and output structure
        mock_allgather.assert_called_once()
        self.assertEqual(len(out_args), 2)
        torch.testing.assert_close(out_args[0], query)
        torch.testing.assert_close(out_args[1], mock_key)

        # Verify value in kwargs and updated
        self.assertIn("value", out_kwargs)
        torch.testing.assert_close(out_kwargs["value"], mock_value)

    @with_comms
    @unittest.mock.patch(
        "torch.distributed.tensor.experimental._context_parallel._attention.flex_cp_allgather"
    )
    def test_flex_input_fn_with_extra_args(self, mock_allgather):
        """Test with mixed positional/keyword and extra arguments"""
        query, key, value = self._create_test_tensors()
        cp_style, device_mesh, mock_key, mock_value = self._setup_mock_and_context(
            mock_allgather, key, value
        )

        # Mix of positional and keyword with extra args
        def score_mod(q, k, b, h, m, n):
            return q

        block_mask = None
        args = (query, key, value, score_mod, block_mask)
        kwargs = {"enable_gqa": False}
        out_args, out_kwargs = cp_style.flex_input_fn(None, args, kwargs, device_mesh)

        # Verify mock called and output structure
        mock_allgather.assert_called_once()
        self.assertEqual(len(out_args), 5)
        torch.testing.assert_close(out_args[0], query)
        torch.testing.assert_close(out_args[1], mock_key)
        torch.testing.assert_close(out_args[2], mock_value)
        self.assertEqual(out_args[3], score_mod)
        self.assertEqual(out_args[4], block_mask)

        # Verify extra kwargs unchanged
        self.assertEqual(out_kwargs["enable_gqa"], False)


class TestContextParallelStyleSDPA(DTensorTestBase):
    """Test suite for _ContextParallel.sdpa_input_fn argument handling"""

    @property
    def world_size(self) -> int:
        return 2

    def _create_test_tensors(self):
        """Helper to create test query, key, value tensors"""
        query = torch.randn(2, 4, 128, 64, device=self.device_type)
        key = torch.randn(2, 4, 128, 64, device=self.device_type)
        value = torch.randn(2, 4, 128, 64, device=self.device_type)
        return query, key, value

    def _setup_context(self):
        """Helper to create CP instance and device mesh"""
        cp_style = _ContextParallel(
            seq_dim=2, attention_type=_ContextParallel.AttentionType.SDPA
        )
        device_mesh = DeviceMesh(self.device_type, torch.arange(0, self.world_size))
        return cp_style, device_mesh

    @with_comms
    def test_sdpa_input_fn_all_positional(self):
        """Test sdpa_input_fn with all positional arguments"""
        query, key, value = self._create_test_tensors()
        cp_style, device_mesh = self._setup_context()

        # Call with all positional args
        args = (query, key, value)
        kwargs = {}
        out_args, out_kwargs = cp_style.sdpa_input_fn(None, args, kwargs, device_mesh)

        # Verify output structure: should be all positional
        self.assertEqual(len(out_args), 3)
        self.assertEqual(len(out_kwargs), 0)

        # Verify all outputs are DTensors
        self.assertIsInstance(out_args[0], DTensor)
        self.assertIsInstance(out_args[1], DTensor)
        self.assertIsInstance(out_args[2], DTensor)

        # Verify DTensors have correct placement (Shard(2) for seq_dim=2)
        from torch.distributed.tensor.placement_types import Shard

        self.assertEqual(out_args[0].placements, [Shard(2)])
        self.assertEqual(out_args[1].placements, [Shard(2)])
        self.assertEqual(out_args[2].placements, [Shard(2)])

    @with_comms
    def test_sdpa_input_fn_all_keyword(self):
        """Test sdpa_input_fn with all keyword arguments"""
        query, key, value = self._create_test_tensors()
        cp_style, device_mesh = self._setup_context()

        # Call with all keyword args
        args = ()
        kwargs = {"query": query, "key": key, "value": value}
        out_args, out_kwargs = cp_style.sdpa_input_fn(None, args, kwargs, device_mesh)

        # Verify output structure: should be all keyword
        self.assertEqual(len(out_args), 0)
        self.assertIn("query", out_kwargs)
        self.assertIn("key", out_kwargs)
        self.assertIn("value", out_kwargs)

        # Verify all outputs are DTensors
        self.assertIsInstance(out_kwargs["query"], DTensor)
        self.assertIsInstance(out_kwargs["key"], DTensor)
        self.assertIsInstance(out_kwargs["value"], DTensor)

        # Verify DTensors have correct placement
        from torch.distributed.tensor.placement_types import Shard

        self.assertEqual(out_kwargs["query"].placements, [Shard(2)])
        self.assertEqual(out_kwargs["key"].placements, [Shard(2)])
        self.assertEqual(out_kwargs["value"].placements, [Shard(2)])

    @with_comms
    def test_sdpa_input_fn_query_positional_kv_keyword(self):
        """Test sdpa_input_fn with query positional, key/value keyword"""
        query, key, value = self._create_test_tensors()
        cp_style, device_mesh = self._setup_context()

        # Query positional, key/value keyword
        args = (query,)
        kwargs = {"key": key, "value": value}
        out_args, out_kwargs = cp_style.sdpa_input_fn(None, args, kwargs, device_mesh)

        # Verify output structure: query should be positional, rest keyword
        self.assertEqual(len(out_args), 1)
        self.assertIn("key", out_kwargs)
        self.assertIn("value", out_kwargs)

        # Verify all outputs are DTensors
        self.assertIsInstance(out_args[0], DTensor)
        self.assertIsInstance(out_kwargs["key"], DTensor)
        self.assertIsInstance(out_kwargs["value"], DTensor)

        # Verify DTensors have correct placement
        from torch.distributed.tensor.placement_types import Shard

        self.assertEqual(out_args[0].placements, [Shard(2)])
        self.assertEqual(out_kwargs["key"].placements, [Shard(2)])
        self.assertEqual(out_kwargs["value"].placements, [Shard(2)])


RingAttentionTestWithLocalTensor = create_local_tensor_test_class(
    RingAttentionTest,
    skipped_tests=[
        # Need to make attention implementation local tensor friendly, e.g.
        # rewrite "rank local" logic
        "test_ring_attention_sdpa",
    ],
)

CPFlexAttentionTestWithLocalTensor = create_local_tensor_test_class(
    CPFlexAttentionTest,
    skipped_tests=[
        # Missing support for batched tensors
        "test_cp_flex_attention_causal_mask",
        "test_cp_flex_attention_document_mask",
    ],
)

TestCPCustomOpsWithLocalTensor = create_local_tensor_test_class(
    TestCPCustomOps,
    skipped_tests=[
        # Missing support for fake tensors
        "test_flex_cp_custom_op",
    ],
)

TestShardingWithLocalTensor = create_local_tensor_test_class(
    TestSharding,
)


class _MockMesh:
    """Minimal DeviceMesh stand-in for non-distributed unit tests."""

    def __init__(self, world_size: int, rank: int, device_type: str = "cpu") -> None:
        self._world_size = world_size
        self._rank = rank
        self.device_type = device_type

    def size(self) -> int:
        return self._world_size

    def get_local_rank(self) -> int:
        return self._rank


def _build_varlen_meta(cu_seq_q: list[int], B: int, seq_len: int) -> VarlenMetadata:
    """Build a global VarlenMetadata from a per-batch cumulative tensor.

    The given ``cu_seq_q`` covers a single batch element ``[0, ..., seq_len]``.
    It is replicated across ``B`` batches with offsets, matching what
    ``create_varlen_metadata_for_document`` produces.
    """
    if cu_seq_q[0] != 0 or cu_seq_q[-1] != seq_len:
        raise ValueError("cu_seq_q must start at 0 and end at seq_len")
    per_batch = torch.tensor(cu_seq_q, dtype=torch.int32)
    # For B batches, concat with offsets and dedupe the boundary.
    parts = [per_batch[:-1] + b * seq_len for b in range(B)]
    parts.append(torch.tensor([B * seq_len], dtype=torch.int32))
    cu = torch.cat(parts)
    seg_lens = torch.diff(cu)
    return VarlenMetadata(
        cu_seq_q=cu,
        cu_seq_k=cu.clone(),
        max_q=int(seg_lens.max().item()),
        max_k=int(seg_lens.max().item()),
    )


class TestCPVarlenMetadata(TestCase):
    """Pure-logic CPU tests for VarlenMetadata._shard_for_cp.

    These do not require a real distributed environment; we mock the
    DeviceMesh with ``_MockMesh`` and iterate over ranks manually.
    """

    def setUp(self) -> None:
        super().setUp()
        # Tests in this class only enable load balancing when they pass an
        # explicit balancer; reset the global flag to make that the default.
        self._saved_enable_lb = _cp_options.enable_load_balance
        _cp_options.enable_load_balance = False

    def tearDown(self) -> None:
        _cp_options.enable_load_balance = self._saved_enable_lb
        super().tearDown()

    def _run(
        self,
        global_meta: VarlenMetadata,
        cp_world_size: int,
        B: int,
        seq_len: int,
        load_balancer_factory=None,
    ) -> list[VarlenMetadata]:
        """Return per-rank metadata for the given global metadata."""
        results = []
        for rank in range(cp_world_size):
            mesh = _MockMesh(cp_world_size, rank)
            lb = (
                load_balancer_factory(seq_len, cp_world_size)
                if load_balancer_factory
                else None
            )
            results.append(
                global_meta._shard_for_cp(
                    mesh,
                    batch_size=B,
                    seq_length=seq_len,
                    load_balancer=lb,
                )
            )
        return results

    @staticmethod
    def _seg_lens(meta: VarlenMetadata) -> tuple[list[int], list[int]]:
        return (
            torch.diff(meta.cu_seq_q).tolist(),
            torch.diff(meta.cu_seq_k).tolist(),
        )

    def test_single_doc_no_loadbalancer(self) -> None:
        # B=1, one doc spanning [0, 32). CP=2 -> rank 0 gets [0, 16),
        # rank 1 gets [16, 32). rank 1 is mid-doc so its seqlen_k = 32.
        meta = _build_varlen_meta([0, 32], B=1, seq_len=32)
        per_rank = self._run(meta, cp_world_size=2, B=1, seq_len=32)
        self.assertEqual(self._seg_lens(per_rank[0]), ([16], [16]))
        self.assertEqual(self._seg_lens(per_rank[1]), ([16], [32]))
        self.assertEqual(per_rank[0].max_q, 16)
        self.assertEqual(per_rank[1].max_k, 32)

    def test_single_doc_headtail(self) -> None:
        # seq_len=32, CP=2, headtail chunks=8 -> rearranged
        # [0..7, 24..31, 8..15, 16..23]. Rank 0 gets [0..7, 24..31] (one
        # contiguous + one tail chunk -> 2 segments). Rank 1 gets [8..23]
        # contiguous -> 1 segment.
        meta = _build_varlen_meta([0, 32], B=1, seq_len=32)
        per_rank = self._run(
            meta,
            cp_world_size=2,
            B=1,
            seq_len=32,
            load_balancer_factory=lambda s, w: _HeadTailLoadBalancer(
                s, w, torch.device("cpu")
            ),
        )
        self.assertEqual(self._seg_lens(per_rank[0]), ([8, 8], [8, 32]))
        self.assertEqual(self._seg_lens(per_rank[1]), ([16], [24]))

    def test_multi_doc_straddling(self) -> None:
        # Worked example: docs [0,10), [10,50), [50,64). CP=2, no LB.
        # rank 0 has [0..32): doc 0 fully + doc 1 prefix [10..32).
        # rank 1 has [32..64): doc 1 mid [32..50) + doc 2 fully [50..64).
        meta = _build_varlen_meta([0, 10, 50, 64], B=1, seq_len=64)
        per_rank = self._run(meta, cp_world_size=2, B=1, seq_len=64)
        self.assertEqual(self._seg_lens(per_rank[0]), ([10, 22], [10, 22]))
        self.assertEqual(self._seg_lens(per_rank[1]), ([18, 14], [40, 14]))

    def test_multi_doc_headtail(self) -> None:
        # Same docs as test_multi_doc_straddling, with headtail balancer
        # (chunk=16). Forward permutation is [0..16, 48..64, 16..48],
        # inverse:
        #   p_orig in [0,16)  -> p_rearr = p_orig
        #   p_orig in [16,48) -> p_rearr = p_orig + 16
        #   p_orig in [48,64) -> p_rearr = p_orig - 32
        # Rank 0 Q = chunks 0+3 = [0..16, 48..64) -> 4 segs:
        #   doc 0 [0..10) (10,10); doc 1 prefix [10..16) (6,6);
        #   doc 1 tail [48..50) (2, 40); doc 2 [50..64) (14,14).
        # Rank 1 Q = chunks 1+2 = [16..48) in doc 1 -> 1 seg (32, 38).
        meta = _build_varlen_meta([0, 10, 50, 64], B=1, seq_len=64)
        per_rank = self._run(
            meta,
            cp_world_size=2,
            B=1,
            seq_len=64,
            load_balancer_factory=lambda s, w: _HeadTailLoadBalancer(
                s, w, torch.device("cpu")
            ),
        )
        self.assertEqual(self._seg_lens(per_rank[0]), ([10, 6, 2, 14], [10, 6, 40, 14]))
        self.assertEqual(self._seg_lens(per_rank[1]), ([32], [38]))

        # k_local_indices compose the K-gather with the inverse permutation.
        # Rank 1 K covers original [10..48) -> [10..16) stays, [16..48) +16.
        expected_rank1 = torch.tensor(
            list(range(10, 16)) + list(range(32, 64)), dtype=torch.long
        )
        self.assertEqual(per_rank[1].k_local_indices, expected_rank1)
        # Rank 0 K-gather covers originals [0..10), [10..16), [10..50), [50..64).
        expected_rank0 = torch.tensor(
            list(range(10))  # seg 0: doc 0
            + list(range(10, 16))  # seg 1: doc 1 prefix
            + list(range(10, 16))  # seg 2 piece [10,16)
            + list(range(32, 64))  # seg 2 piece [16,48) shifted +16
            + [16, 17]  # seg 2 piece [48,50) shifted -32
            + list(range(18, 32)),  # seg 3: doc 2 [50,64) shifted -32
            dtype=torch.long,
        )
        self.assertEqual(per_rank[0].k_local_indices, expected_rank0)

    def test_one_token_segment(self) -> None:
        # 1-token doc + 7-token doc; seq_len=8, CP=2.
        # rank 0 has [0..4): doc 0 [0..1) + doc 1 prefix [1..4).
        # rank 1 has [4..8): doc 1 only, mid-doc.
        meta = _build_varlen_meta([0, 1, 8], B=1, seq_len=8)
        per_rank = self._run(meta, cp_world_size=2, B=1, seq_len=8)
        self.assertEqual(self._seg_lens(per_rank[0]), ([1, 3], [1, 3]))
        self.assertEqual(self._seg_lens(per_rank[1]), ([4], [7]))

    def test_multi_batch_multi_doc_straddling(self) -> None:
        # B=2, per-batch docs [10, 22] -> packed cu_seq_q =
        # [0, 10, 32, 42, 64]. CP=2, shard_len=16.
        # Rank 0 has batch0[0..16) + batch1[0..16):
        #   batch0: doc0 [0..10) + doc1 prefix [10..16) -> (10,10), (6,6)
        #   batch1: doc2 [32..42) + doc3 prefix [42..48) -> (10,10), (6,6)
        # Rank 1 has batch0[16..32) + batch1[16..32):
        #   batch0: doc1 mid [16..32) -> (16, 22) (seqlen_k = 31-10+1)
        #   batch1: doc3 mid [48..64) -> (16, 22) (seqlen_k = 63-42+1)
        # Also asserts k_local_indices for rank 0: must pick out the
        # non-contiguous per-segment K regions ([0..10), [10..16),
        # [32..42), [42..48)) from the packed length-64 K view.
        meta = _build_varlen_meta([0, 10, 32], B=2, seq_len=32)
        per_rank = self._run(meta, cp_world_size=2, B=2, seq_len=32)
        self.assertEqual(self._seg_lens(per_rank[0]), ([10, 6, 10, 6], [10, 6, 10, 6]))
        self.assertEqual(self._seg_lens(per_rank[1]), ([16, 16], [22, 22]))
        expected_rank0_k = torch.tensor(
            list(range(10))
            + list(range(10, 16))
            + list(range(32, 42))
            + list(range(42, 48)),
            dtype=torch.long,
        )
        self.assertEqual(per_rank[0].k_local_indices, expected_rank0_k)

    def test_cp_world_size_4(self) -> None:
        # CP=4 exercises off-by-one in the per-rank boundary math beyond
        # the standard CP=2 cases. seq_len=64, docs [20, 44] -> packed
        # cu_seq_q = [0, 20, 64]. shard_len=16.
        # Rank 0 [0..16): doc 0 prefix -> seqlen_q=16, seqlen_k=16
        # Rank 1 [16..32): doc 0 tail [16..20) + doc 1 prefix [20..32)
        #   -> (4, 20) (seqlen_k = 19-0+1), (12, 12)
        # Rank 2 [32..48): doc 1 mid -> (16, 28) (seqlen_k = 47-20+1)
        # Rank 3 [48..64): doc 1 end -> (16, 44) (seqlen_k = 63-20+1)
        meta = _build_varlen_meta([0, 20, 64], B=1, seq_len=64)
        per_rank = self._run(meta, cp_world_size=4, B=1, seq_len=64)
        self.assertEqual(self._seg_lens(per_rank[0]), ([16], [16]))
        self.assertEqual(self._seg_lens(per_rank[1]), ([4, 12], [20, 12]))
        self.assertEqual(self._seg_lens(per_rank[2]), ([16], [28]))
        self.assertEqual(self._seg_lens(per_rank[3]), ([16], [44]))

    def test_seq_len_divisibility(self) -> None:
        meta = _build_varlen_meta([0, 30], B=1, seq_len=30)
        with self.assertRaisesRegex(ValueError, "divisible"):
            meta._shard_for_cp(_MockMesh(4, 0), batch_size=1, seq_length=30)

    def test_cross_attention_unsupported(self) -> None:
        # cu_seq_q != cu_seq_k must be rejected: only self-attention is
        # supported.
        meta = VarlenMetadata(
            cu_seq_q=torch.tensor([0, 16, 32], dtype=torch.int32),
            cu_seq_k=torch.tensor([0, 20, 40], dtype=torch.int32),
            max_q=16,
            max_k=20,
        )
        with self.assertRaisesRegex(ValueError, "self-attention"):
            meta._shard_for_cp(_MockMesh(2, 0), batch_size=1, seq_length=32)


def _synthetic_doc_lengths(seq_len: int, kind: str, seed: int = 0) -> list[int]:
    """Generate doc lengths summing to seq_len.

    ``kind="mixture"`` (deterministic, hand-checkable):
        ~30 short docs U(16,64), 2 medium docs U(200,320), 1 long doc =
        remainder. Long doc placed mid-sequence (worst for headtail).
    """
    rng = torch.Generator().manual_seed(seed)
    if kind == "mixture":
        short_lens = [
            int(torch.randint(16, 65, (1,), generator=rng).item()) for _ in range(30)
        ]
        medium_lens = [
            int(torch.randint(200, 321, (1,), generator=rng).item()) for _ in range(2)
        ]
        used = sum(short_lens) + sum(medium_lens)
        if used >= seq_len:
            raise ValueError("mixture overflows seq_len; bump seq_len")
        long_len = seq_len - used
        # Place long doc at the midpoint of the short docs.
        mid = len(short_lens) // 2
        return short_lens[:mid] + medium_lens + [long_len] + short_lens[mid:]
    else:
        raise ValueError(f"unknown kind {kind!r}")


def _build_varlen_meta_from_doc_lens(
    doc_lens: list[int], B: int, seq_len: int, device: str = "cpu"
) -> VarlenMetadata:
    """Per-batch doc lens replicated across B with offsets (mirrors
    create_varlen_metadata_for_document)."""
    if sum(doc_lens) != seq_len:
        raise ValueError("doc_lens must sum to seq_len")
    cum = [0]
    for length in doc_lens:
        cum.append(cum[-1] + length)
    parts: list[torch.Tensor] = []
    for b in range(B):
        parts.append(
            torch.tensor(cum[:-1], dtype=torch.int32, device=device) + b * seq_len
        )
    parts.append(torch.tensor([B * seq_len], dtype=torch.int32, device=device))
    cu = torch.cat(parts)
    return VarlenMetadata(
        cu_seq_q=cu,
        cu_seq_k=cu.clone(),
        max_q=int(torch.diff(cu).max().item()),
        max_k=int(torch.diff(cu).max().item()),
    )


def _per_rank_total_work(
    indices: torch.Tensor, cu_seq_q: torch.Tensor, B: int, S: int, W: int
) -> torch.Tensor:
    """Sum of visible-K-token-counts per rank for a given (B, S) permutation.

    Used to compare load-balance quality: a rank's "work" is the sum
    over its Q tokens of (offset_within_doc + 1) under causal masking.
    """
    cu = cu_seq_q.to(torch.long)
    positions = torch.arange(B * S, dtype=torch.long)
    doc_id = torch.searchsorted(cu, positions, right=True) - 1
    work_per_token = positions - cu[doc_id] + 1  # (B*S,)
    work_per_token_2d = work_per_token.view(B, S)
    work_rearr = torch.gather(work_per_token_2d, 1, indices.to(torch.long))
    shard = S // W
    return work_rearr.view(B, W, shard).sum(dim=(0, 2))


class TestVarlenPTRRLoadBalancer(TestCase):
    """Pure-logic CPU tests for _VarlenPTRRLoadBalancer."""

    @staticmethod
    def _check_perm(indices: torch.Tensor) -> None:
        B, S = indices.shape
        for b in range(B):
            torch.testing.assert_close(
                torch.sort(indices[b]).values,
                torch.arange(S, dtype=indices.dtype),
            )

    @staticmethod
    def _check_restore(lb: _VarlenPTRRLoadBalancer) -> None:
        fwd = lb._generate_indices(restore=False).to(torch.long)
        rev = lb._generate_indices(restore=True).to(torch.long)
        for b in range(fwd.shape[0]):
            roundtrip = fwd[b][rev[b]]
            torch.testing.assert_close(
                roundtrip, torch.arange(fwd.shape[1], dtype=torch.long)
            )

    def test_single_doc_hand_balance(self) -> None:
        # B=1, W=2, S=128, BS=32. One full-seq doc.
        # work blocks: sum(1..32)=528, sum(33..64)=1552, sum(65..96)=2576,
        # sum(97..128)=3600. PTRR pairs (heaviest, lightest): rank gets
        # blocks summing to 528+3600 and 1552+2576 -- both 4128.
        S, BS, W = 128, 32, 2
        meta = _build_varlen_meta_from_doc_lens([S], B=1, seq_len=S)
        lb = _VarlenPTRRLoadBalancer(
            meta.cu_seq_q,
            batch_size=1,
            seq_length=S,
            world_size=W,
            block_size=BS,
        )
        idx = lb._generate_indices(restore=False)
        self._check_perm(idx)
        self._check_restore(lb)
        per_rank = _per_rank_total_work(idx, meta.cu_seq_q, B=1, S=S, W=W)
        torch.testing.assert_close(
            per_rank, torch.tensor([4128, 4128], dtype=per_rank.dtype)
        )

    def test_multi_batch_different_doc_structures(self) -> None:
        BS, W = 32, 2
        # batch 0: one doc of 128; batch 1: two docs of 64+64.
        cu = torch.tensor([0, 128, 192, 256], dtype=torch.int32)
        lb = _VarlenPTRRLoadBalancer(
            cu,
            batch_size=2,
            seq_length=128,
            world_size=W,
            block_size=BS,
        )
        idx = lb._generate_indices(restore=False)
        self._check_perm(idx)
        self._check_restore(lb)

    def test_balance_quality_beats_headtail(self) -> None:
        # The point of PTRR. On the synthetic mixture (skewed),
        # PTRR's per-rank-work spread should be < 0.5x headtail's.
        S, BS, W = 4096, 128, 2
        doc_lens = _synthetic_doc_lengths(S, kind="mixture", seed=0)
        meta = _build_varlen_meta_from_doc_lens(doc_lens, B=1, seq_len=S)

        ptrr = _VarlenPTRRLoadBalancer(
            meta.cu_seq_q,
            batch_size=1,
            seq_length=S,
            world_size=W,
            block_size=BS,
        )
        ht = _HeadTailLoadBalancer(S, W, torch.device("cpu"))

        ptrr_work = _per_rank_total_work(
            ptrr._generate_indices(restore=False),
            meta.cu_seq_q,
            B=1,
            S=S,
            W=W,
        )
        ht_work = _per_rank_total_work(
            ht._generate_indices(restore=False),
            meta.cu_seq_q,
            B=1,
            S=S,
            W=W,
        )
        ptrr_spread = (ptrr_work.max() - ptrr_work.min()).item()
        ht_spread = (ht_work.max() - ht_work.min()).item()
        self.assertLess(
            ptrr_spread,
            0.5 * ht_spread,
            msg=(
                f"PTRR spread {ptrr_spread} should be < 0.5 * "
                f"headtail spread {ht_spread}"
            ),
        )

    def test_num_blocks_equals_world_size(self) -> None:
        S, BS, W = 64, 32, 2  # num_blocks = 2 = W
        meta = _build_varlen_meta_from_doc_lens([S], B=1, seq_len=S)
        lb = _VarlenPTRRLoadBalancer(
            meta.cu_seq_q,
            batch_size=1,
            seq_length=S,
            world_size=W,
            block_size=BS,
        )
        idx = lb._generate_indices(restore=False)
        self._check_perm(idx)

    def test_block_size_divisibility_error(self) -> None:
        meta = _build_varlen_meta_from_doc_lens([128], B=1, seq_len=128)
        with self.assertRaisesRegex(ValueError, "divisible by block_size"):
            _VarlenPTRRLoadBalancer(
                meta.cu_seq_q,
                batch_size=1,
                seq_length=128,
                world_size=2,
                block_size=100,
            )

    def test_world_size_divisibility_error(self) -> None:
        # num_blocks = 6, not divisible by W=4.
        meta = _build_varlen_meta_from_doc_lens([192], B=1, seq_len=192)
        with self.assertRaisesRegex(ValueError, "must be divisible by world_size"):
            _VarlenPTRRLoadBalancer(
                meta.cu_seq_q,
                batch_size=1,
                seq_length=192,
                world_size=4,
                block_size=32,
            )

    def test_consistency_error(self) -> None:
        cu_bad = torch.tensor([0, 50], dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "does not match"):
            _VarlenPTRRLoadBalancer(
                cu_bad,
                batch_size=1,
                seq_length=64,
                world_size=2,
                block_size=32,
            )


class CPVarlenAttentionTest(DTensorTestBase):
    """End-to-end correctness for varlen attention under CP.

    Strategy: build identical global Q/K/V and global VarlenMetadata on
    every rank; compute the reference output non-CP. Then shard Q on the
    sequence dim and shard the metadata via ``VarlenMetadata._shard_for_cp``;
    keep K/V replicated (in real training this is what DTensor's
    ``Replicate`` placement on the CP dim achieves). Run varlen_attn on
    each rank and unshard the output to compare to the reference.
    """

    @property
    def world_size(self) -> int:
        return 2

    def _build_global_meta(
        self, B: int, seq_len: int, doc_lens: list[int], device: str
    ) -> VarlenMetadata:
        if sum(doc_lens) != seq_len:
            raise AssertionError("doc_lens must sum to seq_len")
        per_batch = [0]
        running = 0
        for length in doc_lens:
            running += length
            per_batch.append(running)
        # Replicate per-batch boundaries across B with offsets.
        cu = []
        offset = 0
        for _ in range(B):
            cu.extend(p + offset for p in per_batch[:-1])
            offset += seq_len
        cu.append(B * seq_len)
        cu_t = torch.tensor(cu, dtype=torch.int32, device=device)
        seg_lens = torch.diff(cu_t)
        return VarlenMetadata(
            cu_seq_q=cu_t,
            cu_seq_k=cu_t.clone(),
            max_q=int(seg_lens.max().item()),
            max_k=int(seg_lens.max().item()),
        )

    def _run_varlen_cp(
        self,
        *,
        B: int,
        seq_len: int,
        doc_lens: list[int],
        n_heads: int,
        head_dim: int,
        load_balancer_factory,
    ) -> None:
        from torch.nn.attention.varlen import varlen_attn

        torch.manual_seed(1234)
        device = self.device_type

        global_meta = self._build_global_meta(B, seq_len, doc_lens, device)

        # Identical Q/K/V on every rank.
        q_full = torch.randn(
            B,
            seq_len,
            n_heads,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        kv_kwargs = dict(device=device, dtype=torch.bfloat16, requires_grad=True)
        k_full = torch.randn(B, seq_len, n_heads, head_dim, **kv_kwargs)
        v_full = torch.randn(B, seq_len, n_heads, head_dim, **kv_kwargs)

        # Reference: non-CP varlen on packed global tensors.
        ref_q = q_full.reshape(B * seq_len, n_heads, head_dim)
        ref_k = k_full.reshape(B * seq_len, n_heads, head_dim)
        ref_v = v_full.reshape(B * seq_len, n_heads, head_dim)
        ref_out_packed = varlen_attn(
            ref_q,
            ref_k,
            ref_v,
            global_meta.cu_seq_q,
            global_meta.cu_seq_k,
            global_meta.max_q,
            global_meta.max_k,
            window_size=(-1, 0),
        )
        ref_out = ref_out_packed.view(B, seq_len, n_heads, head_dim)
        ref_out.sum().backward()
        ref_q_grad = q_full.grad.detach().clone()
        ref_k_grad = k_full.grad.detach().clone()
        ref_v_grad = v_full.grad.detach().clone()

        # CP setup.
        device_mesh = init_device_mesh(
            device_type=self.device_type,
            mesh_shape=(self.world_size,),
            mesh_dim_names=("cp",),
        )
        load_balancer = (
            load_balancer_factory(
                global_meta=global_meta,
                seq_len=seq_len,
                B=B,
                world_size=self.world_size,
                device=torch.device(device),
            )
            if load_balancer_factory
            else None
        )

        # Shard Q (no requires_grad yet -- set after the to_local() inside
        # _context_parallel_shard so the local tensor is a true leaf).
        q_for_cp = q_full.detach().clone()
        sharded = _context_parallel_shard(
            mesh=device_mesh,
            buffers=[q_for_cp, global_meta],
            seq_dims=[1, 0],  # seq_dim ignored for VarlenMetadata
            load_balancer=load_balancer,
            batch_and_seq=(B, seq_len),
        )
        local_q, local_meta = sharded
        local_q.requires_grad_(True)

        # Simulate production K/V layout under DTensor Replicate-on-CP
        # all-gather: each rank holds the full K/V, but in load-balancer
        # rearranged order (concatenation of rank-local rearranged shards).
        if load_balancer is not None:
            rearrange_full = load_balancer._generate_indices(restore=False)
            if rearrange_full.shape[0] == 1:
                rearrange_full = rearrange_full.expand(B, -1)
            expand_idx = (
                rearrange_full.to(torch.long)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .expand_as(k_full)
            )
            k_local = torch.gather(k_full.detach(), dim=1, index=expand_idx).clone()
            v_local = torch.gather(v_full.detach(), dim=1, index=expand_idx).clone()
        else:
            expand_idx = None
            k_local = k_full.detach().clone()
            v_local = v_full.detach().clone()
        k_local.requires_grad_(True)
        v_local.requires_grad_(True)

        shard_len = seq_len // self.world_size
        local_q_packed = local_q.reshape(B * shard_len, n_heads, head_dim)
        k_packed = k_local.reshape(B * seq_len, n_heads, head_dim)
        v_packed = v_local.reshape(B * seq_len, n_heads, head_dim)
        # Re-pack K/V to the per-segment visible regions; required when
        # the rank's K is the full global K but cu_seq_k describes only
        # the visible portions (which may be non-contiguous for B > 1).
        if local_meta.k_local_indices is not None:
            k_packed = k_packed.index_select(0, local_meta.k_local_indices)
            v_packed = v_packed.index_select(0, local_meta.k_local_indices)
        local_out_packed = varlen_attn(
            local_q_packed,
            k_packed,
            v_packed,
            local_meta.cu_seq_q,
            local_meta.cu_seq_k,
            local_meta.max_q,
            local_meta.max_k,
            window_size=(-1, 0),
        )
        local_out = local_out_packed.view(B, shard_len, n_heads, head_dim)
        local_out.sum().backward()

        # Unshard the local Q grad and the local output to compare.
        seq_dim = 1
        cp_out, cp_q_grad = context_parallel_unshard(
            device_mesh,
            buffers=[local_out, local_q.grad],
            seq_dims=[seq_dim, seq_dim],
            load_balancer=load_balancer,
        )

        # K/V grads live on each rank's (full) K/V copy and cover only the
        # segments this rank's Q attended to. Un-rearrange into original
        # coords (reverse of the forward torch.gather) and sum across CP
        # ranks to reconstruct the full K/V grad.
        if expand_idx is not None:
            cp_k_grad = torch.empty_like(k_local.grad)
            cp_k_grad.scatter_(1, expand_idx, k_local.grad)
            cp_v_grad = torch.empty_like(v_local.grad)
            cp_v_grad.scatter_(1, expand_idx, v_local.grad)
        else:
            cp_k_grad = k_local.grad.detach().clone()
            cp_v_grad = v_local.grad.detach().clone()
        dist.all_reduce(cp_k_grad, group=device_mesh.get_group())
        dist.all_reduce(cp_v_grad, group=device_mesh.get_group())

        atol = 5e-3
        rtol = 5e-3
        torch.testing.assert_close(cp_out, ref_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(cp_q_grad, ref_q_grad, atol=atol, rtol=rtol)
        # K/V grads pick up extra rounding error from the cross-rank sum,
        # so use a bfloat16-appropriate tolerance.
        kv_atol = 5e-2
        kv_rtol = 5e-2
        torch.testing.assert_close(cp_k_grad, ref_k_grad, atol=kv_atol, rtol=kv_rtol)
        torch.testing.assert_close(cp_v_grad, ref_v_grad, atol=kv_atol, rtol=kv_rtol)

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention"
    )
    @with_comms
    def test_cp_varlen_no_loadbalancer(self) -> None:
        # Single doc, no load balancer, B=1 (the simplest path).
        self._run_varlen_cp(
            B=1,
            seq_len=128,
            doc_lens=[128],
            n_heads=4,
            head_dim=32,
            load_balancer_factory=None,
        )

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention"
    )
    @with_comms
    def test_cp_varlen_multi_batch_no_lb(self) -> None:
        # Multi-batch exercises the K re-packing via k_local_indices.
        self._run_varlen_cp(
            B=2,
            seq_len=128,
            doc_lens=[128],
            n_heads=4,
            head_dim=32,
            load_balancer_factory=None,
        )

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention"
    )
    @with_comms
    def test_cp_varlen_multi_doc_no_lb(self) -> None:
        # Multi-doc with straddling boundaries.
        self._run_varlen_cp(
            B=2,
            seq_len=128,
            doc_lens=[40, 88],
            n_heads=4,
            head_dim=32,
            load_balancer_factory=None,
        )

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention"
    )
    @with_comms
    def test_cp_varlen_headtail(self) -> None:
        self._run_varlen_cp(
            B=2,
            seq_len=128,
            doc_lens=[40, 88],
            n_heads=4,
            head_dim=32,
            load_balancer_factory=lambda *,
            global_meta,
            seq_len,
            B,
            world_size,
            device: _HeadTailLoadBalancer(seq_len, world_size, device),
        )

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention"
    )
    @with_comms
    def test_cp_varlen_ptrr(self) -> None:
        # B=1, multi-doc straddling.  Default block_size=32 (small enough
        # for the 128-token test seq).
        self._run_varlen_cp(
            B=1,
            seq_len=128,
            doc_lens=[40, 88],
            n_heads=4,
            head_dim=32,
            load_balancer_factory=lambda *,
            global_meta,
            seq_len,
            B,
            world_size,
            device: _VarlenPTRRLoadBalancer(
                global_meta.cu_seq_q,
                batch_size=B,
                seq_length=seq_len,
                world_size=world_size,
                block_size=32,
            ),
        )

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention"
    )
    @with_comms
    def test_cp_varlen_ptrr_multi_batch(self) -> None:
        # B=2, exercises both per-batch PTRR rearrangement and
        # k_local_indices re-packing.
        self._run_varlen_cp(
            B=2,
            seq_len=128,
            doc_lens=[40, 88],
            n_heads=4,
            head_dim=32,
            load_balancer_factory=lambda *,
            global_meta,
            seq_len,
            B,
            world_size,
            device: _VarlenPTRRLoadBalancer(
                global_meta.cu_seq_q,
                batch_size=B,
                seq_length=seq_len,
                world_size=world_size,
                block_size=32,
            ),
        )


if __name__ == "__main__":
    run_tests()
