# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import itertools
import random
import unittest
from collections.abc import Callable
from typing import Any, ClassVar, Optional

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DeviceMesh
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
    context_parallel,
    context_parallel_unshard,
    set_rotate_method,
)
from torch.distributed.tensor.experimental._context_parallel._cp_custom_ops import (
    flex_cp_allgather,
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
from torch.testing._internal.common_utils import run_tests, skipIfRocm
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

        with CommDebugMode() as comm_mode:
            with sdpa_kernel(backend):
                cp_out = fn_eval(
                    attention,
                    cp_q,
                    cp_k,
                    cp_v,
                    is_causal=is_causal,
                )

            if not compiled and rotater == _RotateMethod.ALL_TO_ALL:
                # Compiler and CommDebugMode do not work well together.
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
    assert total_length % chunk_size == 0 and num_chunks >= num_documents

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
        document_lengths: Optional[list[list[int]]] = None,
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
    ) -> Optional[_LoadBalancer]:
        seq_length = kwargs["seq_length"]
        document_lengths = kwargs["document_lengths"]
        block_mask = kwargs["block_mask"]

        # generate load balancer
        if lb_type == "None":
            load_balancer = None  # no load-balance
        elif lb_type == "_HeadTailLoadBalancer":
            assert isinstance(seq_length, int)
            load_balancer = _HeadTailLoadBalancer(
                seq_length, self.world_size, torch.device(self.device_type)
            )
        elif lb_type == "_PerDocumentHeadTailLoadBalancer":
            assert isinstance(document_lengths, list)
            load_balancer = _PerDocumentHeadTailLoadBalancer(
                document_lengths, self.world_size, torch.device(self.device_type)
            )
        elif lb_type == "_PTRRLoadBalancer":
            assert isinstance(block_mask, BlockMask)
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

        # Create q, k, v tensors with shape (B, nheads, seq_len, dim)
        q = torch.randn(
            B, nheads, seq_len, dim, device=self.device_type, dtype=torch.bfloat16
        )
        k = torch.randn(
            B, nheads, seq_len, dim, device=self.device_type, dtype=torch.bfloat16
        )
        v = torch.randn(
            B, nheads, seq_len, dim, device=self.device_type, dtype=torch.bfloat16
        )
        q_dt = distribute_tensor(q, device_mesh, [Shard(2)])
        k_dt = distribute_tensor(k, device_mesh, [Shard(2)])
        v_dt = distribute_tensor(v, device_mesh, [Shard(2)])

        # Run SDPA with sequence-sharded tensors WITHOUT enabling CP
        # Without CP enabled, DTensor should select a different strategy
        # (not sequence-sharded) because Shard(2) strategy is only available with CP
        out = F.scaled_dot_product_attention(q_dt, k_dt, v_dt)

        # Verify the output is NOT sharded on sequence dimension (dim 2)
        # This proves that CP sharding rules were not used
        self.assertNotEqual(out.placements[0], Shard(2))
        # The output should be replicated or sharded on batch head dimensions.
        self.assertIn(out.placements[0], [Replicate(), Shard(0), Shard(1)])


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


if __name__ == "__main__":
    run_tests()
