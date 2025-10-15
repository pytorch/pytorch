# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import functools
import itertools
import random
import unittest
from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.experimental._attention import (
    _CausalBehavior,
    _context_parallel_buffers,
    _cp_options,
    _DispatchMode,
    _is_causal_behavior,
    _RotateMethod,
    context_parallel,
    context_parallel_unshard,
    set_rotate_method,
)
from torch.distributed.tensor.experimental._load_balancer import (
    _HeadTailLoadBalancer,
    _LoadBalancer,
    _PerDocumentHeadTailLoadBalancer,
)
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    AuxRequest,
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
    DTensorTestBase,
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
                "dispatch_mode": [
                    _DispatchMode.MONKEY_PATCH,
                    _DispatchMode.TORCH_FUNCTION,
                ],
            },
            self._test_ring_attention_sdpa,
        )

    def _test_ring_attention_sdpa(
        self,
        is_causal: bool,
        compiled: bool,
        backend: SDPBackend,
        load_balance: bool,
        rotater: _RotateMethod,
        test_forward_only: bool,
        dispatch_mode: _DispatchMode,
    ) -> None:
        torch.distributed.tensor.experimental._attention._dispatch_mode = dispatch_mode

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
        query_tokens = 64
        context_tokens = 64
        dim = 32
        nheads = 8
        torch.manual_seed(10)
        dtype = (
            torch.bfloat16
            if backend == SDPBackend.FLASH_ATTENTION
            or backend == SDPBackend.CUDNN_ATTENTION
            else torch.float32
        )

        _cp_options.enable_load_balance = load_balance

        q = torch.rand(
            (bs, nheads, self.world_size * query_tokens, dim),
            device=self.device_type,
            dtype=dtype,
            requires_grad=True,
        )
        k = torch.rand(
            (bs, nheads, self.world_size * context_tokens, dim),
            device=self.device_type,
            dtype=dtype,
            requires_grad=True,
        )
        v = torch.rand(
            (bs, nheads, self.world_size * context_tokens, dim),
            device=self.device_type,
            dtype=dtype,
            requires_grad=True,
        )

        # Ensure all ranks have the same initialization data.
        with torch.no_grad():
            dist.broadcast(q, src=0)
            dist.broadcast(k, src=0)
            dist.broadcast(v, src=0)

        with sdpa_kernel(backend):
            out = fn_eval(F.scaled_dot_product_attention, q, k, v, is_causal=is_causal)

        cp_q = q.detach().clone()
        cp_k = k.detach().clone()
        cp_v = v.detach().clone()
        # Theoretically, context_parallel() should not be used to shard
        # parameters because when require_grad is True, resize_ is not
        # allowed. But requires_grad of cp_q, cp_k, and cp_v are False
        # now. So we can just use context_parallel() to shard q, k, v.
        # In reality, context_paralle() should be used to shard the input.
        with context_parallel(
            device_mesh, buffers=(cp_q, cp_k, cp_v), buffer_seq_dims=(2, 2, 2)
        ):
            cp_q.requires_grad = True
            cp_k.requires_grad = True
            cp_v.requires_grad = True
            with CommDebugMode() as comm_mode:
                with sdpa_kernel(backend):
                    if compiled:
                        fn = torch.compile(
                            F.scaled_dot_product_attention,
                            fullgraph=True,
                            backend="aot_eager",
                        )
                    else:
                        fn = F.scaled_dot_product_attention

                    cp_out = fn_eval(fn, cp_q, cp_k, cp_v, is_causal=is_causal)

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

            # Due to numerical error, we need to choose different atol for different
            # attention kernels
            (cp_out,) = context_parallel_unshard(device_mesh, [cp_out], [2])
            atol = (
                1e-08
                if backend == SDPBackend.EFFICIENT_ATTENTION
                else 1e-3 * self.world_size
            )
            self.assertTrue(torch.allclose(out, cp_out, atol=atol))

            if not test_forward_only:
                cp_dq, cp_dk, cp_dv = context_parallel_unshard(
                    device_mesh,
                    [cp_q.grad, cp_k.grad, cp_v.grad],
                    [2, 2, 2],
                )
                atol = (
                    2e-06
                    if backend == SDPBackend.EFFICIENT_ATTENTION
                    else 8e-3 * self.world_size
                )
                self.assertTrue(torch.allclose(q.grad, cp_dq, atol=atol))
                self.assertTrue(torch.allclose(k.grad, cp_dk, atol=atol))
                self.assertTrue(torch.allclose(v.grad, cp_dv, atol=atol))

                cp_q.grad = None
                cp_k.grad = None
                cp_v.grad = None

            cp_q.requires_grad = False
            cp_k.requires_grad = False
            cp_v.requires_grad = False

        torch.distributed.tensor.experimental._attention._dispatch_mode = (
            _DispatchMode.MONKEY_PATCH
        )

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


def length_to_offsets(
    lengths: list[list[int]], device: Union[str, torch.device]
) -> Tensor:
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


class CPFlexAttentionTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _test_cp_flex_attention(
        self,
        *,
        qkv_size: int,
        B: int = 1,
        mask_func: _mask_mod_signature = causal_mask,
        lb: Optional[_LoadBalancer] = None,
        atol: float = 1e-4,
        rtol: float = 1,
    ) -> None:
        torch.cuda.manual_seed(1234)

        dtype = torch.float32
        bs = B if B > 1 else 8
        dim = 32
        nheads = 8

        qkv = [
            torch.rand(
                (bs, nheads, qkv_size, dim),
                device=self.device_type,
                dtype=dtype,
                requires_grad=True,
            )
            for _ in range(3)
        ]

        block_mask = compiled_create_block_mask(
            mask_func,
            B=B,
            H=1,
            Q_LEN=qkv_size,
            KV_LEN=qkv_size,
            device=self.device_type,
        )

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

        # create block_mask for CP
        from torch.distributed.tensor.experimental._attention import (
            create_cp_block_mask,
        )

        if not lb and _cp_options.enable_load_balance:
            # NOTE: when parallelizing `flex_attention`, we require not-None
            # `load_balancer` object be explicitly passed APIs `_context_parallel_shard`
            # and `context_parallel_unshard` if load-balancing is needed.
            lb = _HeadTailLoadBalancer(qkv_size, self.world_size, self.device_type)

        # if load-balance is enabled, reorder input tensor and produce the index tensor
        # NOTE: call create_block_mask() within TorchFunctionMode would cause error in create_fw_bw_graph
        cp_block_mask = create_cp_block_mask(
            mask_func,
            B=B,
            H=1,
            Q_LEN=qkv_size,
            KV_LEN=qkv_size,
            device_mesh=device_mesh,
            load_balancer=lb,
        )

        # shard qkv on seq_dim
        shard_dim = 2

        cp_qkv = _context_parallel_buffers(
            device_mesh,
            buffers=[t.detach().clone() for t in qkv],
            buffer_seq_dims=[shard_dim] * 3,
            load_balancer=lb,
        )
        for t in cp_qkv:
            t.requires_grad = True

        # TODO: remove this once https://github.com/pytorch/pytorch/pull/164500 is merged
        torch.distributed.tensor.experimental._attention._dispatch_mode = (
            _DispatchMode.TORCH_FUNCTION
        )
        with context_parallel(
            device_mesh, buffers=[torch.empty(self.world_size * 2)], buffer_seq_dims=[0]
        ):
            cp_out, cp_aux = compiled_flex_attention(
                *cp_qkv,
                block_mask=cp_block_mask,
                return_aux=AuxRequest(lse=True),
            )

            # backward run
            cp_out.sum().backward()

        # unshard the output
        cp_out, cp_lse = context_parallel_unshard(
            device_mesh,
            buffers=[cp_out, cp_aux.lse],
            seq_dims=[2, 2],
            load_balancer=lb,
        )
        torch.testing.assert_close(cp_out, expect_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(cp_lse, expect_aux.lse, atol=atol, rtol=rtol)

        # unshard the gradient
        cp_qkv_grad = context_parallel_unshard(
            device_mesh,
            buffers=[t.grad for t in cp_qkv],
            seq_dims=[2, 2, 2],
            load_balancer=lb,
        )

        qkv_grad = [t.grad for t in qkv]
        for grad, cp_grad in zip(qkv_grad, cp_qkv_grad):
            torch.testing.assert_close(grad, cp_grad, atol=atol, rtol=rtol)

        # reset CP context dispatch mode to default
        torch.distributed.tensor.experimental._attention._dispatch_mode = (
            _DispatchMode.MONKEY_PATCH
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention"
    )
    def test_cp_flex_attention_causal_mask(self) -> None:
        restore_enable_load_balance = _cp_options.enable_load_balance

        for enable_load_balance in [
            False,  # test w/o load-balancing
            True,  # test w/ the default load-balancing
        ]:
            _cp_options.enable_load_balance = enable_load_balance

            self.run_subtests(
                {
                    "qkv_size": [
                        (256 if enable_load_balance else 128) * self.world_size,
                        2048,
                    ],
                },
                self._test_cp_flex_attention,
            )

            # NOTE: Context Parallel should not be used for small attentions (block_size < 128)
            with self.assertRaisesRegex(AssertionError, "Tensor-likes are not close"):
                self.run_subtests(
                    {"qkv_size": [64 * self.world_size]},
                    self._test_cp_flex_attention,
                )

        _cp_options.enable_load_balance = restore_enable_load_balance

    # TODO: merge with the above test
    @skip_if_lt_x_gpu(2)
    @with_comms
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention"
    )
    def test_cp_flex_attention_document_mask(self) -> None:
        restore_enable_load_balance = _cp_options.enable_load_balance

        random.seed(10)

        # parameters for testing
        doc_count = 28
        enable_load_balance_list = [True, False]
        batch_size_list = [2, 4, 8]
        max_seq_len_list = [
            256 * self.world_size,
            2048,
            # 128 * self.world_size  # NOTE: Mismatched elements: 8 / 131072 (0.0%),
        ]

        # NOTE: Each (enable_load_balance, batch_size, seq_len) tuple introduces 2
        # create_block_mask compilations: 1 for single-rank flex_attention and 1 for
        # CP flex_attention. In order to avoid the "exceeds_recompile_limit" error,
        # we need to increase the cache_size_limit to 12 which is the total number
        # of compilations in our test case.
        torch._dynamo.config.cache_size_limit = (
            2
            * len(enable_load_balance_list)
            * len(batch_size_list)
            * len(max_seq_len_list)
        )

        # TODO: change this for-loop to run_subtests
        # Use a for-loop instead of run_subtests because we need to intialize the mask
        # for each subtest. This can be baked into self._test_cp_flex_attention as
        # a str argument denoting mask type.
        for enable_load_balance, batch_size, max_seq_len in itertools.product(
            enable_load_balance_list, batch_size_list, max_seq_len_list
        ):
            _cp_options.enable_load_balance = enable_load_balance

            # initialize document mask
            lengths = [
                (
                    generate_random_lengths_in_chunks(
                        max_seq_len, doc_count, chunk_size=2 * self.world_size
                    )
                    if enable_load_balance
                    else generate_random_lengths(max_seq_len, doc_count)
                )
                for _ in range(batch_size)
            ]
            offsets = length_to_offsets(lengths, self.device_type)
            document_causal_mask = generate_doc_mask_mod(causal_mask, offsets)

            # generate load balancer
            load_balancer = (
                _PerDocumentHeadTailLoadBalancer(
                    lengths, self.world_size, torch.device(self.device_type)
                )
                if enable_load_balance
                else None
            )

            # construct testing function
            test_func = functools.partial(
                self._test_cp_flex_attention,
                qkv_size=max_seq_len,
                B=batch_size,
                lb=load_balancer,
                mask_func=document_causal_mask,
                atol=1e-6,
            )

            test_func()

        _cp_options.enable_load_balance = restore_enable_load_balance


if __name__ == "__main__":
    run_tests()
