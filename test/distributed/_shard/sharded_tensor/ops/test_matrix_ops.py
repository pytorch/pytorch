# Owner(s): ["oncall: distributed"]

import copy
import itertools
import sys

import torch
from torch.distributed._shard import sharded_tensor, _shard_tensor
from torch.testing._internal.common_distributed import (
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    TEST_WITH_DEV_DBG_ASAN,
    run_tests,
)
from torch.testing._internal.distributed._shard.sharded_tensor import (
    TEST_GPU_NUM,
    ShardedTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed._shard.sharded_tensor._test_ops_common import (
    generate_enumerable_sharding_specs_for_test,
)
from torch.testing._internal.distributed._shard.sharded_tensor._test_st_common import (
    _chunk_sharding_specs_list_for_test,
)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestShardedTensorMatrixOps(ShardedTensorTestBase):
    @with_comms(init_rpc=True)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharded_tensor_contiguous(self):
        specs = _chunk_sharding_specs_list_for_test([0], seed=7)
        for spec in specs:
            st = sharded_tensor.rand(spec, 10, 22, 5, init_rrefs=True)
            st = st.transpose(1, 0)
            st = st.contiguous()
            self.assertTrue(st.is_contiguous())
            self.assertTrue(st.local_tensor().is_contiguous())

    @with_comms(init_rpc=True)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharded_tensor_type_as(self):
        specs = _chunk_sharding_specs_list_for_test([0], seed=7)
        for spec in specs:
            st = sharded_tensor.rand(
                spec, 16, 30, 5, init_rrefs=True, dtype=torch.double
            )
            st_2 = sharded_tensor.rand(
                spec, 16, 30, 5, init_rrefs=True, dtype=torch.float
            )
            st_3 = st.type_as(st_2)
            self.assertEqual(torch.float, st_3.dtype)
            self.assertEqual(torch.float, st_3.local_tensor().dtype)
            st_3 = st.type_as(torch.zeros(10).type(torch.BoolTensor).cuda())
            self.assertEqual(torch.bool, st_3.dtype)
            self.assertEqual(torch.bool, st_3.local_tensor().dtype)

    @with_comms(init_rpc=True)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharded_tensor_transpose(self):
        specs = _chunk_sharding_specs_list_for_test([0, 1, 2], seed=7)
        for spec in specs:
            tensor = torch.rand(15, 27, 16).cuda(self.rank)
            tensor_t = tensor.transpose(0, 1).contiguous()
            spec_n = copy.deepcopy(spec)
            if spec_n.dim in (0, 1):
                spec_n.dim = 1 - spec_n.dim
            st_expected = _shard_tensor(tensor_t, spec_n)
            self.assertTrue(
                torch.allclose(
                    torch.transpose(_shard_tensor(tensor, spec), 0, 1), st_expected
                )
            )
            tensor_t = torch.transpose(tensor, 1, 2).contiguous()
            spec_n = copy.deepcopy(spec)
            if spec_n.dim in (1, 2):
                spec_n.dim = 3 - spec_n.dim
            st_expected = _shard_tensor(tensor_t, spec_n)
            self.assertTrue(
                torch.allclose(_shard_tensor(tensor, spec).transpose(1, 2), st_expected)
            )

    @with_comms(init_rpc=True)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharded_tensor_transpose_error(self):
        enumerable_spec = generate_enumerable_sharding_specs_for_test()[0]
        st = sharded_tensor.rand(
            enumerable_spec, 10, 10, init_rrefs=True, dtype=torch.double
        )
        with self.assertRaisesRegex(
            NotImplementedError,
            "Only ChunkShardingSpec supported for 'transpose'",
        ):
            st.transpose(1, 0)

    @with_comms(init_rpc=True)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharded_tensor_softmax(self):
        specs = _chunk_sharding_specs_list_for_test([0, 2], seed=17)
        for spec in specs:
            tensor = torch.rand(15, 27, 16).cuda(self.rank)
            tensor_n = torch.nn.functional.softmax(tensor, dim=1, dtype=torch.float32)
            st_expected = _shard_tensor(tensor_n, spec)
            self.assertTrue(
                torch.allclose(
                    torch.nn.functional.softmax(
                        _shard_tensor(tensor, spec), dim=1, dtype=torch.float32
                    ),
                    st_expected,
                )
            )

    def _test_masked_fill_with_sizes(self, mask_size, broadcast_style=False):
        specs = _chunk_sharding_specs_list_for_test([0, 1, 2], seed=7)
        for spec in specs:
            tensor = torch.rand(35, 17, 26).cuda(self.rank)
            mask = torch.randint(0, 2, mask_size).type(torch.BoolTensor).cuda(self.rank)
            if broadcast_style:
                mask = mask.unsqueeze(1)
            tensor_m = tensor.masked_fill(mask, 25.0)
            st_expected = _shard_tensor(tensor_m, spec)
            self.assertTrue(
                torch.allclose(
                    _shard_tensor(tensor, spec).masked_fill(mask, 25.0),
                    st_expected,
                )
            )

    @with_comms(init_rpc=True)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharded_tensor_masked_fill(self):
        self._test_masked_fill_with_sizes((35, 17, 26))
        self._test_masked_fill_with_sizes((17, 26))
        self._test_masked_fill_with_sizes((35, 26), broadcast_style=True)
        self._test_masked_fill_with_sizes((26,))

    @with_comms(init_rpc=True)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharded_tensor_masked_fill_error(self):
        specs = _chunk_sharding_specs_list_for_test([1, 2], seed=7)
        for spec in specs:
            st = sharded_tensor.rand(
                spec, 35, 17, 26, init_rrefs=True, dtype=torch.double
            )
            mask = (
                torch.randint(0, 2, (2, 35, 17, 26))
                .type(torch.BoolTensor)
                .cuda(self.rank)
            )
            with self.assertRaisesRegex(
                ValueError,
                "mask dim must not greater than the dim of the sharded tensor.",
            ):
                st.masked_fill(mask, 25.0)
            mask = torch.randint(0, 2, (16, 26)).type(torch.BoolTensor).cuda(self.rank)
            with self.assertRaisesRegex(
                ValueError,
                "The size of mask 0 must match the size of sharded tensor 1 "
                "at non-singleton dimension 0",
            ):
                st.masked_fill(mask, 25.0)

    @with_comms(init_rpc=True)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharded_tensor_view(self):
        specs = _chunk_sharding_specs_list_for_test([0, 0], seed=10)
        for spec in specs:
            tensor = torch.rand(16, 35, 26).cuda(self.rank)
            tensor_v = tensor.view(16, 35, 26).view(4, 4, 35, 26)
            st_expected = _shard_tensor(tensor_v, spec)
            self.assertTrue(
                torch.allclose(
                    _shard_tensor(tensor, spec).view(4, 4, 35, 26),
                    st_expected,
                )
            )
            st_expected = _shard_tensor(tensor, spec)
            self.assertTrue(
                torch.allclose(
                    _shard_tensor(tensor_v, spec).view(16, 35, 26),
                    st_expected,
                )
            )

    @with_comms(init_rpc=True)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharded_tensor_view_error(self):
        for spec in _chunk_sharding_specs_list_for_test([2], seed=7):
            st = sharded_tensor.rand(
                spec, 35, 17, 26, init_rrefs=True, dtype=torch.double
            )
            with self.assertRaisesRegex(
                NotImplementedError,
                "Shape having dim 2 is not supported "
                "for sharded tensor sharded on dim 2.",
            ):
                st.view(35 * 17, 26)
            with self.assertRaisesRegex(
                ValueError,
                r"Shape '\[5, 7, 35, 17, 26\]' is invalid for sharded tensor size 15470.",
            ):
                st.view(5, 7, 35, 17, 26)
            with self.assertRaisesRegex(
                ValueError,
                "Only one dimension can be inferred for sharded view op.",
            ):
                st.view(5, 7, -1, -1)

    @with_comms(init_rpc=True)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharded_tensor_layer_norm(self):
        specs = _chunk_sharding_specs_list_for_test([1, 2], seed=10)
        flags = [True, False]
        for spec, flag in itertools.product(specs, flags):
            tensor = torch.rand(16, 35, 26).cuda(self.rank)
            layer_norm = torch.nn.LayerNorm((35, 26), elementwise_affine=flag).cuda(
                self.rank
            )
            st = layer_norm(_shard_tensor(tensor, spec))
            with torch.no_grad():
                tensor_normed = layer_norm(tensor)
            st_expected = _shard_tensor(tensor_normed, spec)
            self.assertEqual(
                st.local_tensor(),
                st_expected.local_tensor(),
            )
            self.assertTrue(
                torch.allclose(
                    st,
                    st_expected,
                    atol=1e-6,
                )
            )
            st_expected = torch.nn.functional.layer_norm(
                _shard_tensor(tensor, spec),
                (35, 26),
                weight=layer_norm.weight,
                bias=layer_norm.bias,
            )
            self.assertTrue(
                torch.allclose(
                    st,
                    st_expected,
                    atol=1e-6,
                )
            )

    @with_comms(init_rpc=True)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharded_tensor_layer_norm_error(self):
        specs = _chunk_sharding_specs_list_for_test([2], seed=10)
        for spec in specs:
            tensor = torch.rand(16, 35, 26).cuda(self.rank)
            with self.assertRaisesRegex(
                ValueError,
                "normalized_shape dim must not be greater "
                "than the dim of the sharded tensor.",
            ):
                layer_norm = torch.nn.LayerNorm((14, 55, 35, 26)).cuda(self.rank)
                layer_norm(_shard_tensor(tensor, spec))
            with self.assertRaisesRegex(
                ValueError,
                r"Given normalized_shape=\[35\], expected input with shape "
                r"\[\*, 35\], but got input of size \[16, 35, 26\].",
            ):
                layer_norm = torch.nn.LayerNorm((35)).cuda(self.rank)
                layer_norm(_shard_tensor(tensor, spec))


if __name__ == "__main__":
    run_tests()
