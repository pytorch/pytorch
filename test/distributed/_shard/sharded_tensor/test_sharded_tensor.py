# Owner(s): ["oncall: distributed"]

import copy
import io
import itertools
import math
import pickle
import sys

import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d, rpc
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.api import (
    _collect_local_shard,
    _reshard_output,
    _shard_tensor,
    load_with_process_group,
    shard_parameter,
)
from torch.distributed._shard.sharded_tensor import (
    custom_sharded_op_impl,
    pre_load_state_dict_hook,
    Shard,
    ShardedTensor,
    ShardedTensorBase,
    ShardedTensorMetadata,
    state_dict_hook,
)
from torch.distributed._shard.sharded_tensor.api import (
    _create_tensor_from_params,
    TensorProperties,
)
from torch.distributed._shard.sharded_tensor.utils import (
    _parse_and_validate_remote_device,
)
from torch.distributed._shard.sharding_spec import (
    ChunkShardingSpec,
    EnumerableShardingSpec,
    ShardMetadata,
)
from torch.distributed.remote_device import _remote_device
from torch.testing._internal.common_distributed import (
    requires_nccl,
    skip_if_lt_x_gpu,
    spawn_threads_and_init_comms,
    tp_transports,
)
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle_if,
    skipIfRocm,
    TEST_CUDA,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed._shard.sharded_tensor._test_st_common import (
    _chunk_sharding_specs_list_for_test,
    MyShardedModel1,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestShardedTensorMetadata(TestCase):
    def test_serialize_and_deserialize(self):
        shard_metadatas = [
            ShardMetadata(
                shard_offsets=[0, 0],
                shard_sizes=[5, 5],
                placement="rank:0/cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[0, 5],
                shard_sizes=[5, 5],
                placement="rank:1/cuda:1",
            ),
            ShardMetadata(
                shard_offsets=[5, 0],
                shard_sizes=[5, 5],
                placement="rank:2/cuda:2",
            ),
            ShardMetadata(
                shard_offsets=[5, 5],
                shard_sizes=[5, 5],
                placement="rank:3/cuda:3",
            ),
        ]

        dtypes = [
            torch.float,
            torch.double,
            torch.cfloat,
            torch.cdouble,
            torch.half,
            torch.bfloat16,
            torch.uint8,
            torch.int8,
            torch.short,
            torch.int,
            torch.long,
            torch.bool,
        ]

        layouts = [torch.strided, torch.sparse_coo]
        requires_grads = [True, False]
        memory_formats = [
            torch.contiguous_format,
            torch.channels_last,
            torch.preserve_format,
        ]
        pin_memories = [True, False]

        for tensor_properties_input in itertools.product(
            dtypes, layouts, requires_grads, memory_formats, pin_memories
        ):
            (
                dtype,
                layout,
                requires_grad,
                memory_format,
                pin_memory,
            ) = tensor_properties_input

            expected_st_metadata = sharded_tensor.ShardedTensorMetadata(
                shard_metadatas,
                (10, 10),
                TensorProperties(
                    dtype, layout, requires_grad, memory_format, pin_memory
                ),
            )

            pickled_obj = pickle.dumps(expected_st_metadata)
            st_metadata = pickle.loads(pickled_obj)
            self.assertEqual(expected_st_metadata, st_metadata)


class TestCreateTensorFromParams(TestCase):
    @skip_but_pass_in_sandcastle_if(not TEST_CUDA, "CUDA GPU is needed")
    def test_empty(self):
        expected_dtype = torch.double
        tensor_properties = TensorProperties(
            dtype=expected_dtype,
            layout=torch.strided,
            requires_grad=False,
            pin_memory=False,
            memory_format=torch.contiguous_format,
        )
        local_device = torch.device("cuda:0")
        local_tensor = _create_tensor_from_params(
            5, 10, local_device=local_device, tensor_properties=tensor_properties
        )
        self.assertEqual(local_device, local_tensor.device)
        self.assertEqual(expected_dtype, local_tensor.dtype)
        self.assertEqual(torch.strided, local_tensor.layout)
        self.assertEqual(False, local_tensor.requires_grad)


class TestShardParameter(ShardedTensorTestBase):
    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_shard_parameter(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        fc = torch.nn.Linear(12, 12).cuda(self.rank)
        weight_og = fc.weight.clone()
        shard_parameter(fc, "weight", spec)

        # Verify.
        self.assertTrue(isinstance(fc.weight, ShardedTensor))
        local_shards = fc.weight.local_shards()
        self.assertEqual(1, len(local_shards))
        self.assertEqual(torch.Size([3, 12]), local_shards[0].tensor.size())
        self.assertEqual(3, local_shards[0].tensor.size(0))
        self.assertEqual(12, local_shards[0].tensor.size(1))
        self.assertEqual(
            torch.narrow(weight_og, 0, 3 * self.rank, 3), local_shards[0].tensor
        )

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_shard_parameter_errors(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        fc = torch.nn.Linear(12, 12).cuda(self.rank)
        with self.assertRaisesRegex(ValueError, "does not match with src_rank"):
            shard_parameter(fc, "weight", spec, src_rank=self.rank)

        with self.assertRaisesRegex(AttributeError, "has no attribute"):
            shard_parameter(fc, "foo", spec)

        with self.assertRaisesRegex(
            ValueError, "Expected Linear.bias to be a Tensor, but found str"
        ):
            del fc.bias
            fc.bias = "foo"
            shard_parameter(fc, "bias", spec)

        with self.assertRaisesRegex(ValueError, "not a contiguous Tensor"):
            fc.bias = torch.rand(10, 10).cuda(self.rank).t()
            shard_parameter(fc, "bias", spec)

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:{self.rank}/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        with self.assertRaisesRegex(ValueError, "does not match with sharding_spec"):
            shard_parameter(fc, "weight", spec)

        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="rank:0/cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="rank:1/cuda:1",
                ),
            ]
        )
        with self.assertRaisesRegex(NotImplementedError, "not implemented yet!"):
            shard_parameter(fc, "weight", spec)


class TestShardTensor(ShardedTensorTestBase):
    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_shard_tensor(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        tensor = torch.rand(12, 12).cuda(self.rank)
        st = _shard_tensor(tensor, spec)

        # Verify.
        self.assertTrue(isinstance(st, sharded_tensor.ShardedTensor))
        local_shard = st.local_tensor()
        self.assertEqual(1, len(st.local_shards()))
        self.assertEqual(torch.Size([3, 12]), local_shard.size())
        self.assertEqual(torch.narrow(tensor, 0, 3 * self.rank, 3), local_shard)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_shard_tensor_with_empty_shard(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        tensor = torch.rand(9, 12).cuda(self.rank)
        st = _shard_tensor(tensor, spec)

        # Verify.
        self.assertTrue(isinstance(st, sharded_tensor.ShardedTensor))
        sms = st.metadata().shards_metadata
        self.assertEqual(len(sms), 4)
        for sm in sms:
            self.assertTrue(sm.shard_offsets[0] + sm.shard_sizes[0] <= tensor.size(0))

        local_shard = st.local_tensor()
        self.assertEqual(1, len(st.local_shards()))
        if dist.get_rank() < 3:
            self.assertEqual(torch.Size([3, 12]), local_shard.size())
            self.assertEqual(torch.narrow(tensor, 0, 3 * self.rank, 3), local_shard)
        else:
            self.assertEqual(torch.Size([0, 12]), local_shard.size())

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_shard_tensor_errors(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        tensor = torch.rand(12, 12).cuda(self.rank)

        with self.assertRaisesRegex(ValueError, "does not match with src_rank"):
            _shard_tensor(tensor, spec, src_rank=self.rank)

        with self.assertRaisesRegex(ValueError, "not a contiguous Tensor"):
            tensor_t = torch.rand(12, 12).cuda(self.rank).t()
            _shard_tensor(tensor_t, spec)

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:{self.rank}/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        with self.assertRaisesRegex(ValueError, "does not match with sharding_spec"):
            _shard_tensor(tensor, spec)

        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="rank:0/cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="rank:1/cuda:1",
                ),
            ]
        )
        with self.assertRaisesRegex(NotImplementedError, "not implemented yet!"):
            _shard_tensor(tensor, spec)


class TestModuleHookApi(ShardedTensorTestBase):
    class DummyNNModule(torch.nn.Module):
        def __init__(self, spec, tensor_size):
            super().__init__()
            self.st = sharded_tensor.rand(spec, *tensor_size)

        def forward(self):
            return self.st

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_reshard_output(self):
        specs = _chunk_sharding_specs_list_for_test([0, 1], seed=5)
        spec, reshard_spec = specs[0], specs[1]
        test_module = self.DummyNNModule(spec, [24, 12])
        st = test_module()
        local_shard = st.local_tensor()
        pg = dist.distributed_c10d._get_default_group()
        st_compare = ShardedTensor._init_from_local_shards(
            copy.deepcopy(st.local_shards()),
            st.size(),
            process_group=pg,
        )
        st_compare._sharding_spec = copy.deepcopy(spec)
        st_compare.reshard(reshard_spec)
        test_module = _reshard_output(test_module, reshard_spec)
        st = test_module()
        local_shard = st.local_tensor()
        local_shard_compare = st_compare.local_tensor()
        self.assertEqual(local_shard, local_shard_compare)
        self.assertEqual(local_shard.size(0), 24)
        self.assertEqual(local_shard.size(1), 3)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_collect_local_shard(self):
        specs = _chunk_sharding_specs_list_for_test([0], seed=5)
        spec = specs[0]
        test_module = self.DummyNNModule(spec, [23, 15])
        st = test_module()
        local_shard = st.local_tensor()
        test_module = _collect_local_shard(test_module)
        output = test_module()
        self.assertTrue(isinstance(output, torch.Tensor))
        self.assertEqual(local_shard, output)


class TestLocalTensor(ShardedTensorTestBase):
    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_local_tensor(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        st = sharded_tensor.rand(spec, 24, 12)
        local_shard = st.local_tensor()
        self.assertEqual(torch.Size([6, 12]), local_shard.size())
        self.assertEqual(st.local_tensor(), local_shard)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_local_tensor_error(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:1/cuda:1",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:2/cuda:2",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
                "rank:3/cuda:3",
            ],
        )
        st = sharded_tensor.rand(spec, 24, 12)
        with self.assertRaisesRegex(
            NotImplementedError, "Only single local shard is supported."
        ):
            st.local_tensor()


class TestShardedTensorChunked(ShardedTensorTestBase):
    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharded_tensor_metadata(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        st = sharded_tensor.empty(spec, 10, 20, init_rrefs=True)
        st_metadata = st.metadata()
        self.assertEqual(torch.Size([10, 20]), st_metadata.size)
        self.assertEqual(torch.Size([10, 20]), st.size())
        self.assertEqual(torch.float, st.dtype)
        self.assertEqual(torch.strided, st.layout)
        self.assertEqual(False, st.requires_grad)
        self.assertTrue(st.is_contiguous())
        self.assertFalse(st.is_pinned())

        st = sharded_tensor.empty(spec, 10, 20, requires_grad=True, init_rrefs=True)
        self.assertEqual(True, st.requires_grad)

        st = sharded_tensor.empty(spec, 10, 20, dtype=torch.double, init_rrefs=True)
        self.assertEqual(torch.double, st.dtype)

        # Need CPU for pin_memory
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cpu",
                "rank:1/cpu",
                "rank:2/cpu",
                "rank:3/cpu",
            ],
        )

        st = sharded_tensor.empty(spec, 10, 20, pin_memory=True, init_rrefs=True)
        self.assertEqual(True, st.is_pinned())

        # test read only properties, they're read only as we can't simply change
        # the global metadata without changing the underlying shard's properties
        with self.assertRaisesRegex(RuntimeError, "torch function '__set__'"):
            st.requires_grad = True

    @skipIfRocm
    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_complete_world_size(self):
        for dim in [0, -2]:
            spec = ChunkShardingSpec(
                dim=dim,
                placements=[
                    "rank:0/cuda:0",
                    "rank:1/cuda:1",
                    "rank:2/cuda:2",
                    "rank:3/cuda:3",
                ],
            )
            st = sharded_tensor.empty(spec, 10, 20, init_rrefs=True)

            # Validate local shard.
            local_shards = st.local_shards()
            self.assertEqual(1, len(local_shards))
            local_shard = local_shards[0].tensor
            self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
            if self.rank == 3:
                self.assertEqual((1, 20), local_shard.size())
            else:
                self.assertEqual((3, 20), local_shard.size())

            # Validate global metadata.
            st_metadata = st.metadata()
            shards_metadata = st_metadata.shards_metadata
            self.assertEqual(4, len(shards_metadata))

            for rank, shard_metadata in enumerate(shards_metadata):
                self.assertEqual([rank * 3, 0], shard_metadata.shard_offsets)
                if rank == 3:
                    self.assertEqual([1, 20], shard_metadata.shard_sizes)
                else:
                    self.assertEqual([3, 20], shard_metadata.shard_sizes)
                self.assertEqual(
                    f"rank:{rank}/cuda:{rank}", str(shard_metadata.placement)
                )

            # Validate remote shards.
            remote_shards = st.remote_shards()
            self.assertEqual(3, len(remote_shards))

            for rpc_rank, shards in remote_shards.items():
                self.assertEqual(1, len(shards))
                for remote_shard in shards:
                    self.assertEqual(rpc_rank, remote_shard.owner().id)
                    shard = remote_shard.to_here()
                    self.assertEqual(
                        f"rank:{rpc_rank}/cuda:{rpc_rank}",
                        str(shard.metadata.placement),
                    )
                    if rpc_rank == 3:
                        self.assertEqual((1, 20), shard.tensor.size())
                    else:
                        self.assertEqual((3, 20), shard.tensor.size())

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_create_sharded_tensor_with_ones(self):
        """Test sharded_tensor.ones(...)"""

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        h, w = 10, 20
        st = sharded_tensor.ones(spec, h, w)

        # Validate local shard is initialized with torch.ones
        local_shards = st.local_shards()
        self.assertEqual(1, len(local_shards))
        local_shard = local_shards[0].tensor
        self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
        # The split: for rank!=3 ceil(h/4)=3  for rank=3 1
        expected_h = 1 if self.rank == 3 else math.ceil(h / 4)
        self.assertEqual((expected_h, w), local_shard.size())
        self.assertEqual(local_shard, torch.ones(expected_h, w))

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_gather_even(self) -> None:
        """Test _sharded_tensor.gather(...) with evenly distributed._shards"""

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        h, w = 10, 20
        st = sharded_tensor.ones(spec, h, w)

        full_tensor = None
        dst = 1
        if self.rank == dst:
            full_tensor = torch.zeros(
                h,
                w,
                device=torch.device(f"cuda:{dst}"),
            )
        st.gather(dst, full_tensor)

        if self.rank == dst:
            self.assertEqual(full_tensor, torch.ones(h, w))
        else:
            self.assertIsNone(full_tensor)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_gather_uneven(self) -> None:
        """Test _sharded_tensor.gather(...) with unevenly distributed._shards"""

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
            ],
        )
        h, w = 10, 20
        st = sharded_tensor.ones(spec, h, w)

        full_tensor = None
        dst = 1
        if self.rank == dst:
            full_tensor = torch.zeros(
                h,
                w,
                device=torch.device(f"cuda:{dst}"),
            )
        st.gather(dst, full_tensor)

        if self.rank == dst:
            self.assertEqual(full_tensor, torch.ones(h, w))
        else:
            self.assertIsNone(full_tensor)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_create_sharded_tensor_with_zeros(self):
        """Test sharded_tensor.zeros(...)"""

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        h, w = 10, 20
        st = sharded_tensor.zeros(spec, h, w)

        # Validate local shard is initialized with torch.zeros
        local_shards = st.local_shards()
        self.assertEqual(1, len(local_shards))
        local_shard = local_shards[0].tensor
        self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
        # The split: for rank!=3 ceil(h/4)=3  for rank=3 1
        expected_h = 1 if self.rank == 3 else math.ceil(h / 4)
        self.assertEqual((expected_h, w), local_shard.size())
        self.assertEqual(local_shard, torch.zeros(expected_h, w))

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_create_sharded_tensor_with_rand(self):
        """Test sharded_tensor.rand(...)/randn(...)"""

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        h, w = 8, 2
        seed = 1234

        expected_h = 2
        expected_device = torch.device(f"cuda:{self.rank}")
        dtype = torch.double
        torch.manual_seed(seed)
        # Test sharded_tensor.rand creation
        expected = torch.rand(expected_h, w, device=expected_device, dtype=dtype)
        # reset seed to ensure the same random numbers are generated
        torch.manual_seed(seed)
        st = sharded_tensor.rand(spec, h, w, dtype=dtype)

        # Validate local shard is initialized with torch.rand
        local_shards = st.local_shards()
        self.assertEqual(1, len(local_shards))
        local_shard = local_shards[0].tensor
        self.assertEqual(expected_device, local_shard.device)
        self.assertEqual((expected_h, w), local_shard.size())
        self.assertEqual(expected, local_shard)

        # Test sharded_tensor.randn creation
        torch.manual_seed(seed)
        expected_randn = torch.randn(expected_h, w, device=expected_device, dtype=dtype)
        # reset seed to ensure the same random numbers are generated
        torch.manual_seed(seed)
        st_randn = sharded_tensor.randn(spec, h, w, dtype=dtype)

        # Validate local shard is initialized with torch.randn
        local_shards = st_randn.local_shards()
        self.assertEqual(1, len(local_shards))
        local_shard = local_shards[0].tensor
        self.assertEqual(expected_device, local_shard.device)
        self.assertEqual((expected_h, w), local_shard.size())
        self.assertEqual(expected_randn, local_shard)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_create_sharded_tensor_with_full(self):
        """Test sharded_tensor.full(...)"""

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        h, w = 10, 20
        fill_value = 1234
        st = sharded_tensor.full(
            spec, size=(h, w), fill_value=fill_value, dtype=torch.int32
        )

        # Validate local shard is initialized with torch.full
        local_shards = st.local_shards()
        self.assertEqual(1, len(local_shards))
        local_shard = local_shards[0].tensor
        self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
        # The split: for rank!=3 ceil(h/4)=3  for rank=3 1
        expected_h = 1 if self.rank == 3 else math.ceil(h / 4)
        self.assertEqual((expected_h, w), local_shard.size())
        self.assertEqual(
            local_shard,
            torch.full(size=(expected_h, w), fill_value=fill_value, dtype=torch.int32),
        )

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_create_sharded_tensor_like(self):
        """Test tensor like methods, i.e. torch.zeros_like(...), torch.full_like, etc."""

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        h, w = 8, 8
        expected_h = 2
        seed = 1234
        dtype = torch.double
        expected_device = torch.device(f"cuda:{self.rank}")
        st = sharded_tensor.rand(spec, (h, w), dtype=dtype)
        tensor_like_ops = {
            torch.zeros_like: torch.zeros,
            torch.ones_like: torch.ones,
            torch.rand_like: torch.rand,
            torch.randn_like: torch.randn,
            torch.empty_like: torch.empty,
            torch.full_like: torch.full,
        }
        for op, expect_local_op in tensor_like_ops.items():
            if op == torch.full_like:
                # special handle full/full_like as it needs to have additional fill_value arg
                expect_tensor = expect_local_op(
                    (expected_h, w), 8.8, device=expected_device, dtype=dtype
                )
                new_op_st = op(st, 8.8, dtype=dtype)
                self.assertEqual(new_op_st.local_tensor(), expect_tensor)
            elif op == torch.empty_like:
                # empty/empty_like we only compare the shape
                expect_tensor = expect_local_op(
                    expected_h, w, device=expected_device, dtype=dtype
                )
                new_op_st = op(st, dtype=dtype)
                self.assertEqual(new_op_st.local_tensor().shape, expect_tensor.shape)
            else:
                torch.manual_seed(seed)
                expect_tensor = expect_local_op(
                    expected_h, w, device=expected_device, dtype=dtype
                )
                torch.manual_seed(seed)
                new_op_st = op(st, dtype=dtype)
                self.assertEqual(new_op_st.local_tensor(), expect_tensor)

    @skipIfRocm
    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_partial_world_size(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        st = sharded_tensor.empty(spec, 10, 20, init_rrefs=True)

        # Validate local shard.
        local_shards = st.local_shards()
        if self.rank >= 2:
            self.assertEqual(1, len(local_shards))
            local_shard = local_shards[0].tensor
            self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
            self.assertEqual((5, 20), local_shard.size())
        else:
            self.assertEqual(0, len(local_shards))

        # Validate global metadata.
        st_metadata = st.metadata()
        shards_metadata = st_metadata.shards_metadata
        self.assertEqual(2, len(shards_metadata))

        for shard_rank, shard_metadata in enumerate(shards_metadata):
            self.assertEqual([shard_rank * 5, 0], shard_metadata.shard_offsets)
            self.assertEqual([5, 20], shard_metadata.shard_sizes)
            self.assertEqual(
                f"rank:{shard_rank + 2}/cuda:{shard_rank + 2}",
                str(shard_metadata.placement),
            )

        # Validate remote shards.
        remote_shards = st.remote_shards()
        if self.rank >= 2:
            self.assertEqual(1, len(remote_shards))
        else:
            self.assertEqual(2, len(remote_shards))

        for rpc_rank, shards in remote_shards.items():
            self.assertEqual(1, len(shards))
            for remote_shard in shards:
                self.assertEqual(rpc_rank, remote_shard.owner().id)
                shard = remote_shard.to_here()
                self.assertEqual(
                    f"rank:{rpc_rank}/cuda:{rpc_rank}", str(shard.metadata.placement)
                )
                self.assertEqual((5, 20), shard.tensor.size())

    @skipIfRocm
    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_new_group(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        pg = dist.new_group(ranks=[1, 2, 3])
        st = sharded_tensor.empty(spec, 10, 20, process_group=pg, init_rrefs=True)

        # Validate local shard.
        local_shards = st.local_shards()
        if self.rank >= 2:
            self.assertEqual(1, len(local_shards))
            local_shard = local_shards[0].tensor
            self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
            self.assertEqual((5, 20), local_shard.size())
        else:
            self.assertEqual(0, len(local_shards))

        # Validate global metadata.
        st_metadata = st.metadata()
        shards_metadata = st_metadata.shards_metadata
        self.assertEqual(2, len(shards_metadata))

        for shard_rank, shard_metadata in enumerate(shards_metadata):
            self.assertEqual([shard_rank * 5, 0], shard_metadata.shard_offsets)
            self.assertEqual([5, 20], shard_metadata.shard_sizes)
            self.assertEqual(
                f"rank:{shard_rank + 2}/cuda:{shard_rank + 2}",
                str(shard_metadata.placement),
            )

        # Validate remote shards.
        remote_shards = st.remote_shards()
        if self.rank >= 2:
            self.assertEqual(1, len(remote_shards))
        else:
            self.assertEqual(2, len(remote_shards))

        for rpc_rank, shards in remote_shards.items():
            self.assertEqual(1, len(shards))
            for remote_shard in shards:
                shard = remote_shard.to_here()
                self.assertEqual(rpc_rank, remote_shard.owner().id)
                self.assertEqual(
                    f"rank:{rpc_rank}/cuda:{rpc_rank}", str(shard.metadata.placement)
                )
                self.assertEqual((5, 20), shard.tensor.size())

    @skipIfRocm
    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_multiple_local_shards(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        st = sharded_tensor.empty(spec, 16, 20, init_rrefs=True)

        # Validate local shards.
        local_shards = st.local_shards()
        self.assertEqual(2, len(local_shards))
        for local_shard in local_shards:
            self.assertEqual(
                torch.device(f"cuda:{self.rank}"), local_shard.tensor.device
            )
            self.assertEqual((2, 20), local_shard.tensor.size())

        # Validate global metadata.
        st_metadata = st.metadata()
        shards_metadata = st_metadata.shards_metadata
        self.assertEqual(8, len(shards_metadata))

        for shard_idx, shard_metadata in enumerate(shards_metadata):
            self.assertEqual([shard_idx * 2, 0], shard_metadata.shard_offsets)
            self.assertEqual([2, 20], shard_metadata.shard_sizes)
            self.assertEqual(
                f"rank:{shard_idx % 4}/cuda:{shard_idx % 4}",
                str(shard_metadata.placement),
            )

        # Validate remote shards.
        remote_shards = st.remote_shards()
        self.assertEqual(3, len(remote_shards))
        for rpc_rank, shards in remote_shards.items():
            self.assertEqual(2, len(shards))
            for remote_shard in shards:
                shard = remote_shard.to_here()
                self.assertEqual((2, 20), shard.tensor.size())
                self.assertEqual(rpc_rank, remote_shard.owner().id)

    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharding_columns(self):
        self.init_pg()

        for dim in [1, -1]:
            spec = ChunkShardingSpec(
                dim=dim,
                placements=[
                    "rank:0/cuda:0",
                    "rank:1/cuda:1",
                    "rank:2/cuda:2",
                    "rank:3/cuda:3",
                ],
            )

            st = sharded_tensor.empty(spec, 10, 32)

            # Validate local shard.
            local_shards = st.local_shards()
            self.assertEqual(1, len(local_shards))
            local_shard = local_shards[0].tensor
            self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
            self.assertEqual((10, 8), local_shard.size())

            # Validate global metadata.
            st_metadata = st.metadata()
            shards_metadata = st_metadata.shards_metadata
            self.assertEqual(4, len(shards_metadata))

            for rank, shard_metadata in enumerate(shards_metadata):
                self.assertEqual([0, rank * 8], shard_metadata.shard_offsets)
                self.assertEqual([10, 8], shard_metadata.shard_sizes)
                self.assertEqual(
                    f"rank:{rank}/cuda:{rank}", str(shard_metadata.placement)
                )

    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_invalid_sharding(self):
        self.init_pg()

        with self.assertRaisesRegex(
            NotImplementedError, "does not support named dimension"
        ):
            spec = ChunkShardingSpec(dim="H", placements=["rank:1/cuda:1"])
            sharded_tensor.empty(spec, 10, 20)

        for dim in [2, 3, 4, -3, -4, -5]:
            spec = ChunkShardingSpec(dim=dim, placements=["rank:1/cuda:1"])
            with self.assertRaisesRegex(ValueError, "Invalid sharding dim"):
                sharded_tensor.empty(spec, 10, 20)

        spec = ChunkShardingSpec(dim=0, placements=["rank:5/cuda:1"])
        with self.assertRaisesRegex(
            ValueError, "Global rank 5 does not exist in input process group"
        ):
            sharded_tensor.empty(spec, 10, 20)

        spec = ChunkShardingSpec(dim=0, placements=["rank:0/cuda:1"])
        st = sharded_tensor.empty(spec, 10, 20)
        tensor = torch.empty(10, 20)
        with self.assertRaisesRegex(
            RuntimeError, r".*not supported for ShardedTensor!$"
        ):
            torch.add(st, tensor)

        spec = ChunkShardingSpec(dim=0, placements=["rank:0/cuda:1"])
        with self.assertRaisesRegex(
            ValueError, "Only torch.strided layout is currently supported"
        ):
            sharded_tensor.empty(spec, 10, 20, layout=torch.sparse_coo)

        spec = ChunkShardingSpec(dim=0, placements=["rank:0/cuda:1"])
        with self.assertRaisesRegex(
            ValueError,
            "Only torch.contiguous_format memory_format is currently supported",
        ):
            sharded_tensor.empty(spec, 10, 20, memory_format=torch.channels_last)

        spec = ChunkShardingSpec(dim=0, placements=["worker0/cuda:1"])
        with self.assertRaisesRegex(
            RuntimeError, "RPC framework needs to be initialized"
        ):
            sharded_tensor.empty(spec, 10, 20)

        spec = ChunkShardingSpec(dim=0, placements=["rank:0/cuda:1"])
        with self.assertRaisesRegex(
            RuntimeError, "RPC Framework needs to be initialized"
        ):
            st = sharded_tensor.empty(spec, 10, 20, init_rrefs=True)

        with self.assertRaisesRegex(
            RuntimeError, "ShardedTensor created with init_rrefs=False"
        ):
            st = sharded_tensor.empty(spec, 10, 20)
            st.remote_shards()

        self.init_rpc()
        spec = ChunkShardingSpec(dim=0, placements=["workerfoo/cuda:1"])
        with self.assertRaisesRegex(ValueError, "Invalid worker name"):
            sharded_tensor.empty(spec, 10, 20, init_rrefs=True)

    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_invalid_pg_rpc_ranks(self):
        self.init_pg()

        # Init RPC with different ranks.
        rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
            _transports=tp_transports()
        )
        rpc_backend_options.init_method = f"file://{self.file_name}"
        rank = (self.rank + 1) % self.world_size
        rpc.init_rpc(
            name=f"worker{rank}",
            rank=rank,
            world_size=self.world_size,
            rpc_backend_options=rpc_backend_options,
        )

        spec = ChunkShardingSpec(dim=0, placements=["rank:1/cuda:1"])
        with self.assertRaisesRegex(
            ValueError, "Default ProcessGroup and RPC ranks must be the same"
        ):
            sharded_tensor.empty(spec, 10, 20, init_rrefs=True)

    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_insufficient_sharding_dims(self):
        self.init_pg()

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        st = sharded_tensor.empty(spec, 2, 20)

        # Validate local shard.
        local_shards = st.local_shards()
        if self.rank <= 1:
            self.assertEqual(1, len(local_shards))
            local_shard = local_shards[0].tensor
            self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
            self.assertEqual((1, 20), local_shard.size())
        else:
            self.assertEqual(1, len(local_shards))
            local_shard = local_shards[0].tensor
            self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
            self.assertEqual(local_shard.numel(), 0)

        # Validate global metadata.
        st_metadata = st.metadata()
        shards_metadata = st_metadata.shards_metadata
        self.assertEqual(4, len(shards_metadata))

        for shard_rank, shard_metadata in enumerate(shards_metadata):
            self.assertEqual([shard_rank, 0], shard_metadata.shard_offsets)
            self.assertEqual(
                f"rank:{shard_rank}/cuda:{shard_rank}", str(shard_metadata.placement)
            )
            if shard_rank <= 1:
                self.assertEqual([1, 20], shard_metadata.shard_sizes)
            else:
                self.assertEqual([0, 20], shard_metadata.shard_sizes)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharded_tensor_sizes(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        # Test with *args
        st = sharded_tensor.empty(spec, 10, 20, init_rrefs=True)
        self.assertEqual(torch.Size([10, 20]), st.size())

        # Test with single *args
        st = sharded_tensor.empty(spec, 10, init_rrefs=True)
        self.assertEqual(torch.Size([10]), st.size())

        # Test with list
        st = sharded_tensor.empty(spec, [10, 20], init_rrefs=True)
        self.assertEqual(torch.Size([10, 20]), st.size())

        # Test with tuple
        st = sharded_tensor.empty(spec, (10, 20), init_rrefs=True)
        self.assertEqual(torch.Size([10, 20]), st.size())

        # Test with row size
        st = sharded_tensor.empty(spec, (10, 20), init_rrefs=True)
        self.assertEqual(st.size(0), 10)

        # Test with col size
        st = sharded_tensor.empty(spec, (10, 20), init_rrefs=True)
        self.assertEqual(st.size(1), 20)

        # Test with negative indexed size
        st = sharded_tensor.empty(spec, (10, 20), init_rrefs=True)
        self.assertEqual(st.size(-1), 20)

        # Test with dim/ndim
        self.assertEqual(st.dim(), 2)
        self.assertEqual(st.ndim, 2)
        # Test with invalid input
        st = sharded_tensor.empty(spec, (10, 20), init_rrefs=True)
        with self.assertRaisesRegex(IndexError, "Dimension out of range"):
            st.size(-3)
        with self.assertRaisesRegex(IndexError, "Dimension out of range"):
            st.size(2)

        with self.assertRaises(TypeError):
            st = sharded_tensor.empty(spec, "foo")

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_state_dict(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        m = MyShardedModel1(spec)

        # Test save
        m._register_state_dict_hook(state_dict_hook)
        buffer = io.BytesIO()
        mod_state_dict = m.state_dict()
        mod_state_keys = mod_state_dict.keys()
        self.assertTrue("sharded_tensor1" in mod_state_keys)
        self.assertTrue("submodule.sharded_tensor2" in mod_state_keys)
        torch.save(mod_state_dict, buffer)

        # Test load.
        module_load = MyShardedModel1()
        module_load._register_load_state_dict_pre_hook(pre_load_state_dict_hook, True)

        buffer.seek(0)
        # weights_only=False as ShardedTensor weights_only is already tested in TestFSDPStateDict.test_torch_save_load
        state_dict_deser = torch.load(buffer, weights_only=False)
        module_load.load_state_dict(state_dict_deser, strict=False)

        module_load._register_state_dict_hook(state_dict_hook)
        loaded_dict_keys = module_load.state_dict().keys()
        self.assertTrue("sharded_tensor1" in loaded_dict_keys)
        self.assertTrue("submodule.sharded_tensor2" in loaded_dict_keys)
        # Verify after load.
        self.assertTrue(torch.equal(m.sharded_tensor1, module_load.sharded_tensor1))
        self.assertTrue(
            torch.equal(
                m.submodule.sharded_tensor2, module_load.submodule.sharded_tensor2
            )
        )

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_state_dict_new_group(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:2/cuda:0",
                "rank:3/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        pg = dist.new_group([2, 3])

        m = MyShardedModel1(spec, pg)

        # Test save
        m._register_state_dict_hook(state_dict_hook)
        buffer = io.BytesIO()
        torch.save(m.state_dict(), buffer)

        # Test load.
        module_load = MyShardedModel1(spec=None, group=pg)
        module_load._register_load_state_dict_pre_hook(pre_load_state_dict_hook, True)

        buffer.seek(0)
        with load_with_process_group(pg):
            # ShardedTensor weights_only is already tested in TestFSDPStateDict.test_torch_save_load
            state_dict_deser = torch.load(buffer, weights_only=False)
            module_load.load_state_dict(state_dict_deser, strict=False)

        # Verify after load.
        self.assertTrue(torch.equal(m.sharded_tensor1, module_load.sharded_tensor1))
        self.assertTrue(
            torch.equal(
                m.submodule.sharded_tensor2, module_load.submodule.sharded_tensor2
            )
        )

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_state_dict_no_sharded_tensors(self):
        # Verify hooks don't affect modules with no ShardedTensors.
        m = torch.nn.Linear(10, 10)

        # Test save
        state_dict_before = m.state_dict()
        m._register_state_dict_hook(state_dict_hook)
        buffer = io.BytesIO()
        torch.save(m.state_dict(), buffer)
        self.assertEqual(state_dict_before, m.state_dict())

        # Test load.
        module_load = torch.nn.Linear(10, 10)
        module_load._register_load_state_dict_pre_hook(pre_load_state_dict_hook, True)

        buffer.seek(0)
        state_dict_deser = torch.load(buffer)
        module_load.load_state_dict(state_dict_deser, strict=False)

        # Verify after load.
        self.assertEqual(m.weight, module_load.weight)
        self.assertEqual(m.bias, module_load.bias)

    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_load_state_dict_errors(self):
        self.init_rpc()

        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            init_method=f"file://{self.file_name}",
        )

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        m = MyShardedModel1(spec)

        # Test save
        m._register_state_dict_hook(state_dict_hook)
        buffer = io.BytesIO()
        torch.save(m.state_dict(), buffer)

        pg = dist.new_group(ranks=[0, 2, 3])

        buffer.seek(0)
        if self.rank != 0:
            with self.assertRaisesRegex(RuntimeError, "Local rank at save time was"):
                with load_with_process_group(pg):
                    # ShardedTensor weights_only is already tested in TestFSDPStateDict.test_torch_save_load
                    torch.load(buffer, weights_only=False)
        else:
            with self.assertRaisesRegex(
                RuntimeError, "Local world size at save time was"
            ):
                with load_with_process_group(pg):
                    # ShardedTensor weights_only is already tested in TestFSDPStateDict.test_torch_save_load
                    torch.load(buffer, weights_only=False)

        dist.destroy_process_group()
        buffer.seek(0)
        with self.assertRaisesRegex(
            RuntimeError, "Need to initialize default process group"
        ):
            # ShardedTensor weights_only is already tested in TestFSDPStateDict.test_torch_save_load
            torch.load(buffer, weights_only=False)
        rpc.shutdown()

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_cleanup(self):
        def create_tensors():
            spec = ChunkShardingSpec(
                dim=0,
                placements=[
                    "rank:0/cuda:0",
                    "rank:1/cuda:1",
                    "rank:2/cuda:2",
                    "rank:3/cuda:3",
                ],
            )
            sharded_tensor.empty(spec, 10, 20, init_rrefs=True)
            sharded_tensor.empty(spec, 10, 20)

        create_tensors()
        self.assertEqual(0, len(sharded_tensor.api._sharded_tensor_map))


class TestShardedTensorEnumerable(ShardedTensorTestBase):
    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharded_tensor_metadata(self):
        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="rank:0/cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[0, 5],
                    shard_sizes=[5, 5],
                    placement="rank:1/cuda:1",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="rank:2/cuda:2",
                ),
                ShardMetadata(
                    shard_offsets=[5, 5],
                    shard_sizes=[5, 5],
                    placement="rank:3/cuda:3",
                ),
            ]
        )

        st = sharded_tensor.empty(spec, 10, 10, init_rrefs=True)
        st_metadata = st.metadata()
        self.assertEqual(torch.Size([10, 10]), st_metadata.size)
        self.assertEqual(torch.float, st.dtype)
        self.assertEqual(torch.strided, st.layout)
        self.assertEqual(False, st.requires_grad)
        self.assertTrue(st.is_contiguous())
        self.assertFalse(st.is_pinned())

        st = sharded_tensor.empty(spec, 10, 10, requires_grad=True, init_rrefs=True)
        self.assertEqual(True, st.requires_grad)

        st = sharded_tensor.empty(spec, 10, 10, dtype=torch.double, init_rrefs=True)
        self.assertEqual(torch.double, st.dtype)

        # Need CPU for pin_memory
        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="rank:0/cpu",
                ),
                ShardMetadata(
                    shard_offsets=[0, 5],
                    shard_sizes=[5, 5],
                    placement="rank:1/cpu",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="rank:2/cpu",
                ),
                ShardMetadata(
                    shard_offsets=[5, 5],
                    shard_sizes=[5, 5],
                    placement="rank:3/cpu",
                ),
            ]
        )

        st = sharded_tensor.empty(spec, 10, 10, pin_memory=True, init_rrefs=True)
        self.assertTrue(st.is_pinned())

    @skipIfRocm
    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_grid_sharding(self):
        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="rank:0/cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[0, 5],
                    shard_sizes=[5, 5],
                    placement="rank:1/cuda:1",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="rank:2/cuda:2",
                ),
                ShardMetadata(
                    shard_offsets=[5, 5],
                    shard_sizes=[5, 5],
                    placement="rank:3/cuda:3",
                ),
            ]
        )

        st = sharded_tensor.empty(spec, 10, 10, init_rrefs=True)
        self.assertEqual((10, 10), st.size())
        self.assertEqual(1, len(st.local_shards()))

        # Verify local shard.
        local_shard = st.local_shards()[0]
        self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.tensor.device)
        self.assertEqual((5, 5), local_shard.tensor.size())

        # Verify local shard metadata.
        self.assertEqual(
            (self.rank // 2 * 5, (self.rank % 2) * 5),
            local_shard.metadata.shard_offsets,
        )
        self.assertEqual((5, 5), local_shard.metadata.shard_sizes)
        self.assertEqual(
            f"rank:{self.rank}/cuda:{self.rank}", str(local_shard.metadata.placement)
        )

        # Verify global metadata.
        st_metadata = st.metadata()
        shards_metadata = st_metadata.shards_metadata
        self.assertEqual(4, len(shards_metadata))
        for rank, shard_metadata in enumerate(shards_metadata):
            self.assertEqual(
                (rank // 2 * 5, (rank % 2) * 5), shard_metadata.shard_offsets
            )
            self.assertEqual((5, 5), shard_metadata.shard_sizes)
            self.assertEqual(f"rank:{rank}/cuda:{rank}", str(shard_metadata.placement))

        # Validate remote shards.
        remote_shards = st.remote_shards()
        self.assertEqual(3, len(remote_shards))

        for rpc_rank, shards in remote_shards.items():
            self.assertEqual(1, len(shards))
            for remote_shard in shards:
                self.assertEqual(rpc_rank, remote_shard.owner().id)
                shard = remote_shard.to_here()
                self.assertEqual((5, 5), shard.tensor.size())

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_create_sharded_tensor_with_ones(self):
        """Test sharded_tensor.ones(...)"""

        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="rank:0/cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[0, 5],
                    shard_sizes=[5, 5],
                    placement="rank:1/cuda:1",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="rank:2/cuda:2",
                ),
                ShardMetadata(
                    shard_offsets=[5, 5],
                    shard_sizes=[5, 5],
                    placement="rank:3/cuda:3",
                ),
            ]
        )

        st = sharded_tensor.ones(spec, 10, 10, init_rrefs=True)
        self.assertEqual((10, 10), st.size())
        self.assertEqual(1, len(st.local_shards()))

        # Verify local shard is initialized with torch.ones
        local_shard = st.local_shards()[0]
        self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.tensor.device)
        self.assertEqual((5, 5), local_shard.tensor.size())
        self.assertEqual(local_shard.tensor, torch.ones(5, 5))

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_gather_even(self) -> None:
        """Test _sharded_tensor.gather(...) with evenly distributed._shards"""

        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="rank:0/cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[0, 5],
                    shard_sizes=[5, 5],
                    placement="rank:1/cuda:1",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="rank:2/cuda:2",
                ),
                ShardMetadata(
                    shard_offsets=[5, 5],
                    shard_sizes=[5, 5],
                    placement="rank:3/cuda:3",
                ),
            ]
        )

        h, w = 10, 10
        st = sharded_tensor.ones(spec, h, w, init_rrefs=True)

        full_tensor = None
        dst = 0
        if self.rank == dst:
            full_tensor = torch.zeros(h, w, device=torch.device(f"cuda:{dst}"))
        st.gather(dst, full_tensor)

        if self.rank == dst:
            self.assertEqual(full_tensor, torch.ones(h, w))
        else:
            self.assertIsNone(full_tensor)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_gather_uneven(self) -> None:
        """Test _sharded_tensor.gather(...) with unevenly distributed._shards"""

        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="rank:0/cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[0, 5],
                    shard_sizes=[5, 5],
                    placement="rank:1/cuda:1",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="rank:0/cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[5, 5],
                    shard_sizes=[5, 5],
                    placement="rank:3/cuda:3",
                ),
            ]
        )

        h, w = 10, 10
        st = sharded_tensor.ones(spec, h, w, init_rrefs=True)

        full_tensor = None
        dst = 0
        if self.rank == dst:
            full_tensor = torch.zeros(h, w, device=torch.device(f"cuda:{dst}"))
        st.gather(dst, full_tensor)

        if self.rank == dst:
            self.assertEqual(full_tensor, torch.ones(h, w))
        else:
            self.assertIsNone(full_tensor)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharded_tensor_to_cpu(self):
        cpu_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cpu",
                "rank:1/cpu",
                "rank:2/cpu",
                "rank:3/cpu",
            ],
        )
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        h, w = 10, 20
        gloo_pg = dist.new_group(backend="gloo")

        # CPU sharded tensor should return the same instance (no copy)
        st_cpu = sharded_tensor.zeros(cpu_spec, h, w, process_group=gloo_pg)
        new_st_cpu = st_cpu.cpu()
        self.assertTrue(st_cpu is new_st_cpu)

        # GPU sharded tensor to cpu
        st = sharded_tensor.zeros(spec, h, w)
        # test ability to move st to CPU
        spec_before_move = st.sharding_spec()
        new_st = st.cpu(process_group=gloo_pg)
        # return a copy of original st
        self.assertFalse(st is new_st)
        # check the spec is still ChunkShardingSpec
        spec_after_move = new_st.sharding_spec()
        self.assertIsInstance(spec_after_move, ChunkShardingSpec)
        self.assertIsInstance(new_st._process_group, distributed_c10d.ProcessGroup)
        # test specs before and after the move almost the same except placement device
        self.assertEqual(spec_before_move.dim, spec_after_move.dim)
        self.assertEqual(
            len(spec_before_move.placements), len(spec_after_move.placements)
        )
        for i, remote_device_after in enumerate(spec_after_move.placements):
            remote_device_before = spec_before_move.placements[i]
            self.assertEqual(remote_device_before.rank(), remote_device_after.rank())
            self.assertEqual(str(remote_device_after.device()), "cpu")

        # ensure metadata also get changed to CPU
        metas = new_st.metadata().shards_metadata
        for meta in metas:
            self.assertEqual(str(meta.placement.device()), "cpu")

        # Test if a mixed sharded tensor (ShardedTensor with different devices) to cpu
        mixed_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cpu",
                "rank:1/cpu",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        st = sharded_tensor.zeros(mixed_spec, h, w, process_group=gloo_pg)
        new_st = st.cpu()
        # return a copy of original st
        self.assertFalse(st is new_st)
        # check the spec is still ChunkShardingSpec
        spec_after_move = new_st.sharding_spec()
        self.assertIsInstance(spec_after_move, ChunkShardingSpec)
        # test specs before and after the move almost the same except placement device
        self.assertEqual(mixed_spec.dim, spec_after_move.dim)
        self.assertEqual(len(mixed_spec.placements), len(spec_after_move.placements))
        for i, remote_device_after in enumerate(spec_after_move.placements):
            remote_device_before = mixed_spec.placements[i]
            self.assertEqual(remote_device_before.rank(), remote_device_after.rank())
            self.assertEqual(str(remote_device_after.device()), "cpu")

        # ensure metadata also get changed to CPU
        metas = new_st.metadata().shards_metadata
        for meta in metas:
            self.assertEqual(str(meta.placement.device()), "cpu")

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharded_tensor_to_cuda(self):
        cpu_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cpu",
                "rank:1/cpu",
                "rank:2/cpu",
                "rank:3/cpu",
            ],
        )
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        h, w = 10, 20
        # CUDA sharded tensor should return a new ShardedTensor, but same
        # local shards(no movements)
        st_cuda = sharded_tensor.zeros(spec, h, w)
        new_st_cuda = st_cuda.cuda()
        self.assertTrue(st_cuda is not new_st_cuda)
        self.assertTrue(st_cuda.local_tensor() is new_st_cuda.local_tensor())

        gloo_pg = dist.new_group(backend="gloo")

        # CPU sharded tensor to GPU
        st_cpu = sharded_tensor.zeros(cpu_spec, h, w, process_group=gloo_pg)
        # test ability to move st to GPU
        spec_before_move = st_cpu.sharding_spec()
        new_st_gpu = st_cpu.cuda()
        # check the spec is still ChunkShardingSpec
        spec_after_move = new_st_gpu.sharding_spec()
        self.assertIsInstance(spec_after_move, ChunkShardingSpec)
        # test specs before and after the move almost the same except placement device
        self.assertEqual(spec_before_move.dim, spec_after_move.dim)
        self.assertEqual(
            len(spec_before_move.placements), len(spec_after_move.placements)
        )
        for i, remote_device_after in enumerate(spec_after_move.placements):
            remote_device_before = spec_before_move.placements[i]
            self.assertEqual(remote_device_before.rank(), remote_device_after.rank())
            self.assertEqual(str(remote_device_before.device().type), "cpu")
            self.assertEqual(str(remote_device_after.device().type), "cuda")

        # ensure metadata also get changed to GPU
        metas = new_st_gpu.metadata().shards_metadata
        for meta in metas:
            self.assertEqual(str(meta.placement.device().type), "cuda")

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharded_tensor_to_test(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        h, w = 10, 20
        # CUDA sharded tensor should return a new ShardedTensor, but same
        # local shards(no movements)
        st = sharded_tensor.zeros(spec, h, w)
        # test same dtype, device return itself
        st_self = st.to(dtype=st.dtype, device="cuda")
        self.assertTrue(st_self is st)

        # test dtype to
        st_16 = st.to(torch.float16)
        self.assertFalse(st_16 is st)
        self.assertEqual(st_16.dtype, torch.float16)
        # test device to
        st_cpu = st.to(device=torch.device("cpu"))
        self.assertFalse(st_cpu is st)
        self.assertEqual(st_cpu.local_tensor().device.type, "cpu")
        st_cuda = st_cpu.to(device=torch.device("cuda"))
        self.assertEqual(st_cuda.local_tensor().device.type, "cuda")
        # non-kwarg device to
        st_cuda = st_cpu.to(torch.device("cuda"))
        self.assertEqual(st_cuda.local_tensor().device.type, "cuda")
        st_cpu = st_cuda.to(torch.device("cpu"))
        self.assertEqual(st_cpu.local_tensor().device.type, "cpu")
        # with string like device conversion
        st_cpu = st_cuda.to("cpu")
        self.assertEqual(st_cpu.local_tensor().device.type, "cpu")
        st_cuda = st_cpu.to("cuda")
        self.assertEqual(st_cuda.local_tensor().device.type, "cuda")
        # with int like device conversion
        st_cpu = st_cuda.to("cpu")
        self.assertEqual(st_cpu.local_tensor().device.type, "cpu")
        st_cuda = st_cpu.to(self.rank)
        self.assertEqual(st_cuda.local_tensor().device.type, "cuda")

        # test tensor to
        cuda_tensor = torch.randn(3, 4, dtype=torch.float16, device="cuda")
        st_cuda = st.to(cuda_tensor)
        self.assertFalse(st_cuda is st)
        self.assertEqual(st_cuda.dtype, torch.float16)

        cuda_tensor = torch.randn(3, 4, dtype=torch.float16, device="cuda:2")
        st_cuda = st.to(cuda_tensor)
        self.assertEqual(st_cuda.dtype, torch.float16)

        # test dtype and device together
        st_cpu_16 = st.to("cpu", torch.float16)
        self.assertEqual(st_cpu_16.dtype, torch.float16)
        self.assertEqual(st_cpu_16.local_tensor().device.type, "cpu")

        st_cuda_32 = st_cpu_16.to("cuda", torch.float32)
        self.assertEqual(st_cuda_32.dtype, torch.float32)
        self.assertEqual(st_cuda_32.local_tensor().device.type, "cuda")

        # test pass additional process group
        gloo_pg = dist.new_group(backend="gloo")
        st_gloo = st.to(device="cpu", process_group=gloo_pg)
        self.assertFalse(st_gloo is st)
        self.assertEqual(st_gloo.local_tensor().device.type, "cpu")
        self.assertEqual(st_gloo._process_group, gloo_pg)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharded_tensor_device(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        h, w = 10, 20
        # CUDA sharded tensor should return a new ShardedTensor, but same
        # local shards(no movements)
        st = sharded_tensor.zeros(spec, h, w)
        current_device = torch.device(torch.cuda.current_device())
        self.assertEqual(current_device, st.device)

        # test after to cpu, device get changed
        cpu_device = torch.device("cpu")
        st_cpu = st.to(device=cpu_device)
        self.assertEqual(st_cpu.device, cpu_device)

    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_uneven_shards(self):
        self.init_pg()

        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[2, 4],
                    placement="rank:0/cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[0, 4],
                    shard_sizes=[4, 2],
                    placement="rank:1/cuda:1",
                ),
                ShardMetadata(
                    shard_offsets=[2, 0],
                    shard_sizes=[4, 4],
                    placement="rank:2/cuda:2",
                ),
                ShardMetadata(
                    shard_offsets=[4, 4],
                    shard_sizes=[2, 2],
                    placement="rank:3/cuda:3",
                ),
            ]
        )

        st = sharded_tensor.empty(spec, 6, 6)
        self.assertEqual((6, 6), st.size())
        self.assertEqual(1, len(st.local_shards()))

        def verify_size(rank, tensor_dims):
            if rank == 0:
                self.assertEqual((2, 4), tensor_dims)
            elif rank == 1:
                self.assertEqual((4, 2), tensor_dims)
            elif rank == 2:
                self.assertEqual((4, 4), tensor_dims)
            elif rank == 3:
                self.assertEqual((2, 2), tensor_dims)

        def verify_offsets(rank, offsets):
            if rank == 0:
                self.assertEqual((0, 0), offsets)
            elif rank == 1:
                self.assertEqual((0, 4), offsets)
            elif rank == 2:
                self.assertEqual((2, 0), offsets)
            elif rank == 3:
                self.assertEqual((4, 4), offsets)

        # Verify local shard.
        local_shard = st.local_shards()[0]
        self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.tensor.device)
        verify_size(self.rank, local_shard.tensor.size())

        # Verify local shard metadata.
        verify_offsets(self.rank, local_shard.metadata.shard_offsets)
        verify_size(self.rank, local_shard.metadata.shard_sizes)
        self.assertEqual(
            f"rank:{self.rank}/cuda:{self.rank}", str(local_shard.metadata.placement)
        )

        # Verify global metadata.
        st_metadata = st.metadata()
        shards_metadata = st_metadata.shards_metadata
        self.assertEqual(4, len(shards_metadata))
        for rank, shard_metadata in enumerate(shards_metadata):
            verify_offsets(rank, shard_metadata.shard_offsets)
            verify_size(rank, shard_metadata.shard_sizes)
            self.assertEqual(f"rank:{rank}/cuda:{rank}", str(shard_metadata.placement))

    @skipIfRocm
    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_partial_world_size(self):
        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="rank:0/cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="rank:1/cuda:1",
                ),
            ]
        )

        st = sharded_tensor.empty(spec, 10, 5, init_rrefs=True)
        self.assertEqual((10, 5), st.size())
        if self.rank <= 1:
            self.assertEqual(1, len(st.local_shards()))
        else:
            self.assertEqual(0, len(st.local_shards()))

        if self.rank <= 1:
            # Verify local shard.
            local_shard = st.local_shards()[0]
            self.assertEqual(
                torch.device(f"cuda:{self.rank}"), local_shard.tensor.device
            )
            self.assertEqual((5, 5), local_shard.tensor.size())

            # Verify local shard metadata.
            self.assertEqual((self.rank * 5, 0), local_shard.metadata.shard_offsets)
            self.assertEqual((5, 5), local_shard.metadata.shard_sizes)
            self.assertEqual(
                f"rank:{self.rank}/cuda:{self.rank}",
                str(local_shard.metadata.placement),
            )

        # Verify global metadata.
        st_metadata = st.metadata()
        shards_metadata = st_metadata.shards_metadata
        self.assertEqual(2, len(shards_metadata))
        for rank, shard_metadata in enumerate(shards_metadata):
            self.assertEqual((rank * 5, 0), shard_metadata.shard_offsets)
            self.assertEqual((5, 5), shard_metadata.shard_sizes)
            self.assertEqual(f"rank:{rank}/cuda:{rank}", str(shard_metadata.placement))

        # Validate remote shards.
        remote_shards = st.remote_shards()
        if self.rank <= 1:
            self.assertEqual(1, len(remote_shards))
        else:
            self.assertEqual(2, len(remote_shards))

        for rpc_rank, shards in remote_shards.items():
            self.assertEqual(1, len(shards))

            for remote_shard in shards:
                self.assertEqual(rpc_rank, remote_shard.owner().id)
                shard = remote_shard.to_here()
                self.assertEqual((5, 5), shard.tensor.size())

    @skipIfRocm
    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_new_group(self):
        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="rank:1/cuda:1",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="rank:3/cuda:3",
                ),
            ]
        )

        pg = dist.new_group(ranks=[1, 2, 3])

        st = sharded_tensor.empty(spec, 10, 5, process_group=pg, init_rrefs=True)
        self.assertEqual((10, 5), st.size())
        if self.rank == 1 or self.rank == 3:
            # Verify local shard.
            local_shard = st.local_shards()[0]
            self.assertEqual(
                torch.device(f"cuda:{self.rank}"), local_shard.tensor.device
            )
            self.assertEqual((5, 5), local_shard.tensor.size())

            # Verify local shard metadata.
            self.assertEqual(
                (self.rank // 2 * 5, 0), local_shard.metadata.shard_offsets
            )
            self.assertEqual((5, 5), local_shard.metadata.shard_sizes)
            self.assertEqual(
                f"rank:{self.rank}/cuda:{self.rank}",
                str(local_shard.metadata.placement),
            )

        # Verify global metadata.
        st_metadata = st.metadata()
        shards_metadata = st_metadata.shards_metadata
        self.assertEqual(2, len(shards_metadata))
        for rank, shard_metadata in enumerate(shards_metadata):
            self.assertEqual((rank * 5, 0), shard_metadata.shard_offsets)
            self.assertEqual((5, 5), shard_metadata.shard_sizes)
            self.assertEqual(
                f"rank:{rank * 2 + 1}/cuda:{rank * 2 + 1}",
                str(shard_metadata.placement),
            )

        # Validate remote shards.
        remote_shards = st.remote_shards()
        if self.rank == 1 or self.rank == 3:
            self.assertEqual(1, len(remote_shards))
        else:
            self.assertEqual(2, len(remote_shards))

        for rpc_rank, shards in remote_shards.items():
            self.assertEqual(1, len(shards))

            for remote_shard in shards:
                self.assertEqual(rpc_rank, remote_shard.owner().id)
                shard = remote_shard.to_here()
                self.assertEqual((5, 5), shard.tensor.size())

    @skipIfRocm
    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_multiple_local_shards(self):
        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="rank:0/cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[0, 5],
                    shard_sizes=[5, 5],
                    placement="rank:1/cuda:1",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="rank:0/cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[5, 5],
                    shard_sizes=[5, 5],
                    placement="rank:1/cuda:1",
                ),
            ]
        )

        st = sharded_tensor.empty(spec, 10, 10, init_rrefs=True)
        self.assertEqual((10, 10), st.size())

        if self.rank <= 1:
            self.assertEqual(2, len(st.local_shards()))

            # Verify local shards.
            for idx, local_shard in enumerate(st.local_shards()):
                self.assertEqual(
                    torch.device(f"cuda:{self.rank}"), local_shard.tensor.device
                )
                self.assertEqual((5, 5), local_shard.tensor.size())

                # Verify local shard metadata.
                self.assertEqual(
                    (idx * 5, self.rank * 5), local_shard.metadata.shard_offsets
                )
                self.assertEqual((5, 5), local_shard.metadata.shard_sizes)
                self.assertEqual(
                    f"rank:{self.rank}/cuda:{self.rank}",
                    str(local_shard.metadata.placement),
                )
        else:
            self.assertEqual(0, len(st.local_shards()))

        # Verify global metadata.
        st_metadata = st.metadata()
        shards_metadata = st_metadata.shards_metadata
        self.assertEqual(4, len(shards_metadata))
        for shard_rank, shard_metadata in enumerate(shards_metadata):
            self.assertEqual(
                (shard_rank // 2 * 5, (shard_rank % 2) * 5),
                shard_metadata.shard_offsets,
            )
            self.assertEqual((5, 5), shard_metadata.shard_sizes)
            self.assertEqual(
                f"rank:{shard_rank % 2}/cuda:{shard_rank % 2}",
                str(shard_metadata.placement),
            )

        # Validate remote shards.
        remote_shards = st.remote_shards()
        if self.rank <= 1:
            self.assertEqual(1, len(remote_shards))
        else:
            self.assertEqual(2, len(remote_shards))

        for rpc_rank, shards in remote_shards.items():
            self.assertEqual(2, len(shards))
            for remote_shard in shards:
                self.assertEqual(rpc_rank, remote_shard.owner().id)
                shard = remote_shard.to_here()
                self.assertEqual((5, 5), shard.tensor.size())

    @skipIfRocm
    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_with_rpc_names(self):
        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="worker0/cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[0, 5],
                    shard_sizes=[5, 5],
                    placement="worker1/cuda:1",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="worker2/cuda:2",
                ),
                ShardMetadata(
                    shard_offsets=[5, 5],
                    shard_sizes=[5, 5],
                    placement="worker3/cuda:3",
                ),
            ]
        )

        st = sharded_tensor.empty(spec, 10, 10, init_rrefs=True)
        self.assertEqual((10, 10), st.size())
        self.assertEqual(1, len(st.local_shards()))

        # Verify local shard.
        local_shard = st.local_shards()[0]
        self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.tensor.device)
        self.assertEqual((5, 5), local_shard.tensor.size())

        # Verify local shard metadata.
        self.assertEqual(
            (self.rank // 2 * 5, (self.rank % 2) * 5),
            local_shard.metadata.shard_offsets,
        )
        self.assertEqual((5, 5), local_shard.metadata.shard_sizes)
        self.assertEqual(
            f"worker{self.rank}/cuda:{self.rank}", str(local_shard.metadata.placement)
        )

        # Verify global metadata.
        st_metadata = st.metadata()
        shards_metadata = st_metadata.shards_metadata
        self.assertEqual(4, len(shards_metadata))
        for rank, shard_metadata in enumerate(shards_metadata):
            self.assertEqual(
                (rank // 2 * 5, (rank % 2) * 5), shard_metadata.shard_offsets
            )
            self.assertEqual((5, 5), shard_metadata.shard_sizes)
            self.assertEqual(f"worker{rank}/cuda:{rank}", str(shard_metadata.placement))

        # Validate remote shards.
        remote_shards = st.remote_shards()
        self.assertEqual(3, len(remote_shards))

        for rpc_rank, shards in remote_shards.items():
            self.assertEqual(1, len(shards))
            for remote_shard in shards:
                self.assertEqual(rpc_rank, remote_shard.owner().id)
                shard = remote_shard.to_here()
                self.assertEqual((5, 5), shard.tensor.size())


class TestShardedTensorFromLocalTensor(ShardedTensorTestBase):
    def _generate_st_from_chunk_local_tensor(self, st_size, sharding_spec):
        tensor_meta = sharding_spec.build_metadata(st_size, TensorProperties())
        pg = dist.distributed_c10d._get_default_group()

        local_tensor = None
        local_shard_metadata = None
        rank_to_metadata = {}
        for shard_metadata in tensor_meta.shards_metadata:
            rank, device = _parse_and_validate_remote_device(
                pg, shard_metadata.placement
            )
            rank_to_metadata[rank] = shard_metadata
            if rank == self.rank:
                local_tensor = torch.rand(shard_metadata.shard_sizes).cuda(device)
                local_shard_metadata = shard_metadata

        # TODO: figure out what the API should behave when some rank have no shard
        # see https://github.com/pytorch/pytorch/issues/73133
        if local_tensor is None:
            raise AssertionError("Expected local_tensor to not be None")
        st = ShardedTensor._init_from_local_tensor(
            local_tensor,
            sharding_spec,
            st_size,
            init_rrefs=True,
        )
        self.assertEqual(tuple(st_size), st.size())
        self.assertEqual(1, len(st.local_shards()))

        # Verify local shard.
        local_shard = st.local_shards()[0]
        self.assertEqual(st.local_tensor(), local_tensor)
        self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.tensor.device)

        # Verify local shard metadata.
        self.assertEqual(
            local_shard_metadata.shard_offsets, local_shard.metadata.shard_offsets
        )
        self.assertEqual(
            local_shard_metadata.shard_sizes, local_shard.metadata.shard_sizes
        )
        self.assertEqual(local_shard_metadata.placement, local_shard.metadata.placement)

        # Verify global metadata.
        st_shards_metadata = st.metadata().shards_metadata
        self.assertEqual(self.world_size, len(st_shards_metadata))
        self.assertEqual(tensor_meta.shards_metadata, st_shards_metadata)

        # Validate remote shards.
        remote_shards = st.remote_shards()
        self.assertEqual(self.world_size - 1, len(remote_shards))
        for rpc_rank, shards in remote_shards.items():
            self.assertEqual(1, len(shards))
            for remote_shard in shards:
                self.assertEqual(rpc_rank, remote_shard.owner().id)
                # If remote shard does not exist, to_here() will throw exception.
                if tensor_meta.shards_metadata[rpc_rank]:
                    shard = remote_shard.to_here()
                    self.assertEqual(
                        rank_to_metadata[rpc_rank].shard_sizes, shard.tensor.size()
                    )

    @skipIfRocm
    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_init_from_local_tensor(self):
        chunk_specs = _chunk_sharding_specs_list_for_test([0, 1, 1, 0], seed=31)
        for spec in chunk_specs:
            self._generate_st_from_chunk_local_tensor([20, 10], spec)
            self._generate_st_from_chunk_local_tensor([21, 11], spec)
            self._generate_st_from_chunk_local_tensor([23, 16], spec)
            self._generate_st_from_chunk_local_tensor([44, 16, 8], spec)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_init_from_local_tensor_errors(self):
        enumerable_sharding_spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="rank:0/cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="rank:1/cuda:1",
                ),
            ]
        )
        st_size = [24, 12]
        local_tensor = torch.rand(*st_size).cuda(self.rank)
        with self.assertRaisesRegex(ValueError, "do not cover the entire tensor"):
            ShardedTensor._init_from_local_tensor(
                local_tensor,
                enumerable_sharding_spec,
                st_size,
            )
        chunk_specs = _chunk_sharding_specs_list_for_test([0], seed=31)
        with self.assertRaisesRegex(
            ValueError, "local_tensor is not a contiguous Tensor."
        ):
            ShardedTensor._init_from_local_tensor(
                local_tensor.t(),
                chunk_specs[0],
                st_size,
            )


class TestShardedTensorFromLocalShards(ShardedTensorTestBase):
    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_local_shards(self):
        shard_offsets = [(self.rank // 2) * 5, (self.rank % 2) * 5]
        local_shard_metadata = ShardMetadata(
            shard_offsets=shard_offsets,
            shard_sizes=[5, 5],
            placement=f"rank:{self.rank}/cuda:{self.rank}",
        )

        local_tensor = torch.randn(5, 5, device=f"cuda:{self.rank}")
        local_shard = sharded_tensor.Shard(local_tensor, local_shard_metadata)
        local_shard_from_offsets = sharded_tensor.Shard.from_tensor_and_offsets(
            local_tensor, shard_offsets=shard_offsets, rank=self.rank
        )
        self.assertEqual(local_shard.metadata, local_shard_from_offsets.metadata)

        wrong_local_shard_metadata = ShardMetadata(
            shard_offsets=shard_offsets,
            shard_sizes=[6, 5],
            placement=f"rank:{self.rank}/cuda:{self.rank}",
        )
        with self.assertRaisesRegex(ValueError, "Shard tensor size does not match"):
            sharded_tensor.Shard(local_tensor, metadata=wrong_local_shard_metadata)

    @skipIfRocm
    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_init_from_local_shards(self):
        local_shard_metadata = ShardMetadata(
            shard_offsets=[(self.rank // 2) * 5, (self.rank % 2) * 5],
            shard_sizes=[5, 5],
            placement=f"rank:{self.rank}/cuda:{self.rank}",
        )

        local_shards = [
            sharded_tensor.Shard(
                torch.randn(5, 5, device=f"cuda:{self.rank}"), local_shard_metadata
            )
        ]

        st = sharded_tensor.init_from_local_shards(
            local_shards, [10, 10], init_rrefs=True
        )
        self.assertEqual((10, 10), st.size())
        self.assertEqual(1, len(st.local_shards()))

        # Verify local shard.
        local_shard = st.local_shards()[0]
        self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.tensor.device)
        self.assertEqual((5, 5), local_shard.tensor.size())

        # Verify local shard metadata.
        self.assertEqual(
            (self.rank // 2 * 5, (self.rank % 2) * 5),
            local_shard.metadata.shard_offsets,
        )
        self.assertEqual((5, 5), local_shard.metadata.shard_sizes)
        self.assertEqual(
            f"rank:{self.rank}/cuda:{self.rank}", str(local_shard.metadata.placement)
        )

        # Verify global metadata.
        shards_metadata = st.metadata().shards_metadata
        self.assertEqual(4, len(shards_metadata))
        for rank, shard_metadata in enumerate(shards_metadata):
            self.assertEqual(
                (rank // 2 * 5, (rank % 2) * 5), shard_metadata.shard_offsets
            )
            self.assertEqual((5, 5), shard_metadata.shard_sizes)
            self.assertEqual(f"rank:{rank}/cuda:{rank}", str(shard_metadata.placement))

        # Validate remote shards.
        remote_shards = st.remote_shards()
        self.assertEqual(3, len(remote_shards))

        for rpc_rank, shards in remote_shards.items():
            self.assertEqual(1, len(shards))
            for remote_shard in shards:
                self.assertEqual(rpc_rank, remote_shard.owner().id)
                shard = remote_shard.to_here()
                self.assertEqual((5, 5), shard.tensor.size())

    @skipIfRocm
    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_recalc_for_metadata(self):
        shard_sizes = [0, 5]  # test 2 different shard sizes
        for shard_size in shard_sizes:
            local_shard_metadata = ShardMetadata(
                shard_offsets=[0, 0],
                shard_sizes=[shard_size, shard_size],
                placement=f"rank:{self.rank}/cuda:{self.rank}",
            )

            local_shards = [
                sharded_tensor.Shard(
                    torch.randn(shard_size, shard_size, device=f"cuda:{self.rank}"),
                    local_shard_metadata,
                )
            ]

            st = sharded_tensor.init_from_local_shards(local_shards, None, None)
            self.assertEqual((shard_size * 4, shard_size), st.size())
            self.assertEqual(1, len(st.local_shards()))

            # Verify local shard.
            local_shard = st.local_shards()[0]
            self.assertEqual(
                torch.device(f"cuda:{self.rank}"), local_shard.tensor.device
            )
            self.assertEqual((shard_size, shard_size), local_shard.tensor.size())

            # Verify local shard metadata.
            self.assertEqual(
                (self.rank * shard_size, 0),
                local_shard.metadata.shard_offsets,
            )
            self.assertEqual((shard_size, shard_size), local_shard.metadata.shard_sizes)
            self.assertEqual(
                f"rank:{self.rank}/cuda:{self.rank}",
                str(local_shard.metadata.placement),
            )

            # Verify global metadata.
            shards_metadata = st.metadata().shards_metadata
            self.assertEqual(4, len(shards_metadata))
            for rank, shard_metadata in enumerate(shards_metadata):
                self.assertEqual((rank * shard_size, 0), shard_metadata.shard_offsets)
                self.assertEqual((shard_size, shard_size), shard_metadata.shard_sizes)
                self.assertEqual(
                    f"rank:{rank}/cuda:{rank}", str(shard_metadata.placement)
                )

            with self.assertRaises(ValueError):
                st = sharded_tensor.init_from_local_shards(local_shards)

    @skipIfRocm
    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_init_from_local_shards_with_different_glb_size(self):
        wrong_offset_local_shard_metadata = ShardMetadata(
            shard_offsets=[0, 0],
            shard_sizes=[5, 5],
            placement=f"rank:{self.rank}/cuda:{self.rank}",
        )

        wrong_offset_local_shards = [
            sharded_tensor.Shard(
                torch.randn(5, 5, device=f"cuda:{self.rank}"),
                wrong_offset_local_shard_metadata,
            )
        ]
        with self.assertRaises(ValueError):
            sharded_tensor.init_from_local_shards(wrong_offset_local_shards, 0, 0)

        local_shard_metadata = ShardMetadata(
            shard_offsets=[self.rank * 5, 0],
            shard_sizes=[5, 5],
            placement=f"rank:{self.rank}/cuda:{self.rank}",
        )

        local_shards = [
            sharded_tensor.Shard(
                torch.randn(5, 5, device=f"cuda:{self.rank}"), local_shard_metadata
            )
        ]
        with self.assertRaises(ValueError):
            sharded_tensor.init_from_local_shards(local_shards, 0, 0)

    @skipIfRocm
    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_non_rw_sharded_recalc_for_metadata(self):
        local_shard_metadata = ShardMetadata(
            shard_offsets=[(self.rank // 2) * 5, (self.rank % 2) * 5],
            shard_sizes=[5, 5],
            placement=f"rank:{self.rank}/cuda:{self.rank}",
        )

        local_shards = [
            sharded_tensor.Shard(
                torch.randn(5, 5, device=f"cuda:{self.rank}"), local_shard_metadata
            )
        ]

        st = sharded_tensor.init_from_local_shards(local_shards, None, 5)
        if self.rank == 0:
            self.assertEqual(
                st.local_shards()[0].metadata.shard_offsets,
                local_shard_metadata.shard_offsets,
            )
        else:
            self.assertNotEqual(
                st.local_shards()[0].metadata.shard_offsets,
                local_shard_metadata.shard_offsets,
            )
        self.assertEqual(
            st.local_shards()[0].metadata.shard_sizes, local_shard_metadata.shard_sizes
        )
        self.assertEqual(
            st.local_shards()[0].metadata.placement, local_shard_metadata.placement
        )

    @skip_if_lt_x_gpu(4)
    def test_st_base_init_from_local_shards_and_global_metadata(self):
        world_size = 4
        shards_metadata = []
        shards = []
        for rank in range(world_size):
            local_shard_metadata = ShardMetadata(
                shard_offsets=[(rank // 2) * 5, (rank % 2) * 5],
                shard_sizes=[5, 5],
                placement=f"rank:{rank}/cuda:{rank}",
            )
            shards_metadata.append(local_shard_metadata)
            shards.append(
                sharded_tensor.Shard(
                    torch.randn(5, 5, device=f"cuda:{rank}"), local_shard_metadata
                )
            )

        tensor_properties = TensorProperties(
            dtype=torch.get_default_dtype(),
            layout=torch.strided,
            requires_grad=False,
            memory_format=torch.contiguous_format,
            pin_memory=False,
        )

        sharded_tensor_metadata = sharded_tensor.ShardedTensorMetadata(
            shards_metadata=shards_metadata,
            size=torch.Size([10, 10]),
            tensor_properties=tensor_properties,
        )

        st_base = sharded_tensor.ShardedTensorBase._init_from_local_shards_and_global_metadata(
            shards, sharded_tensor_metadata=sharded_tensor_metadata
        )
        self.assertEqual(4, len(st_base.local_shards()))

        # Verify local shard of st_base
        local_shard = st_base.local_shards()[0]
        self.assertEqual(torch.device("cuda:0"), local_shard.tensor.device)
        self.assertEqual((5, 5), local_shard.tensor.size())

        # Verify local shard metadata.
        self.assertEqual(
            (0, 0),
            local_shard.metadata.shard_offsets,
        )
        self.assertEqual((5, 5), local_shard.metadata.shard_sizes)
        self.assertEqual("rank:0/cuda:0", str(local_shard.metadata.placement))

        # Verify global metadata.
        shards_metadata = st_base.metadata().shards_metadata
        self.assertEqual(4, len(shards_metadata))
        for rank, shard_metadata in enumerate(shards_metadata):
            self.assertEqual(
                (rank // 2 * 5, (rank % 2) * 5), shard_metadata.shard_offsets
            )
            self.assertEqual((5, 5), shard_metadata.shard_sizes)
            self.assertEqual(f"rank:{rank}/cuda:{rank}", str(shard_metadata.placement))

    @skipIfRocm
    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_init_from_local_shards_and_global_metadata_with_all_zeros(self):
        local_shard_metadata = ShardMetadata(
            shard_offsets=[0, 0],
            shard_sizes=[0, 0],
            placement=f"rank:{self.rank}/cuda:{self.rank}",
        )

        shards_metadata = []
        for r in range(self.world_size):
            if r == self.rank:
                shards_metadata.append(local_shard_metadata)
            else:
                shards_metadata.append(
                    ShardMetadata(
                        shard_offsets=[0, 0],
                        shard_sizes=[0, 0],
                        placement=f"rank:{r}/cuda:{r}",
                    )
                )

        local_shards = [
            sharded_tensor.Shard(
                torch.randn(0, 0, device=f"cuda:{self.rank}"), local_shard_metadata
            )
        ]

        tensor_properties = TensorProperties(
            dtype=torch.get_default_dtype(),
            layout=torch.strided,
            requires_grad=False,
            memory_format=torch.contiguous_format,
            pin_memory=False,
        )

        sharded_tensor_metadata = sharded_tensor.ShardedTensorMetadata(
            shards_metadata=shards_metadata,
            size=torch.Size([0, 0]),
            tensor_properties=tensor_properties,
        )

        st = ShardedTensor._init_from_local_shards_and_global_metadata(
            local_shards,
            sharded_tensor_metadata,
        )

        self.assertEqual((0, 0), st.size())
        self.assertEqual(1, len(st.local_shards()))

        # Verify local shard.
        local_shard = st.local_shards()[0]
        self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.tensor.device)
        self.assertEqual((0, 0), local_shard.tensor.size())

        # Verify local shard metadata.
        self.assertEqual(
            (0, 0),
            local_shard.metadata.shard_offsets,
        )
        self.assertEqual((0, 0), local_shard.metadata.shard_sizes)
        self.assertEqual(
            f"rank:{self.rank}/cuda:{self.rank}", str(local_shard.metadata.placement)
        )

        # Verify global metadata.
        shards_metadata = st.metadata().shards_metadata
        self.assertEqual(4, len(shards_metadata))
        for rank, shard_metadata in enumerate(shards_metadata):
            self.assertEqual((0, 0), shard_metadata.shard_offsets)
            self.assertEqual((0, 0), shard_metadata.shard_sizes)
            self.assertEqual(f"rank:{rank}/cuda:{rank}", str(shard_metadata.placement))

    @skipIfRocm
    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_init_from_local_shards_and_global_metadata_with_local_view(self):
        # testing cases where we create ST with local view, meaning we initialize other rank's metadata with 0s
        shard_offsets = [0, 1]  # valid, invalid
        for shard_offset in shard_offsets:
            local_shard_metadata = ShardMetadata(
                shard_offsets=[shard_offset, 0],
                shard_sizes=[5, 5],
                placement=f"rank:{self.rank}/cuda:{self.rank}",
            )

            shards_metadata = []
            for r in range(self.world_size):
                if r == self.rank:
                    shards_metadata.append(local_shard_metadata)
                else:
                    shards_metadata.append(
                        ShardMetadata(
                            shard_offsets=[0 if r < self.rank else 5, 0],
                            shard_sizes=[0, 0],
                            placement=f"rank:{r}/cuda:{r}",
                        )
                    )

            local_shards = [
                sharded_tensor.Shard(
                    torch.randn(5, 5, device=f"cuda:{self.rank}"), local_shard_metadata
                )
            ]

            tensor_properties = TensorProperties(
                dtype=torch.get_default_dtype(),
                layout=torch.strided,
                requires_grad=False,
                memory_format=torch.contiguous_format,
                pin_memory=False,
            )

            sharded_tensor_metadata = sharded_tensor.ShardedTensorMetadata(
                shards_metadata=shards_metadata,
                size=torch.Size([5, 5]),
                tensor_properties=tensor_properties,
            )
            if shard_offset == 0:
                # valid case
                st = ShardedTensor._init_from_local_shards_and_global_metadata(
                    local_shards,
                    sharded_tensor_metadata,
                )
            else:
                # invalid case
                with self.assertRaises(ValueError):
                    ShardedTensor._init_from_local_shards_and_global_metadata(
                        local_shards,
                        sharded_tensor_metadata,
                    )
                return

            self.assertEqual((5, 5), st.size())
            self.assertEqual(1, len(st.local_shards()))

            # Verify local shard.
            local_shard = st.local_shards()[0]
            self.assertEqual(
                torch.device(f"cuda:{self.rank}"), local_shard.tensor.device
            )
            self.assertEqual((5, 5), local_shard.tensor.size())

            # Verify local shard metadata.
            self.assertEqual(
                (0, 0),
                local_shard.metadata.shard_offsets,
            )
            self.assertEqual((5, 5), local_shard.metadata.shard_sizes)
            self.assertEqual(
                f"rank:{self.rank}/cuda:{self.rank}",
                str(local_shard.metadata.placement),
            )

            # Verify global metadata.
            shards_metadata = st.metadata().shards_metadata
            self.assertEqual(4, len(shards_metadata))
            for rank, shard_metadata in enumerate(shards_metadata):
                self.assertEqual(
                    (0 if rank <= self.rank else 5, 0), shard_metadata.shard_offsets
                )
                if rank == self.rank:
                    self.assertEqual((5, 5), shard_metadata.shard_sizes)
                else:
                    self.assertEqual((0, 0), shard_metadata.shard_sizes)
                self.assertEqual(
                    f"rank:{rank}/cuda:{rank}", str(shard_metadata.placement)
                )

    @skipIfRocm
    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_init_from_local_shards_and_global_metadata(self):
        local_shard_metadata = ShardMetadata(
            shard_offsets=[(self.rank // 2) * 5, (self.rank % 2) * 5],
            shard_sizes=[5, 5],
            placement=f"rank:{self.rank}/cuda:{self.rank}",
        )

        shards_metadata = []
        for r in range(self.world_size):
            if r == self.rank:
                shards_metadata.append(local_shard_metadata)
            else:
                shards_metadata.append(
                    ShardMetadata(
                        shard_offsets=[(r // 2) * 5, (r % 2) * 5],
                        shard_sizes=[5, 5],
                        placement=f"rank:{r}/cuda:{r}",
                    )
                )

        local_shards = [
            sharded_tensor.Shard(
                torch.randn(5, 5, device=f"cuda:{self.rank}"), local_shard_metadata
            )
        ]

        tensor_properties = TensorProperties(
            dtype=torch.get_default_dtype(),
            layout=torch.strided,
            requires_grad=False,
            memory_format=torch.contiguous_format,
            pin_memory=False,
        )

        sharded_tensor_metadata = sharded_tensor.ShardedTensorMetadata(
            shards_metadata=shards_metadata,
            size=torch.Size([10, 10]),
            tensor_properties=tensor_properties,
        )

        st = ShardedTensor._init_from_local_shards_and_global_metadata(
            local_shards,
            sharded_tensor_metadata,
            init_rrefs=True,
        )
        self.assertEqual((10, 10), st.size())
        self.assertEqual(1, len(st.local_shards()))

        # Verify local shard.
        local_shard = st.local_shards()[0]
        self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.tensor.device)
        self.assertEqual((5, 5), local_shard.tensor.size())

        # Verify local shard metadata.
        self.assertEqual(
            (self.rank // 2 * 5, (self.rank % 2) * 5),
            local_shard.metadata.shard_offsets,
        )
        self.assertEqual((5, 5), local_shard.metadata.shard_sizes)
        self.assertEqual(
            f"rank:{self.rank}/cuda:{self.rank}", str(local_shard.metadata.placement)
        )

        # Verify global metadata.
        shards_metadata = st.metadata().shards_metadata
        self.assertEqual(4, len(shards_metadata))
        for rank, shard_metadata in enumerate(shards_metadata):
            self.assertEqual(
                (rank // 2 * 5, (rank % 2) * 5), shard_metadata.shard_offsets
            )
            self.assertEqual((5, 5), shard_metadata.shard_sizes)
            self.assertEqual(f"rank:{rank}/cuda:{rank}", str(shard_metadata.placement))

        # Validate remote shards.
        remote_shards = st.remote_shards()
        self.assertEqual(3, len(remote_shards))

        for rpc_rank, shards in remote_shards.items():
            self.assertEqual(1, len(shards))
            for remote_shard in shards:
                self.assertEqual(rpc_rank, remote_shard.owner().id)
                shard = remote_shard.to_here()
                self.assertEqual((5, 5), shard.tensor.size())

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_init_from_local_shards_new_group(self):
        new_pg = dist.new_group(ranks=[1, 2, 3])

        if self.rank != 0:
            local_shard_metadata = ShardMetadata(
                shard_offsets=[5 * (self.rank - 1), 0],
                shard_sizes=[5, 5],
                placement=f"rank:{self.rank}/cuda:{self.rank}",
            )
            local_shards = [
                sharded_tensor.Shard(
                    torch.randn(5, 5, device=f"cuda:{self.rank}"), local_shard_metadata
                )
            ]

            st = sharded_tensor.init_from_local_shards(
                local_shards, [15, 5], process_group=new_pg
            )

            # Verify local shard.
            local_shard = st.local_shards()[0]
            self.assertEqual(
                torch.device(f"cuda:{self.rank}"), local_shard.tensor.device
            )
            self.assertEqual((5, 5), local_shard.tensor.size())

            # Verify local shard metadata.
            self.assertEqual(
                ((self.rank - 1) * 5, 0), local_shard.metadata.shard_offsets
            )
            self.assertEqual((5, 5), local_shard.metadata.shard_sizes)
            self.assertEqual(
                f"rank:{self.rank}/cuda:{self.rank}",
                str(local_shard.metadata.placement),
            )

            # Verify global metadata.
            st_metadata = st.metadata()
            shards_metadata = st_metadata.shards_metadata
            self.assertEqual(3, len(shards_metadata))
            for rank, shard_metadata in enumerate(shards_metadata):
                self.assertEqual((rank * 5, 0), shard_metadata.shard_offsets)
                self.assertEqual((5, 5), shard_metadata.shard_sizes)
                self.assertEqual(
                    f"rank:{rank + 1}/cuda:{rank + 1}", str(shard_metadata.placement)
                )

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_init_from_local_shards_invalid_local_shards(self):
        local_shard_metadata = ShardMetadata(
            shard_offsets=[(self.rank // 2) * 5, (self.rank % 2) * 5],
            shard_sizes=[5, 5],
            placement=f"rank:{self.rank}/cuda:{self.rank}",
        )

        indices = [[0, 1, 1], [2, 0, 2]]
        values = [3.2, 4.5, 5.8]
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, (5, 5), device=f"cuda:{self.rank}"
        )

        empty_local_shards = []
        with self.assertRaisesRegex(ValueError, "have no local shards on all ranks"):
            sharded_tensor.init_from_local_shards(
                empty_local_shards, [10, 10], init_rrefs=True
            )

        wrong_layout_shards = [
            sharded_tensor.Shard(sparse_tensor, local_shard_metadata)
        ]
        with self.assertRaisesRegex(
            ValueError, "Only torch.strided layout is currently supported"
        ):
            sharded_tensor.init_from_local_shards(
                wrong_layout_shards, [10, 10], init_rrefs=True
            )

        wrong_memory_format_shards = [
            sharded_tensor.Shard(
                torch.randn(5, 5, device=f"cuda:{self.rank}").t(), local_shard_metadata
            )
        ]
        with self.assertRaisesRegex(
            ValueError,
            "Only torch.contiguous_format memory_format is currently supported",
        ):
            sharded_tensor.init_from_local_shards(
                wrong_memory_format_shards, [10, 10], init_rrefs=True
            )

        with self.assertRaisesRegex(ValueError, "Shard tensor size does not match"):
            sharded_tensor.Shard(
                torch.randn(2, 3, device=f"cuda:{self.rank}"), local_shard_metadata
            )

        with self.assertRaisesRegex(
            ValueError, "Local shard tensor device does not match"
        ):
            sharded_tensor.Shard(torch.randn(5, 5), local_shard_metadata)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_init_from_local_shards_invalid_property_cross_ranks(self):
        local_shard_metadata = ShardMetadata(
            shard_offsets=[(self.rank // 2) * 5, (self.rank % 2) * 5],
            shard_sizes=[5, 5],
            placement=f"rank:{self.rank}/cuda:{self.rank}",
        )
        tensor_overall_size = [10, 10] if self.rank == 0 else [10, 5]
        wrong_dtype_shards = [
            sharded_tensor.Shard(
                torch.ones(5, 5, device=f"cuda:{self.rank}"), local_shard_metadata
            )
        ]
        with self.assertRaisesRegex(
            ValueError,
            "ShardedTensor global_size property does not match from different ranks!",
        ):
            sharded_tensor.init_from_local_shards(
                wrong_dtype_shards, tensor_overall_size, init_rrefs=True
            )

        tensor_dtype = torch.int if self.rank == 0 else torch.float32
        wrong_dtype_shards = [
            sharded_tensor.Shard(
                torch.ones(5, 5, device=f"cuda:{self.rank}", dtype=tensor_dtype),
                local_shard_metadata,
            )
        ]
        with self.assertRaisesRegex(
            ValueError,
            "ShardedTensor dtype property does not match from different ranks!",
        ):
            sharded_tensor.init_from_local_shards(
                wrong_dtype_shards, [10, 10], init_rrefs=True
            )

        tensor_requires_grad = self.rank == 0
        wrong_requires_grad_shards = [
            sharded_tensor.Shard(
                torch.randn(
                    5, 5, device=f"cuda:{self.rank}", requires_grad=tensor_requires_grad
                ),
                local_shard_metadata,
            )
        ]
        with self.assertRaisesRegex(
            ValueError,
            "ShardedTensor requires_grad property does not match from different ranks!",
        ):
            sharded_tensor.init_from_local_shards(
                wrong_requires_grad_shards, [10, 10], init_rrefs=True
            )

        local_shard_metadata = ShardMetadata(
            shard_offsets=[(self.rank // 2) * 5, (self.rank % 2) * 5],
            shard_sizes=[5, 5],
            placement=f"rank:{self.rank}/cpu",
        )

    @with_comms(init_rpc=False, backend="gloo")
    @skip_if_lt_x_gpu(4)
    def test_init_from_local_shards_invalid_pin_memory(self):
        # pin memory can only be on dense cpu
        local_shard_metadata = ShardMetadata(
            shard_offsets=[(self.rank // 2) * 5, (self.rank % 2) * 5],
            shard_sizes=[5, 5],
            placement=f"rank:{self.rank}/cpu",
        )
        wrong_pin_memory_local_shards = [
            sharded_tensor.Shard(
                torch.randn(5, 5, pin_memory=True), local_shard_metadata
            ),
            sharded_tensor.Shard(
                torch.randn(5, 5, pin_memory=False), local_shard_metadata
            ),
        ]
        with self.assertRaisesRegex(
            ValueError, "Local shards' tensor pin_memory property need to be the same"
        ):
            sharded_tensor.init_from_local_shards(
                wrong_pin_memory_local_shards, [10, 10], init_rrefs=True
            )

        tensor_pin_memory = self.rank == 0
        wrong_pin_memory_shards_cross_ranks = [
            sharded_tensor.Shard(
                torch.randn(5, 5, pin_memory=tensor_pin_memory), local_shard_metadata
            )
        ]
        with self.assertRaisesRegex(
            ValueError,
            "ShardedTensor pin_memory property does not match from different ranks!",
        ):
            sharded_tensor.init_from_local_shards(
                wrong_pin_memory_shards_cross_ranks, [10, 10], init_rrefs=True
            )

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_init_from_local_shards_invalid_shards_overlap(self):
        local_shard_size = [5, 5] if self.rank != 0 else [6, 6]
        local_shard_metadata = ShardMetadata(
            shard_offsets=[(self.rank // 2) * 5, (self.rank % 2) * 5],
            shard_sizes=local_shard_size,
            placement=f"rank:{self.rank}/cuda:{self.rank}",
        )

        local_shards = [
            sharded_tensor.Shard(
                torch.randn(local_shard_size, device=f"cuda:{self.rank}"),
                local_shard_metadata,
            )
        ]

        with self.assertRaisesRegex(ValueError, "overlap"):
            sharded_tensor.init_from_local_shards(
                local_shards, [10, 10], init_rrefs=True
            )

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_init_from_local_shards_invalid_shards_gaps(self):
        local_shard_size = [5, 5] if self.rank != 0 else [4, 4]
        local_shard_metadata = ShardMetadata(
            shard_offsets=[(self.rank // 2) * 5, (self.rank % 2) * 5],
            shard_sizes=local_shard_size,
            placement=f"rank:{self.rank}/cuda:{self.rank}",
        )

        local_shards = [
            sharded_tensor.Shard(
                torch.randn(local_shard_size, device=f"cuda:{self.rank}"),
                local_shard_metadata,
            )
        ]

        with self.assertRaisesRegex(ValueError, "does not match tensor volume"):
            sharded_tensor.init_from_local_shards(
                local_shards, [10, 10], init_rrefs=True
            )

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_init_from_local_shards_and_global_metadata_invalid_shards(self):
        local_shard_metadata = ShardMetadata(
            shard_offsets=[(self.rank // 2) * 5, (self.rank % 2) * 5],
            shard_sizes=[5, 5],
            placement=f"rank:{self.rank}/cuda:{self.rank}",
        )

        shards_metadata = []
        for r in range(self.world_size):
            if r == self.rank:
                shards_metadata.append(local_shard_metadata)
            else:
                shards_metadata.append(
                    ShardMetadata(
                        shard_offsets=[(r // 2) * 5, (r % 2) * 5],
                        shard_sizes=[5, 5],
                        placement=f"rank:{r}/cuda:{r}",
                    )
                )

        tensor_properties = TensorProperties(
            dtype=torch.get_default_dtype(),
            layout=torch.strided,
            requires_grad=False,
            memory_format=torch.contiguous_format,
            pin_memory=False,
        )

        sharded_tensor_metadata = sharded_tensor.ShardedTensorMetadata(
            shards_metadata=shards_metadata,
            size=torch.Size([10, 10]),
            tensor_properties=tensor_properties,
        )

        empty_local_shards = []
        with self.assertRaisesRegex(
            RuntimeError, "does not match number of local shards metadata"
        ):
            ShardedTensor._init_from_local_shards_and_global_metadata(
                empty_local_shards, sharded_tensor_metadata
            )

        wrong_num_shards = [
            sharded_tensor.Shard(
                torch.randn(5, 5, device=f"cuda:{self.rank}"), local_shard_metadata
            ),
            sharded_tensor.Shard(
                torch.randn(5, 5, device=f"cuda:{self.rank}"), local_shard_metadata
            ),
        ]
        with self.assertRaisesRegex(
            RuntimeError, "does not match number of local shards metadata"
        ):
            ShardedTensor._init_from_local_shards_and_global_metadata(
                wrong_num_shards, sharded_tensor_metadata
            )

        with self.assertRaisesRegex(
            ValueError, "Shard tensor size does not match with metadata.shard_lengths"
        ):
            sharded_tensor.Shard(
                torch.randn(2, 3, device=f"cuda:{self.rank}"), local_shard_metadata
            )

        with self.assertRaisesRegex(
            ValueError,
            "Local shard tensor device does not match with local Shard's placement",
        ):
            sharded_tensor.Shard(torch.randn(5, 5), local_shard_metadata)

        wrong_dtype_shards = [
            sharded_tensor.Shard(
                torch.ones(5, 5, device=f"cuda:{self.rank}", dtype=torch.int),
                local_shard_metadata,
            )
        ]
        with self.assertRaisesRegex(
            ValueError, "Local shards' tensor dtype property is incompatible with"
        ):
            ShardedTensor._init_from_local_shards_and_global_metadata(
                wrong_dtype_shards, sharded_tensor_metadata
            )

        indices = [[0, 1, 1], [2, 0, 2]]
        values = [3.2, 4.5, 5.8]
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, (5, 5), device=f"cuda:{self.rank}"
        )

        wrong_layout_shards = [
            sharded_tensor.Shard(sparse_tensor, local_shard_metadata)
        ]
        with self.assertRaisesRegex(
            ValueError, "Local shards' tensor layout property is incompatible with"
        ):
            ShardedTensor._init_from_local_shards_and_global_metadata(
                wrong_layout_shards, sharded_tensor_metadata
            )

        wrong_requires_grad_shards = [
            sharded_tensor.Shard(
                torch.randn(5, 5, device=f"cuda:{self.rank}", requires_grad=True),
                local_shard_metadata,
            )
        ]
        with self.assertRaisesRegex(
            ValueError,
            "Local shards' tensor requires_grad property is incompatible with",
        ):
            ShardedTensor._init_from_local_shards_and_global_metadata(
                wrong_requires_grad_shards, sharded_tensor_metadata
            )

        wrong_memory_format_shards = [
            sharded_tensor.Shard(
                torch.randn(5, 5, device=f"cuda:{self.rank}").t(), local_shard_metadata
            )
        ]
        with self.assertRaisesRegex(
            ValueError,
            "Only torch.contiguous_format memory_format is currently supported",
        ):
            ShardedTensor._init_from_local_shards_and_global_metadata(
                wrong_memory_format_shards, sharded_tensor_metadata
            )
        # pin_memory can only be on CPU
        local_shard_metadata.placement = _remote_device(f"rank:{self.rank}/cpu")
        wrong_pin_memory_shards = [
            sharded_tensor.Shard(
                torch.randn(5, 5, pin_memory=True), local_shard_metadata
            )
        ]
        with self.assertRaisesRegex(
            ValueError, "Local shards' tensor pin_memory property is incompatible with"
        ):
            ShardedTensor._init_from_local_shards_and_global_metadata(
                wrong_pin_memory_shards, sharded_tensor_metadata
            )


class TestShardedTensorCustomOps(ShardedTensorTestBase):
    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_custom_op(self):
        @custom_sharded_op_impl(torch.asin)
        def my_sharded_asin(types, args, kwargs, process_group):
            return torch.asin(args[0].local_shards()[0].tensor)

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        st = sharded_tensor.rand(spec, 10, 10)
        res = torch.asin(st)
        self.assertEqual(res, torch.asin(st.local_shards()[0].tensor))

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_custom_op_override(self):
        t = torch.rand(10, 10).cuda(self.rank)

        from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op

        @custom_sharding_spec_op(ChunkShardingSpec, torch.nn.functional.linear)
        def my_sharded_linear(types, args, kwargs, process_group):
            return t

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        m = torch.nn.Linear(32, 16).cuda(self.rank)
        shard_parameter(m, "weight", spec)

        result = m(torch.rand(15, 32).cuda(self.rank))
        self.assertEqual(t, result)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_custom_op_errors(self):
        with self.assertRaisesRegex(TypeError, "expects signature"):

            @custom_sharded_op_impl(torch.nn.functional.linear)
            def my_op1(types, args, kwargs, process_group, random_param):
                pass

        with self.assertRaisesRegex(TypeError, "expects signature"):

            @custom_sharded_op_impl(torch.nn.functional.linear)
            def my_op2(types):
                pass


class TestShardMetadata(ShardedTensorTestBase):
    @with_comms
    @requires_nccl()
    def test_shard_metadata_init(self):
        pg = dist.distributed_c10d._get_default_group()

        md = ShardMetadata([10], [0])
        self.assertIsNone(md.placement)
        with self.assertRaisesRegex(ValueError, "remote device is None"):
            _parse_and_validate_remote_device(pg, md.placement)

        # String placement gets converted by ctor
        md = ShardMetadata([10], [0], "rank:0/cpu")
        self.assertEqual(md.placement, _remote_device("rank:0/cpu"))
        rank, device = _parse_and_validate_remote_device(pg, md.placement)
        self.assertEqual(0, rank)
        self.assertEqual(device, torch.device("cpu"))

    @with_comms
    @requires_nccl()
    def test_create_shard_with_no_placement(self):
        md = ShardMetadata([0], [10])
        shard = Shard(torch.zeros(10), md)
        self.assertIsNone(shard.metadata.placement)


class TestShardedTensorSubGroupInit(TestCase):
    @spawn_threads_and_init_comms(world_size=4)
    def test_sub_process_group_sharded_tensor_init(self):
        world_pg = dist.GroupMember.WORLD
        rank = dist.get_rank()

        sub_group_sz = 2
        sub_pg_ranks = [r for r in range(4) if r % sub_group_sz == rank % sub_group_sz]
        sub_pg = dist.new_group(
            sub_pg_ranks,
            backend=dist.get_backend(world_pg),
            use_local_synchronization=True,
        )
        dist.barrier(sub_pg)

        ShardedTensor._init_from_local_shards(
            [
                Shard(
                    tensor=torch.tensor([1, 2, 3], device="meta"),
                    metadata=ShardMetadata(
                        shard_offsets=[3 * (rank // sub_group_sz)],
                        shard_sizes=[3],
                        placement=f"rank:{rank}/meta",
                    ),
                )
            ],
            6,
            process_group=sub_pg,
        )

    @spawn_threads_and_init_comms(world_size=4)
    def test_sub_process_group_placement_validation(self):
        world_pg = dist.GroupMember.WORLD
        self.assertIsNotNone(world_pg)
        rank = dist.get_rank()

        sub_group_sz = 2
        sub_pg_ranks = [r for r in range(4) if r % sub_group_sz == rank % sub_group_sz]
        sub_pg = dist.new_group(
            sub_pg_ranks,
            backend=dist.get_backend(world_pg),
            use_local_synchronization=True,
        )
        dist.barrier(sub_pg)

        for r in sub_pg_ranks:
            _parse_and_validate_remote_device(
                sub_pg, _remote_device(f"rank:{r}/cuda:{r % sub_group_sz}")
            )


class TestCreateTensorNoProcessGroupMode(TestCase):
    def test_init_from_local_shards_and_global_metadata(self):
        st_metadata: ShardedTensorMetadata = ShardedTensorMetadata(
            shards_metadata=[
                ShardMetadata(
                    shard_offsets=[0, 0], shard_sizes=[2, 2], placement="rank:0/cpu"
                ),
                ShardMetadata(
                    shard_offsets=[2, 0], shard_sizes=[2, 2], placement="rank:1/cpu"
                ),
            ],
            size=torch.Size([4, 2]),
        )
        st_local_shards: list[Shard] = []
        for shard_metadata in st_metadata.shards_metadata:
            st_local_shards.append(
                Shard(
                    tensor=torch.zeros(
                        shard_metadata.shard_sizes,
                        device=shard_metadata.placement.device(),
                    ),
                    metadata=shard_metadata,
                )
            )

        ShardedTensorBase._init_from_local_shards_and_global_metadata(
            local_shards=st_local_shards,
            sharded_tensor_metadata=st_metadata,
        )

    def test_non_contiguous_local_shards(self):
        st_metadata: ShardedTensorMetadata = ShardedTensorMetadata(
            shards_metadata=[
                ShardMetadata(
                    shard_offsets=[0, 0], shard_sizes=[2, 2], placement="rank:0/cpu"
                ),
                ShardMetadata(
                    shard_offsets=[2, 0], shard_sizes=[2, 2], placement="rank:1/cpu"
                ),
            ],
            size=torch.Size([4, 2]),
        )
        st_local_shards: list[Shard] = []
        src = torch.randn(4, 2)
        for shard_metadata in st_metadata.shards_metadata:
            offsets = shard_metadata.shard_offsets
            sizes = shard_metadata.shard_sizes
            st_local_shards.append(
                Shard(
                    tensor=src[
                        offsets[0] : offsets[0] + sizes[0],
                        offsets[1] : offsets[1] + sizes[1],
                    ],
                    metadata=shard_metadata,
                )
            )

        ShardedTensorBase._init_from_local_shards_and_global_metadata(
            local_shards=st_local_shards,
            sharded_tensor_metadata=st_metadata,
        )


if __name__ == "__main__":
    run_tests()
