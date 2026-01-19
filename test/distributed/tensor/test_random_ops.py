# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import itertools

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.tensor._random as random
from torch.distributed._local_tensor import LocalTensor, maybe_run_for_local_tensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
from torch.distributed.tensor._random import (
    is_rng_supported_mesh,
    manual_seed,
    OffsetBasedRNGTracker,
)
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
    DTensorTestBase,
    skip_if_lt_x_gpu,
    skip_unless_torch_gpu,
    with_comms,
)
from torch.utils._typing_utils import not_none


def get_generator_seed_for_device_type(device_type: str):
    from torch.distributed._local_tensor import (
        get_generator_seed_for_device_type as _get_seed,
    )

    return _get_seed(device_type)


class DistTensorRandomInitTest(DTensorTestBase):
    def _run_init_op(self, init_op, *args, **kwargs):
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]
        input_size = (8, 4)

        # NOTE: currently random initialization on gpu device has different
        # behavior from other devices. Unify the test once the behavior is unified.
        if not is_rng_supported_mesh(device_mesh):
            input_tensor = torch.randn(*input_size, device=self.device_type)
            dtensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
            local_tensor_clone = torch.clone(input_tensor)
            torch.manual_seed(self.rank)
            local_tensor_clone = init_op(local_tensor_clone, *args, **kwargs)
            torch.manual_seed(self.rank)
            dtensor = init_op(dtensor, *args, **kwargs)
            self.assertEqual(local_tensor_clone, dtensor.to_local())
        else:
            # create DTensor from Tensor
            _tensor = torch.empty(*input_size, device=self.device_type)
            dtensor = distribute_tensor(_tensor, device_mesh, [Shard(1)])

            # DTensor random init
            dtensor = init_op(dtensor, *args, **kwargs)
            local_tensor = dtensor.to_local()

            # compare with local tensors from other ranks
            for other_rank in range(self.world_size):
                if self.rank != other_rank:
                    slice_idx = (
                        slice(input_size[0]),
                        slice(
                            other_rank * input_size[1], (other_rank + 1) * input_size[1]
                        ),
                    )
                    # other rank should have a different local tensor
                    self.assertNotEqual(dtensor.full_tensor()[slice_idx], local_tensor)

    @with_comms
    def test_init_ops(self):
        self._run_init_op(
            torch.nn.init.kaiming_uniform_,
            a=0,
            mode="fan_in",
            nonlinearity="leaky_relu",
        )
        self._run_init_op(torch.nn.init.normal_, mean=1.5, std=0.8)
        self._run_init_op(torch.nn.init.uniform_, a=0, b=1.2)

        for dtype in (torch.float32, torch.float16):
            self._run_init_op(torch.rand_like, dtype=dtype)
            self._run_init_op(torch.randn_like, dtype=dtype)
            self._run_init_op(torch.randint_like, low=0, high=100, dtype=dtype)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_init_with_user_generator(self):
        device_mesh = self.build_device_mesh()
        torch.manual_seed(42)
        rng = torch.Generator(device=self.device_type).manual_seed(42)
        t1 = torch.distributed.tensor.empty(
            (8, 3), device_mesh=device_mesh, placements=[Shard(0)]
        )
        t2 = torch.distributed.tensor.empty(
            (8, 3), device_mesh=device_mesh, placements=[Shard(0)]
        )
        for i in range(2):
            # run a second time, to make sure that `rng`'s offset-state is advancing on the second usage
            torch.nn.init.uniform_(t1, 0.0, 1.0)
            torch.nn.init.uniform_(t2, 0.0, 1.0, rng)
            self.assertEqual(t1.full_tensor(), t2.full_tensor(), f"Failed at {i=}")

        # ensure that we do not cache the 'seed' from the first time we see it in DTensor
        # this is a behavior change, DTensor used to cache the generator state and not modify the original generator,
        # now it modifies the original generator instead.
        torch.manual_seed(55)
        rng.manual_seed(55)
        torch.nn.init.uniform_(t1, 0.0, 1.0)
        torch.nn.init.uniform_(t2, 0.0, 1.0, rng)
        self.assertEqual(t1.full_tensor(), t2.full_tensor())

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_meta_tensor_init(self):
        # test suite sets each rank's seed to the same value.
        # The DTensor random ops will use the same generator as the default one on the device.

        # Note: this behavior changed, and now the guideline is to set the same RNG seed on all SPMD ranks.
        torch.get_device_module(self.device_type).manual_seed(0)
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        size = [1024, 2048]
        meta_dtensor = distribute_tensor(
            torch.empty(*size, device="meta"), device_mesh, [Replicate()]
        )

        # Test 1: enable the distribute region for RNG (by default)
        self.assertTrue(meta_dtensor.is_meta)
        # Tensor meta init
        dtensor = torch.empty_like(meta_dtensor, device=self.device_type)
        dtensor.uniform_()
        # check `distribute_region_enabled` is set to True by default
        self.assertTrue(random._rng_tracker.distribute_region_enabled)

        # allgather the local tensors
        gathered_local_tensors = funcol.all_gather_tensor(
            dtensor.to_local(), gather_dim=0, group=(device_mesh, 0)
        )

        @maybe_run_for_local_tensor
        def compute_rankwise_if_local_tensor(gathered_local_tensors, rank):
            # the tensor slice on the current rank
            self_slice = slice(1024 * rank, 1024 * rank + 1024)

            # compare with local tensors from other ranks
            for other_rank in range(self.world_size):
                # the RNG result on each rank are the same because they're replicated
                if rank != other_rank:
                    # other rank should have an identical local tensor
                    other_slice = slice(1024 * other_rank, 1024 * other_rank + 1024)
                    self.assertEqual(
                        gathered_local_tensors[self_slice, :],
                        gathered_local_tensors[other_slice, :],
                    )

        compute_rankwise_if_local_tensor(gathered_local_tensors.wait(), self.rank)

        # Test 2: disable the distribute region for RNG
        self.assertTrue(meta_dtensor.is_meta)
        # Tensor meta init
        dtensor = torch.empty_like(meta_dtensor, device=self.device_type)
        random._rng_tracker.distribute_region_enabled = False
        dtensor.uniform_()
        # check `distribute_region_enabled` is set to False
        self.assertTrue(not random._rng_tracker.distribute_region_enabled)

        # allgather the local tensors
        local_tensor = funcol.all_gather_tensor(
            dtensor.to_local(), gather_dim=0, group=(device_mesh, 0)
        )

        compute_rankwise_if_local_tensor(local_tensor.wait(), self.rank)

    @with_comms
    @skip_unless_torch_gpu
    def test_tp_model_meta_init(self):
        # initialize the 1-d device mesh for TP
        tp_mesh = init_device_mesh(self.device_type, mesh_shape=(self.world_size,))

        # model meta init
        with torch.device("meta"):
            model = torch.nn.Linear(self.world_size, self.world_size, bias=False)
            self.assertEqual(model.weight.device, torch.device("meta"))
            parallelize_module(model, tp_mesh, ColwiseParallel())
            if random._rng_tracker is not None:
                random._rng_tracker.distribute_region_enabled = True

            self.assertEqual(model.weight.device, torch.device("meta"))

        # actual initialization
        device = torch.device(
            self.device_type, torch.get_device_module(self.device_type).current_device()
        )
        model.to_empty(device=device)
        model.reset_parameters()
        self.assertTrue(
            random._rng_tracker is not None
            and isinstance(random._rng_tracker, OffsetBasedRNGTracker)
        )
        self.assertEqual(model.weight.device, device)
        assert isinstance(model.weight, DTensor)

        # gather all the shards to compare initialization results
        WORLD = torch.distributed.group.WORLD
        assert WORLD is not None
        weight_local = model.weight.to_local()
        weight_gather = funcol.all_gather_tensor(
            weight_local,
            gather_dim=0,
            group=WORLD,
        )

        @maybe_run_for_local_tensor
        def compute_rankwise_if_local_tensor(weight_local, weight_gather, rank):
            # verify the weights are initialized differently on all ranks
            for other_rank in range(self.world_size):
                if rank != other_rank:
                    self.assertNotEqual(
                        weight_local,
                        weight_gather[other_rank : other_rank + 1, :],
                    )

        compute_rankwise_if_local_tensor(weight_local, weight_gather.wait(), self.rank)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_fsdp_tp_model_meta_init(self):
        # initialize the 2-d device mesh
        global_mesh = init_device_mesh(
            self.device_type,
            mesh_shape=(self.world_size // 2, 2),
            mesh_dim_names=("dp", "tp"),
        )
        dp_mesh, tp_mesh = global_mesh["dp"], global_mesh["tp"]

        # model meta init
        with torch.device("meta"):
            model = torch.nn.Linear(self.world_size, self.world_size, bias=False)
            self.assertEqual(model.weight.device, torch.device("meta"))
            parallelize_module(model, tp_mesh, ColwiseParallel())
            if random._rng_tracker is not None:
                random._rng_tracker.distribute_region_enabled = True

            fully_shard(model, mesh=dp_mesh)
            self.assertEqual(model.weight.device, torch.device("meta"))

        # actual initialization
        device = torch.device(
            self.device_type, torch.get_device_module(self.device_type).current_device()
        )
        model.to_empty(device=device)
        model.reset_parameters()
        self.assertTrue(
            random._rng_tracker is not None
            and isinstance(random._rng_tracker, OffsetBasedRNGTracker)
        )
        self.assertEqual(model.weight.device, device)
        assert isinstance(model.weight, DTensor)

        # gather all the shards to compare initialization results
        WORLD = torch.distributed.group.WORLD
        assert WORLD is not None
        weight_local = model.weight.to_local()
        weight_gather = funcol.all_gather_tensor(
            weight_local,
            gather_dim=0,
            group=WORLD,
        )

        @maybe_run_for_local_tensor
        def compute_rankwise_if_local_tensor(weight_local, weight_gather, rank):
            # verify the weights are initialized differently on all ranks
            for other_rank in range(self.world_size):
                if rank != other_rank:
                    self.assertNotEqual(
                        weight_local,
                        weight_gather[other_rank : other_rank + 1, :],
                    )

        compute_rankwise_if_local_tensor(weight_local, weight_gather.wait(), self.rank)

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_dtensor_init_helper_tensor_meta_strides(self):
        """Test that DTensorSpec.tensor_meta has correct strides in _distribute_region."""
        import torch.distributed.tensor._random as random_module

        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        captured_specs = []

        class SpecCapturingTracker:
            """Wrapper to intercept _distribute_region and capture DTensorSpec."""

            def __init__(self, tracker):
                self._tracker = tracker

            def _distribute_region(self, spec):
                captured_specs.append(spec.tensor_meta)
                return self._tracker._distribute_region(spec)

        # Initialize the RNG tracker
        torch.manual_seed(42)
        torch.distributed.tensor.randn(
            (8, 8), device_mesh=device_mesh, placements=[Shard(0)]
        )

        # Wrap tracker to capture specs
        random_module._rng_tracker = SpecCapturingTracker(random_module._rng_tracker)

        torch.distributed.tensor.randn(
            (8, 8), device_mesh=device_mesh, placements=[Shard(0)]
        )

        self.assertEqual(len(captured_specs), 1)
        self.assertEqual(captured_specs[0].shape, torch.Size([8, 8]))
        self.assertEqual(captured_specs[0].stride, (8, 1))


class DistTensorRandomOpTest(DTensorTestBase):
    @with_comms
    @skip_unless_torch_gpu
    def test_rng_tracker_init(self):
        torch.manual_seed(self.rank)
        seed_local = (
            torch.zeros_like(torch.empty(1), device=self.device_type)
            + torch.initial_seed()
        )
        torch.distributed.broadcast(seed_local, src=0)
        # if local tensor, it should automatically reconcile after the broadcast
        # since all virtual ranks should have rank 0's initial_seed()
        seed_from_rank_0 = seed_local

        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        # seed synchronization now does NOT happen after the first `distribute_tensor`
        # call
        dt = distribute_tensor(
            torch.empty([self.world_size], device=self.device_type),
            device_mesh,
            [Shard(0)],
        )
        self.assertTrue(random._rng_tracker is None)
        # seed synchronization only happens after `manual_seed` or the first DTensor
        # random op call
        dt.uniform_(0, 1)

        # We do not maintain the copy of the seed in dtensor, but we do mutate the global rng state
        # since we now always pull it fresh from the local device generator
        self.assertEqual(
            seed_from_rank_0, get_generator_seed_for_device_type(self.device_type)
        )

    @with_comms
    @skip_unless_torch_gpu
    def test_manual_seed(self):
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # in the case of calling ``torch.distributed.tensor._random.manual_seed``,
        # no seed synchronization should happen since we fully trust the users' input
        # and will not override the value.
        comm_mode = CommDebugMode()
        with comm_mode:
            # Test 1: set different seed on different ranks
            # RNG tracker should not be initialized until DTensor ``manual_seed``
            # is called.
            self.assertTrue(random._rng_tracker is None)
            manual_seed(self.rank, device_mesh)
            # RNG tracker should already be initialized
            self.assertTrue(random._rng_tracker is not None)
            self.assertEqual(
                self.rank, get_generator_seed_for_device_type(self.device_type)
            )

            # Test 2: set same seed on different ranks
            manual_seed(1234, device_mesh)
            self.assertEqual(1234, get_generator_seed_for_device_type(self.device_type))

        self.assertEqual(comm_mode.get_total_counts(), 0)

    @with_comms
    @skip_unless_torch_gpu
    def test_manual_seed_submesh(self):
        @maybe_run_for_local_tensor
        def compute_rankwise_if_local_tensor(rank):
            # the current rank is not a part of the mesh
            single_rank_device_mesh = DeviceMesh(
                self.device_type, [(rank + 1) % self.world_size], _rank=rank
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "manual_seed requires the current rank to be a part of the device mesh",
            ):
                manual_seed(rank, single_rank_device_mesh)

        compute_rankwise_if_local_tensor(self.rank)

    @with_comms
    @skip_unless_torch_gpu
    def test_pipeline_parallel_manual_seed(self):
        # This test is to verify the `manual_seed` API works as expected in the
        # pipeline parallel setting.
        world_mesh = init_device_mesh(
            self.device_type,
            (self.world_size // 2, 2),
            mesh_dim_names=("pp", "spmd"),
        )
        pp_mesh = world_mesh["pp"]
        pp_rank = pp_mesh.get_local_rank()  # rank 0,1 = 0; rank 2,3 = 1
        spmd_mesh = world_mesh["spmd"]

        # set the seed for each pipeline stage to 123 + pp_rank
        manual_seed(123 + pp_rank, spmd_mesh)
        # dtensor no longer stores a copy of the seed, but it mutates the device's generator so we can check that
        self.assertEqual(
            123 + pp_rank, get_generator_seed_for_device_type(self.device_type)
        )

        # mimic initializing a model weight sharded on the SPMD mesh
        spmd_dtensor = torch.distributed.tensor.ones(
            2 * spmd_mesh.size(), 2, device_mesh=spmd_mesh, placements=[Shard(0)]
        )
        torch.nn.init.normal_(spmd_dtensor)

        # gather all the shards to compare initialization results
        WORLD = torch.distributed.group.WORLD
        assert WORLD is not None
        tensor_gather = funcol.all_gather_tensor(
            spmd_dtensor.to_local(),
            gather_dim=0,
            group=WORLD,
        )

        # verify the weights are initialized differently on all ranks
        for other_rank in range(self.world_size):
            if self.rank != other_rank:
                self.assertNotEqual(
                    spmd_dtensor,
                    tensor_gather[2 * other_rank : 2 * (other_rank + 1), :],
                )

    @with_comms
    @skip_unless_torch_gpu
    def test_deterministic_dropout_1d(self):
        # test suite sets each rank's seed to the same value but in actual
        # execution the default random seed will be different (a random value).
        # The DTensor random ops will use the same random seed even though the
        # torch random generator keeps different seeds on ranks.
        torch.manual_seed(self.rank)
        # TODO: add test before/after enabling distribute region
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        size = [4, 4]

        dtensor = distribute_tensor(
            torch.empty(*size, device=self.device_type), device_mesh, [Shard(1)]
        )

        # a random op call shifts the offset
        dtensor.uniform_(0, 1)

        # the dtensor is now replicate on all ranks
        dtensor = dtensor.redistribute(device_mesh, [Replicate()])

        dropout = torch.nn.Dropout(p=0.2)
        dtensor = dropout(dtensor)

        # allgather the local tensors
        local_tensor = funcol.all_gather_tensor(
            dtensor.to_local(), gather_dim=0, group=(device_mesh, 0)
        )

        @maybe_run_for_local_tensor
        def compute_rankwise_if_local_tensor(local_tensor, rank):
            # compare with local tensors from other ranks
            self_slice = slice(4 * rank, 4 * rank + 4)
            for other_rank in range(self.world_size):
                if rank != other_rank:
                    # other rank should have an identical local tensor
                    other_slice = slice(4 * other_rank, 4 * other_rank + 4)
                    self.assertEqual(
                        local_tensor[self_slice, :],
                        local_tensor[other_slice, :],
                    )

        compute_rankwise_if_local_tensor(local_tensor, self.rank)

    @with_comms
    @skip_unless_torch_gpu
    def test_deterministic_rand_1d(self):
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        size = [4, 4 * self.world_size]

        for fn in [
            torch.distributed.tensor.rand,
            torch.distributed.tensor.randn,
        ]:
            dtensor = fn(size, device_mesh=device_mesh, placements=[Shard(1)])
            local_tensor = funcol.all_gather_tensor(
                dtensor.to_local(), gather_dim=0, group=(device_mesh, 0)
            )

            @maybe_run_for_local_tensor
            def compute_rankwise_if_local_tensor(local_tensor, rank):
                # compare with local tensors from other ranks
                self_slice = slice(4 * rank, 4 * rank + 4)
                for other_rank in range(self.world_size):
                    if rank != other_rank:
                        # other rank should have an identical local tensor for replicate placement
                        other_slice = slice(4 * other_rank, 4 * other_rank + 4)
                        self.assertNotEqual(
                            local_tensor[self_slice, :],
                            local_tensor[other_slice, :],
                        )

            compute_rankwise_if_local_tensor(local_tensor, self.rank)

            # we should set manual seed to the same value on all SPMD ranks
            torch.manual_seed(0)
            dtensor = fn(size, device_mesh=device_mesh, placements=[Replicate()])
            local_tensor = funcol.all_gather_tensor(
                dtensor.to_local(), gather_dim=0, group=(device_mesh, 0)
            )

            @maybe_run_for_local_tensor
            def compute_rankwise_if_local_tensor(local_tensor, rank):
                # compare with local tensors from other ranks
                self_slice = slice(4 * rank, 4 * rank + 4)
                for other_rank in range(self.world_size):
                    if rank != other_rank:
                        # other rank should have an identical local tensor for replicate placement
                        other_slice = slice(4 * other_rank, 4 * other_rank + 4)
                        self.assertEqual(
                            local_tensor[self_slice, :],
                            local_tensor[other_slice, :],
                        )

            compute_rankwise_if_local_tensor(local_tensor, self.rank)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_deterministic_uniform_2d(self):
        mesh = torch.arange(self.world_size).reshape(2, 2)
        device_mesh = DeviceMesh(self.device_type, mesh)
        dtensor = distribute_tensor(
            torch.empty(
                *[self.world_size for _ in mesh.size()], device=self.device_type
            ),
            device_mesh,
            [Replicate(), Replicate()],
        )

        placements_list = [  # this list of placements should be enough to cover
            [Shard(0), Shard(1)],
            [Shard(1), Shard(0)],
            [Shard(0), Replicate()],
            [Replicate(), Shard(0)],
            [Shard(1), Replicate()],
            [Replicate(), Shard(1)],
            [Replicate(), Replicate()],
        ]

        shard_index_list = [
            {0: 0, 1: 1, 2: 2, 3: 3},
            {0: 0, 1: 2, 2: 1, 3: 3},
            {0: 0, 1: 0, 2: 1, 3: 1},
            {0: 0, 1: 1, 2: 0, 3: 1},
            {0: 0, 1: 0, 2: 1, 3: 1},
            {0: 0, 1: 1, 2: 0, 3: 1},
            {0: 0, 1: 0, 2: 0, 3: 0},
        ]

        coordinate = device_mesh.get_coordinate()
        assert coordinate is not None

        for placements, shard_index in zip(placements_list, shard_index_list):
            dtensor = dtensor.redistribute(device_mesh, placements)

            # random op call
            dtensor.uniform_(0, 1)

            # check shard information is correct
            shard_coord = [
                coordinate[mesh_dim] if mesh_dim >= 0 else 0
                for mesh_dim in dtensor._spec.dim_map
            ]

            shard_size = [
                device_mesh.size(mesh_dim) if mesh_dim >= 0 else 1
                for mesh_dim in dtensor._spec.dim_map
            ]

            shard_linear_idx = random._rng_tracker._calc_shard_linear_idx(
                shard_coord, shard_size
            )

            @maybe_run_for_local_tensor
            def check_shard_index(shard_linear_idx, rank):
                self.assertEqual(shard_linear_idx, shard_index[rank])

            check_shard_index(shard_linear_idx, self.rank)

            # compute local size and offset
            _, local_shard_offset = compute_local_shape_and_global_offset(
                dtensor.shape, device_mesh, placements
            )

            # get the local shard size and local shard offset for each shard
            # local_shard_list_on_dim[i] has the list of all shards on that dim
            # as a tuple (local_shard_offset, local_shard_size)
            dtensor_shape = dtensor.shape
            local_shard_list_on_dim: list[list[tuple[int, int]]] = [
                [(0, l)] for l in dtensor_shape
            ]
            for idx, placement in enumerate(placements):
                if isinstance(placement, Shard):
                    mesh_dim_size = device_mesh.size(idx)
                    shard_dim = placement.dim
                    local_shard_list_on_dim[shard_dim] = []
                    for shard_idx_on_dim in range(mesh_dim_size):
                        (
                            shard_size,
                            shard_offset,
                        ) = placement._local_shard_size_and_offset(
                            dtensor_shape[shard_dim],
                            mesh_dim_size,
                            shard_idx_on_dim,
                        )
                        local_shard_list_on_dim[shard_dim].append(
                            (not_none(shard_offset), shard_size)
                        )

            local_shard_comb = itertools.product(*local_shard_list_on_dim)

            # the local shard
            local_tensor = dtensor.to_local()
            # allgather the local tensors
            full_tensor = dtensor.full_tensor()

            full_tensor = (
                full_tensor.reconcile()
                if isinstance(full_tensor, LocalTensor)
                else full_tensor
            )

            @maybe_run_for_local_tensor
            def blockwise_iter_if_localtensor(local_tensor, local_shard_offset):
                # compare local tensor with each other shard
                for other_local_shard in local_shard_comb:
                    other_local_shard_offset, _ = zip(*other_local_shard)
                    slice_idx = [
                        slice(offset, offset + size)
                        for offset, size in other_local_shard
                    ]
                    if local_shard_offset == other_local_shard_offset:
                        self.assertEqual(full_tensor[tuple(slice_idx)], local_tensor)
                    else:
                        self.assertNotEqual(full_tensor[tuple(slice_idx)], local_tensor)

            blockwise_iter_if_localtensor(local_tensor, local_shard_offset)

    def test_philox_state_seed_roundtrip(self):
        """
        Test that _PhiloxState seed can be read and re-set without error.

        This test addresses the issue where reading a seed value from the state
        (which uses uint64 view) and then re-setting it would fail with:
        OverflowError: can't convert negative int to unsigned

        The fix ensures the seed getter uses uint64 view, preventing negative
        values from appearing when the high bit is set.
        """
        from torch.distributed.tensor._random import _PhiloxState

        state = torch.zeros(16, dtype=torch.uint8, device="cpu")
        philox = _PhiloxState(state)
        test_seed = 2**63 + 42  # This has the sign bit set when viewed as int64
        philox.seed = test_seed
        philox.seed = philox.seed


class DistTensorRandomOpsTest3D(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    @skip_if_lt_x_gpu(8)
    @with_comms
    def test_hsdp_tp_model_meta_init(self):
        # initialize the 3-d device mesh
        global_mesh = init_device_mesh(
            self.device_type,
            mesh_shape=(self.world_size // 4, 2, 2),
            mesh_dim_names=("dp_replicate", "dp_shard", "tp"),
        )
        tp_mesh = global_mesh["tp"]
        dp_mesh = global_mesh["dp_replicate", "dp_shard"]

        # model meta init
        with torch.device("meta"):
            model = torch.nn.Linear(self.world_size, self.world_size, bias=False)
            self.assertEqual(model.weight.device, torch.device("meta"))
            parallelize_module(model, tp_mesh, ColwiseParallel())
            if random._rng_tracker is not None:
                random._rng_tracker.distribute_region_enabled = True

            fully_shard(model, mesh=dp_mesh)
            self.assertEqual(model.weight.device, torch.device("meta"))

        # actual initialization
        device = torch.device(
            self.device_type, torch.get_device_module(self.device_type).current_device()
        )
        model.to_empty(device=device)
        model.reset_parameters()
        self.assertTrue(
            random._rng_tracker is not None
            and isinstance(random._rng_tracker, OffsetBasedRNGTracker)
        )
        self.assertEqual(model.weight.device, device)
        assert isinstance(model.weight, DTensor)

        # gather all the shards to compare initialization results
        WORLD = torch.distributed.group.WORLD
        assert WORLD is not None
        weight_local = model.weight.to_local()
        weight_gather = funcol.all_gather_tensor(
            weight_local,
            gather_dim=0,
            group=WORLD,
        )

        weight_gather = weight_gather.wait()

        weight_gather = (
            weight_gather.reconcile()
            if isinstance(weight_gather, LocalTensor)
            else weight_gather
        )

        @maybe_run_for_local_tensor
        def compute_rankwise_if_local_tensor(weight_local, rank):
            # verify the weights are initialized differently on all ranks
            shard_dim_0_len = self.world_size // 4
            for other_rank in range(self.world_size):
                other_rank_dim_0_start = other_rank * shard_dim_0_len
                other_rank_dim_0_end = other_rank_dim_0_start + shard_dim_0_len
                if rank % 4 != other_rank % 4:
                    self.assertNotEqual(
                        weight_local,
                        weight_gather[other_rank_dim_0_start:other_rank_dim_0_end, :],
                    )
                else:
                    self.assertEqual(
                        weight_local,
                        weight_gather[other_rank_dim_0_start:other_rank_dim_0_end, :],
                    )

        compute_rankwise_if_local_tensor(weight_local, self.rank)


DistTensorRandomInitTestWithLocalTensor = create_local_tensor_test_class(
    DistTensorRandomInitTest,
)

DistTensorRandomOpTestWithLocalTensor = create_local_tensor_test_class(
    DistTensorRandomOpTest,
)

DistTensorRandomOpsTest3DWithLocalTensor = create_local_tensor_test_class(
    DistTensorRandomOpsTest3D,
)

if __name__ == "__main__":
    run_tests()
