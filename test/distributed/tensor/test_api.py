# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import contextlib
import tempfile

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor._dtensor_spec import ShardOrderEntry
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorContinuousTestBase,
    DTensorTestBase,
    with_comms,
)


class MyModel(nn.Module):
    def __init__(self, n_features, n_layers, device):
        super().__init__()
        self.seq = nn.Sequential(
            *[nn.Linear(n_features, n_features, device=device) for _ in range(n_layers)]
        )

    def forward(self, x):
        return self.seq(x)

    def reset_parameters(self):
        for m in self.seq:
            m.reset_parameters()


c10d_ops = torch.ops.c10d


class DTensorAPITest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        # hard code world size to 4 as we need to test
        # at least with 2d mesh
        return 4

    @with_comms
    def test_distribute_tensor_rank(self):
        comm_mode = CommDebugMode()

        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]

        for requires_grad in [True, False]:
            tensor_to_shard = torch.randn(
                3 * self.world_size, 3, requires_grad=requires_grad
            )
            with comm_mode:
                dist_tensor = distribute_tensor(
                    tensor_to_shard, device_mesh, shard_spec
                )
                self.assertEqual(comm_mode.get_comm_counts()[c10d_ops.scatter_], 1)
            self.assertEqual(dist_tensor.size(), torch.Size([3 * self.world_size, 3]))
            local_tensor = dist_tensor.to_local()
            self.assertEqual(local_tensor.size(), torch.Size([3, 3]))
            if requires_grad:
                self.assertTrue(dist_tensor.requires_grad)
                self.assertTrue(dist_tensor.is_leaf)

        # test negative dim
        shard_minus_spec = [Shard(-1)]
        tensor_to_shard = torch.randn(3, 3 * self.world_size)
        dist_tensor = distribute_tensor(tensor_to_shard, device_mesh, shard_minus_spec)
        self.assertEqual(dist_tensor.placements[0].dim, 1)

        placement_combs = [[Shard(0)], [Shard(1)], [Replicate()]]
        # test src_data_rank == 1
        # set seed differently for each rank
        torch.manual_seed(self.rank)
        for placement in placement_combs:
            tensor_to_distribute = torch.randn(3 * self.world_size, 3 * self.world_size)
            dtensor = distribute_tensor(
                tensor_to_distribute, device_mesh, placement, src_data_rank=1
            )
            full_dtensor = dtensor.full_tensor()
            if self.rank == 1:
                self.assertEqual(full_dtensor, tensor_to_distribute)

        # test src_data_rank = None, make sure it does not have communication
        with comm_mode:
            for placement in placement_combs:
                if isinstance(placement[0], Shard):
                    shard_dim = placement[0].dim
                    shape = [3, 3]
                    shape[shard_dim] *= self.world_size
                    tensor_to_distribute = torch.randn(*shape)
                else:
                    tensor_to_distribute = torch.randn(3, 3)

                dtensor = distribute_tensor(
                    tensor_to_distribute, device_mesh, placement, src_data_rank=None
                )
                self.assertEqual(dtensor.to_local().shape, (3, 3))
        self.assertEqual(comm_mode.get_total_counts(), 0)

    @with_comms
    def test_distribute_tensor_errors(self):
        device_mesh = DeviceMesh(
            self.device_type, torch.arange(self.world_size).reshape(2, 2)
        )
        tensor_shape = [3 * self.world_size, 3 * self.world_size]
        tensor_to_distribute = torch.randn(*tensor_shape)

        with self.assertRaisesRegex(ValueError, "must have the same length"):
            shard_spec = [Shard(0)]
            distribute_tensor(tensor_to_distribute, device_mesh, shard_spec)

        with self.assertRaisesRegex(RuntimeError, "distribute leaf tensor"):
            shard_spec = [Shard(0), Shard(0)]
            global_tensor = torch.randn(*tensor_shape, requires_grad=True)
            global_tensor_to_distribute = global_tensor + 2
            distribute_tensor(global_tensor_to_distribute, device_mesh, shard_spec)

        spec = [Shard(0), Shard(1)]
        dtensor = distribute_tensor(tensor_to_distribute, device_mesh, spec)

        with self.assertRaisesRegex(ValueError, "to a different device mesh"):
            new_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
            distribute_tensor(dtensor, new_mesh, [Shard(0)])

        with self.assertRaisesRegex(ValueError, "to a different placements"):
            new_spec = [Shard(0), Replicate()]
            distribute_tensor(dtensor, device_mesh, new_spec)

    @with_comms
    def test_distribute_tensor_uneven_sharding(self):
        device_mesh = self.build_device_mesh()
        input_sizes_and_shard_dims = [
            ((self.world_size * 3 + 1, 3, 3), 0),
            ((self.world_size * 3 + 2, 3, 3), 0),
            ((3, self.world_size * 3 + 1, 3), 1),
            ((3, self.world_size * 3 + 2, 3), 1),
            ((3, 3, self.world_size * 3 + 1), 2),
            ((3, 3, self.world_size * 3 + 2), 2),
        ]
        for input_size, shard_dim in input_sizes_and_shard_dims:
            shard_spec = [Shard(shard_dim)]
            tensor_to_shard = torch.randn(input_size)
            splitted_tensor_list = list(
                torch.chunk(tensor_to_shard, self.world_size, dim=shard_dim)
            )
            dist_tensor = distribute_tensor(tensor_to_shard, device_mesh, shard_spec)
            self.assertEqual(dist_tensor.size(), torch.Size(input_size))
            local_tensor = dist_tensor.to_local()
            self.assertEqual(local_tensor, splitted_tensor_list[self.rank])

    @with_comms
    def test_distribute_module(self):
        device_mesh = self.build_device_mesh()
        # fully shard all linear modules on dim 0
        module_to_shard = MyModel(5 * self.world_size, 20, device=self.device_type)
        shard_spec = [Shard(0)]

        def shard_fn(name, module, device_mesh):
            if isinstance(module, nn.Linear):
                for name, param in module.named_parameters():
                    dist_param = torch.nn.Parameter(
                        distribute_tensor(param, device_mesh, shard_spec)
                    )
                    module.register_parameter(name, dist_param)

        sharded_module = distribute_module(module_to_shard, device_mesh, shard_fn)
        for param in sharded_module.parameters():
            self.assertIsInstance(param, DTensor)
            self.assertEqual(param.placements, shard_spec)

        replica_spec = [Replicate()]
        # fully replicate all modules without passing in partition_fn
        module_to_replicate = MyModel(5, 20, device=self.device_type)
        replica_module = distribute_module(module_to_replicate, device_mesh)
        for param in replica_module.parameters():
            self.assertIsInstance(param, DTensor)
            self.assertEqual(param.placements, replica_spec)

        # fully replicate all modules by passing in partition_fn
        def replicate_fn(name, module, device_mesh):
            if isinstance(module, nn.Linear):
                for name, param in module.named_parameters():
                    dist_param = torch.nn.Parameter(
                        distribute_tensor(param, device_mesh, replica_spec)
                    )
                    module.register_parameter(name, dist_param)

        module_to_replicate = MyModel(5, 20, device=self.device_type)
        replica_module = distribute_module(
            module_to_replicate, device_mesh, replicate_fn
        )
        for param in replica_module.parameters():
            self.assertIsInstance(param, DTensor)
            self.assertEqual(param.placements, replica_spec)

        # only shard part of module, and rest of module should be replicate
        def shard_fn(name, module, device_mesh):
            if isinstance(module, nn.Linear) and (name == "seq.0" or name == "seq.8"):
                for name, param in module.named_parameters():
                    dist_param = torch.nn.Parameter(
                        distribute_tensor(param, device_mesh, shard_spec)
                    )
                    module.register_parameter(name, dist_param)

        module_to_distribute = MyModel(5 * self.world_size, 20, device=self.device_type)
        dist_module = distribute_module(module_to_distribute, device_mesh, shard_fn)
        for name, param in dist_module.named_parameters():
            self.assertIsInstance(param, DTensor)
            if name.startswith(("seq.0", "seq.8")):
                self.assertEqual(param.placements, shard_spec)
            else:
                self.assertEqual(param.placements, replica_spec)

    @with_comms
    def test_distribute_module_input_fn_output_fn(self):
        device_mesh = self.build_device_mesh()

        # fully replicate all linear modules
        module_to_replicate = MyModel(20, 1, device=self.device_type)

        # mark input sharding on dim 0
        def input_fn(mod, inputs, device_mesh):
            return DTensor.from_local(inputs[0], device_mesh, [Shard(0)])

        def output_fn(mod, outputs, device_mesh):
            assert isinstance(outputs, DTensor)
            return outputs.to_local()

        replica_module = distribute_module(
            module_to_replicate,
            device_mesh,
            input_fn=input_fn,
            output_fn=output_fn,
        )

        input_tensor = torch.randn(5, 20, device=self.device_type)
        local_out = replica_module(input_tensor)
        self.assertIsInstance(local_out, torch.Tensor)
        self.assertNotIsInstance(local_out, DTensor)

        # full replicate (even on inputs)
        model = MyModel(10, 10, device=self.device_type)

        def replicate_input_fn(mod, inputs, device_mesh):
            return DTensor.from_local(inputs[0], device_mesh, [Replicate()])

        replica_model = distribute_module(
            model,
            device_mesh,
            input_fn=replicate_input_fn,
        )
        input = torch.randn(10, 10, requires_grad=True)
        output = replica_model(input)
        output.sum().backward()
        param_grad = next(iter(replica_model.parameters())).grad
        self.assertTrue(isinstance(param_grad, DTensor))
        self.assertTrue(isinstance(param_grad.placements[0], Replicate))

    @with_comms
    def test_distribute_module_input_fn_output_fn_warning(self):
        device_mesh = self.build_device_mesh()

        # fully replicate all linear modules
        module_to_replicate = MyModel(20, 1, device=self.device_type)

        # mark input sharding on dim 0
        def input_fn(inputs, device_mesh):
            return DTensor.from_local(inputs[0], device_mesh, [Shard(0)])

        def output_fn(outputs, device_mesh):
            assert isinstance(outputs, DTensor)
            return outputs.to_local()

        with self.assertWarnsRegex(FutureWarning, "Deprecating"):
            replica_module = distribute_module(
                module_to_replicate,
                device_mesh,
                input_fn=input_fn,
                output_fn=output_fn,
            )

        input_tensor = torch.randn(5, 20, device=self.device_type)
        local_out = replica_module(input_tensor)
        self.assertIsInstance(local_out, torch.Tensor)
        self.assertNotIsInstance(local_out, DTensor)

    @with_comms
    def test_distribute_module_casting(self):
        device_mesh = self.build_device_mesh()

        # check DTensor casting
        dt = DTensor.from_local(torch.rand(10), device_mesh, [Replicate()])
        dt = dt.to(torch.bfloat16)
        self.assertEqual(dt.dtype, torch.bfloat16)
        self.assertEqual(dt._local_tensor.dtype, torch.bfloat16)

        # check distribute_tensor casting
        dt = distribute_tensor(torch.rand(10), device_mesh, [Replicate()])
        dt = dt.to(torch.bfloat16)
        self.assertEqual(dt.dtype, torch.bfloat16)
        self.assertEqual(dt._local_tensor.dtype, torch.bfloat16)

        # check distribute_module casting
        model = MyModel(10, 10, device=self.device_type)
        replica_model = distribute_module(
            model,
            device_mesh,
        )
        replica_model = replica_model.to(torch.bfloat16)
        self.assertEqual(replica_model.seq[0].weight.dtype, torch.bfloat16)
        self.assertEqual(
            replica_model.seq[0].weight._local_tensor.dtype, torch.bfloat16
        )

        # check autocast
        # `distribute_module` is an in-place operation, so we need to create a
        # new model
        model = MyModel(10, 10, device=self.device_type)
        dt = distribute_tensor(torch.rand(10), device_mesh, [Replicate()])
        replica_model = distribute_module(
            model,
            device_mesh,
        )
        with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
            output = replica_model(dt)
        self.assertEqual(output.dtype, torch.bfloat16)

    @with_comms
    def test_distribute_module_meta(self):
        # If  the model is too big, the user may first the create entire model on the meta device and then initialize
        # it on the device in the partition function.
        device_mesh = self.build_device_mesh()

        # fully shard all parameters on dim 0
        module_to_shard = MyModel(5 * self.world_size, 20, device="meta")

        shard_spec = [Shard(0)]

        def shard_fn(name, module, device_mesh):
            for param_name, param in module._parameters.items():
                dist_param = distribute_tensor(param, device_mesh, shard_spec)
                dist_param = torch.empty_like(
                    dist_param, device=device_mesh.device_type
                )
                module.register_parameter(param_name, torch.nn.Parameter(dist_param))

        sharded_module = distribute_module(module_to_shard, device_mesh, shard_fn)
        for param in sharded_module.parameters():
            self.assertIsInstance(param, DTensor)
            self.assertFalse(param.is_meta)
            self.assertTrue(param.device.type == device_mesh.device_type)

    @with_comms
    def test_checkpoint_apis_check_partial_placement(self):
        device_mesh = self.build_device_mesh()
        tensor = torch.randn(5, 5, device=self.device_type)
        dtensor = DTensor.from_local(tensor, device_mesh, [Partial()])
        with self.assertRaisesRegex(
            ValueError, "Any checkpointing related operations are not supported for"
        ):
            dtensor.__create_write_items__("fqn", None)

        with self.assertRaisesRegex(
            ValueError, "Any checkpointing related operations are not supported for"
        ):
            dtensor.__create_chunk_list__()

        with self.assertRaisesRegex(
            ValueError, "Any checkpointing related operations are not supported for"
        ):
            dtensor.__get_tensor_shard__(0)

        # Ideally we should not allow checkpointing related operations for DTensor
        with self.assertRaisesRegex(
            dcp.api.CheckpointException,
            "Any checkpointing related operations are not supported for",
        ):
            dcp.save({"fqn": dtensor}, checkpoint_id=tempfile.mkdtemp())


class DTensorDeviceOrderAPITest(DTensorContinuousTestBase):
    world_size = 4

    @property
    def device(self):
        return f"{DTensorContinuousTestBase.device_type()}:{self.rank}"

    def build_device_mesh(self, mesh_shape=None) -> DeviceMesh:
        if mesh_shape is None:
            mesh_shape = (2, self.world_size // 2)
        return init_device_mesh(DTensorContinuousTestBase.device_type(), mesh_shape)

    def test_neither_placements_nor_shard_order(self):
        """Test that neither placements nor shard_order, use default"""
        mesh = self.build_device_mesh((2, self.world_size // 2))
        input_tensor = torch.randn(8, 6, 5, device=self.device)
        input_tensor_dt = distribute_tensor(input_tensor, mesh)
        self.assertEqual(
            input_tensor_dt.placements, [Replicate() for _ in range(mesh.ndim)]
        )
        self.assertEqual(input_tensor_dt.shard_order, ())
        input_tensor_dt.redistribute(mesh, (Shard(0), Shard(0)))
        input_tensor_dt.redistribute(mesh)
        self.assertEqual(
            input_tensor_dt.placements, [Replicate() for _ in range(mesh.ndim)]
        )
        self.assertEqual(input_tensor_dt.shard_order, ())

    @parametrize(
        "placements, shard_order_dict, should_pass",
        [
            [(Shard(0), Shard(0)), {0: [0], 1: [1]}, False],
            [(Shard(0), Shard(0)), {0: [0]}, False],
            [(Shard(0), Shard(0)), {0: [0, 1]}, True],
            [(Shard(0), Shard(0)), {0: [1, 0]}, True],
            [(Shard(1), Shard(0)), {0: [1], 1: [0]}, True],
            [(Shard(1), Shard(0)), {0: [0], 1: [1]}, False],
            [(Shard(1), Shard(2)), {1: [0], 2: [1]}, True],
            [(Replicate(), Shard(2)), {2: [1]}, True],
            [(Replicate(), Replicate()), {}, True],
            [(Shard(0), Shard(0)), {}, False],
            [(Shard(0), Shard(0)), None, True],
        ],
    )
    def test_conflict_placements_and_shard_order(
        self, placements, shard_order_dict, should_pass
    ):
        """Test that providing conflict placements and shard_order raises an error."""
        mesh = self.build_device_mesh((2, self.world_size // 2))
        input_tensor = torch.randn(8, 6, 5, device=self.device)
        test_context = (
            contextlib.nullcontext()
            if should_pass
            else self.assertRaisesRegex(
                AssertionError,
                "Conflict sharding annotation",
            )
        )
        with test_context:
            distribute_tensor(
                input_tensor, mesh, placements=placements, shard_order=shard_order_dict
            )

    @parametrize(
        "placements, expected_shard_order_tuple",
        [
            [
                (Shard(0), Shard(1)),
                (ShardOrderEntry(0, (0,)), ShardOrderEntry(1, (1,))),
            ],
            [(Shard(0), Shard(0)), (ShardOrderEntry(0, (0, 1)),)],
            [
                (Shard(1), Shard(2)),
                (ShardOrderEntry(1, (0,)), ShardOrderEntry(2, (1,))),
            ],
            [(Replicate(), Shard(2)), (ShardOrderEntry(2, (1,)),)],
            [(Replicate(), Replicate()), ()],
        ],
    )
    def test_only_placements_provided(self, placements, expected_shard_order_tuple):
        """Test that providing only placements works correctly."""
        mesh = self.build_device_mesh((2, self.world_size // 2))
        input_tensor = torch.randn(8, 6, 5, device=self.device)
        input_tensor_dt = distribute_tensor(input_tensor, mesh, placements)
        self.assertEqual(input_tensor_dt.placements, tuple(placements))
        self.assertEqual(input_tensor_dt.full_tensor(), input_tensor)
        self.assertEqual(input_tensor_dt.shard_order, expected_shard_order_tuple)

    @parametrize(
        "expected_placements, shard_order_dict",
        [
            [(Shard(0), Shard(1)), {0: [0], 1: [1]}],
            [(Shard(0), Shard(0)), {0: [0, 1]}],
            [(Shard(0), Shard(0)), {0: [1, 0]}],
            [(Shard(1), Shard(2)), {1: [0], 2: [1]}],
            [(Replicate(), Shard(2)), {2: [1]}],
            [(Replicate(), Replicate()), {}],
            [(Replicate(), Replicate()), {0: []}],  # allow empty_shard_order_sequences
        ],
    )
    def test_only_shard_order_provided(self, expected_placements, shard_order_dict):
        """Test that providing only shard_order works correctly."""
        mesh = self.build_device_mesh((2, self.world_size // 2))
        input_tensor = torch.randn(8, 6, 5, device=self.device)
        input_tensor_dt = distribute_tensor(
            input_tensor, mesh, shard_order=shard_order_dict
        )
        self.assertEqual(input_tensor_dt.placements, expected_placements)
        self.assertEqual(input_tensor_dt.full_tensor(), input_tensor)

        # all replicate tensor, test for redistribution
        input_tensor_dt = distribute_tensor(input_tensor, mesh)
        input_tensor_dt = input_tensor_dt.redistribute(
            mesh, shard_order=shard_order_dict
        )
        self.assertEqual(input_tensor_dt.placements, expected_placements)
        self.assertEqual(input_tensor_dt.full_tensor(), input_tensor)

    @parametrize(
        "placements, shard_order_dict, should_pass",
        [
            [(Shard(0), Shard(0)), {0: [1, 0]}, True],
            [None, {0: [1], 1: [0]}, True],
            [(Shard(1), Shard(2)), {1: [0], 2: [2]}, False],
            [(Shard(1), Shard(2)), {1: [0], 2: [-1]}, False],
            [(Shard(1), Shard(2)), {1: [0], -1: [1]}, True],
            [None, {1: [0, 1]}, True],
            [None, {1: [1, -3]}, False],
        ],
    )
    def test_out_of_range_shard_order(self, placements, shard_order_dict, should_pass):
        """Test that providing only shard_order works correctly."""
        mesh = self.build_device_mesh((2, self.world_size // 2))
        input_tensor = torch.randn(8, 6, 5, device=self.device)
        test_context = (
            contextlib.nullcontext()
            if should_pass
            else self.assertRaisesRegex(
                IndexError,
                "`shard_order` is out of range for placements",
            )
        )
        with test_context:
            distribute_tensor(
                input_tensor, mesh, placements=placements, shard_order=shard_order_dict
            )
        # all replicate tensor, test for redistribution
        input_tensor_dt = distribute_tensor(input_tensor, mesh)
        with test_context:
            input_tensor_dt.redistribute(
                mesh, placements=placements, shard_order=shard_order_dict
            )

    @parametrize(
        "placements, shard_order_dict, should_pass",
        [
            [(Shard(0), Shard(0)), {-3: [1, 0]}, True],
            [(Shard(0), Shard(0)), {0: [0], -3: [1]}, False],
            [(Shard(0), Shard(0)), {0: [0, 1], -3: []}, False],
            [(Shard(0), Shard(0)), {0: [1, 0]}, True],
        ],
    )
    def test_duplicated_tensor_dim_shard_order(
        self, placements, shard_order_dict, should_pass
    ):
        """Test that providing only shard_order works correctly."""
        mesh = self.build_device_mesh((2, self.world_size // 2))
        input_tensor = torch.randn(8, 6, 5, device=self.device)
        test_context = (
            contextlib.nullcontext()
            if should_pass
            else self.assertRaisesRegex(
                ValueError,
                r"both normalized tensor dim * and un-normalized tensor dim * are specified in shard_order",
            )
        )
        with test_context:
            distribute_tensor(
                input_tensor, mesh, placements=placements, shard_order=shard_order_dict
            )
        # all replicate tensor, test for redistribution
        input_tensor_dt = distribute_tensor(input_tensor, mesh)
        with test_context:
            input_tensor_dt.redistribute(
                mesh, placements=placements, shard_order=shard_order_dict
            )

    @parametrize(
        "placements, shard_order_dict, should_pass",
        [
            [(Shard(0), Shard(0)), {0: [1, 0]}, True],
            [(Shard(0), Shard(0)), {3: [1, 0]}, False],
            [None, {3: [1, 0]}, False],
        ],
    )
    def test_shard_order_out_of_tensor_rank_spec(
        self, placements, shard_order_dict, should_pass
    ):
        """Test that providing only shard_order works correctly."""
        mesh = self.build_device_mesh((2, self.world_size // 2))
        input_tensor = torch.randn(8, 6, 5, device=self.device)
        test_context = (
            contextlib.nullcontext()
            if should_pass
            else self.assertRaisesRegex(
                ValueError,
                "`shard_order` is out of range for tensor_rank",
            )
        )
        with test_context:
            distribute_tensor(
                input_tensor, mesh, placements=placements, shard_order=shard_order_dict
            )
        # all replicate tensor, test for redistribution
        input_tensor_dt = distribute_tensor(input_tensor, mesh)
        with test_context:
            input_tensor_dt.redistribute(
                mesh, placements=placements, shard_order=shard_order_dict
            )

    def test_placement_length_validation_edge_cases(self):
        """Test edge cases for placement length validation."""
        mesh = self.build_device_mesh((2, self.world_size // 2))
        input_tensor = torch.randn(8, 6, 5, device=self.device)

        # Empty placements
        with self.assertRaisesRegex(
            ValueError,
            "`placements` must have the same length",
        ):
            distribute_tensor(input_tensor, mesh, placements=[])

        # Too many placements
        with self.assertRaisesRegex(
            ValueError,
            "`placements` must have the same length",
        ):
            distribute_tensor(
                input_tensor,
                mesh,
                placements=[
                    Shard(0),
                    Shard(1),
                    Replicate(),
                ],  # mesh.ndim = 2, but 3 placements
            )

    @parametrize(
        "placements, shard_order_dict, should_pass",
        [
            [(Shard(0), Shard(2)), {0: [0], 2: [1]}, True],
            [(Shard(0), Shard(2)), None, True],
            [(Shard(-3), Shard(2)), None, True],
            [(Shard(-4), Shard(2)), None, False],
            [(Shard(-4), Shard(3)), None, False],
        ],
    )
    def test_placement_out_of_tensor_rank_spec(
        self, placements, shard_order_dict, should_pass
    ):
        """Test that providing only shard_order works correctly."""
        mesh = self.build_device_mesh((2, self.world_size // 2))
        input_tensor = torch.randn(8, 6, 5, device=self.device)
        test_context = (
            contextlib.nullcontext()
            if should_pass
            else self.assertRaisesRegex(
                ValueError,
                "`placements` is out of range for tensor_rank",
            )
        )
        with test_context:
            distribute_tensor(
                input_tensor, mesh, placements=placements, shard_order=shard_order_dict
            )
        # all replicate tensor, test for redistribution
        input_tensor_dt = distribute_tensor(input_tensor, mesh)
        with test_context:
            input_tensor_dt.redistribute(
                mesh, placements=placements, shard_order=shard_order_dict
            )

    def test_empty_shard_order_creates_replicated_dtensor(self):
        """Test that empty shard_order creates a replicated DTensor."""
        mesh = self.build_device_mesh((2, self.world_size // 2))
        input_tensor = torch.randn(8, 6, 5, device=self.device)
        empty_shard_order = {}

        dt_empty_shard_order = distribute_tensor(
            input_tensor, mesh, shard_order=empty_shard_order
        )
        expected_default_placements = (Replicate(), Replicate())
        self.assertEqual(dt_empty_shard_order.placements, expected_default_placements)
        self.assertEqual(dt_empty_shard_order.full_tensor(), input_tensor)
        # test for redistribution
        dt_empty_shard_order = dt_empty_shard_order.redistribute(mesh, shard_order={})
        self.assertEqual(dt_empty_shard_order.placements, expected_default_placements)
        self.assertEqual(dt_empty_shard_order.full_tensor(), input_tensor)

    @parametrize(
        "placements, expected_shard_order_tuple",
        [
            [
                (Shard(0), Shard(1)),
                (ShardOrderEntry(0, (0,)), ShardOrderEntry(1, (1,))),
            ],
            [(Shard(0), Shard(0)), (ShardOrderEntry(0, (0, 1)),)],
            [
                (Shard(1), Shard(2)),
                (ShardOrderEntry(1, (0,)), ShardOrderEntry(2, (1,))),
            ],
            [(Replicate(), Shard(2)), (ShardOrderEntry(2, (1,)),)],
            [(Replicate(), Replicate()), ()],
        ],
    )
    def test_redistribute_with_placements_only(
        self, placements, expected_shard_order_tuple
    ):
        """Test redistribution using placements only."""
        mesh = self.build_device_mesh((2, self.world_size // 2))
        input_tensor = torch.randn(8, 6, 5, device=self.device)
        dt_default = distribute_tensor(
            input_tensor, mesh, placements=(Replicate(), Replicate())
        )
        dt_redist_placements = dt_default.redistribute(mesh, placements)
        self.assertEqual(dt_redist_placements.placements, placements)
        self.assertEqual(dt_redist_placements.full_tensor(), input_tensor)
        self.assertEqual(dt_redist_placements.shard_order, expected_shard_order_tuple)

    @parametrize(
        "expected_placements, shard_order_dict",
        [
            [(Shard(0), Shard(1)), {0: [0], 1: [1]}],
            [(Shard(0), Shard(0)), {0: [0, 1]}],
            [(Shard(0), Shard(0)), {0: [1, 0]}],
            [(Shard(1), Shard(2)), {1: [0], 2: [1]}],
            [(Replicate(), Shard(2)), {2: [1]}],
            [(Replicate(), Replicate()), {}],
        ],
    )
    def test_redistribute_with_shard_order_only(
        self, expected_placements, shard_order_dict
    ):
        """Test redistribution using shard_order only."""
        mesh = self.build_device_mesh((2, self.world_size // 2))
        input_tensor = torch.randn(8, 6, 5, device=self.device)
        dt_default = distribute_tensor(
            input_tensor, mesh, placements=(Replicate(), Replicate())
        )
        dt_redist_shard_order = dt_default.redistribute(
            mesh, shard_order=shard_order_dict
        )
        self.assertEqual(dt_redist_shard_order.placements, expected_placements)
        self.assertEqual(dt_redist_shard_order.full_tensor(), input_tensor)

    def test_special_placement_with_shard_order(self):
        """Test special placement when specify shard_order together."""
        mesh = self.build_device_mesh((2, self.world_size // 2))
        input_tensor = torch.randn(8, 6, 5, device=self.device)
        # test _StridedShard
        dt_default = distribute_tensor(
            input_tensor,
            mesh,
            placements=(_StridedShard(0, split_factor=2), Replicate()),
        )
        # _StridedShard doesn't have shard_order
        self.assertEqual(dt_default.shard_order, ())

        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot specify both `placements` and `shard_order` when `placements` contains `_StridedShard`!",
        ):
            distribute_tensor(
                input_tensor,
                mesh,
                placements=(_StridedShard(0, split_factor=2), Replicate()),
                shard_order={0: [0]},
            )
        # all replicate tensor, test for redistribution
        input_tensor_dt = distribute_tensor(input_tensor, mesh)
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot specify both `placements` and `shard_order` when `placements` contains `_StridedShard`!",
        ):
            input_tensor_dt.redistribute(
                mesh,
                placements=(_StridedShard(0, split_factor=2), Shard(1)),
                shard_order={0: [0]},
            )

        # test Partial
        gathered_tensor = DTensor.from_local(
            input_tensor, mesh, placements=(Partial(), Shard(0))
        )
        self.assertEqual(gathered_tensor.placements, (Partial(), Shard(0)))
        self.assertEqual(gathered_tensor.shard_order, (ShardOrderEntry(0, (1,)),))

        # can redistribute to Partial from Partial
        dt_redist_shard_order = gathered_tensor.redistribute(
            mesh, placements=(Partial(), Shard(1)), shard_order={1: [1]}
        )

        # doesn't allow create new Partial
        with self.assertRaisesRegex(
            RuntimeError,
            "redistributing to Partial is for internal use only",
        ):
            gathered_tensor.redistribute(mesh, placements=(Partial(), Partial()))

        # can redistribute from Partial
        dt_redist_shard_order = gathered_tensor.redistribute(mesh, shard_order={1: [0]})
        self.assertEqual(dt_redist_shard_order.placements, (Shard(1), Replicate()))


instantiate_parametrized_tests(DTensorDeviceOrderAPITest)


if __name__ == "__main__":
    run_tests()
