# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch
from torch.distributed._tensor import DeviceMesh
from torch.distributed._tensor.op_schema import OpSchema

from torch.distributed._tensor.ops.common_rules import (
    einop_rule,
    pointwise_rule,
    reduction_rule,
)
from torch.distributed._tensor.placement_types import DTensorSpec, TensorMeta
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

aten = torch.ops.aten


class CommonRulesTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        # hard code world size to 4 as we need to test
        # at least with 2d mesh
        return 4

    def _gen_tensor_meta(self, shape):
        empty_tensor = torch.empty(shape)
        return TensorMeta(
            empty_tensor.shape,
            empty_tensor.stride(),
            empty_tensor.dtype,
        )

    @with_comms
    def test_einop_basic_propagation(self):
        # plain einsum, mm
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        mm_call = aten.mm.default
        # propagate col-wise sharding
        mat1, mat2 = [-1, -1], [-1, 0]

        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 4]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([4, 8]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(mm_call, (mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [-1, 0])

        # propagate row-wise sharding
        mat1, mat2 = [0, -1], [-1, -1]
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(mm_call, (mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [0, -1])

        # generate partial
        mat1, mat2 = [-1, 0], [0, -1]
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(mm_call, (mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertTrue(output_spec.placements[0].is_partial())

    @with_comms
    def test_einop_pointwise_propagation(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        add_call = aten.add.Tensor
        # addition
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 8]))
        mat1 = [0, -1]
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        output_sharding = einop_rule(
            "ij,ij->ij", OpSchema(add_call, (mat1_spec, mat1_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [0, -1])

        # broadcast addition
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 8]))
        mat1 = [-1, 0, -1]
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )

        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([2]))
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, [-1], [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = einop_rule(
            "ijk,k->ijk", OpSchema(add_call, (mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [-1, 0, -1])

        # broadcast to a common shape
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 8, 8]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([1, 8]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, [0, -1, -1], [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, [-1, -1], [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = einop_rule(
            "ijk,1k->ijk", OpSchema(add_call, (mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [0, -1, -1])

    @with_comms
    def test_einop_merge_sharding(self):
        # 2d mesh einop merge sharding
        mesh_shape = torch.arange(self.world_size).reshape(
            self.world_size // 2, self.world_size // 2
        )
        mesh = DeviceMesh(self.device_type, mesh_shape)

        mm_call = aten.mm.default

        mat1, mat2 = [0, -1], [-1, 1]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 4]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([4, 8]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(mm_call, (mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [0, 1])

    @with_comms
    def test_einop_linearity(self):
        mesh_shape = torch.arange(self.world_size).reshape(
            self.world_size // 2, self.world_size // 2
        )
        mesh = DeviceMesh(self.device_type, mesh_shape)

        mm_call = aten.mm.default

        mat1, mat2 = [0, -1], [-1, -1]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 4]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([4, 8]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [1], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        # if not turn on linearity, partial sum is not eligible to propagate, we return
        # suggestion to reshard inputs with no partial sum (i.e. all_reduce one input)
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(mm_call, (mat1_spec, mat2_spec), {})
        )
        self.assertIsNone(output_sharding.output_spec)
        suggestions = output_sharding.schema_suggestions
        self.assertIsNotNone(suggestions)
        suggested_spec = suggestions[0].args_schema[0]
        self.assertFalse(suggested_spec.placements[1].is_partial())

        # einop prop with linearity on mm, should give back suggestion
        # on converting placements to partial
        output_sharding = einop_rule(
            "mk,kn->mn",
            OpSchema(mm_call, (mat1_spec, mat2_spec), {}),
            linearity=True,
        )
        self.assertIsNone(output_sharding.output_spec)
        suggestions = output_sharding.schema_suggestions
        self.assertIsNotNone(suggestions)
        mat2_spec = suggestions[0].args_schema[1]
        # mat2 mesh dim 1 should become partial now!
        self.assertTrue(mat2_spec.placements[1].is_partial())

        # einop prop with linearity on point-wise, should give back suggestion
        # on converting placements to partial
        add_call = aten.add.Tensor
        mat1, mat2 = [0, -1], [0, -1]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 6]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([8, 6]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [1], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )

        output_sharding = einop_rule(
            "ij,ij->ij",
            OpSchema(add_call, (mat1_spec, mat2_spec), {}),
            linearity=True,
        )
        self.assertIsNone(output_sharding.output_spec)
        suggestions = output_sharding.schema_suggestions
        self.assertIsNotNone(suggestions)
        mat2_spec = suggestions[0].args_schema[1]
        # mat2 mesh dim 1 should become partial now!
        self.assertTrue(mat2_spec.placements[1].is_partial())

    @with_comms
    def test_einop_multi_sharding_on_mesh_dim(self):
        # einop prop with multi sharding on same mesh dim
        mesh_shape = torch.arange(self.world_size)
        mesh = DeviceMesh(self.device_type, mesh_shape)

        mm_call = aten.mm.default
        mat1, mat2 = [0, -1], [0, -1]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 12]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([12, 4]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(mm_call, (mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNone(output_spec)
        self.assertIsNotNone(output_sharding.schema_suggestions)

        # ensure that the suggestion is to reshard the second
        # arg by all_gather its tensor dim sharding
        schema_suggestion = output_sharding.schema_suggestions[0]
        self.assertEqual(schema_suggestion.args_schema[0].dim_map, [0, -1])
        self.assertEqual(schema_suggestion.args_schema[1].dim_map, [-1, -1])

    @with_comms
    def test_einop_errors(self):
        mesh_shape = torch.arange(self.world_size).reshape(
            self.world_size // 2, self.world_size // 2
        )
        mesh = DeviceMesh(self.device_type, mesh_shape)

        add_call = aten.add.Tensor
        mat1, mat2 = [0, -1], [1, -1]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 4]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([8, 4]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )

        with self.assertRaisesRegex(RuntimeError, "sharded two different ways:"):
            einop_rule("ij,ij->ij", OpSchema(add_call, (mat1_spec, mat2_spec), {}))

    @with_comms
    def test_pointwise_rules_broadcasting(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        where_call = aten.where.self
        inp1, inp2, inp3 = [0], [], [-1, -1]
        inp1_tensor_meta = self._gen_tensor_meta(torch.Size([8]))
        inp2_tensor_meta = self._gen_tensor_meta(torch.Size([]))
        inp3_tensor_meta = self._gen_tensor_meta(torch.Size([1, 1]))
        condition = DTensorSpec.from_dim_map(
            mesh, inp1, [], tensor_meta=inp1_tensor_meta
        )
        self_tensor = DTensorSpec.from_dim_map(
            mesh, inp2, [], tensor_meta=inp2_tensor_meta
        )
        other_tensor = DTensorSpec.from_dim_map(
            mesh, inp3, [], tensor_meta=inp3_tensor_meta
        )
        # propagate point-wise sharding with broadcasting
        output_sharding = pointwise_rule(
            OpSchema(where_call, (condition, self_tensor, other_tensor), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [-1, 0])

    @with_comms
    def test_pointwise_rules_suggestion(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        lerp_call = aten.lerp.Scalar
        # propagate point-wise sharding
        inp1, inp2 = [-1, -1], [-1, 0]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 4]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([8, 4]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, inp1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, inp2, [], tensor_meta=mat2_tensor_meta
        )
        # adding a positional argument -1 to arg schema
        output_sharding = pointwise_rule(
            OpSchema(lerp_call, (mat1_spec, mat2_spec, -1), {})
        )
        self.assertIsNone(output_sharding.output_spec)
        self.assertIsNotNone(output_sharding.schema_suggestions)

        # ensure that the suggestion from pointwise rules still have
        # the positional args that are not DTensorSpec
        schema_suggestion = output_sharding.schema_suggestions[0]
        self.assertEqual(len(schema_suggestion.args_schema), 3)
        self.assertEqual(schema_suggestion.args_schema[2], -1)

    @with_comms
    def test_pointwise_multi_sharding_on_mesh_dim(self):
        # 2d mesh pointwise sharding
        mesh_shape = torch.arange(self.world_size).reshape(
            self.world_size // 2, self.world_size // 2
        )
        mesh = DeviceMesh(self.device_type, mesh_shape)

        add_call = aten.add.Tensor

        # basic case to test implicit broadcasting shape alignment
        mat1, mat2 = [-1, 0], [0]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([20, 6]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([6]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = pointwise_rule(OpSchema(add_call, (mat1_spec, mat2_spec), {}))
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [-1, 0])

        # more advanced case that needs reshard one input to align sharding
        mat1, mat2 = [0, -1, -1, 1], [0, -1, 1]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([12, 1, 1, 8]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([12, 4, 8]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = pointwise_rule(OpSchema(add_call, (mat1_spec, mat2_spec), {}))
        output_spec = output_sharding.output_spec
        self.assertIsNone(output_spec)
        self.assertIsNotNone(output_sharding.schema_suggestions)

        # ensure that the suggestion is to reshard the first
        # arg by all_gather first tensor dim sharding
        schema_suggestion = output_sharding.schema_suggestions[0]
        self.assertEqual(schema_suggestion.args_schema[0].dim_map, [-1, -1, -1, 1])
        self.assertEqual(schema_suggestion.args_schema[1].dim_map, mat2)

    @with_comms
    def test_pointwise_enforce_sharding_multi_sharding_on_mesh_dim(self):
        # 2d mesh pointwise sharding
        mesh_shape = torch.arange(self.world_size).reshape(
            self.world_size // 2, self.world_size // 2
        )
        mesh = DeviceMesh(self.device_type, mesh_shape)

        add_call = aten.add_.Tensor

        # more advanced case that needs reshard one input to align sharding
        mat1, mat2 = [0, -1, 1], [-1, -1, 0]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([12, 4, 8]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([12, 1, 8]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = pointwise_rule(OpSchema(add_call, (mat1_spec, mat2_spec), {}))
        output_spec = output_sharding.output_spec
        self.assertIsNone(output_spec)
        self.assertIsNotNone(output_sharding.schema_suggestions)

        # ensure that the suggestion is to reshard the second
        # arg as we should enforce the sharding of the first arg
        schema_suggestion = output_sharding.schema_suggestions[0]
        self.assertEqual(schema_suggestion.args_schema[0].dim_map, mat1)
        self.assertEqual(schema_suggestion.args_schema[1].dim_map, mat1)

    @with_comms
    def test_reduction_rule(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        sum_call = aten.sum.default
        # reduction on a 2d mat
        mat1 = [0, -1]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 4]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        # reduction on dim 0
        output_sharding_0 = reduction_rule(
            OpSchema(sum_call, (mat1_spec, 0), {}),
            dims=[0],
            reduction_linear=True,
        )
        self.assertIsNotNone(output_sharding_0.output_spec)
        self.assertEqual(output_sharding_0.output_spec.dim_map, [-1])
        # pending sum on dim 0
        self.assertEqual(output_sharding_0.output_spec.sums, [0])

        # reduction on dim 1
        output_sharding_1 = reduction_rule(
            OpSchema(sum_call, (mat1_spec, 1), {}),
            dims=[1],
            reduction_linear=True,
        )
        self.assertIsNotNone(output_sharding_1.output_spec)
        self.assertEqual(output_sharding_1.output_spec.dim_map, [0])
        self.assertEqual(output_sharding_1.output_spec.sums, [])

        # full reduction if not specify dim
        output_sharding_all_dim = reduction_rule(
            OpSchema(sum_call, (mat1_spec,), {}),
            dims=[0, 1],
            reduction_linear=True,
        )
        self.assertIsNotNone(output_sharding_all_dim.output_spec)
        self.assertEqual(output_sharding_all_dim.output_spec.dim_map, [])
        # pending sum on mesh
        self.assertEqual(output_sharding_all_dim.output_spec.sums, [0])


if __name__ == "__main__":
    run_tests()
