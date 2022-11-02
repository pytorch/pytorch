# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch
from torch.testing._internal.common_utils import run_tests
from torchgen.model import FunctionSchema
from torch.distributed._tensor.dispatch import OpSchema

from torch.distributed._tensor.ops.common_rules import (
    einop_rule,
    reduction_rule,
    pointwise_rule,
)
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.testing._internal.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.distributed._tensor import DeviceMesh


class CommonRulesTest(DTensorTestBase):
    def parse_schema(self, schema_str):
        return FunctionSchema.parse(schema_str)

    @with_comms
    def test_einop_basic_propagation(self):
        # plain einsum, mm
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        func_schema = self.parse_schema(
            "aten::mm(Tensor self, Tensor mat2) -> Tensor"
        )
        # propagate col-wise sharding
        mat1, mat2 = [-1, -1], [-1, 0]
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], shape=torch.Size([8, 4])
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], shape=torch.Size([4, 8])
        )
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(func_schema, (mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [-1, 0])
        self.assertEqual(output_spec.shape, torch.Size([8, 8]))

        # propagate row-wise sharding
        mat1, mat2 = [0, -1], [-1, -1]
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], shape=torch.Size([8, 4])
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], shape=torch.Size([4, 8])
        )
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(func_schema, (mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [0, -1])
        self.assertEqual(output_spec.shape, torch.Size([8, 8]))

        # generate partial
        mat1, mat2 = [-1, 0], [0, -1]
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], shape=torch.Size([8, 4])
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], shape=torch.Size([4, 8])
        )
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(func_schema, (mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertTrue(output_spec.placements[0].is_partial())
        self.assertEqual(output_spec.shape, torch.Size([8, 8]))

    @with_comms
    def test_einop_pointwise_propagation(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        func_schema = self.parse_schema(
            "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"
        )
        # addition
        mat1 = [0, -1]
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], shape=torch.Size([8, 8])
        )
        output_sharding = einop_rule(
            "ij,ij->ij", OpSchema(func_schema, (mat1_spec, mat1_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [0, -1])
        self.assertEqual(output_spec.shape, torch.Size([8, 8]))

        # broadcast addition
        mat1 = [-1, 0, -1]
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], shape=torch.Size([8, 4, 2])
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, [-1], [], shape=torch.Size([2])
        )
        output_sharding = einop_rule(
            "ijk,k->ijk", OpSchema(func_schema, (mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [-1, 0, -1])
        self.assertEqual(output_spec.shape, torch.Size([8, 4, 2]))

        # broadcast to a common shape
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, [0, -1, -1], [], shape=torch.Size([8, 8, 8])
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, [-1, -1], [], shape=torch.Size([1, 8])
        )
        output_sharding = einop_rule(
            "ijk,1k->ijk", OpSchema(func_schema, (mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [0, -1, -1])
        self.assertEqual(output_spec.shape, torch.Size([8, 8, 8]))

    @with_comms
    def test_einop_merge_sharding(self):
        # 2d mesh einop merge sharding
        mesh_shape = torch.arange(self.world_size).reshape(
            self.world_size // 2, self.world_size // 2
        )
        mesh = DeviceMesh(self.device_type, mesh_shape)

        func_schema = self.parse_schema(
            "aten::mm(Tensor self, Tensor mat2) -> Tensor"
        )

        mat1, mat2 = [0, -1], [-1, 1]
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], shape=torch.Size([8, 4])
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], shape=torch.Size([4, 8])
        )
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(func_schema, (mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [0, 1])
        self.assertEqual(output_spec.shape, torch.Size([8, 8]))

    @with_comms
    def test_einop_linearity(self):
        mesh_shape = torch.arange(self.world_size).reshape(
            self.world_size // 2, self.world_size // 2
        )
        mesh = DeviceMesh(self.device_type, mesh_shape)

        mm_func_schema = self.parse_schema(
            "aten::mm(Tensor self, Tensor mat2) -> Tensor"
        )

        mat1, mat2 = [0, -1], [-1, -1]
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [1], shape=torch.Size([8, 4])
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], shape=torch.Size([4, 8])
        )
        # if not turn on linearity, partial sum is not eligible to propagate, we return
        # suggestion to reshard inputs with no partial sum (i.e. all_reduce one input)
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(mm_func_schema, (mat1_spec, mat2_spec), {})
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
            OpSchema(mm_func_schema, (mat1_spec, mat2_spec), {}),
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
        add_func_schema = self.parse_schema(
            "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"
        )
        mat1, mat2 = [0, -1], [0, -1]
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [1], shape=torch.Size([8, 6])
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], shape=torch.Size([8, 6])
        )

        output_sharding = einop_rule(
            "ij,ij->ij",
            OpSchema(add_func_schema, (mat1_spec, mat2_spec), {}),
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

        func_schema = self.parse_schema(
            "aten::mm(Tensor self, Tensor mat2) -> Tensor"
        )
        mat1, mat2 = [0, -1], [0, -1]
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], shape=torch.Size([8, 12])
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], shape=torch.Size([12, 4])
        )
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(func_schema, (mat1_spec, mat2_spec), {})
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

        func_schema = self.parse_schema(
            "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"
        )
        mat1, mat2 = [0, -1], [1, -1]
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], shape=torch.Size([8, 4])
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], shape=torch.Size([8, 4])
        )

        with self.assertRaisesRegex(
            RuntimeError, "sharded two different ways:"
        ):
            einop_rule(
                "ij,ij->ij", OpSchema(func_schema, (mat1_spec, mat2_spec), {})
            )

    @with_comms
    def test_pointwise_rules_broadcasting(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        func_schema = self.parse_schema(
            "where.self(Tensor condition, Tensor self, Tensor other) -> Tensor"
        )
        inp1, inp2, inp3 = [0], [], [-1, -1]
        condition = DTensorSpec.from_dim_map(
            mesh, inp1, [], shape=torch.Size([8])
        )
        self_tensor = DTensorSpec.from_dim_map(
            mesh, inp2, [], shape=torch.Size([])
        )
        other_tensor = DTensorSpec.from_dim_map(
            mesh, inp3, [], shape=torch.Size([1, 1])
        )
        # propagate point-wise sharding with broadcasting
        output_sharding = pointwise_rule(
            OpSchema(func_schema, (condition, self_tensor, other_tensor), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [-1, 0])
        self.assertEqual(output_spec.shape, [1, 8])

    @with_comms
    def test_pointwise_rules_suggestion(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        func_schema = self.parse_schema(
            "aten::lerp.Scalar(Tensor self, Tensor end, Scalar weight) -> Tensor"
        )
        # propagate point-wise sharding
        inp1, inp2 = [-1, -1], [-1, 0]
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, inp1, [], shape=torch.Size([8, 4])
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, inp2, [], shape=torch.Size([8, 4])
        )
        # adding a positional argument -1 to arg schema
        output_sharding = pointwise_rule(
            OpSchema(func_schema, (mat1_spec, mat2_spec, -1), {})
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

        func_schema = self.parse_schema(
            "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"
        )

        # basic case to test implicit broadcasting shape alignment
        mat1, mat2 = [-1, 0], [0]
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], shape=torch.Size([20, 6])
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], shape=torch.Size([6])
        )
        output_sharding = pointwise_rule(
            OpSchema(func_schema, (mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [-1, 0])

        # more advanced case that needs reshard one input to align sharding
        mat1, mat2 = [0, -1, -1, 1], [0, -1, 1]
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], shape=torch.Size([12, 1, 1, 8])
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], shape=torch.Size([12, 4, 8])
        )
        output_sharding = pointwise_rule(
            OpSchema(func_schema, (mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNone(output_spec)
        self.assertIsNotNone(output_sharding.schema_suggestions)

        # ensure that the suggestion is to reshard the first
        # arg by all_gather first tensor dim sharding
        schema_suggestion = output_sharding.schema_suggestions[0]
        self.assertEqual(
            schema_suggestion.args_schema[0].dim_map, [-1, -1, -1, 1]
        )
        self.assertEqual(schema_suggestion.args_schema[1].dim_map, mat2)

    @with_comms
    def test_pointwise_enforce_sharding_multi_sharding_on_mesh_dim(self):
        # 2d mesh pointwise sharding
        mesh_shape = torch.arange(self.world_size).reshape(
            self.world_size // 2, self.world_size // 2
        )
        mesh = DeviceMesh(self.device_type, mesh_shape)

        func_schema = self.parse_schema(
            "aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)"
        )

        # more advanced case that needs reshard one input to align sharding
        mat1, mat2 = [0, -1, 1], [-1, -1, 0]
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], shape=torch.Size([12, 4, 8])
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], shape=torch.Size([12, 1, 8])
        )
        output_sharding = pointwise_rule(
            OpSchema(func_schema, (mat1_spec, mat2_spec), {})
        )
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

        func_schema = self.parse_schema(
            "aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor"
        )
        # reduction on a 2d mat
        mat1 = [0, -1]
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], shape=torch.Size([8, 4])
        )
        # reduction on dim 0
        output_sharding_0 = reduction_rule(
            OpSchema(func_schema, (mat1_spec, 0), {}),
            dims=[0],
            reduction_linear=True,
        )
        self.assertIsNotNone(output_sharding_0.output_spec)
        self.assertEqual(output_sharding_0.output_spec.dim_map, [-1])
        # pending sum on dim 0
        self.assertEqual(output_sharding_0.output_spec.sums, [0])
        self.assertEqual(output_sharding_0.output_spec.shape, torch.Size([4]))

        # reduction on dim 1
        output_sharding_1 = reduction_rule(
            OpSchema(func_schema, (mat1_spec, 1), {}),
            dims=[1],
            reduction_linear=True,
        )
        self.assertIsNotNone(output_sharding_1.output_spec)
        self.assertEqual(output_sharding_1.output_spec.dim_map, [0])
        self.assertEqual(output_sharding_1.output_spec.sums, [])
        self.assertEqual(output_sharding_1.output_spec.shape, torch.Size([8]))

        # full reduction if not specify dim
        output_sharding_all_dim = reduction_rule(
            OpSchema(func_schema, (mat1_spec,), {}),
            dims=[0, 1],
            reduction_linear=True,
        )
        self.assertIsNotNone(output_sharding_all_dim.output_spec)
        self.assertEqual(output_sharding_all_dim.output_spec.dim_map, [])
        # pending sum on mesh
        self.assertEqual(output_sharding_all_dim.output_spec.sums, [0])
        self.assertEqual(
            output_sharding_all_dim.output_spec.shape, torch.Size([])
        )


if __name__ == "__main__":
    run_tests()
