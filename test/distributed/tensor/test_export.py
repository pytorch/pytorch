import contextlib

import torch
import torch.distributed as dist
from torch._dynamo.functional_export import _dynamo_graph_capture_for_export
from torch._functorch.aot_autograd import aot_export_joint_with_descriptors
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    requires_cuda,
    run_tests,
    TestCase,
)
from torch.testing._internal.distributed._tensor.common_dtensor import MLPModule
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils._debug_mode import DebugMode


class SimpleModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.mlp_0 = MLPModule(device)
        self.mlp_1 = MLPModule(device)

    def forward(self, input):
        return self.mlp_1(self.mlp_0(input))


def strict_export_and_aot_export_joint_with_descriptors(model, inputs):
    # install_free_tensors is required for dynamo to work
    with torch._dynamo.config.patch(install_free_tensors=True):
        with torch._export.utils._disable_aten_to_metadata_assertions():
            ep = torch.export.export(model, (inputs,), strict=True)

    # joint_gm produced here is missing the backward region, due to incompatiblility
    # between ep.module() and aot_export_joint_with_descriptors.
    # Keeping this here to show the issue.
    # return aot_export_joint_with_descriptors_alone(ep.module(), inputs)
    return ep.graph_module


def graph_capture_and_aot_export_joint_with_descriptors(model, inputs):
    # TODO: switch to use the official graph_capture API once it is ready
    gm = _dynamo_graph_capture_for_export(model)(inputs)
    return aot_export_joint_with_descriptors_alone(gm, inputs)


def aot_export_joint_with_descriptors_alone(model, inputs):
    with contextlib.ExitStack() as stack:
        joint_with_descriptors = aot_export_joint_with_descriptors(
            stack,
            model,
            (inputs,),
        )
        return joint_with_descriptors.graph_module


@requires_cuda
class DTensorExportTest(TestCase):
    def tearDown(self):
        super().tearDown()
        dist.destroy_process_group()

    def setUp(self):
        super().setUp()
        self.world_size = 8
        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=self.world_size, store=store
        )
        self.device_type = "cuda"

    @parametrize(
        "export_fn",
        [
            strict_export_and_aot_export_joint_with_descriptors,
            # graph_capture_and_aot_export_joint_with_descriptors,
            # aot_export_joint_with_descriptors_alone,
        ],
    )
    def test_dtensor_module(
        self,
        export_fn,
    ):
        dp_degree = 2
        tp_degree = self.world_size // dp_degree

        # 2-D mesh is [dp, tp]
        mesh_2d = init_device_mesh(
            self.device_type,
            mesh_shape=(dp_degree, tp_degree),
            mesh_dim_names=["dp", "tp"],
        )

        model = SimpleModel(self.device_type)
        parallelize_plan = {
            "mlp_0.net1": ColwiseParallel(),
            "mlp_0.net2": RowwiseParallel(),
            "mlp_1.net1": ColwiseParallel(),
            "mlp_1.net2": RowwiseParallel(),
        }
        tp_model = parallelize_module(model, mesh_2d["tp"], parallelize_plan)

        inputs = torch.rand(20, 10, device=self.device_type)
        inputs = distribute_tensor(inputs, mesh_2d["tp"], placements=[Replicate()])

        # with DebugMode(record_torchfunction=False) as debug_mode:
        #     out = tp_model(inputs)

        joint_gm = export_fn(tp_model, inputs)




instantiate_parametrized_tests(DTensorExportTest)


if __name__ == "__main__":
    run_tests()
