# Owner(s): ["oncall: distributed"]

from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase
from torch.testing._internal.two_tensor import TwoTensor


class BorrowedTwoTensor(TwoTensor):
    @staticmethod
    def __new__(cls, a, b, *args, **kwargs):
        with torch.inference_mode(a.is_inference()):
            return TwoTensor.__new__(cls, a, b, *args, **kwargs)

    def fsdp_get_unsharded_view(self, mesh, module, mp_policy):
        if mesh.size() != 1:
            raise AssertionError("The unsharded view requires one shard rank")
        return BorrowedTwoTensor(self.a, self.b)


class TestFullyShardUnshardedViews(TestCase):
    def setUp(self):
        super().setUp()
        dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)

    def tearDown(self):
        dist.destroy_process_group()
        super().tearDown()

    def _make_model(self, device, subclass, meta, reshard_after_forward):
        with torch.device("meta" if meta else device):
            model = nn.Linear(8, 4)
            if subclass:
                model.weight = nn.Parameter(
                    BorrowedTwoTensor(
                        model.weight.detach(), model.weight.detach().clone()
                    ),
                    requires_grad=False,
                )
            model.requires_grad_(False)
        fully_shard(
            model,
            mesh=init_device_mesh(torch.device(device).type, (1,)),
            reshard_after_forward=reshard_after_forward,
        )
        if meta:
            model.to_empty(device=device)
        state = model.state_dict()
        for tensor in state.values():
            tensor.fill_(0.5)
        model.load_state_dict(state)
        return model

    @parametrize("subclass", [False, True])
    @parametrize("meta", [False, True])
    @parametrize("reshard_after_forward", [False, True])
    def test_views_survive_reshard_and_weight_loading(
        self, device, subclass, meta, reshard_after_forward
    ):
        model = self._make_model(device, subclass, meta, reshard_after_forward)
        local = model.weight.to_local()
        source = (local.a, local.b) if subclass else (local,)
        pointers = [tensor.data_ptr() for tensor in source]
        inputs = torch.ones(2, 8, device=device)
        with (
            torch.no_grad(),
            patch.object(
                FSDPParam,
                "init_all_gather_outputs",
                side_effect=AssertionError(
                    "Borrowed views must not allocate gather outputs"
                ),
            ),
        ):
            model(inputs)
            group = fully_shard.state(model)._fsdp_param_group
            param = group.fsdp_params[0]
            borrowed = param._unsharded_param
            buffers = (borrowed.a, borrowed.b) if subclass else (borrowed,)
            self.assertEqual([tensor.data_ptr() for tensor in buffers], pointers)
            for version in range(1, 4):
                model.reshard()
                state = model.state_dict()
                self.assertEqual([tensor.data_ptr() for tensor in buffers], pointers)
                self.assertTrue(
                    all(tensor.untyped_storage().nbytes() > 0 for tensor in buffers)
                )
                update = {
                    name: torch.full_like(tensor, version)
                    for name, tensor in state.items()
                }
                model.load_state_dict(update)
                self.assertIs(param._unsharded_param, borrowed)
                for tensor in buffers:
                    self.assertEqual(tensor, torch.full_like(tensor, version))
                output = model(inputs)
                outputs = (output.a, output.b) if subclass else (output,)
                for tensor in outputs:
                    self.assertEqual(tensor, torch.full_like(tensor, 9 * version))
                self.assertEqual([tensor.data_ptr() for tensor in buffers], pointers)
                for fsdp_param in group.fsdp_params:
                    self.assertTrue(fsdp_param._unsharded_param_is_view)
                    self.assertEqual(fsdp_param.all_gather_outputs, [])
                    self.assertEqual(fsdp_param._unsharded_inner_tensors, [])

    @parametrize(
        "requires_grad,param_dtype", [(True, torch.float32), (False, torch.bfloat16)]
    )
    def test_training_and_casts_keep_independent_storage(
        self, device, requires_grad, param_dtype
    ):
        model = nn.Linear(8, 4, device=device).requires_grad_(requires_grad)
        fully_shard(
            model,
            mesh=init_device_mesh(torch.device(device).type, (1,)),
            mp_policy=MixedPrecisionPolicy(param_dtype=param_dtype),
            reshard_after_forward=False,
        )
        source = model.weight.to_local()
        model(torch.ones(2, 8, device=device))
        param = fully_shard.state(model)._fsdp_param_group.fsdp_params[0]
        self.assertFalse(param._unsharded_param_is_view)
        self.assertNotEqual(param._unsharded_param.data_ptr(), source.data_ptr())
        self.assertEqual(param._unsharded_param.dtype, param_dtype)

    @onlyCUDA
    def test_cpu_offload_keeps_independent_storage(self, device):
        model = nn.Linear(8, 4, device=device).requires_grad_(False)
        fully_shard(
            model,
            mesh=init_device_mesh("cuda", (1,)),
            offload_policy=CPUOffloadPolicy(pin_memory=False),
            reshard_after_forward=False,
        )
        model(torch.ones(2, 8, device=device))
        param = fully_shard.state(model)._fsdp_param_group.fsdp_params[0]
        self.assertFalse(param._unsharded_param_is_view)
        self.assertEqual(param.sharded_param.device.type, "cpu")
        self.assertEqual(param._unsharded_param.device.type, "cuda")

    @parametrize("invalid", [False, True])
    def test_extension_can_decline_and_must_preserve_metadata(self, device, invalid):
        model = nn.Linear(8, 4, device=device).requires_grad_(False)
        fully_shard(model, mesh=init_device_mesh(torch.device(device).type, (1,)))
        local = model.weight._local_tensor
        local.fsdp_get_unsharded_view = lambda *_: local.double() if invalid else None
        inputs = torch.ones(2, 8, device=device)
        if invalid:
            with self.assertRaisesRegex(
                ValueError, "parameter's local shape, dtype, and device"
            ):
                model(inputs)
        else:
            model(inputs)
            param = fully_shard.state(model)._fsdp_param_group.fsdp_params[0]
            self.assertFalse(param._unsharded_param_is_view)

    def test_replaced_storage_recreates_view(self, device):
        model = self._make_model(device, False, False, False)
        inputs = torch.ones(2, 8, device=device)
        with torch.no_grad():
            model(inputs)
            old_weight = model.weight
            state = {
                name: torch.full_like(tensor, 2)
                for name, tensor in model.state_dict().items()
            }
            model.load_state_dict(state, assign=True)
            source = model.weight.to_local()
            self.assertNotEqual(source.data_ptr(), old_weight.data_ptr())
            self.assertEqual(model(inputs), torch.full((2, 4), 18.0, device=device))
            self.assertEqual(model.weight.data_ptr(), source.data_ptr())

    @onlyCUDA
    @parametrize("subclass", [False, True])
    def test_graph_replay_reads_loaded_weights_without_unshard(self, device, subclass):
        model = self._make_model(device, subclass, True, False)
        inputs = torch.ones(2, 8, device=device)
        with torch.inference_mode():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    model(inputs)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = model(inputs)
        with torch.no_grad():
            for version in range(1, 4):
                state = {
                    name: torch.full_like(tensor, version)
                    for name, tensor in model.state_dict().items()
                }
                model.load_state_dict(state)
                graph.replay()
                outputs = (captured.a, captured.b) if subclass else (captured,)
                for output in outputs:
                    self.assertEqual(output, torch.full_like(output, 9 * version))
            graph.reset()


instantiate_device_type_tests(
    TestFullyShardUnshardedViews, globals(), only_for=("cpu", "cuda")
)


if __name__ == "__main__":
    run_tests()
