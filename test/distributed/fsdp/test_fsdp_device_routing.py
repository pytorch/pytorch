# Owner(s): ["oncall: distributed"]

import contextlib
import tempfile
from datetime import timedelta
from unittest import mock

import torch
import torch.distributed as dist
from torch.testing._internal import common_fsdp as fsdp
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TestFSDPDeviceRouting(fsdp.FSDPTestMixin, TestCase):
    hw_classification = HardwareClassification.CPU
    device_type = "cpu"

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @contextlib.contextmanager
    def _cpu_process_group(self):
        if not dist.is_gloo_available():
            self.skipTest("Gloo is required")
        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group(
                "gloo",
                init_method=f"file://{directory}/store",
                rank=0,
                world_size=1,
                timeout=timedelta(seconds=30),
            )
            try:
                yield
            finally:
                if dist.is_initialized():
                    dist.destroy_process_group()

    @parametrize("wrapper", ["ddp", "fsdp"])
    def test_cpu_amp_updates_parameters(self, wrapper):
        with self._cpu_process_group(), mock.patch.object(fsdp, "DEVICE_TYPE", "cuda"):
            raw = fsdp.TransformerWithSharedParams.init(
                self.process_group,
                fsdp.FSDPInitMode.NO_FSDP,
                fsdp.DEVICEInitMode.DEVICE_BEFORE,
                deterministic=True,
                device="cpu",
            )
            model = (
                fsdp.DDP(raw)
                if wrapper == "ddp"
                else fsdp.FSDP(raw, device_id=torch.device("cpu"))
            )
            before = [p.detach().clone() for p in model.parameters()]
            observed = []

            def observe(module, inputs):
                observed.append(
                    (torch.is_autocast_enabled("cpu"), [x.device.type for x in inputs])
                )

            handle = model.register_forward_pre_hook(observe)
            try:
                loss = self._train_for_several_steps(
                    model, 2, True, enable_sharded_grad_scaler=True
                )
            finally:
                handle.remove()
            self.assertEqual(loss.device.type, "cpu")
            self.assertEqual(loss.dtype, torch.float32)
            self.assertEqual(len(observed), 2)
            self.assertTrue(
                all(
                    enabled and all(d == "cpu" for d in devices)
                    for enabled, devices in observed
                )
            )
            self.assertTrue(
                any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
            )

    def test_cpu_parity_without_autocast(self):
        with self._cpu_process_group(), mock.patch.object(fsdp, "DEVICE_TYPE", "cuda"):
            self._test_fsdp_parity(
                fsdp.NestedWrappedModule,
                fsdp.FSDPInitMode.RECURSIVE,
                fsdp.DEVICEInitMode.DEVICE_BEFORE,
                save_model=False,
            )

    @parametrize("fake_pg", [False, True])
    def test_cpu_worker_on_accelerator_host(self, fake_pg):
        class CPUWorker(fsdp.FSDPTest):
            device_type = "cpu"
            world_size = 3

            def run_test(self, name, pipe):
                pass

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(torch.accelerator, "is_available", return_value=True)
            )
            stack.enter_context(
                mock.patch.object(torch.accelerator, "device_count", return_value=1)
            )
            bind = stack.enter_context(
                mock.patch.object(torch.accelerator, "set_device_index")
            )
            init = stack.enter_context(mock.patch.object(dist, "init_process_group"))
            barrier = stack.enter_context(mock.patch.object(dist, "barrier"))
            destroy = stack.enter_context(
                mock.patch.object(dist, "destroy_process_group")
            )
            stack.enter_context(mock.patch.object(torch._dynamo, "reset"))
            stack.enter_context(mock.patch.object(fsdp, "set_rng_seed"))
            CPUWorker._run(2, "runTest", "unused", None, fake_pg=fake_pg)
        self.assertEqual(
            init.call_args.kwargs["backend"], "fake" if fake_pg else "gloo"
        )
        self.assertEqual(init.call_args.kwargs["world_size"], 3)
        bind.assert_not_called()
        barrier.assert_has_calls([mock.call(), mock.call()])
        self.assertEqual(barrier.call_count, 2)
        destroy.assert_called_once()

    @parametrize("fake_pg", [False, True])
    def test_accelerator_worker_barrier_device_ids(self, fake_pg):
        class AcceleratorWorker(fsdp.FSDPTest):
            device_type = "xpu"
            world_size = 2

            def backend(self, device):
                self.assertEqual(device, "xpu")
                return "xccl"

            def run_test(self, name, pipe):
                pass

        device_module = mock.Mock()
        device_module.device_count.return_value = 4
        with contextlib.ExitStack() as stack:
            get_module = stack.enter_context(
                mock.patch.object(
                    torch, "get_device_module", return_value=device_module
                )
            )
            host_count = stack.enter_context(
                mock.patch.object(torch.accelerator, "device_count", return_value=1)
            )
            host_bind = stack.enter_context(
                mock.patch.object(torch.accelerator, "set_device_index")
            )
            init = stack.enter_context(mock.patch.object(dist, "init_process_group"))
            barrier = stack.enter_context(mock.patch.object(dist, "barrier"))
            destroy = stack.enter_context(
                mock.patch.object(dist, "destroy_process_group")
            )
            reset = stack.enter_context(mock.patch.object(torch._dynamo, "reset"))
            stack.enter_context(mock.patch.object(fsdp, "set_rng_seed"))
            AcceleratorWorker._run(1, "runTest", "unused", None, fake_pg=fake_pg)
        get_module.assert_called_once_with("xpu")
        device_module.set_device.assert_called_once_with(1)
        host_count.assert_not_called()
        host_bind.assert_not_called()
        self.assertEqual(
            init.call_args.kwargs["backend"], "fake" if fake_pg else "xccl"
        )
        expected = mock.call() if fake_pg else mock.call(device_ids=[1])
        barrier.assert_has_calls([expected, expected])
        self.assertEqual(barrier.call_count, 2)
        self.assertEqual(reset.call_count, 2)
        destroy.assert_called_once()

    def test_legacy_callable_and_device_base(self):
        class Legacy(fsdp.FSDPTestContinuous):
            @classmethod
            def device_type(cls):
                return "cpu"

        class DeviceBase:
            device_type = "cpu"

        class Generated(DeviceBase, Legacy):
            pass

        with mock.patch.object(fsdp, "DEVICE_TYPE", "cuda"):
            self.assertEqual(object.__new__(Legacy)._device_type, "cpu")
            self.assertEqual(object.__new__(Generated)._device_type, "cpu")

    @parametrize(
        "kwargs",
        [
            {"init_kwargs": {"device": "cuda"}},
            {"device_id": "cuda:0"},
            {"device_id": 0},
        ],
    )
    def test_conflicting_devices(self, kwargs):
        class CPU(fsdp.FSDPTestMixin):
            device_type = "cpu"

        with self.assertRaisesRegex(ValueError, "conflicts"):
            CPU()._test_fsdp_parity(
                fsdp.NestedWrappedModule,
                fsdp.FSDPInitMode.RECURSIVE,
                fsdp.DEVICEInitMode.DEVICE_BEFORE,
                **kwargs,
            )

    def test_scaler_none_uses_test_device(self):
        class CPU(fsdp.FSDPTestMixin):
            device_type = "cpu"

        model = torch.nn.Linear(2, 2)
        kwargs = {"device": None}
        with mock.patch.object(
            fsdp, "ShardedGradScaler", side_effect=RuntimeError("stop at scaler")
        ) as scaler:
            with self.assertRaisesRegex(RuntimeError, "stop at scaler"):
                CPU()._train_for_several_steps(
                    model, 1, False, sharded_grad_scaler_kwargs=kwargs
                )
        self.assertEqual(scaler.call_args.kwargs["device"], "cpu")
        self.assertEqual(kwargs, {"device": None})

    @parametrize("device", [None, "cuda", "cpu"])
    def test_moe_parent_device_propagation(self, device):
        def parent(model, *args, **kwargs):
            torch.nn.Module.__init__(model)
            model.rank = 0
            model.world_size = 1

        device_id = "cpu" if device == "cpu" else "cuda:0"
        with contextlib.ExitStack() as stack:
            ctor = stack.enter_context(
                mock.patch.object(
                    fsdp.NestedWrappedModule,
                    "__init__",
                    autospec=True,
                    side_effect=parent,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    fsdp, "FSDP", side_effect=lambda model, *args, **kwargs: model
                )
            )
            stack.enter_context(
                mock.patch.object(dist, "new_group", return_value=object())
            )
            stack.enter_context(mock.patch.object(fsdp, "_MOE_EXPERT_GROUPS", {}))
            fsdp.MixtureOfExperts(
                fsdp.DummyProcessGroup(0, 1),
                True,
                fsdp.DEVICEInitMode.DEVICE_NEVER,
                0,
                False,
                device_id=device_id,
                **({"device": device} if device is not None else {}),
                cpu_offload=fsdp.CPUOffload(True),
            )
        if device == "cpu":
            self.assertEqual(ctor.call_args.kwargs["device_id"], "cpu")
        else:
            self.assertNotIn("device_id", ctor.call_args.kwargs)
        self.assertNotIn("cpu_offload", ctor.call_args.kwargs)

    @parametrize(
        "model_class",
        [
            fsdp.TransformerWithSharedParams,
            fsdp.NestedWrappedModule,
            fsdp.AlwaysWrapNestedWrappedModule,
            fsdp.NonUniformReqGradNWM,
            fsdp.NestedWrappedModuleWithDelay,
            fsdp.MixtureOfExperts,
        ],
    )
    def test_selected_cpu_model(self, model_class):
        with mock.patch.object(fsdp, "DEVICE_TYPE", "cuda"):
            model = model_class.init(
                fsdp.DummyProcessGroup(0, 1),
                fsdp.FSDPInitMode.NO_FSDP,
                fsdp.DEVICEInitMode.DEVICE_BEFORE,
                device="cpu",
            )
            inputs = model.get_input(torch.device("cpu"))
            output = model(*inputs)
        self.assertTrue(all(p.device.type == "cpu" for p in model.parameters()))
        self.assertEqual(output.device.type, "cpu")


instantiate_parametrized_tests(TestFSDPDeviceRouting)

if __name__ == "__main__":
    run_tests()
