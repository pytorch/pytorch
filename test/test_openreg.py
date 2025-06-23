# Owner(s): ["module: PrivateUse1"]

import os
import tempfile
import types
import unittest

import psutil
import pytorch_openreg  # noqa: F401

import torch
from torch.testing._internal.common_utils import (
    IS_LINUX,
    run_tests,
    skipIfTorchDynamo,
    skipIfXpu,
    TestCase,
)


class TestPrivateUse1(TestCase):
    """Tests of third-parth device integration mechinasm based PrivateUse1"""

    def test_backend_name(self):
        self.assertEqual(torch._C._get_privateuse1_backend_name(), "openreg")
        # backend can be renamed to the same name multiple times
        torch.utils.rename_privateuse1_backend("openreg")
        with self.assertRaisesRegex(RuntimeError, "has already been set"):  # type: ignore[misc]
            torch.utils.rename_privateuse1_backend("dev")

    def test_backend_module_registration(self):
        def generate_faked_module():
            return types.ModuleType("fake_module")

        with self.assertRaisesRegex(RuntimeError, "Expected one of cpu"):  # type: ignore[misc]
            torch._register_device_module("dev", generate_faked_module())
        with self.assertRaisesRegex(RuntimeError, "The runtime module of"):  # type: ignore[misc]
            torch._register_device_module("openreg", generate_faked_module())

    def test_backend_generate_methods(self):
        with self.assertRaisesRegex(RuntimeError, "The custom device module of"):  # type: ignore[misc]
            torch.utils.generate_methods_for_privateuse1_backend()  # type: ignore[misc]

        self.assertTrue(hasattr(torch.Tensor, "is_openreg"))
        self.assertTrue(hasattr(torch.Tensor, "openreg"))
        self.assertTrue(hasattr(torch.TypedStorage, "is_openreg"))
        self.assertTrue(hasattr(torch.TypedStorage, "openreg"))
        self.assertTrue(hasattr(torch.UntypedStorage, "is_openreg"))
        self.assertTrue(hasattr(torch.UntypedStorage, "openreg"))
        self.assertTrue(hasattr(torch.nn.Module, "openreg"))
        self.assertTrue(hasattr(torch.nn.utils.rnn.PackedSequence, "is_openreg"))
        self.assertTrue(hasattr(torch.nn.utils.rnn.PackedSequence, "openreg"))

    def test_backend_module_function(self):
        with self.assertRaisesRegex(RuntimeError, "Try to call torch.openreg"):  # type: ignore[misc]
            torch.utils.backend_registration._get_custom_mod_func("func_name_")  # type: ignore[misc]
        self.assertTrue(
            torch.utils.backend_registration._get_custom_mod_func("device_count")() == 2  # type: ignore[misc]
        )

    @skipIfTorchDynamo()
    def test_backend_operator_registration(self):
        self.assertTrue(
            torch._C._dispatch_has_kernel_for_dispatch_key(
                "aten::empty.memory_format", torch.DispatchKey.PrivateUse1
            )
        )
        x = torch.empty(3, 3, device="openreg")
        self.assertTrue(x.device.type, "openreg")
        self.assertTrue(x.shape, torch.Size([3, 3]))

    def test_backend_dispatchstub(self):
        x_cpu = torch.randn(2, 2, 3, dtype=torch.float32, device="cpu")
        x_openreg = x_cpu.to("openreg")

        y_cpu = torch.abs(x_cpu)
        y_openreg = torch.abs(x_openreg)
        self.assertEqual(y_cpu, y_openreg.cpu())

        o_cpu = torch.randn(2, 2, 6, dtype=torch.float32, device="cpu")
        o_openreg = o_cpu.to("openreg")
        # output operand with resize flag is False in TensorIterator.
        torch.abs(x_cpu, out=o_cpu[:, :, 0:6:2])
        torch.abs(x_openreg, out=o_openreg[:, :, 0:6:2])
        self.assertEqual(o_cpu, o_openreg.cpu())

        # output operand with resize flag is True in TensorIterator and
        # convert output to contiguous tensor in TensorIterator.
        torch.abs(x_cpu, out=o_cpu[:, :, 0:6:3])
        torch.abs(x_openreg, out=o_openreg[:, :, 0:6:3])
        self.assertEqual(o_cpu, o_openreg.cpu())

    def test_backend_tensor_type(self):
        dtypes_map = {
            torch.bool: "torch.openreg.BoolTensor",
            torch.double: "torch.openreg.DoubleTensor",
            torch.float32: "torch.openreg.FloatTensor",
            torch.half: "torch.openreg.HalfTensor",
            torch.int32: "torch.openreg.IntTensor",
            torch.int64: "torch.openreg.LongTensor",
            torch.int8: "torch.openreg.CharTensor",
            torch.short: "torch.openreg.ShortTensor",
            torch.uint8: "torch.openreg.ByteTensor",
        }

        for dtype, str in dtypes_map.items():
            x = torch.empty(4, 4, dtype=dtype, device="openreg")
            self.assertTrue(x.type() == str)

    # Note that all dtype-d Tensor objects here are only for legacy reasons
    # and should NOT be used.
    def test_backend_type_methods(self):
        # Tensor
        tensor_cpu = torch.randn([8]).float()
        self.assertEqual(tensor_cpu.type(), "torch.FloatTensor")

        tensor_openreg = tensor_cpu.openreg()
        self.assertEqual(tensor_openreg.type(), "torch.openreg.FloatTensor")

        # Storage
        storage_cpu = tensor_cpu.storage()
        self.assertEqual(storage_cpu.type(), "torch.FloatStorage")

        tensor_openreg = tensor_cpu.openreg()
        storage_openreg = tensor_openreg.storage()
        self.assertEqual(storage_openreg.type(), "torch.storage.TypedStorage")

        class CustomFloatStorage:
            @property
            def __module__(self):
                return "torch." + torch._C._get_privateuse1_backend_name()

            @property
            def __name__(self):
                return "FloatStorage"

        try:
            torch.openreg.FloatStorage = CustomFloatStorage()
            self.assertEqual(storage_openreg.type(), "torch.openreg.FloatStorage")

            # test custom int storage after defining FloatStorage
            tensor_openreg = tensor_cpu.int().openreg()
            storage_openreg = tensor_openreg.storage()
            self.assertEqual(storage_openreg.type(), "torch.storage.TypedStorage")
        finally:
            torch.openreg.FloatStorage = None

    def test_backend_tensor_methods(self):
        x = torch.empty(4, 4)
        self.assertFalse(x.is_openreg)  # type: ignore[misc]

        y = x.openreg(torch.device("openreg"))  # type: ignore[misc]
        self.assertTrue(y.is_openreg)  # type: ignore[misc]
        z = x.openreg(torch.device("openreg:0"))  # type: ignore[misc]
        self.assertTrue(z.is_openreg)  # type: ignore[misc]
        n = x.openreg(0)  # type: ignore[misc]
        self.assertTrue(n.is_openreg)  # type: ignore[misc]

    @unittest.skip("Need to support Parameter in openreg")
    def test_backend_module_methods(self):
        class FakeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.x = torch.nn.Parameter(torch.randn(3, 3))

            def forward(self):
                pass

        module = FakeModule()
        self.assertEqual(module.x.device.type, "cpu")
        module.openreg()  # type: ignore[misc]
        self.assertEqual(module.x.device.type, "openreg")

    @unittest.skip("Need to support untyped_storage in openreg")
    def test_backend_storage_methods(self):
        x = torch.empty(4, 4)

        x_cpu = x.storage()
        self.assertFalse(x_cpu.is_openreg)  # type: ignore[misc]
        x_openreg = x_cpu.openreg()  # type: ignore[misc]
        self.assertTrue(x_openreg.is_openreg)  # type: ignore[misc]

        y = torch.empty(4, 4)

        y_cpu = y.untyped_storage()
        self.assertFalse(y_cpu.is_openreg)  # type: ignore[misc]
        y_openreg = y_cpu.openreg()  # type: ignore[misc]
        self.assertTrue(y_openreg.is_openreg)  # type: ignore[misc]

    def test_backend_packed_sequence_methods(self):
        x = torch.rand(5, 3)
        y = torch.tensor([1, 1, 1, 1, 1])

        z_cpu = torch.nn.utils.rnn.PackedSequence(x, y)
        self.assertFalse(z_cpu.is_openreg)  # type: ignore[misc]

        z_openreg = z_cpu.openreg()  # type: ignore[misc]
        self.assertTrue(z_openreg.is_openreg)  # type: ignore[misc]


class TestOpenReg(TestCase):
    """Tests of mimick accelerator named OpenReg based on PrivateUse1"""

    # Stream & Event
    def test_stream_synchronize(self):
        stream = torch.Stream(device="openreg:1")
        stream.synchronize()
        self.assertEqual(True, stream.query())

    def test_stream_wait_stream(self):
        stream_1 = torch.Stream(device="openreg:0")
        stream_2 = torch.Stream(device="openreg:1")
        # Does not crash!
        stream_2.wait_stream(stream_1)

    @skipIfTorchDynamo()
    def test_record_event(self):
        stream = torch.Stream(device="openreg:1")
        event1 = stream.record_event()
        self.assertNotEqual(0, event1.event_id)
        event2 = stream.record_event()
        self.assertNotEqual(0, event2.event_id)
        self.assertNotEqual(event1.event_id, event2.event_id)

    @skipIfTorchDynamo()
    def test_event_elapsed_time(self):
        stream = torch.Stream(device="openreg:1")
        e1 = torch.Event(device="openreg:1", enable_timing=True)
        e1.record(stream)
        e2 = torch.Event(device="openreg:1", enable_timing=True)
        e2.record(stream)

        e2.synchronize()
        self.assertTrue(e2.query())

        ms = e1.elapsed_time(e2)
        self.assertTrue(ms > 0)

    @skipIfTorchDynamo()
    def test_stream_wait_event(self):
        s1 = torch.Stream(device="openreg")
        s2 = torch.Stream(device="openreg")
        e = s1.record_event()
        s2.wait_event(e)

    @skipIfTorchDynamo()
    def test_event_wait_stream(self):
        s1 = torch.Stream(device="openreg")
        s2 = torch.Stream(device="openreg")
        e1 = s1.record_event()
        e1.wait(s2)

    # Copy
    def test_cross_device_copy(self):
        a = torch.rand(10)
        b = a.to(device="openreg").add(2).to(device="cpu")
        self.assertEqual(b, a + 2)

    def test_copy_same_device(self):
        a = torch.ones(10, device="openreg").clone()
        self.assertEqual(a, torch.ones(10, device="openreg"))

    def test_cross_diff_devices_copy(self):
        a = torch.ones(10, device="openreg:0").to(device="openreg:1").to(device="cpu")
        self.assertEqual(a, torch.ones(10))

    # RNG
    def test_generator(self):
        generator = torch.Generator(device="openreg:1")
        self.assertEqual(generator.device.type, "openreg")
        self.assertEqual(generator.device.index, 1)

    def test_rng_state(self):
        state = torch.openreg.get_rng_state(0)  # type: ignore[misc]
        torch.openreg.set_rng_state(state, 0)  # type: ignore[misc]

    def test_manual_seed(self):
        torch.openreg.manual_seed_all(2024)  # type: ignore[misc]
        self.assertEqual(torch.openreg.initial_seed(), 2024)  # type: ignore[misc]

    # Autograd
    @unittest.skipIf(not IS_LINUX, "Only works on linux")
    def test_autograd_init(self):
        # Make sure autograd is initialized
        torch.ones(2, requires_grad=True, device="openreg").sum().backward()

        pid = os.getpid()
        task_path = f"/proc/{pid}/task"
        all_threads = psutil.Process(pid).threads()

        all_thread_names = set()

        for t in all_threads:
            with open(f"{task_path}/{t.id}/comm") as file:
                thread_name = file.read().strip()
            all_thread_names.add(thread_name)

        for i in range(torch.accelerator.device_count()):
            self.assertIn(f"pt_autograd_{i}", all_thread_names)

    # Storage & Pin Memory
    @skipIfTorchDynamo("unsupported aten.is_pinned.default")
    def test_pin_memory(self):
        tensor = torch.randn(10)
        self.assertFalse(tensor.is_pinned())
        pinned_tensor = tensor.pin_memory()
        self.assertTrue(pinned_tensor.is_pinned())
        slice_tensor = pinned_tensor[2:5]
        self.assertTrue(slice_tensor.is_pinned())

        tensor = torch.randn(10)
        storage = tensor.storage()
        self.assertFalse(storage.is_pinned("openreg"))
        pinned_storage = storage.pin_memory("openreg")
        self.assertTrue(pinned_storage.is_pinned("openreg"))

        tensor = torch.randn(10)
        untyped_storage = tensor.untyped_storage()
        self.assertFalse(untyped_storage.is_pinned("openreg"))
        pinned_untyped_storage = untyped_storage.pin_memory("openreg")
        self.assertTrue(pinned_untyped_storage.is_pinned("openreg"))

    @skipIfTorchDynamo("unsupported aten.is_pinned.default")
    def test_rewrapped_storage(self):
        pinned_a = torch.randn(10).pin_memory()
        rewrapped_a = torch.tensor((), dtype=torch.float32).set_(
            pinned_a.untyped_storage()[2:],
            size=(5,),
            stride=(1,),
            storage_offset=0,
        )
        self.assertTrue(rewrapped_a.is_pinned())
        self.assertNotEqual(pinned_a.data_ptr(), rewrapped_a.data_ptr())

    # Serialization
    @unittest.skip(
        "Temporarily disable due to the tiny differences between clang++ and g++ in defining static variable in inline function,"
        "this pr can fix this, https://github.com/pytorch/pytorch/pull/147095"
    )
    def test_serialization(self):
        storage = torch.UntypedStorage(4, device=torch.device("openreg"))
        self.assertEqual(torch.serialization.location_tag(storage), "openreg:0")

        storage = torch.UntypedStorage(4, device=torch.device("openreg:0"))
        self.assertEqual(torch.serialization.location_tag(storage), "openreg:0")

        storage_cpu = torch.empty(4, 4).storage()
        storage_openreg = torch.serialization.default_restore_location(
            storage_cpu, "openreg:0"
        )
        self.assertTrue(storage_openreg.is_openreg)  # type: ignore[misc]

        tensor = torch.empty(3, 3, device="openreg")
        self.assertEqual(torch._utils.get_tensor_metadata(tensor), {})  # type: ignore[misc]
        metadata = {"version_number": True, "format_number": True}
        torch._utils.set_tensor_metadata(tensor, metadata)  # type: ignore[misc]
        self.assertEqual(torch._utils.get_tensor_metadata(tensor), metadata)  # type: ignore[misc]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.pt")
            torch.save(tensor, path)

            tensor_openreg = torch.load(path)
            self.assertTrue(tensor_openreg.is_openreg)
            self.assertEqual(torch._utils.get_tensor_metadata(tensor_openreg), metadata)  # type: ignore[misc]

            tensor_cpu = torch.load(path, map_location="cpu")
            self.assertFalse(tensor_cpu.is_openreg)
            self.assertEqual(torch._utils.get_tensor_metadata(tensor_cpu), {})  # type: ignore[misc]

    # Opeartors
    def test_factory(self):
        x = torch.empty(3, device="openreg")
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.shape, torch.Size([3]))

        y = torch.zeros(3, device="openreg")
        self.assertEqual(y.device.type, "openreg")
        self.assertEqual(y.shape, torch.Size([3]))

        z = torch.tensor((), device="openreg")
        self.assertEqual(z.device.type, "openreg")
        self.assertEqual(z.shape, torch.Size([0]))

    def test_fake_tensor(self):
        with torch._subclasses.fake_tensor.FakeTensorMode():
            a = torch.empty(1, device="openreg")
            b = torch.empty(1, device="openreg:0")
            result = a + b  # noqa: F841

    def test_named_tensor(self):
        return torch.empty([2, 3, 4, 5], device="openreg", names=["N", "C", "H", "W"])

    def test_printing(self):
        a = torch.ones(20, device="openreg")
        # Does not crash!
        str(a)

    def test_data_dependent_output(self):
        cpu_a = torch.randn(10)
        a = cpu_a.to(device="openreg")
        mask = a.gt(0)
        out = torch.masked_select(a, mask)

        self.assertEqual(out, cpu_a.masked_select(cpu_a.gt(0)))

    def test_expand(self):
        x = torch.tensor([[1], [2], [3]], device="openreg")
        y = x.expand(3, 2)
        self.assertEqual(y.to(device="cpu"), torch.tensor([[1, 1], [2, 2], [3, 3]]))
        self.assertEqual(x.data_ptr(), y.data_ptr())

    def test_resize(self):
        tensor_cpu = torch.randn([4, 4])

        tensor_openreg = tensor_cpu.openreg()
        self.assertTrue(tensor_openreg.size() == torch.Size([4, 4]))

        storage_openreg = tensor_openreg.storage()
        self.assertTrue(storage_openreg.size() == 16)

        tensor_openreg.resize_(2, 2, 2, 2)
        self.assertTrue(tensor_openreg.size() == torch.Size([2, 2, 2, 2]))

        storage_openreg = tensor_openreg.storage()
        self.assertTrue(storage_openreg.size() == 16)

    # Quantize
    @skipIfXpu(msg="missing kernel for openreg")
    def test_quantize(self):
        x = torch.randn(3, 4, 5, dtype=torch.float32, device="openreg")
        quantized_tensor = torch.quantize_per_tensor(x, 0.1, 10, torch.qint8)
        self.assertEqual(quantized_tensor.device, torch.device("openreg:0"))
        self.assertEqual(quantized_tensor.dtype, torch.qint8)

    # custom autograd
    def test_compile_autograd_function_returns_self(self):
        in_ref = torch.randn(4, device="openreg", requires_grad=True)
        out_ref = torch.ops.openreg.custom_autograd_fn_returns_self(in_ref)
        out_ref.sum().backward()

        in_test = in_ref.detach().clone().requires_grad_(True)
        # TODO(FFFrog): Need to support inductor for OpenReg first.
        out_test = torch.compile(backend="aot_eager")(
            torch.ops.openreg.custom_autograd_fn_returns_self
        )(in_test)
        out_test.sum().backward()

        self.assertEqual(out_ref, out_test)
        self.assertEqual(in_ref.grad, in_test.grad)

    @skipIfTorchDynamo("Temporary disabled due to torch._ops.OpOverloadPacket")
    def test_compile_autograd_function_aliasing(self):
        in_ref = torch.randn(4, device="openreg", requires_grad=True)
        out_ref = torch.ops.openreg.custom_autograd_fn_aliasing(in_ref)
        out_ref.sum().backward()

        in_test = in_ref.detach().clone().requires_grad_(True)
        # TODO(FFFrog): Need to support inductor for OpenReg first.
        out_test = torch.compile(backend="aot_eager")(
            torch.ops.openreg.custom_autograd_fn_aliasing
        )(in_test)
        out_test.sum().backward()

        self.assertEqual(out_ref, out_test)
        self.assertEqual(in_ref.grad, in_test.grad)


if __name__ == "__main__":
    run_tests()
