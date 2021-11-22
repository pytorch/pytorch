from torch.testing._internal.common_utils import TestCase, run_tests
from torch._mock_dispatcher.library import Library
from torch._mock_dispatcher.dispatcher import Dispatcher
from torch._mock_dispatcher.dispatch_key import DispatchKey, getDispatchTableIndexForDispatchKey
from torch._mock_dispatcher.dispatch_key_set import DispatchKeySet

# Below are some basic expecttests and runtime tests for the python implementation of the dispatcher.
# These tests aren't comprehensive, because the goal of writing the dispatcher in python
# is mostly about being able to prototype dispatcher changes before moving them into C++.
#
# You can prototype a change to the python dispatcher, make sure it passes these tests for sanity,
# and then try the change in C++. If any tests fail in C++, we should be able to quickly
# replicate the test here as needed.
class TestMockDispatchKey(TestCase):

    def test_undefined(self):
        ks = DispatchKeySet.from_keys([])
        self.assertExpectedInline(ks.to_padded_bin_str(), """0000000000000000000000000000000000000000000000000000000000000000""")
        self.assertEqual(0, getDispatchTableIndexForDispatchKey(ks.highestPriorityTypeId()))

    def test_cpu(self):
        ks = DispatchKeySet.from_keys([DispatchKey.CPU])
        self.assertExpectedInline(ks.to_padded_bin_str(), """0000000000000000000000000000000000000000000000000000000000000001""")
        self.assertEqual(1, getDispatchTableIndexForDispatchKey(ks.highestPriorityTypeId()))

    def test_cuda(self):
        ks = DispatchKeySet.from_keys([DispatchKey.CUDA])
        self.assertExpectedInline(ks.to_padded_bin_str(), """0000000000000000000000000000000000000000000000000000000000000010""")
        self.assertEqual(2, getDispatchTableIndexForDispatchKey(ks.highestPriorityTypeId()))

    def test_cpu_cuda(self):
        ks = DispatchKeySet.from_keys([DispatchKey.CPU, DispatchKey.CUDA])
        self.assertExpectedInline(ks.to_padded_bin_str(), """0000000000000000000000000000000000000000000000000000000000000011""")
        self.assertEqual(2, getDispatchTableIndexForDispatchKey(ks.highestPriorityTypeId()))

    def test_cpu_cuda_autograd(self):
        ks = DispatchKeySet.from_keys([DispatchKey.CUDA, DispatchKey.AutogradCUDA, DispatchKey.CPU, DispatchKey.AutogradCPU])
        self.assertExpectedInline(ks.to_padded_bin_str(), """0000000000000000000000000001100000000000000000000000000000000011""")
        self.assertEqual(37, getDispatchTableIndexForDispatchKey(ks.highestPriorityTypeId()))

    def test_batched_named_xla_cpu(self):
        ks = DispatchKeySet.from_keys([DispatchKey.Batched, DispatchKey.Named, DispatchKey.XLA, DispatchKey.CPU])
        self.assertExpectedInline(ks.to_padded_bin_str(), """0000000000000001000000000000001000000000000000000000000000100001""")
        self.assertEqual(49, getDispatchTableIndexForDispatchKey(ks.highestPriorityTypeId()))



class TestMockDispatch(TestCase):

    def setUp(self):
        Dispatcher.singleton().reset()
        self.lib_cpu = Library("test", DispatchKey.CPU)
        self.lib_cuda = Library("test", DispatchKey.CUDA)
        self.lib_composite_implicit = Library("test", DispatchKey.CompositeImplicitAutograd)
        self.lib_composite_explicit = Library("test", DispatchKey.CompositeExplicitAutograd)
        self._foo1_cpu = 0
        self._foo1_cuda = 0
        self._foo1_composite_implicit = 0
        self._foo1_composite_explicit = 0

    def cpu_incr(self):
        self._foo1_cpu += 1

    def cuda_incr(self):
        self._foo1_cuda += 1

    def composite_implicit_incr(self):
        self._foo1_composite_implicit += 1

    def composite_explicit_incr(self):
        self._foo1_composite_explicit += 1

    def test_noop(self) -> None:
        self.lib_cpu.impl("foo1", self.cpu_incr)
        self.lib_cuda.impl("foo1", self.cuda_incr)
        Dispatcher.singleton().dumpRuntimeState("foo1")
        self.assertExpectedInline(Dispatcher.singleton().dumpRuntimeState("foo1"), '''\
name: foo1

Undefined: _missing_fn
CPU: cpu_incr
CUDA: cuda_incr
HIP: _missing_fn
FPGA: _missing_fn
ORT: _missing_fn
XLA: _missing_fn
Lazy: _missing_fn
Vulkan: _missing_fn
Metal: _missing_fn
XPU: _missing_fn
MKLDNN: _missing_fn
OpenGL: _missing_fn
OpenCL: _missing_fn
IDEEP: _missing_fn
QuantizedCPU: _missing_fn
QuantizedCUDA: _missing_fn
QuantizedXPU: _missing_fn
CustomRNGKeyId: _missing_fn
MkldnnCPU: _missing_fn
SparseCPU: _missing_fn
SparseCUDA: _missing_fn
SparseCsrCPU: _missing_fn
SparseCsrCUDA: _missing_fn
SparseHIP: _missing_fn
SparseXPU: _missing_fn
NestedTensor: _missing_fn
MLC: _missing_fn
HPU: _missing_fn
PrivateUse1: _missing_fn
PrivateUse2: _missing_fn
PrivateUse3: _missing_fn
Meta: _missing_fn
BackendSelect: _missing_fn
Named: _missing_fn
AutogradOther: _missing_fn
AutogradCPU: _missing_fn
AutogradCUDA: _missing_fn
AutogradXLA: _missing_fn
AutogradLazy: _missing_fn
AutogradNestedTensor: _missing_fn
AutogradMLC: _missing_fn
AutogradHPU: _missing_fn
AutogradXPU: _missing_fn
AutogradPrivateUse1: _missing_fn
AutogradPrivateUse2: _missing_fn
AutogradPrivateUse3: _missing_fn
Tracer: _missing_fn
Autocast: _missing_fn
Batched: _missing_fn
VmapMode: _missing_fn
TESTING_ONLY_GenericWrapper: _missing_fn
TESTING_ONLY_GenericMode: _missing_fn
NumDispatchKeys: _missing_fn''')

        self.assertEqual(0, self._foo1_cpu)
        self.assertEqual(0, self._foo1_cuda)
        self.assertEqual(0, self._foo1_composite_implicit)
        self.assertEqual(0, self._foo1_composite_explicit)

    def test_cpu_and_cuda_mixed(self) -> None:
        self.lib_cpu.impl("foo1", self.cpu_incr)
        self.lib_cuda.impl("foo1", self.cuda_incr)
        self.assertExpectedInline(Dispatcher.singleton().dumpRuntimeState("foo1"), '''\
name: foo1

Undefined: _missing_fn
CPU: cpu_incr
CUDA: cuda_incr
HIP: _missing_fn
FPGA: _missing_fn
ORT: _missing_fn
XLA: _missing_fn
Lazy: _missing_fn
Vulkan: _missing_fn
Metal: _missing_fn
XPU: _missing_fn
MKLDNN: _missing_fn
OpenGL: _missing_fn
OpenCL: _missing_fn
IDEEP: _missing_fn
QuantizedCPU: _missing_fn
QuantizedCUDA: _missing_fn
QuantizedXPU: _missing_fn
CustomRNGKeyId: _missing_fn
MkldnnCPU: _missing_fn
SparseCPU: _missing_fn
SparseCUDA: _missing_fn
SparseCsrCPU: _missing_fn
SparseCsrCUDA: _missing_fn
SparseHIP: _missing_fn
SparseXPU: _missing_fn
NestedTensor: _missing_fn
MLC: _missing_fn
HPU: _missing_fn
PrivateUse1: _missing_fn
PrivateUse2: _missing_fn
PrivateUse3: _missing_fn
Meta: _missing_fn
BackendSelect: _missing_fn
Named: _missing_fn
AutogradOther: _missing_fn
AutogradCPU: _missing_fn
AutogradCUDA: _missing_fn
AutogradXLA: _missing_fn
AutogradLazy: _missing_fn
AutogradNestedTensor: _missing_fn
AutogradMLC: _missing_fn
AutogradHPU: _missing_fn
AutogradXPU: _missing_fn
AutogradPrivateUse1: _missing_fn
AutogradPrivateUse2: _missing_fn
AutogradPrivateUse3: _missing_fn
Tracer: _missing_fn
Autocast: _missing_fn
Batched: _missing_fn
VmapMode: _missing_fn
TESTING_ONLY_GenericWrapper: _missing_fn
TESTING_ONLY_GenericMode: _missing_fn
NumDispatchKeys: _missing_fn''')

        Dispatcher.singleton().call("foo1", [DispatchKeySet(DispatchKey.CPU), DispatchKeySet(DispatchKey.CPU)])
        Dispatcher.singleton().call("foo1", [DispatchKeySet(DispatchKey.CUDA), DispatchKeySet(DispatchKey.CUDA)])
        Dispatcher.singleton().call("foo1", [DispatchKeySet(DispatchKey.CPU), DispatchKeySet(DispatchKey.CUDA)])
        self.assertEqual(1, self._foo1_cpu)
        self.assertEqual(2, self._foo1_cuda)
        self.assertEqual(0, self._foo1_composite_implicit)
        self.assertEqual(0, self._foo1_composite_explicit)

    def test_other_backends_not_registered(self) -> None:
        self.lib_cpu.impl("foo1", self.cpu_incr)
        self.lib_cuda.impl("foo1", self.cuda_incr)
        with self.assertRaises(AssertionError):
            Dispatcher.singleton().call("foo1", [DispatchKeySet(DispatchKey.XLA)])

    def test_autograd_not_registered(self) -> None:
        self.lib_cpu.impl("foo1", self.cpu_incr)
        self.lib_cuda.impl("foo1", self.cuda_incr)
        with self.assertRaises(AssertionError):
            Dispatcher.singleton().call("foo1", [DispatchKeySet(DispatchKey.AutogradCPU)])


    def test_composite_implicit_key(self) -> None:
        self.lib_cpu.impl("foo1", self.cpu_incr)
        self.lib_composite_implicit.impl("foo1", self.composite_implicit_incr)
        self.assertExpectedInline(Dispatcher.singleton().dumpRuntimeState("foo1"), '''\
name: foo1

Undefined: composite_implicit_incr
CPU: cpu_incr
CUDA: composite_implicit_incr
HIP: _missing_fn
FPGA: _missing_fn
ORT: composite_implicit_incr
XLA: composite_implicit_incr
Lazy: composite_implicit_incr
Vulkan: _missing_fn
Metal: _missing_fn
XPU: composite_implicit_incr
MKLDNN: _missing_fn
OpenGL: _missing_fn
OpenCL: _missing_fn
IDEEP: _missing_fn
QuantizedCPU: _missing_fn
QuantizedCUDA: _missing_fn
QuantizedXPU: _missing_fn
CustomRNGKeyId: _missing_fn
MkldnnCPU: _missing_fn
SparseCPU: _missing_fn
SparseCUDA: _missing_fn
SparseCsrCPU: _missing_fn
SparseCsrCUDA: _missing_fn
SparseHIP: _missing_fn
SparseXPU: _missing_fn
NestedTensor: _missing_fn
MLC: composite_implicit_incr
HPU: composite_implicit_incr
PrivateUse1: composite_implicit_incr
PrivateUse2: composite_implicit_incr
PrivateUse3: composite_implicit_incr
Meta: composite_implicit_incr
BackendSelect: _missing_fn
Named: _missing_fn
AutogradOther: _ambiguous_autograd_fn
AutogradCPU: _missing_fn
AutogradCUDA: composite_implicit_incr
AutogradXLA: composite_implicit_incr
AutogradLazy: composite_implicit_incr
AutogradNestedTensor: composite_implicit_incr
AutogradMLC: composite_implicit_incr
AutogradHPU: composite_implicit_incr
AutogradXPU: composite_implicit_incr
AutogradPrivateUse1: composite_implicit_incr
AutogradPrivateUse2: composite_implicit_incr
AutogradPrivateUse3: composite_implicit_incr
Tracer: _missing_fn
Autocast: _missing_fn
Batched: _missing_fn
VmapMode: _missing_fn
TESTING_ONLY_GenericWrapper: _missing_fn
TESTING_ONLY_GenericMode: _missing_fn
NumDispatchKeys: _missing_fn''')

        Dispatcher.singleton().call("foo1", [DispatchKeySet(DispatchKey.CPU)])
        Dispatcher.singleton().call("foo1", [DispatchKeySet(DispatchKey.XLA)])
        Dispatcher.singleton().call("foo1", [DispatchKeySet(DispatchKey.AutogradCUDA)])
        self.assertEqual(1, self._foo1_cpu)
        self.assertEqual(0, self._foo1_cuda)
        self.assertEqual(2, self._foo1_composite_implicit)
        self.assertEqual(0, self._foo1_composite_explicit)

        # All backend + autograd kernels work, but not the other keys
        with self.assertRaises(AssertionError):
            Dispatcher.singleton().call("foo1", [DispatchKeySet(DispatchKey.VmapMode)])

        # AutogradCPU doesn't work, because we've registered an explicit CPU kernel
        with self.assertRaises(AssertionError):
            Dispatcher.singleton().call("foo1", [DispatchKeySet(DispatchKey.AutogradCPU)])

if __name__ == '__main__':
    run_tests()
