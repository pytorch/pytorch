from torch.testing._internal.common_utils import TestCase, run_tests
from torch._mock_dispatcher.library import Library
from torch._mock_dispatcher.dispatcher import Dispatcher
from torch._mock_dispatcher.dispatch_key import DispatchKey, isRuntimeDispatchKey
from torch._mock_dispatcher.dispatch_key_set import DispatchKeySet, getDispatchTableIndexForDispatchKeySet, getRuntimeDispatchKeySet

# Below are some basic expecttests and runtime tests for the python implementation of the dispatcher.
# These tests aren't comprehensive, because the goal of writing the dispatcher in python
# is mostly about being able to prototype dispatcher changes before moving them into C++.
#
# You can prototype a change to the python dispatcher, make sure it passes these tests for sanity,
# and then try the change in C++. If any tests fail in C++, we should be able to quickly
# replicate the test here as needed.
class TestMockDispatchKey(TestCase):

    def assert_has_keys(self, keyset, list_of_keys):
        for contained_key in list_of_keys:
            self.assertTrue(keyset.has(contained_key))
        other_keys = [k for k in DispatchKey if
                      k not in list_of_keys
                      and isRuntimeDispatchKey(k)
                      and k != DispatchKey.Undefined]
        for other_key in other_keys:
            self.assertFalse(keyset.has(other_key))

    def test_has_dense_cuda(self):
        ks = DispatchKeySet.from_keys([DispatchKey.CUDA])
        # CUDA logically represents "Dense functionality, CUDABit backend"
        contained_keys = [DispatchKey.CUDA, DispatchKey.Dense, DispatchKey.CUDABit]
        self.assert_has_keys(ks, contained_keys)

    def test_has_dense_private_use(self):
        ks = DispatchKeySet.from_keys([DispatchKey.PrivateUse3])
        # PrivateUse3 logically represents "Dense functionality, PrivateUse3Bit backend"
        contained_keys = [DispatchKey.PrivateUse3, DispatchKey.Dense, DispatchKey.PrivateUse3Bit]
        self.assert_has_keys(ks, contained_keys)

    def test_has_autograd_cuda(self):
        ks = DispatchKeySet.from_keys([DispatchKey.AutogradCUDA])
        # AutogradCUDA logically represents "Autograd functionality, CUDABit backend"
        contained_keys = [DispatchKey.AutogradCUDA, DispatchKey.Autograd, DispatchKey.CUDABit]
        self.assert_has_keys(ks, contained_keys)

    def test_has_dense_and_autograd_cuda(self):
        ks = DispatchKeySet.from_keys([DispatchKey.AutogradCUDA, DispatchKey.CUDA])
        # Dense + AutogradCUDA logically represents "Autograd and Dense functionality, CUDABit backend"
        contained_keys = [DispatchKey.AutogradCUDA, DispatchKey.CUDA,
                          DispatchKey.Autograd, DispatchKey.Dense, DispatchKey.CUDABit]
        self.assert_has_keys(ks, contained_keys)

    def test_has_dense_cpu_and_autograd_xla(self):
        ks = DispatchKeySet.from_keys([DispatchKey.AutogradXLA, DispatchKey.CPU])
        # Since we have mixed device and functionality inputs, our keyset ends up representing their cross product.
        # (Both XLA and CPU backends, both Autograd and Dense functionality)
        contained_keys = [DispatchKey.AutogradXLA, DispatchKey.XLA, DispatchKey.AutogradCPU, DispatchKey.CPU,
                          DispatchKey.Autograd, DispatchKey.XLABit, DispatchKey.Dense, DispatchKey.CPUBit]
        self.assert_has_keys(ks, contained_keys)

    def test_has_fpga_and_autograd_other(self):
        ks = DispatchKeySet.from_keys([DispatchKey.FPGA, DispatchKey.AutogradOther])
        # FPGA doesn't get a backend bit. AutogradOther also isn't part of "normal" autograd,
        # and doesn't get an autograd functionality bit.
        contained_keys = [DispatchKey.FPGA, DispatchKey.AutogradOther]
        self.assert_has_keys(ks, contained_keys)

    def test_undefined(self):
        ks = DispatchKeySet.from_keys([])
        self.assertExpectedInline(ks.to_padded_bin_str(), """0000000000000000000000000000000000000000000000000000000000000000""")
        self.assertEqual(0, getDispatchTableIndexForDispatchKeySet(ks))

    def test_cpu(self):
        ks = DispatchKeySet.from_keys([DispatchKey.CPU])
        self.assertExpectedInline(ks.to_padded_bin_str(), """0000000000000000000000000000000000000000000000000001000000000001""")
        self.assertEqual(1, getDispatchTableIndexForDispatchKeySet(ks))

    def test_cuda(self):
        ks = DispatchKeySet.from_keys([DispatchKey.CUDA])
        self.assertExpectedInline(ks.to_padded_bin_str(), """0000000000000000000000000000000000000000000000000001000000000010""")
        self.assertEqual(2, getDispatchTableIndexForDispatchKeySet(ks))

    def test_private_use(self):
        ks = DispatchKeySet.from_keys([DispatchKey.PrivateUse3])
        self.assertExpectedInline(ks.to_padded_bin_str(), """0000000000000000000000000000000000000000000000000001100000000000""")
        self.assertEqual(12, getDispatchTableIndexForDispatchKeySet(ks))

    def test_cpu_cuda(self):
        ks = DispatchKeySet.from_keys([DispatchKey.CPU, DispatchKey.CUDA])
        self.assertExpectedInline(ks.to_padded_bin_str(), """0000000000000000000000000000000000000000000000000001000000000011""")
        self.assertEqual(2, getDispatchTableIndexForDispatchKeySet(ks))

    def test_fake_backend(self):
        # "fake" backends are backends that we don't want to waste runtime table space giving per-backend functionality to.
        # Instead, they take up a single functionality bit in the bitset.
        # FPGA offset start = 12
        # backend offset = 0 (masked because FPGA is not a per-backend functionality)
        # total = 12
        ks = DispatchKeySet.from_keys([DispatchKey.FPGA])
        self.assertExpectedInline(ks.to_padded_bin_str(), """0000000000000000000000000000000000000000000000000010000000000000""")
        self.assertEqual(13, getDispatchTableIndexForDispatchKeySet(ks))

    def test_fake_backend_with_autograd(self):
        # Since fake backends don't get the ability to customize autograd,
        # we ensure that they dispatch to the "AutogradOther" kernel.
        # This test assumes that this is set up correctly on the tensor class, and just tests the DispatchKeySet calculation bit.
        ks = DispatchKeySet.from_keys([DispatchKey.FPGA, DispatchKey.AutogradOther])
        self.assertExpectedInline(ks.to_padded_bin_str(), """0000000000000000000000000000000001000000000000000010000000000000""")
        # offset calculation:
        # - (Undefined) -> 0
        # - (Dense) -> 1 (...12)
        # - (FPGA...Meta) -> 13...23
        # - (Quantized) -> 24 (...35)
        # - (Sparse) -> 36 (...47)
        # - (SparseCsrCPU) -> 48
        # - (SparseCsrCUDA) -> 49
        # - (BackendSelect) -> 50
        # - (Named) -> 51
        # - (AutogradOther) -> 52
        # Final index = 52
        self.assertEqual(52, getDispatchTableIndexForDispatchKeySet(ks))

    def test_sparse_cuda(self):
        # "fake" backends are backends that we don't want to waste runtime table space giving per-backend functionality to.
        # Instead, they take up a single functionality bit in the bitset.
        ks = DispatchKeySet.from_keys([DispatchKey.SparseCUDA])
        self.assertExpectedInline(ks.to_padded_bin_str(), """0000000000000000000000000000000000000010000000000000000000000010""")
        # offset calculation:
        # - (Undefined) -> 0
        # - (Dense) -> 1 (...12)
        # - (FPGA...Meta) -> 13...23
        # - (Quantized) -> 24 (...35)
        # - (Sparse) -> 36 (...47)
        # Final index = 36 (functionality offset) + 1 (backend index) = 37
        self.assertEqual(37, getDispatchTableIndexForDispatchKeySet(ks))

    def test_cpu_xla_autograd(self):
        ks = DispatchKeySet.from_keys([DispatchKey.XLA, DispatchKey.AutogradXLA, DispatchKey.CPU, DispatchKey.AutogradCPU])
        self.assertExpectedInline(ks.to_padded_bin_str(), """0000000000000000000000000000000010000000000000000001000000001001""")
        # Autograd offset start = 53
        # xla backend offset = 3
        # total = 56
        self.assertEqual(56, getDispatchTableIndexForDispatchKeySet(ks))

    def test_batched_named_xla_cpu(self):
        # Batched offset start = 67
        # xla backend offset = 0 (masked out because batched is not a per-backend functionality
        # total = 67
        ks = DispatchKeySet.from_keys([DispatchKey.Batched, DispatchKey.Named, DispatchKey.XLA, DispatchKey.CPU])
        self.assertExpectedInline(ks.to_padded_bin_str(), """0000000000000000000000000000010000100000000000000001000000001001""")
        self.assertEqual(67, getDispatchTableIndexForDispatchKeySet(ks))

    def test_alias_keysets(self):
        ks1 = getRuntimeDispatchKeySet(DispatchKey.CompositeImplicitAutograd)
        ks2 = getRuntimeDispatchKeySet(DispatchKey.CompositeExplicitAutograd)
        ks3 = getRuntimeDispatchKeySet(DispatchKey.AutogradAlias)
        # CompositeImplicitAutograd functionality bits: Autograd, AutogradOther, Dense
        # CompositeExlicitAutograd functionality bits: Dense
        # AutogradAlias functionality bits: Autograd, AutogradOther
        # Also notice two weird discrepencies:
        # - CompositeExplicitAutograd is missing the NestedTensor backend bit.
        # - AutogradAlias is missing the ORT (fake functionality) bit.
        self.assertExpectedInline(ks1.to_padded_bin_str(), """0000000000000000000000000000000011000000100000000101111111111011""")
        self.assertExpectedInline(ks2.to_padded_bin_str(), """0000000000000000000000000000000000000000100000000101111110111011""")
        self.assertExpectedInline(ks3.to_padded_bin_str(), """0000000000000000000000000000000011000000000000000000111111111011""")

class TestMockDispatch(TestCase):

    def setUp(self):
        Dispatcher.singleton().reset()
        self.lib_cpu = Library("test", DispatchKey.CPU)
        self.lib_cuda = Library("test", DispatchKey.CUDA)
        self.lib_cpu_autograd = Library("test", DispatchKey.AutogradCPU)
        self.lib_cuda_autograd = Library("test", DispatchKey.AutogradCUDA)
        self.lib_composite_implicit = Library("test", DispatchKey.CompositeImplicitAutograd)
        self.lib_composite_explicit = Library("test", DispatchKey.CompositeExplicitAutograd)
        self.lib_undefined = Library("test", DispatchKey.Undefined)
        self._foo1_cpu = 0
        self._foo1_cuda = 0
        self._foo1_cpu_autograd = 0
        self._foo1_cuda_autograd = 0
        self._foo1_composite_implicit = 0
        self._foo1_composite_explicit = 0

    def cpu_incr(self):
        self._foo1_cpu += 1

    def cuda_incr(self):
        self._foo1_cuda += 1

    def cpu_autograd_incr(self):
        self._foo1_cpu_autograd += 1

    def cuda_autograd_incr(self):
        self._foo1_cuda_autograd += 1

    def composite_implicit_incr(self):
        self._foo1_composite_implicit += 1

    def composite_explicit_incr(self):
        self._foo1_composite_explicit += 1

    def undefined_error(self):
        raise AssertionError("undefined")

    def test_noop(self) -> None:
        self.lib_cpu.impl("foo1", self.cpu_incr)
        self.lib_cuda.impl("foo1", self.cuda_incr)
        self.assertExpectedInline(Dispatcher.singleton().dumpRuntimeState("foo1"), '''\
name: foo1

Undefined: _missing_fn
CPU: cpu_incr
CUDA: cuda_incr
HIP: _missing_fn
XLA: _missing_fn
Lazy: _missing_fn
XPU: _missing_fn
NestedTensor: _missing_fn
MLC: _missing_fn
HPU: _missing_fn
PrivateUse1: _missing_fn
PrivateUse2: _missing_fn
PrivateUse3: _missing_fn
FPGA: _missing_fn
ORT: _missing_fn
Vulkan: _missing_fn
Metal: _missing_fn
MKLDNN: _missing_fn
OpenGL: _missing_fn
OpenCL: _missing_fn
IDEEP: _missing_fn
CustomRNGKeyId: _missing_fn
MkldnnCPU: _missing_fn
Meta: _missing_fn
QuantizedCPU: _missing_fn
QuantizedCUDA: _missing_fn
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
QuantizedXPU: _missing_fn
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
SparseCPU: _missing_fn
SparseCUDA: _missing_fn
SparseHIP: _missing_fn
Key does not exist: [None]
Key does not exist: [None]
SparseXPU: _missing_fn
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
SparseCsrCPU: _missing_fn
SparseCsrCUDA: _missing_fn
BackendSelect: _missing_fn
Named: _missing_fn
AutogradOther: _missing_fn
AutogradCPU: _missing_fn
AutogradCUDA: _missing_fn
Key does not exist: [None]
AutogradXLA: _missing_fn
AutogradLazy: _missing_fn
AutogradXPU: _missing_fn
AutogradNestedTensor: _missing_fn
AutogradMLC: _missing_fn
AutogradHPU: _missing_fn
AutogradPrivateUse1: _missing_fn
AutogradPrivateUse2: _missing_fn
AutogradPrivateUse3: _missing_fn
Tracer: _missing_fn
Autocast: _missing_fn
Batched: _missing_fn
VmapMode: _missing_fn
TESTING_ONLY_GenericWrapper: _missing_fn
TESTING_ONLY_GenericMode: _missing_fn''')

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
XLA: _missing_fn
Lazy: _missing_fn
XPU: _missing_fn
NestedTensor: _missing_fn
MLC: _missing_fn
HPU: _missing_fn
PrivateUse1: _missing_fn
PrivateUse2: _missing_fn
PrivateUse3: _missing_fn
FPGA: _missing_fn
ORT: _missing_fn
Vulkan: _missing_fn
Metal: _missing_fn
MKLDNN: _missing_fn
OpenGL: _missing_fn
OpenCL: _missing_fn
IDEEP: _missing_fn
CustomRNGKeyId: _missing_fn
MkldnnCPU: _missing_fn
Meta: _missing_fn
QuantizedCPU: _missing_fn
QuantizedCUDA: _missing_fn
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
QuantizedXPU: _missing_fn
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
SparseCPU: _missing_fn
SparseCUDA: _missing_fn
SparseHIP: _missing_fn
Key does not exist: [None]
Key does not exist: [None]
SparseXPU: _missing_fn
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
SparseCsrCPU: _missing_fn
SparseCsrCUDA: _missing_fn
BackendSelect: _missing_fn
Named: _missing_fn
AutogradOther: _missing_fn
AutogradCPU: _missing_fn
AutogradCUDA: _missing_fn
Key does not exist: [None]
AutogradXLA: _missing_fn
AutogradLazy: _missing_fn
AutogradXPU: _missing_fn
AutogradNestedTensor: _missing_fn
AutogradMLC: _missing_fn
AutogradHPU: _missing_fn
AutogradPrivateUse1: _missing_fn
AutogradPrivateUse2: _missing_fn
AutogradPrivateUse3: _missing_fn
Tracer: _missing_fn
Autocast: _missing_fn
Batched: _missing_fn
VmapMode: _missing_fn
TESTING_ONLY_GenericWrapper: _missing_fn
TESTING_ONLY_GenericMode: _missing_fn''')

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

    def test_undefined(self) -> None:
        self.lib_cpu.impl("foo1", self.cpu_incr)
        self.lib_cuda.impl("foo1", self.cuda_incr)
        with self.assertRaises(AssertionError):
            Dispatcher.singleton().call("foo1", [])

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
XLA: composite_implicit_incr
Lazy: composite_implicit_incr
XPU: composite_implicit_incr
NestedTensor: composite_implicit_incr
MLC: composite_implicit_incr
HPU: composite_implicit_incr
PrivateUse1: composite_implicit_incr
PrivateUse2: composite_implicit_incr
PrivateUse3: composite_implicit_incr
FPGA: _missing_fn
ORT: composite_implicit_incr
Vulkan: _missing_fn
Metal: _missing_fn
MKLDNN: _missing_fn
OpenGL: _missing_fn
OpenCL: _missing_fn
IDEEP: _missing_fn
CustomRNGKeyId: _missing_fn
MkldnnCPU: _missing_fn
Meta: composite_implicit_incr
QuantizedCPU: _missing_fn
QuantizedCUDA: _missing_fn
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
QuantizedXPU: _missing_fn
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
SparseCPU: _missing_fn
SparseCUDA: _missing_fn
SparseHIP: _missing_fn
Key does not exist: [None]
Key does not exist: [None]
SparseXPU: _missing_fn
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
SparseCsrCPU: _missing_fn
SparseCsrCUDA: _missing_fn
BackendSelect: _missing_fn
Named: _missing_fn
AutogradOther: _ambiguous_autograd_fn
AutogradCPU: _missing_fn
AutogradCUDA: composite_implicit_incr
Key does not exist: [None]
AutogradXLA: composite_implicit_incr
AutogradLazy: composite_implicit_incr
AutogradXPU: composite_implicit_incr
AutogradNestedTensor: composite_implicit_incr
AutogradMLC: composite_implicit_incr
AutogradHPU: composite_implicit_incr
AutogradPrivateUse1: composite_implicit_incr
AutogradPrivateUse2: composite_implicit_incr
AutogradPrivateUse3: composite_implicit_incr
Tracer: _missing_fn
Autocast: _missing_fn
Batched: _missing_fn
VmapMode: _missing_fn
TESTING_ONLY_GenericWrapper: _missing_fn
TESTING_ONLY_GenericMode: _missing_fn''')

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

        # The current behavior of the system is actually that CompositeImplicitAutograd kernels
        # get registered to the Undefined dispatch key.
        # (This probably doesn't matter in practice, since they'll end up redispatching to a base op's
        # undefined kernel and erroring out a bit later).
        Dispatcher.singleton().call("foo1", [])
        self.assertEqual(3, self._foo1_composite_implicit)

    def test_fallthrough_simple(self) -> None:
        # Give every backend the same fallthrough kernel.
        # This should cause us to hit the fast-path logic for calculating fallthrough keys.
        autograd_libs = []
        for dispatch_key_idx in range(DispatchKey.AutogradCPU.value, DispatchKey.AutogradPrivateUse3.value + 1):
            lib = Library("test", DispatchKey(dispatch_key_idx))
            lib.impl_fallthrough("foo1")
            autograd_libs.append(lib)

        self.lib_cpu.impl("foo1", self.cpu_incr)
        self.lib_cuda.impl("foo1", self.cuda_incr)

        Dispatcher.singleton().call("foo1", [DispatchKeySet(DispatchKey.AutogradCPU), DispatchKeySet(DispatchKey.CPU)])
        self.assertEqual(1, self._foo1_cpu)
        Dispatcher.singleton().call("foo1", [DispatchKeySet(DispatchKey.AutogradCUDA), DispatchKeySet(DispatchKey.CUDA)])
        self.assertEqual(1, self._foo1_cuda)

    def test_fallbacks_and_fallthroughs(self) -> None:
        # give cpu / cuda ordinary kernels
        self.lib_cpu.impl("foo1", self.cpu_incr)
        self.lib_cuda.impl("foo1", self.cuda_incr)

        self.lib_composite_implicit.impl("foo1", self.composite_implicit_incr)

        # per-backend fallthrough
        # give cuda autograd a fallthrough, but give cpu autograd a fallback
        self.lib_cuda_autograd.impl_fallthrough("foo1")
        self.lib_cpu_autograd.fallback(self.cpu_autograd_incr)

        # Give undefined its own function so we can easily tell when the undefined kernel is called.
        # Otherwise, the undefined key will get a composite kernel.
        self.lib_undefined.impl("foo1", self.undefined_error)

        self.assertExpectedInline(Dispatcher.singleton().dumpRuntimeState("foo1"), '''\
name: foo1

Undefined: undefined_error
CPU: cpu_incr
CUDA: cuda_incr
HIP: _missing_fn
XLA: composite_implicit_incr
Lazy: composite_implicit_incr
XPU: composite_implicit_incr
NestedTensor: composite_implicit_incr
MLC: composite_implicit_incr
HPU: composite_implicit_incr
PrivateUse1: composite_implicit_incr
PrivateUse2: composite_implicit_incr
PrivateUse3: composite_implicit_incr
FPGA: _missing_fn
ORT: composite_implicit_incr
Vulkan: _missing_fn
Metal: _missing_fn
MKLDNN: _missing_fn
OpenGL: _missing_fn
OpenCL: _missing_fn
IDEEP: _missing_fn
CustomRNGKeyId: _missing_fn
MkldnnCPU: _missing_fn
Meta: composite_implicit_incr
QuantizedCPU: _missing_fn
QuantizedCUDA: _missing_fn
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
QuantizedXPU: _missing_fn
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
SparseCPU: _missing_fn
SparseCUDA: _missing_fn
SparseHIP: _missing_fn
Key does not exist: [None]
Key does not exist: [None]
SparseXPU: _missing_fn
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
Key does not exist: [None]
SparseCsrCPU: _missing_fn
SparseCsrCUDA: _missing_fn
BackendSelect: _missing_fn
Named: _missing_fn
AutogradOther: _ambiguous_autograd_fn
AutogradCPU: cpu_autograd_incr
AutogradCUDA: _fallthrough_fn
Key does not exist: [None]
AutogradXLA: composite_implicit_incr
AutogradLazy: composite_implicit_incr
AutogradXPU: composite_implicit_incr
AutogradNestedTensor: composite_implicit_incr
AutogradMLC: composite_implicit_incr
AutogradHPU: composite_implicit_incr
AutogradPrivateUse1: composite_implicit_incr
AutogradPrivateUse2: composite_implicit_incr
AutogradPrivateUse3: composite_implicit_incr
Tracer: _missing_fn
Autocast: _missing_fn
Batched: _missing_fn
VmapMode: _missing_fn
TESTING_ONLY_GenericWrapper: _missing_fn
TESTING_ONLY_GenericMode: _missing_fn''')

        Dispatcher.singleton().call("foo1", [DispatchKeySet(DispatchKey.CPU)])
        self.assertEqual(1, self._foo1_cpu)
        Dispatcher.singleton().call("foo1", [DispatchKeySet(DispatchKey.CUDA)])
        self.assertEqual(1, self._foo1_cuda)
        # AutogradCUDA doesn't work, because we've registered an a fallthrough kernel to it.
        with self.assertRaises(AssertionError):
            Dispatcher.singleton().call("foo1", [DispatchKeySet(DispatchKey.AutogradCUDA)])
            # And [AutogradCUDA, AutogradCPU] doesn't work because:
            # (1) CUDA is the highest priority backend bit, so we look at CUDA fallthrough keys
            # (2) CUDA has a fallthrough registered to Autograd, so we skip the Autograd functionality bit
            # (3) there are no other functionality bits set so we hit the Undefined kernel
            Dispatcher.singleton().call("foo1", [DispatchKeySet(DispatchKey.AutogradCUDA), DispatchKeySet(DispatchKey.AutogradCPU)])
        # AutogradCPU works though, because we've registered a boxed fallback to it.
        Dispatcher.singleton().call("foo1", [DispatchKeySet(DispatchKey.AutogradCPU)])
        self.assertEqual(1, self._foo1_cpu_autograd)

        # AutogradCPU has higher priority than CPU, so we hit AutogradCPU
        Dispatcher.singleton().call("foo1", [DispatchKeySet(DispatchKey.AutogradCPU), DispatchKeySet(DispatchKey.CPU)])
        self.assertEqual(2, self._foo1_cpu_autograd)

        # AutogradCUDA normally has higher priority, but AutogradCUDA has a fallthrough so we hit CUDA
        Dispatcher.singleton().call("foo1", [DispatchKeySet(DispatchKey.AutogradCUDA), DispatchKeySet(DispatchKey.CUDA)])
        self.assertEqual(2, self._foo1_cuda)

        # Mixed device: AutogradCUDA normally has priority, but because it has a fallthrough registered then we go to CUDA
        # The "Autograd" bit is zero'd out by the fallthrough key bitset
        # This behavior is DIFFERENT from the old behavior: previously, we would go to AutogradCPU.
        Dispatcher.singleton().call("foo1", [
            DispatchKeySet(DispatchKey.AutogradCUDA),
            DispatchKeySet(DispatchKey.AutogradCPU),
            DispatchKeySet(DispatchKey.CUDA),
            DispatchKeySet(DispatchKey.CPU)])
        self.assertEqual(3, self._foo1_cuda)

if __name__ == '__main__':
    run_tests()
