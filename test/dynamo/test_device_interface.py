# Owner(s): ["module: dynamo"]
import torch
import torch._dynamo.test_case
from torch._dynamo.device_interface import (
    CpuInterface,
    CudaInterface,
    DeviceInterface,
    get_interface_for_device,
    get_registered_device_interfaces,
    MpsInterface,
    MtiaInterface,
    TpuInterface,
    XpuInterface,
    BackendFeature,
)


# The capabilities each in-tree DeviceInterface is expected to advertise.
# Keep in sync with the overrides in torch/_dynamo/device_interface.py: this
# table is the executable spec of "backend_features matches real capability".
EXPECTED_FEATURES: dict[type[DeviceInterface], set[BackendFeature]] = {
    DeviceInterface: set(),
    CudaInterface: {
        BackendFeature.FOREACH,
        BackendFeature.BUCKETIZE,
        BackendFeature.SCAN,
        BackendFeature.SORT,
        BackendFeature.TRITON_TEMPLATES,
        BackendFeature.GPU,
        BackendFeature.ONLINE_SOFTMAX,
    },
    XpuInterface: {
        BackendFeature.FOREACH,
        BackendFeature.TRITON_TEMPLATES,
        BackendFeature.GPU,
        BackendFeature.ONLINE_SOFTMAX,
    },
    MtiaInterface: {BackendFeature.GPU},
    MpsInterface: {BackendFeature.GPU},
    CpuInterface: {BackendFeature.FOREACH, BackendFeature.SORT},
    # TpuInterface deliberately does not override backend_features: the
    # historical GPU_TYPES list never contained "tpu".
    TpuInterface: set(),
}

# Device types that appeared in the hardcoded torch._inductor.utils.GPU_TYPES
# list before it was derived from BackendFeature.GPU.
HISTORICAL_GPU_TYPES = {"cuda", "mps", "xpu", "mtia"}

# Backends whose online-softmax lowering the inductor pattern used to gate on
# via `device_type in ["cuda", "xpu"]`.
HISTORICAL_ONLINE_SOFTMAX_TYPES = {"cuda", "xpu"}


class TestBackendFeatures(torch._dynamo.test_case.TestCase):
    """
    Unit tests for DeviceInterface.backend_features(). These are pure capability
    queries: none of them may require the corresponding device to be available.
    """

    def test_base_default_is_empty(self):
        # Unknown / not-yet-migrated backends stay conservative.
        self.assertEqual(DeviceInterface.backend_features(), set())
        self.assertEqual(DeviceInterface.backend_features(None), set())

    def test_in_tree_overrides(self):
        for iface, expected in EXPECTED_FEATURES.items():
            with self.subTest(interface=iface.__name__):
                self.assertEqual(iface.backend_features(None), expected)

    def test_tpu_does_not_override(self):
        # TpuInterface inherits the conservative base implementation. If it ever
        # gains an override, the expected-capability table and the GPU_TYPES
        # regression test both need updating.
        self.assertNotIn("backend_features", TpuInterface.__dict__)

    def test_gpu_membership_matches_historical_gpu_types(self):
        # BackendFeature.GPU is the new semantic root of GPU_TYPES.
        advertises_gpu = {
            name
            for name, iface in get_registered_device_interfaces()
            if ":" not in name and BackendFeature.GPU in iface.backend_features(None)
        }
        self.assertEqual(advertises_gpu, HISTORICAL_GPU_TYPES)

    def test_online_softmax_membership_matches_historical_list(self):
        advertises_online_softmax = {
            name
            for name, iface in get_registered_device_interfaces()
            if ":" not in name
            and BackendFeature.ONLINE_SOFTMAX in iface.backend_features(None)
        }
        self.assertEqual(advertises_online_softmax, HISTORICAL_ONLINE_SOFTMAX_TYPES)

    def test_cpu_is_not_a_gpu(self):
        self.assertNotIn(BackendFeature.GPU, CpuInterface.backend_features(None))

    def test_registry_dispatch_is_consistent(self):
        # Querying through the registry yields the same set as the class itself.
        for name, iface in get_registered_device_interfaces():
            with self.subTest(device=name):
                self.assertEqual(
                    get_interface_for_device(name).backend_features(None),
                    iface.backend_features(None),
                )
                if iface in EXPECTED_FEATURES:
                    self.assertEqual(
                        iface.backend_features(None), EXPECTED_FEATURES[iface]
                    )

    def test_every_registered_interface_is_covered(self):
        # Guards against a new in-tree interface landing without an entry in
        # EXPECTED_FEATURES (which would silently skip it above).
        for name, iface in get_registered_device_interfaces():
            with self.subTest(device=name):
                self.assertIn(iface, EXPECTED_FEATURES)

    def test_indexed_device_names_match_their_base_type(self):
        # "cuda:0" and "cuda" must advertise the same capabilities.
        for name, iface in get_registered_device_interfaces():
            if ":" not in name:
                continue
            base = name.split(":", 1)[0]
            with self.subTest(device=name):
                self.assertEqual(
                    iface.backend_features(None),
                    get_interface_for_device(base).backend_features(None),
                )

    def test_accepts_device_argument_forms(self):
        # The device argument is accepted (and currently ignored) in every form,
        # and no device needs to be present for the query to succeed.
        for device in (None, "cuda", torch.device("cuda"), torch.device("cuda", 0), 0):
            with self.subTest(device=device):
                self.assertEqual(
                    CudaInterface.backend_features(device),
                    EXPECTED_FEATURES[CudaInterface],
                )

    def test_returns_fresh_set_each_call(self):
        # A caller must not be able to corrupt a backend's advertised capability
        # by mutating a set it was handed.
        features = CudaInterface.backend_features(None)
        features.add(BackendFeature.INPLACE_BUFFERS)
        self.assertEqual(
            CudaInterface.backend_features(None), EXPECTED_FEATURES[CudaInterface]
        )

    def test_advertised_features_are_enum_members(self):
        for iface in EXPECTED_FEATURES:
            with self.subTest(interface=iface.__name__):
                for feature in iface.backend_features(None):
                    self.assertIsInstance(feature, BackendFeature)

    def test_out_of_tree_subclass_defaults_conservative(self):
        # An OOT backend that does not opt in advertises nothing, and in
        # particular is not treated as a GPU.
        class OutOfTreeInterface(DeviceInterface):
            pass

        self.assertEqual(OutOfTreeInterface.backend_features(None), set())
        self.assertNotIn(BackendFeature.GPU, OutOfTreeInterface.backend_features(None))

    def test_out_of_tree_subclass_can_advertise(self):
        # An OOT backend advertises by overriding the method -- no monkeypatching
        # of upstream symbols required.
        class OutOfTreeGpuInterface(DeviceInterface):
            @staticmethod
            def backend_features(device=None):
                return {BackendFeature.GPU, BackendFeature.FOREACH}

        self.assertIn(BackendFeature.GPU, OutOfTreeGpuInterface.backend_features(None))
        self.assertNotIn(
            BackendFeature.SORT, OutOfTreeGpuInterface.backend_features(None)
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
