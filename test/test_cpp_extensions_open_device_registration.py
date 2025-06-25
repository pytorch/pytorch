# Owner(s): ["module: cpp-extensions"]

import _codecs
import io
import os
import unittest
from unittest.mock import patch

import numpy as np
import pytorch_openreg  # noqa: F401

import torch
import torch.testing._internal.common_utils as common
import torch.utils.cpp_extension
from torch.serialization import safe_globals
from torch.testing._internal.common_utils import TemporaryFileName


@unittest.skipIf(common.TEST_XPU, "XPU does not support cppextension currently")
@common.markDynamoStrictTest
class TestCppExtensionOpenRegistration(common.TestCase):
    """Tests Open Device Registration with C++ extensions."""

    module = None

    def setUp(self):
        super().setUp()

        # cpp extensions use relative paths. Those paths are relative to
        # this file, so we'll change the working directory temporarily
        self.old_working_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        assert self.module is not None

    def tearDown(self):
        super().tearDown()

        # return the working directory (see setUp)
        os.chdir(self.old_working_dir)

    @classmethod
    def setUpClass(cls):
        common.remove_cpp_extensions_build_root()

        cls.module = torch.utils.cpp_extension.load(
            name="custom_device_extension",
            sources=[
                "cpp_extensions/open_registration_extension.cpp",
            ],
            extra_include_paths=["cpp_extensions"],
            extra_cflags=["-g"],
            verbose=True,
        )

    def test_open_device_scalar_type_fallback(self):
        z_cpu = torch.Tensor([[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]).to(torch.int64)
        z = torch.triu_indices(3, 3, device="openreg")
        self.assertEqual(z_cpu, z)

    def test_open_device_tensor_type_fallback(self):
        # create tensors located in custom device
        x = torch.Tensor([[1, 2, 3], [2, 3, 4]]).to("openreg")
        y = torch.Tensor([1, 0, 2]).to("openreg")
        # create result tensor located in cpu
        z_cpu = torch.Tensor([[0, 2, 1], [1, 3, 2]])
        # Check that our device is correct.
        device = self.module.custom_device()
        self.assertTrue(x.device == device)
        self.assertFalse(x.is_cpu)

        # call sub op, which will fallback to cpu
        z = torch.sub(x, y)
        self.assertEqual(z_cpu, z)

        # call index op, which will fallback to cpu
        z_cpu = torch.Tensor([3, 1])
        y = torch.Tensor([1, 0]).long().to("openreg")
        z = x[y, y]
        self.assertEqual(z_cpu, z)

    def test_open_device_tensorlist_type_fallback(self):
        # create tensors located in custom device
        v_openreg = torch.Tensor([1, 2, 3]).to("openreg")
        # create result tensor located in cpu
        z_cpu = torch.Tensor([2, 4, 6])
        # create tensorlist for foreach_add op
        x = (v_openreg, v_openreg)
        y = (v_openreg, v_openreg)
        # Check that our device is correct.
        device = self.module.custom_device()
        self.assertTrue(v_openreg.device == device)
        self.assertFalse(v_openreg.is_cpu)

        # call _foreach_add op, which will fallback to cpu
        z = torch._foreach_add(x, y)
        self.assertEqual(z_cpu, z[0])
        self.assertEqual(z_cpu, z[1])

        # call _fused_adamw_ with undefined tensor.
        self.module.fallback_with_undefined_tensor()

    @common.skipIfTorchDynamo()
    @unittest.skipIf(
        np.__version__ < "1.25",
        "versions < 1.25 serialize dtypes differently from how it's serialized in data_legacy_numpy",
    )
    def test_open_device_numpy_serialization(self):
        """
        This tests the legacy _rebuild_device_tensor_from_numpy serialization path
        """
        device = self.module.custom_device()

        # Legacy data saved with _rebuild_device_tensor_from_numpy on f80ed0b8 via

        # with patch.object(torch._C, "_has_storage", return_value=False):
        #     x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device)
        #     x_foo = x.to(device)
        #     sd = {"x": x_foo}
        #     rebuild_func = x_foo._reduce_ex_internal(default_protocol)[0]
        #     self.assertTrue(
        #         rebuild_func is torch._utils._rebuild_device_tensor_from_numpy
        #     )
        #     with open("foo.pt", "wb") as f:
        #         torch.save(sd, f)

        data_legacy_numpy = (
            b"PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x10\x00\x12\x00archive/data.pklFB\x0e\x00ZZZZZZZZZZZZZZ\x80\x02}q\x00X\x01"
            b"\x00\x00\x00xq\x01ctorch._utils\n_rebuild_device_tensor_from_numpy\nq\x02(cnumpy.core.m"
            b"ultiarray\n_reconstruct\nq\x03cnumpy\nndarray\nq\x04K\x00\x85q\x05c_codecs\nencode\nq\x06"
            b"X\x01\x00\x00\x00bq\x07X\x06\x00\x00\x00latin1q\x08\x86q\tRq\n\x87q\x0bRq\x0c(K\x01K\x02K"
            b"\x03\x86q\rcnumpy\ndtype\nq\x0eX\x02\x00\x00\x00f4q\x0f\x89\x88\x87q\x10Rq\x11(K\x03X\x01"
            b"\x00\x00\x00<q\x12NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tq\x13b\x89h\x06X\x1c\x00\x00"
            b"\x00\x00\x00\xc2\x80?\x00\x00\x00@\x00\x00@@\x00\x00\xc2\x80@\x00\x00\xc2\xa0@\x00\x00\xc3"
            b"\x80@q\x14h\x08\x86q\x15Rq\x16tq\x17bctorch\nfloat32\nq\x18X\t\x00\x00\x00openreg:0q\x19\x89"
            b"tq\x1aRq\x1bs.PK\x07\x08\xdfE\xd6\xcaS\x01\x00\x00S\x01\x00\x00PK\x03\x04\x00\x00\x08"
            b"\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x11\x00.\x00"
            b"archive/byteorderFB*\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZlittlePK\x07\x08"
            b"\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0f\x00=\x00archive/versionFB9\x00"
            b"ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00"
            b"\x00\x02\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x1e\x002\x00archive/.data/serialization_idFB.\x00ZZZZZZZZZZZZZ"
            b"ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ0636457737946401051300000025273995036293PK\x07\x08\xee(\xcd"
            b"\x8d(\x00\x00\x00(\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00"
            b"\xdfE\xd6\xcaS\x01\x00\x00S\x01\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00archive/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00"
            b"\x00\x00\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00\x11\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\xa3\x01\x00\x00archive/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08\x00"
            b"\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x0f\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x16\x02\x00\x00archive/versionPK\x01\x02\x00\x00\x00\x00\x08"
            b"\x08\x00\x00\x00\x00\x00\x00\xee(\xcd\x8d(\x00\x00\x00(\x00\x00\x00\x1e\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x92\x02\x00\x00archive/.data/serialization_idPK\x06"
            b"\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00"
            b"\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x06\x01\x00\x00\x00\x00\x00\x008\x03\x00"
            b"\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00>\x04\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00"
            b"PK\x05\x06\x00\x00\x00\x00\x04\x00\x04\x00\x06\x01\x00\x008\x03\x00\x00\x00\x00"
        )
        buf_data_legacy_numpy = io.BytesIO(data_legacy_numpy)

        with safe_globals(
            [
                (np.core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct")
                if np.__version__ >= "2.1"
                else np.core.multiarray._reconstruct,
                np.ndarray,
                np.dtype,
                _codecs.encode,
                np.dtypes.Float32DType,
            ]
        ):
            sd_loaded = torch.load(buf_data_legacy_numpy, weights_only=True)
            buf_data_legacy_numpy.seek(0)
            # Test map_location
            sd_loaded_cpu = torch.load(
                buf_data_legacy_numpy, weights_only=True, map_location="cpu"
            )
        expected = torch.tensor(
            [[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device
        )
        self.assertEqual(sd_loaded["x"].cpu(), expected.cpu())
        self.assertFalse(sd_loaded["x"].is_cpu)
        self.assertTrue(sd_loaded_cpu["x"].is_cpu)

    def test_open_device_cpu_serialization(self):
        torch.utils.rename_privateuse1_backend("openreg")
        device = self.module.custom_device()
        default_protocol = torch.serialization.DEFAULT_PROTOCOL

        with patch.object(torch._C, "_has_storage", return_value=False):
            x = torch.randn(2, 3)
            x_openreg = x.to(device)
            sd = {"x": x_openreg}
            rebuild_func = x_openreg._reduce_ex_internal(default_protocol)[0]
            self.assertTrue(
                rebuild_func is torch._utils._rebuild_device_tensor_from_cpu_tensor
            )
            # Test map_location
            with TemporaryFileName() as f:
                torch.save(sd, f)
                sd_loaded = torch.load(f, weights_only=True)
                # Test map_location
                sd_loaded_cpu = torch.load(f, weights_only=True, map_location="cpu")
            self.assertFalse(sd_loaded["x"].is_cpu)
            self.assertEqual(sd_loaded["x"].cpu(), x)
            self.assertTrue(sd_loaded_cpu["x"].is_cpu)

            # Test metadata_only
            with TemporaryFileName() as f:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Cannot serialize tensors on backends with no storage under skip_data context manager",
                ):
                    with torch.serialization.skip_data():
                        torch.save(sd, f)

    def test_open_device_dlpack(self):
        t = torch.randn(2, 3).to("openreg")
        capsule = torch.utils.dlpack.to_dlpack(t)
        t1 = torch.from_dlpack(capsule)
        self.assertTrue(t1.device == t.device)
        t = t.to("cpu")
        t1 = t1.to("cpu")
        self.assertEqual(t, t1)


if __name__ == "__main__":
    common.run_tests()
