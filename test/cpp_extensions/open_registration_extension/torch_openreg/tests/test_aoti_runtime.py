# Owner(s): ["module: PrivateUse1"]

import ctypes
import subprocess
import sys
from pathlib import Path

import torch
from torch._inductor.codecache import _get_cpp_wrapper_header
from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils.cpp_extension import load_inline, TORCH_LIB_PATH


def load_libtorch_cpu():
    if sys.platform == "win32":
        libname = "torch_cpu.dll"
    elif sys.platform == "darwin":
        libname = "libtorch_cpu.dylib"
    else:
        libname = "libtorch_cpu.so"
    return ctypes.CDLL(str(Path(TORCH_LIB_PATH) / libname))


class TestAOTIShims(TestCase):
    """New aoti_torch_* shims backed by the OpenReg-registered accelerator."""

    @classmethod
    def setUpClass(cls):
        cls.lib = load_libtorch_cpu()

    def test_synchronize_device(self):
        self.assertEqual(self.lib.aoti_torch_synchronize_device(0), 0)


# Drives the AOTI runtime header paths a PrivateUse1 model.so would take:
# parse_device_str, RAII_privateuse1Malloc (constant blob allocation) and
# privateuse1_memcpy (constant H2D/D2D fill). OpenReg device memory is
# mprotect(PROT_NONE)-protected, so any stray host memcpy on these paths
# crashes instead of silently writing through the wrong pointer.
CPP_SOURCE = """
#include <torch/csrc/inductor/aoti_runtime/model_base.h>

std::vector<int64_t> parse_device(std::string device_str) {
  int32_t device_type = -1;
  int32_t device_idx = -1;
  torch::aot_inductor::parse_device_str(device_str, device_type, device_idx);
  return {device_type, device_idx};
}

int64_t privateuse1_device_type() {
  return aoti_torch_device_type_privateuse1();
}

int64_t cpu_device_type() {
  return aoti_torch_device_type_cpu();
}

int64_t cuda_device_type() {
  return aoti_torch_device_type_cuda();
}

torch::Tensor blob_roundtrip(torch::Tensor src, bool via_second_blob) {
  TORCH_CHECK(src.device().is_cpu());
  TORCH_CHECK(src.is_contiguous());
  const int64_t nbytes = src.nbytes();
  const int32_t pu1 = aoti_torch_device_type_privateuse1();
  auto blob = RAII_privateuse1Malloc(nbytes, 0);
  privateuse1_memcpy(blob.get(), src.data_ptr(), nbytes, 0);
  void* read_src = blob.get();
  RAIIDataPtr second_blob;
  if (via_second_blob) {
    second_blob = RAII_privateuse1Malloc(nbytes, 0);
    privateuse1_memcpy(second_blob.get(), blob.get(), nbytes, 0, pu1, 0);
    read_src = second_blob.get();
  }
  auto options = torch::dtype(torch::kUInt8).device(torch::kPrivateUse1, 0);
  auto device_bytes = torch::from_blob(read_src, {nbytes}, options);
  return device_bytes.cpu().view(src.scalar_type()).reshape(src.sizes());
}
"""


class TestAOTIRuntimePrivateUse1(TestCase):
    """AOTI model.so runtime header behavior for a PrivateUse1 backend."""

    @classmethod
    def setUpClass(cls):
        # cpp_builder passes -DUSE_PRIVATEUSE1 when compiling for such a device
        cls.mod = load_inline(
            name="aoti_runtime_openreg_test",
            cpp_sources=CPP_SOURCE,
            functions=[
                "parse_device",
                "privateuse1_device_type",
                "cpu_device_type",
                "cuda_device_type",
                "blob_roundtrip",
            ],
            extra_cflags=["-DUSE_PRIVATEUSE1"],
            with_cuda=False,
            verbose=False,
        )

    def test_parse_device_str(self):
        pu1 = self.mod.privateuse1_device_type()
        cpu = self.mod.cpu_device_type()
        cuda = self.mod.cuda_device_type()
        self.assertEqual(self.mod.parse_device("openreg"), [pu1, -1])
        self.assertEqual(self.mod.parse_device("openreg:0"), [pu1, 0])
        self.assertEqual(self.mod.parse_device("openreg:1"), [pu1, 1])
        # In-tree device strings must keep taking the regex path.
        self.assertEqual(self.mod.parse_device("cpu"), [cpu, -1])
        self.assertEqual(self.mod.parse_device("cuda:3"), [cuda, 3])

    def test_parse_device_str_invalid(self):
        bad = ["openregx", "openreg:", "openreg:x", "openreg:0x", "npu:0", ""]
        for device_str in bad:
            with self.assertRaisesRegex(RuntimeError, "Invalid device"):
                self.mod.parse_device(device_str)

    def test_constant_blob_h2d_roundtrip(self):
        # load_constants path: host constants -> device blob -> read back.
        src = torch.randn(1024)
        self.assertEqual(self.mod.blob_roundtrip(src, False), src)

    def test_constant_blob_d2d_roundtrip(self):
        # update_constant_buffer path: adds a device-to-device blob hop.
        src = torch.arange(4096, dtype=torch.int32)
        self.assertEqual(self.mod.blob_roundtrip(src, True), src)


CPU_ONLY_SCRIPT = """
import sys
from torch.utils.cpp_extension import load_inline

mod = load_inline(
    name="aoti_runtime_cpu_only_test",
    cpp_sources=sys.argv[1],
    functions=["parse_device"],
    with_cuda=False,
)
try:
    mod.parse_device("openreg")
except RuntimeError as e:
    assert "Invalid device" in str(e), e
else:
    raise AssertionError("openreg accepted without USE_PRIVATEUSE1")
"""


class TestAOTIRuntimeWithoutPrivateUse1(TestCase):
    def test_privateuse1_device_rejected(self):
        # Without USE_PRIVATEUSE1 (every CPU model.so) the PrivateUse1 paths
        # and the shims they call are compiled out. Needs its own process: the
        # inline parse_device_str of the two builds would be merged by the
        # dynamic linker.
        cpp_source = CPP_SOURCE.split("torch::Tensor blob_roundtrip")[0]
        subprocess.run(
            [sys.executable, "-c", CPU_ONLY_SCRIPT, cpp_source],
            check=True,
            capture_output=True,
            text=True,
        )


class TestAOTICodegenPrivateUse1(TestCase):
    def test_cpp_wrapper_header(self):
        # The registered backend name has no header of its own.
        for device in ("openreg", "openreg:1"):
            aot = "torch/csrc/inductor/aoti_include/privateuse1.h"
            jit = "torch/csrc/inductor/cpp_wrapper/privateuse1.h"
            self.assertEqual(_get_cpp_wrapper_header(device, aot_mode=True), aot)
            self.assertEqual(_get_cpp_wrapper_header(device, aot_mode=False), jit)
        cuda = "torch/csrc/inductor/aoti_include/cuda.h"
        self.assertEqual(_get_cpp_wrapper_header("cuda:0", aot_mode=True), cuda)

    def test_include_path_hooks(self):
        # Subclass overrides of these hooks see the real device name; the
        # privateuse1 mapping happens in the defaults.
        aot = "#include <torch/csrc/inductor/aoti_include/%s.h>"
        include_path = CppWrapperCpu.get_device_include_path_aot
        self.assertEqual(include_path("openreg"), aot % "privateuse1")
        self.assertEqual(include_path("cuda"), aot % "cuda")


if __name__ == "__main__":
    run_tests()
