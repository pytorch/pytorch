"""checks for explicitly requested build options (gh-114388).

a USE_* option set with -D or thru the enviroment has to fail the
configure when we cant provide it, and has to keep falling back to OFF
when nobody asked for it. the scratch project here runs the real
cmake/EnvVarForwarding.cmake, the real option declarations pulled out of
CMakeLists.txt, and the real loaders and fallback sites from
cmake/Dependencies.cmake, so we exercise the actual wiring instead of a
copy of it.

USE_XPU isnt covered: xpu.cmake does find_package(SYCLToolkit REQUIRED)
which fails the configure on its own before the not-found guard is even
reached.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROCM_PATH = "C:/opt/rocm" if sys.platform == "win32" else "/opt/rocm"
HAVE_CUDA = bool(shutil.which("nvcc")) or os.path.exists("/usr/local/cuda")
HAVE_ROCM = os.path.exists(ROCM_PATH) or bool(shutil.which("rocm-sdk"))
IS_LINUX = sys.platform.startswith("linux")
CLEAN_ENV = re.compile(r"^(USE_|BUILD_|CMAKE_|ROCM|HIP|CUDA|CUDNN|NCCL)")

# the exact text cuda.cmake has always printed, the shared helper has to
# reproduce it verbatim
CUDA_GUARD_MESSAGE = (
    "PyTorch: CUDA was explicitly requested (USE_CUDA=1) but cannot be found. "
    "Please check your CUDA installation, ensure CUDA toolkit is installed, "
    "and that CUDA_HOME or CMAKE_CUDA_COMPILER is set correctly. "
    "If you want to build without CUDA, please set USE_CUDA=0."
)

# the option declarations from CMakeLists.txt, from the sanitizer options
# (the USE_CUDA declaration looks at USE_TSAN) down to the USE_NCCL
# declaration, then the fallback chain from Dependencies.cmake in the
# order it runs there
HARNESS = """
cmake_minimum_required(VERSION 3.27)
project(explicit_request {langs})
set(CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO ON)
find_package(Python COMPONENTS Interpreter REQUIRED)
include(CMakeDependentOption)
include("{repo}/cmake/public/utils.cmake")
include("{repo}/cmake/EnvVarForwarding.cmake")
{options}
if(USE_CUDA)
  set(CAFFE2_USE_CUDA ${{USE_CUDA}})
  include("{repo}/cmake/public/cuda.cmake")
  if(NOT CAFFE2_USE_CUDA)
    caffe2_update_option(USE_CUDA OFF)
  endif()
endif()
if(USE_ROCM)
  include("{repo}/cmake/public/LoadHIP.cmake")
  if(NOT PYTORCH_FOUND_HIP)
    caffe2_update_option(USE_ROCM OFF)
  endif()
endif()
{nccl_block}
message(STATUS "RESULT USE_CUDA=${{USE_CUDA}} USE_ROCM=${{USE_ROCM}} "
               "USE_NCCL=${{USE_NCCL}}")
"""


def extract(text: str, pattern: str, what: str) -> str:
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    if match is None:
        raise AssertionError(f"{what} not found")
    return match[0]


def build_harness() -> str:
    cmake_lists = (REPO / "CMakeLists.txt").read_text()
    options = extract(
        cmake_lists,
        r"^option\(USE_ASAN .*?"
        r"^cmake_dependent_option\(USE_NCCL .*?OFF\)",
        "USE_CUDA..USE_NCCL option block",
    )
    dependencies = (REPO / "cmake" / "Dependencies.cmake").read_text()
    nccl_block = extract(
        dependencies,
        r"^if\(USE_NCCL\)\n.*?\n\n# ---\[ NCCL EP",
        "USE_NCCL fallback block",
    )
    nccl_block = nccl_block.split("# ---[", 1)[0].rstrip()
    nccl_block = nccl_block.replace(
        "${CMAKE_CURRENT_LIST_DIR}", (REPO / "cmake").as_posix())
    # LoadHIP's found path pulls in hip-config.cmake which wants C/CXX
    langs = "C CXX" if HAVE_ROCM else "NONE"
    return HARNESS.format(langs=langs, repo=REPO.as_posix(), options=options,
                          nccl_block=nccl_block)


class TestExplicitRequestFallbacks(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cmake = shutil.which("cmake")
        if cmake is None:
            raise unittest.SkipTest("cmake is not installed")
        cls.cmake = cmake
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.tmp = Path(tmp.name)
        (cls.tmp / "CMakeLists.txt").write_text(build_harness())

    def configure(self, *args: str, env=None, build: str = "build"):
        run_env = {k: v for k, v in os.environ.items() if not CLEAN_ENV.match(k)}
        run_env.update(env or {})
        out = self.tmp / build
        return subprocess.run(
            [self.cmake, "-S", str(self.tmp), "-B", str(out),
             f"-DPython_EXECUTABLE={sys.executable}", *args],
            env=run_env, capture_output=True, text=True)

    def assert_ok(self, option: str, enabled: bool, *args: str, **kwargs) -> None:
        result = self.configure(*args, **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        match = re.search(rf'RESULT .*?\b{option}=(\S*)', result.stdout)
        if match is None:
            self.fail(f"no RESULT line in:\n{result.stdout}")
        value = match[1].upper()
        self.assertEqual(value in ("ON", "1", "TRUE", "YES", "Y"), enabled, value)

    def assert_fails(self, message: str, *args: str, **kwargs) -> None:
        result = self.configure(*args, **kwargs)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(message, " ".join(result.stderr.split()))

    def skip_if_cuda_installed(self) -> None:
        if HAVE_CUDA:
            self.skipTest("CUDA is installed, so it cannot be missing")

    # --- CUDA: the guard predates this change, the helper has to keep its
    # text

    def test_cuda_explicit_request_fails(self) -> None:
        self.skip_if_cuda_installed()
        self.assert_fails(CUDA_GUARD_MESSAGE, "-DUSE_CUDA=ON", build="cuda-d-on")
        self.assert_fails(CUDA_GUARD_MESSAGE, env={"USE_CUDA": "1"},
                          build="cuda-env")

    def test_cuda_default_falls_back(self) -> None:
        self.skip_if_cuda_installed()
        self.assert_ok("USE_CUDA", False, build="cuda-default")
        self.assert_ok("USE_CUDA", False, "-DUSE_CUDA=OFF", build="cuda-d-off")
        # a second default configure in the same dir must not read the
        # fallback-cached OFF as an explicit request
        self.assert_ok("USE_CUDA", False, build="cuda-default")

    # --- ROCm: new guard, split by what the platform lets you reach

    def test_rocm_explicit_request_on_unsupported_platform(self) -> None:
        if IS_LINUX or sys.platform == "win32":
            self.skipTest("USE_ROCM can be provided on this platform")
        self.assert_fails(
            "ROCm was explicitly requested (USE_ROCM=1) but is only "
            "supported on Linux and Windows.",
            "-DUSE_ROCM=ON", build="rocm-platform")

    def test_rocm_explicit_request_not_found(self) -> None:
        if not IS_LINUX:
            self.skipTest("the LoadHIP fallback is only reached on Linux")
        if HAVE_ROCM:
            self.skipTest("ROCm is installed, so it cannot be missing")
        message = "ROCm was explicitly requested (USE_ROCM=1) but cannot be found."
        self.assert_fails(message, "-DUSE_ROCM=ON", build="rocm-d-on")
        self.assert_fails(message, env={"USE_ROCM": "1"}, build="rocm-env")
        self.assert_ok("USE_ROCM", False, build="rocm-default")

    def test_rocm_explicit_request_persists_in_cache(self) -> None:
        if IS_LINUX and HAVE_ROCM:
            self.skipTest("ROCm is installed, so it cannot be missing")
        if IS_LINUX:
            message = "ROCm was explicitly requested (USE_ROCM=1) but cannot be found."
        else:
            message = "ROCm was explicitly requested (USE_ROCM=1) but is only " \
                      "supported on Linux and Windows."
        # the -D entry stays in the cache so a later default run in the
        # same dir fails too, untill the user overrides it
        self.assert_fails(message, "-DUSE_ROCM=ON", build="rocm-persist")
        self.assert_fails(message, build="rocm-persist")
        self.assert_ok("USE_ROCM", False, "-DUSE_ROCM=OFF", build="rocm-persist")

    # --- NCCL: new guard at the Dependencies.cmake fallback site, plus an
    # early refusal for platforms the declaration condition rejects

    def test_nccl_explicit_request(self) -> None:
        if IS_LINUX:
            if HAVE_CUDA:
                self.skipTest("CUDA is installed, so the no-backend fallback "
                              "is not reachable")
            self.assert_fails(
                "NCCL was explicitly requested (USE_NCCL=1) but requires a "
                "CUDA or ROCm build.",
                "-DUSE_NCCL=ON", build="nccl-d-on")
            self.assert_fails(
                "NCCL was explicitly requested (USE_NCCL=1) but requires a "
                "CUDA or ROCm build.",
                env={"USE_NCCL": "1"}, build="nccl-env")
        else:
            self.assert_fails(
                "NCCL was explicitly requested (USE_NCCL=1) but requires "
                "USE_DISTRIBUTED on a non-Apple UNIX platform.",
                "-DUSE_NCCL=ON", build="nccl-platform")

    def test_nccl_default_falls_back(self) -> None:
        self.assert_ok("USE_NCCL", False, build="nccl-default")
        self.assert_ok("USE_NCCL", False, "-DUSE_NCCL=OFF", build="nccl-d-off")

    # --- a configure that dies before a fallback site runs must not leave
    # the declaration defaults looking like user requests on the next run
    # in the same dir

    def test_rocm_fatal_does_not_poison_cuda(self) -> None:
        if IS_LINUX and HAVE_ROCM:
            self.skipTest("ROCm is installed, so it cannot be missing")
        if IS_LINUX:
            message = "ROCm was explicitly requested (USE_ROCM=1) but cannot be found."
        else:
            message = "ROCm was explicitly requested (USE_ROCM=1) but is only " \
                      "supported on Linux and Windows."
        self.assert_fails(message, "-DUSE_ROCM=ON", build="poison-rocm")
        # following the error's advice must not trip the CUDA guard on a
        # box with no CUDA either
        self.assert_ok("USE_ROCM", False, "-DUSE_ROCM=OFF", build="poison-rocm")

    def test_cuda_fatal_does_not_poison_nccl(self) -> None:
        self.skip_if_cuda_installed()
        self.assert_fails(CUDA_GUARD_MESSAGE, "-DUSE_CUDA=ON", build="poison-cuda")
        # following the CUDA error's advice must not trip the NCCL guard
        self.assert_ok("USE_NCCL", False, "-DUSE_CUDA=OFF", build="poison-cuda")


if __name__ == "__main__":
    unittest.main()
