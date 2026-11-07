"""Configure-time checks for explicitly requested GPU backends (gh-114388).

USE_ROCM / USE_CUDA set with -D or through the environment must fail the
configure when the toolkit cannot be found; the default must keep falling back
to OFF. The test project runs the real EnvVarForwarding.cmake, the backend
options from CMakeLists.txt and the real backend loaders, followed by the
fallback that cmake/Dependencies.cmake applies.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ROCM_PLATFORM = sys.platform.startswith("linux") or sys.platform == "win32"
ROCM_PATH = "C:/opt/rocm" if sys.platform == "win32" else "/opt/rocm"
# ROCM_PATH is removed from the child environment, so only these can find ROCm.
FOUND = {
    "ROCM": os.path.exists(ROCM_PATH) or bool(shutil.which("rocm-sdk")),
    "CUDA": bool(shutil.which("nvcc")) or os.path.exists("/usr/local/cuda"),
}
NAME = {"ROCM": "ROCm", "CUDA": "CUDA"}
OTHER = {"ROCM": "CUDA", "CUDA": "ROCM"}
CLEAN_ENV = re.compile(r"^(USE_|BUILD_|CMAKE_|ROCM|HIP|CUDA|PYTORCH_ROCM_ARCH)")

# From the USE_CUDA explicit-set check through the USE_ROCM option, as in
# CMakeLists.txt. Loaders follow the order and fallback of Dependencies.cmake;
# hip-config needs C/CXX (FindThreads) once ROCm is found.
HARNESS = """
cmake_minimum_required(VERSION 3.27)
project(explicit_backend {langs})
set(CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO ON)
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
message(STATUS "RESULT USE_CUDA=${{USE_CUDA}} USE_ROCM=${{USE_ROCM}}")
"""


class TestExplicitBackendRequest(unittest.TestCase):
    def setUp(self) -> None:
        cmake = shutil.which("cmake")
        if cmake is None:
            self.skipTest("cmake is not installed")
        self.cmake = cmake
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        text = (REPO / "CMakeLists.txt").read_text()
        start = r"^if\(DEFINED CACHE\{USE_CUDA\}\)$"
        end = r"^cmake_dependent_option\(USE_ROCM .*?$"
        options = re.search(f"{start}.*?{end}", text, re.MULTILINE | re.DOTALL)
        if options is None:
            raise AssertionError("USE_CUDA/USE_ROCM options not found")
        langs = "C CXX" if FOUND["ROCM"] else "NONE"
        harness = HARNESS.format(langs=langs, repo=REPO.as_posix(), options=options[0])
        (self.tmp / "CMakeLists.txt").write_text(harness)

    def configure(self, backend: str, *args: str, env=None, build: str = "build"):
        # Only the backend under test is configured; the other one is disabled.
        args = (f"-DUSE_{OTHER[backend]}=OFF", *args)
        run_env = {k: v for k, v in os.environ.items() if not CLEAN_ENV.match(k)}
        run_env.update(env or {})
        python = f"-DPython_EXECUTABLE={sys.executable}"
        out = str(self.tmp / build)
        cmd = [self.cmake, "-S", str(self.tmp), "-B", out, python, *args]
        return subprocess.run(cmd, env=run_env, capture_output=True, text=True)

    def ok(self, backend: str, enabled: bool, *args: str, **kwargs) -> None:
        result = self.configure(backend, *args, **kwargs)
        self.assertEqual(result.returncode, 0, result.stderr)
        match = re.search(rf"\bUSE_{backend}=(\S*)", result.stdout)
        if match is None:
            self.fail(result.stdout)
        value = match[1].upper()
        self.assertEqual(value in ("ON", "1", "TRUE", "YES", "Y"), enabled, value)

    def fails(self, backend: str, *args: str, **kwargs) -> None:
        result = self.configure(backend, *args, **kwargs)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        message = f"{NAME[backend]} was explicitly requested"
        self.assertIn(message, " ".join(result.stderr.split()))

    def skip_unless_missing(self, backend: str) -> None:
        if backend == "ROCM" and not ROCM_PLATFORM:
            self.skipTest("USE_ROCM is only available on Linux and Windows")
        if FOUND[backend]:
            self.skipTest(f"{backend} is installed, so it cannot be missing")

    def test_missing_toolkit(self) -> None:
        for b in ("ROCM", "CUDA"):
            with self.subTest(backend=b):
                self.skip_unless_missing(b)
                self.ok(b, False, build=f"{b}-default")
                self.ok(b, False, f"-DUSE_{b}=OFF", build=f"{b}-d-off")
                self.ok(b, False, env={f"USE_{b}": "0"}, build=f"{b}-env-off")
                self.fails(b, f"-DUSE_{b}=ON", build=f"{b}-d-on")
                self.fails(b, env={f"USE_{b}": "1"}, build=f"{b}-env-on")

    def test_reconfigure_rocm_without_toolkit(self) -> None:
        self.skip_unless_missing("ROCM")
        # The fallback caches USE_ROCM=OFF, so a later default run is not an
        # explicit ON; an explicit ON stays in the cache until overridden.
        self.ok("ROCM", False)
        self.ok("ROCM", False)
        self.fails("ROCM", "-DUSE_ROCM=ON")
        self.fails("ROCM")
        self.ok("ROCM", False, "-DUSE_ROCM=OFF")
        self.fails("ROCM", env={"USE_ROCM": "1"})

    @unittest.skipIf(ROCM_PLATFORM, "USE_ROCM is available on this platform")
    def test_explicit_rocm_is_ignored_on_unsupported_platform(self) -> None:
        self.ok("ROCM", False, "-DUSE_ROCM=ON")

    @unittest.skipUnless(ROCM_PLATFORM and os.path.exists(ROCM_PATH), "needs ROCm")
    def test_installed_rocm_is_found(self) -> None:
        env = {"PYTORCH_ROCM_ARCH": os.environ.get("PYTORCH_ROCM_ARCH", "gfx90a")}
        for name, args in (("default", ()), ("explicit", ("-DUSE_ROCM=ON",))):
            self.ok("ROCM", True, *args, env=env, build=name)


if __name__ == "__main__":
    unittest.main()
