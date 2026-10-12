# Owner(s): ["oncall: distributed"]

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@unittest.skipIf(shutil.which("cmake") is None, "requires CMake")
@instantiate_parametrized_tests
class TestNcclEpBuild(TestCase):
    @parametrize(
        "submodule,entry",
        [("nccl-extensions", "nccl_ep/CMakeLists.txt"), ("gloo", "CMakeLists.txt")],
    )
    @parametrize("populated", [False, True])
    def test_submodule_verification(self, submodule, entry, populated):
        repo = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".gitmodules").write_text(
                f'[submodule "third_party/{submodule}"]\n'
                f"    path = third_party/{submodule}\n"
            )
            asmjit = root / "third_party/fbgemm/external/asmjit"
            asmjit.mkdir(parents=True)
            (asmjit / "CMakeLists.txt").touch()
            if populated:
                expected = root / "third_party" / submodule / entry
                expected.parent.mkdir(parents=True, exist_ok=True)
                expected.touch()
            (root / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.18)\n"
                "project(submodule_verification NONE)\n"
                "set(USE_SYSTEM_NCCL ON)\n"
                f'include("{repo / "cmake/PreBuildSteps.cmake"}")\n'
            )
            result = subprocess.run(
                ["cmake", "-S", str(root), "-B", str(root / "build")],
                capture_output=True,
                text=True,
            )
            output = result.stdout + result.stderr
            if populated:
                self.assertEqual(result.returncode, 0, output)
            else:
                self.assertNotEqual(result.returncode, 0, output)
                self.assertRegex(output, r"appears\s+incomplete")

    @parametrize("system_nccl", [False, True])
    def test_source_and_linkage(self, system_nccl):
        repo = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build = root / "build"
            (root / "CMakeLists.txt").write_text(
                """\
cmake_minimum_required(VERSION 3.18)
project(nccl_ep_configuration NONE)
include(ExternalProject)
set(PROJECT_SOURCE_DIR "${PYTORCH_SOURCE_DIR}")
set(CUDA_VERSION 13.4)
set(TORCH_CUDA_ARCH_LIST "9.0")
include("${PROJECT_SOURCE_DIR}/cmake/public/utils.cmake")
include(
  "${PROJECT_SOURCE_DIR}/cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake")
set(NCCL_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/system_nccl/include")
add_custom_target(nccl_external)
include("${PROJECT_SOURCE_DIR}/cmake/External/nccl_ep.cmake")
ExternalProject_Get_Property(nccl_ep_external CMAKE_ARGS)
get_target_property(deps nccl_ep_external MANUALLY_ADDED_DEPENDENCIES)
if(NOT deps)
  set(deps "")
endif()
file(WRITE "${CMAKE_BINARY_DIR}/ep_config.txt"
  "args=${CMAKE_ARGS}\n"
  "library=${NCCL_EP_LIBRARIES}\n"
  "include=${NCCL_EP_INCLUDE_DIRS}\n"
  "jit_home=${NCCL_EP_JIT_HOME}\n"
  "dependencies=${deps}\n")
"""
            )
            env = os.environ.copy()
            env.pop("NCCL_INCLUDE_DIR", None)
            result = subprocess.run(
                [
                    "cmake",
                    "-S",
                    str(root),
                    "-B",
                    str(build),
                    f"-DPYTORCH_SOURCE_DIR={repo}",
                    f"-DUSE_SYSTEM_NCCL={'ON' if system_nccl else 'OFF'}",
                ],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            config = dict(
                line.split("=", 1)
                for line in (build / "ep_config.txt").read_text().splitlines()
            )
            self.assertIn(
                f"-DNCCL_EP_SOURCE_DIR={repo / 'third_party/nccl-extensions/nccl_ep'}",
                config["args"].split(";"),
            )
            library = "libnccl_ep.so" if system_nccl else "libnccl_ep.a"
            self.assertEqual(config["library"], str(build / "nccl/lib" / library))
            self.assertEqual(config["include"], str(build / "nccl/include"))
            self.assertEqual(
                config["jit_home"], "" if system_nccl else str(build / "nccl")
            )
            self.assertEqual("nccl_external" in config["dependencies"], not system_nccl)


if __name__ == "__main__":
    run_tests()
