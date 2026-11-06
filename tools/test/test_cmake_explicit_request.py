"""checks for explicitly requested build options (gh-114388).

a USE_* option set with -D or through the environment has to fail the
configure when we cannot provide it, and has to keep falling back to OFF
when nobody asked for it. the scratch project here runs the real
cmake/EnvVarForwarding.cmake, the real option declarations pulled out of
CMakeLists.txt, and the real NCCL fallback block from
cmake/Dependencies.cmake, so those are exercised as wired. the CUDA and
ROCm fallback wiring below is a hand-written copy of the real blocks, so
changes there are not caught here.

USE_XPU is not covered: xpu.cmake does find_package(SYCLToolkit REQUIRED)
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
        r"^option\(USE_ASAN .*?(?=^cmake_dependent_option\(USE_RCCL)",
        "USE_ASAN..USE_NCCL option block",
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

    def edit_cache_value(self, build: str, option: str, value: str) -> None:
        # what ccmake / cmake-gui do to CMakeCache.txt: rewrite the value
        # line, keep the help-string comment above it
        cache = self.tmp / build / "CMakeCache.txt"
        text = cache.read_text()
        patched, count = re.subn(
            rf"^{option}:BOOL=\w+$", f"{option}:BOOL={value}", text,
            flags=re.MULTILINE)
        if count != 1:
            self.fail(f"expected one {option}:BOOL entry, found {count}")
        cache.write_text(patched)

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

    def test_forced_value_wins_over_cached_on(self) -> None:
        # cmake_dependent_option applies the force value with a plain
        # set() in the caller's scope, so a wrapper that is a function
        # would eat it and leave the cached ON visible after the call.
        # USE_TSAN turns USE_CUDA off, USE_DISTRIBUTED=OFF turns USE_NCCL
        # off, even with the ON value already in the cache. the NCCL half
        # has to be seeded with a real configure first: a fresh dir has
        # no cached USE_NCCL for the force to beat, so the assertion
        # would pass even if the wrapper dropped the forced value on the
        # floor. the cached ON from the seed run is a declaration default
        # (stable since the declaration last ran), so it must not read
        # as an explicit request either.
        self.assert_ok("USE_CUDA", False, "-DUSE_CUDA=ON", "-DUSE_TSAN=ON",
                       build="force-cuda")
        self.assert_ok("USE_NCCL", False, "-DUSE_DISTRIBUTED=OFF",
                       build="force-nccl")
        # seed the cache by hand instead of by configuring: on a box with
        # no backend the fallback caches USE_NCCL=OFF during a seed run,
        # and the forced-OFF assertion below would pass even if the
        # wrapper dropped the forced value on the floor. the hand-written
        # entry is what a configure that settled on ON leaves behind
        # (declaration help string, cached ON, matching _LAST_SEEN stamp,
        # no request marker), so the force has a real cached ON to beat
        build = self.tmp / "force-nccl-seed"
        build.mkdir()
        (build / "CMakeCache.txt").write_text(
            "//Use NCCL\n"
            "USE_NCCL:BOOL=ON\n"
            "\n"
            "_USE_NCCL_LAST_SEEN:INTERNAL=ON\n")
        self.assert_ok("USE_NCCL", False, "-DUSE_DISTRIBUTED=OFF",
                       build="force-nccl-seed")
        # the forced OFF lives in a normal variable while the cache keeps
        # the ON entry, so the second run is where a stamp that recorded
        # the normal variable would read the gap as a user edit and fail
        # the configure through the early refusal. it has to stay quiet
        self.assert_ok("USE_NCCL", False, "-DUSE_DISTRIBUTED=OFF",
                       build="force-nccl-seed")

    def test_cuda_explicit_request_persists_in_cache(self) -> None:
        self.skip_if_cuda_installed()
        # option() stamps its doc string over an adopted entry during the
        # same configure, and the cache is saved even when the guard fails
        # it, so the next run cannot tell the entry apart from a
        # declaration default by its help string anymore. the INTERNAL
        # marker has to remember the request across runs, env-forwarded
        # requests included (gh-114388)
        self.assert_fails(CUDA_GUARD_MESSAGE, env={"USE_CUDA": "1"},
                          build="cuda-env-persist")
        self.assert_fails(CUDA_GUARD_MESSAGE, env={"USE_CUDA": "1"},
                          build="cuda-env-persist")
        self.assert_fails(CUDA_GUARD_MESSAGE, build="cuda-env-persist")
        self.assert_ok("USE_CUDA", False, "-DUSE_CUDA=OFF",
                       build="cuda-env-persist")
        # taking the request back also clears the marker for later runs
        self.assert_ok("USE_CUDA", False, build="cuda-env-persist")
        cache = (self.tmp / "cuda-env-persist" / "CMakeCache.txt").read_text()
        self.assertIn("_USE_CUDA_EXPLICITLY_SET:INTERNAL=OFF", cache)

    def test_cuda_explicit_request_cleared_by_unset(self) -> None:
        self.skip_if_cuda_installed()
        # cmake -U (or "Remove Entry" in cmake-gui) is the standard way
        # back to the default, so the request must not survive it
        self.assert_fails(CUDA_GUARD_MESSAGE, "-DUSE_CUDA=ON", build="cuda-unset")
        self.assert_ok("USE_CUDA", False, "-U", "USE_CUDA", build="cuda-unset")
        cache = (self.tmp / "cuda-unset" / "CMakeCache.txt").read_text()
        self.assertNotIn("_USE_CUDA_EXPLICITLY_SET:INTERNAL=", cache)

    # --- ROCm: new guard, split by what the platform lets you reach

    def test_rocm_explicit_request_on_unsupported_platform(self) -> None:
        if IS_LINUX or sys.platform == "win32":
            self.skipTest("USE_ROCM can be provided on this platform")
        self.assert_fails(
            "ROCm was explicitly requested (USE_ROCM=1) but is only "
            "supported on Linux and Windows.",
            "-DUSE_ROCM=ON", build="rocm-platform")

    def test_rocm_explicit_request_not_found(self) -> None:
        if not (IS_LINUX or sys.platform == "win32"):
            self.skipTest("the LoadHIP fallback is only reached on Linux and Windows")
        if HAVE_ROCM:
            self.skipTest("ROCm is installed, so it cannot be missing")
        message = "ROCm was explicitly requested (USE_ROCM=1) but cannot be found."
        self.assert_fails(message, "-DUSE_ROCM=ON", build="rocm-d-on")
        self.assert_fails(message, env={"USE_ROCM": "1"}, build="rocm-env")
        self.assert_ok("USE_ROCM", False, build="rocm-default")

    def test_rocm_explicit_request_persists_in_cache(self) -> None:
        if IS_LINUX and HAVE_ROCM:
            self.skipTest("ROCm is installed, so it cannot be missing")
        if IS_LINUX or sys.platform == "win32":
            message = "ROCm was explicitly requested (USE_ROCM=1) but cannot be found."
        else:
            message = "ROCm was explicitly requested (USE_ROCM=1) but is only " \
                      "supported on Linux and Windows."
        # the -D entry stays in the cache so a later default run in the
        # same dir fails too, until the user overrides it
        self.assert_fails(message, "-DUSE_ROCM=ON", build="rocm-persist")
        self.assert_fails(message, build="rocm-persist")
        self.assert_ok("USE_ROCM", False, "-DUSE_ROCM=OFF", build="rocm-persist")

    # --- NCCL: new guard at the Dependencies.cmake fallback site, plus an
    # early refusal for platforms the declaration condition rejects

    def test_nccl_explicit_request(self) -> None:
        if IS_LINUX:
            if HAVE_CUDA or HAVE_ROCM:
                self.skipTest("a backend is installed, so the no-backend "
                              "fallback is not reachable")
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
        if HAVE_CUDA or HAVE_ROCM:
            self.skipTest("a backend is installed, so NCCL stays available")
        self.assert_ok("USE_NCCL", False, build="nccl-default")
        self.assert_ok("USE_NCCL", False, "-DUSE_NCCL=OFF", build="nccl-d-off")

    def test_nccl_explicit_request_no_backend(self) -> None:
        if not IS_LINUX:
            self.skipTest("only Linux reaches the no-backend early refusal")
        message = ("NCCL was explicitly requested (USE_NCCL=1) but requires "
                   "a CUDA or ROCm build.")
        # USE_ROCM defaults to ON at its declaration on Linux, so both
        # backends have to be turned off explicitly to reach the case
        # where the NCCL declaration condition is already false
        self.assert_fails(message, "-DUSE_NCCL=ON", "-DUSE_CUDA=0",
                          "-DUSE_ROCM=0", build="nccl-no-backend")
        self.assert_fails(message, env={"USE_NCCL": "1", "USE_CUDA": "0",
                                        "USE_ROCM": "0"},
                          build="nccl-no-backend-env")

    def test_xccl_explicit_request_no_xpu(self) -> None:
        # the USE_XCCL declaration is wrapped now, so an explicit request
        # against a disabled XPU has to fail at declaration time instead
        # of falling back silently at the Dependencies.cmake site
        self.assert_fails(
            "XCCL was explicitly requested (USE_XCCL=1) but requires an "
            "XPU build.",
            "-DUSE_XCCL=ON", "-DUSE_XPU=OFF", build="xccl-no-xpu")

    # --- a configure that dies before a fallback site runs must not leave
    # the declaration defaults looking like user requests on the next run
    # in the same dir

    def test_rocm_fatal_does_not_poison_cuda(self) -> None:
        if IS_LINUX and HAVE_ROCM:
            self.skipTest("ROCm is installed, so it cannot be missing")
        if IS_LINUX or sys.platform == "win32":
            message = "ROCm was explicitly requested (USE_ROCM=1) but cannot be found."
        else:
            message = "ROCm was explicitly requested (USE_ROCM=1) but is only " \
                      "supported on Linux and Windows."
        # on macOS the refusal fires at declaration time, before the
        # harness's CUDA fallback has run, so this is the case where a
        # leftover USE_CUDA=ON default could look like a request on the
        # next run. on Linux the refusal comes from LoadHIP after that
        # fallback already wrote USE_CUDA=OFF, and the same assertions
        # cover the fatal-then-reconfigure flow there
        self.assert_fails(message, "-DUSE_ROCM=ON", build="poison-rocm")
        # following the error's advice must not trip the CUDA guard on a
        # box with no CUDA either
        self.assert_ok("USE_ROCM", False, "-DUSE_ROCM=OFF", build="poison-rocm")

    def test_cuda_fatal_does_not_poison_nccl(self) -> None:
        self.skip_if_cuda_installed()
        if IS_LINUX and HAVE_ROCM:
            self.skipTest("ROCm is installed, so NCCL stays available "
                          "through rccl")
        self.assert_fails(CUDA_GUARD_MESSAGE, "-DUSE_CUDA=ON", build="poison-cuda")
        # following the CUDA error's advice must not trip the NCCL guard
        self.assert_ok("USE_NCCL", False, "-DUSE_CUDA=OFF", build="poison-cuda")

    # --- cache edits: ccmake / cmake-gui / hand edits of CMakeCache.txt
    # keep the entry's help string, so the help-string signal alone cannot
    # see them. the _LAST_SEEN stamp the wrappers write after every
    # declaration is what makes them visible: option() never rewrites the
    # value of an entry it adopts and the fallbacks only ever write OFF
    # after their guard had its chance, so a value that moved under the
    # declaration's help string is the user's hand (gh-114388's symptom)

    def test_cache_edit_to_on_records_request(self) -> None:
        self.skip_if_cuda_installed()
        # default run: the CUDA fallback writes USE_CUDA=OFF, keeping the
        # declaration's help string on the entry
        self.assert_ok("USE_CUDA", False, build="ccmake-on")
        self.edit_cache_value("ccmake-on", "USE_CUDA", "ON")
        # flipping it back ON in ccmake is a request: the guard has to
        # fire instead of falling back silently again
        self.assert_fails(CUDA_GUARD_MESSAGE, build="ccmake-on")
        # and taking it back to OFF in the cache withdraws it again
        self.edit_cache_value("ccmake-on", "USE_CUDA", "OFF")
        self.assert_ok("USE_CUDA", False, build="ccmake-on")

    def test_initial_cache_script_with_docstring_left_alone(self) -> None:
        self.skip_if_cuda_installed()
        # a -C script that reuses the declaration's doc string is
        # indistinguishable from a build dir from before the stamping:
        # entry, help string, and no stamp. signal 2 has to leave both
        # alone rather than read the missing stamp as a value, so the
        # request is not recorded and the option falls back
        init = self.tmp / "init-uses-docstring.cmake"
        init.write_text('set(USE_CUDA ON CACHE BOOL "Use CUDA")\n')
        self.assert_ok("USE_CUDA", False, "-C", str(init),
                       build="c-script-docstring")

    def test_pre_stamping_cache_dir_reconfigures_cleanly(self) -> None:
        # a build directory from before the _LAST_SEEN stamping carries
        # cached values with no stamp to compare against; the first
        # reconfigure must not record phantom requests (the bot's repro:
        # an old CUDA dir with USE_NCCL=ON cached, reconfigured with
        # USE_CUDA=0, used to stop at the NCCL refusal)
        build = self.tmp / "pre-stamping-dir"
        build.mkdir()
        (build / "CMakeCache.txt").write_text(
            "//Use NCCL\n"
            "USE_NCCL:BOOL=ON\n")
        self.assert_ok("USE_NCCL", False, "-DUSE_CUDA=0", "-DUSE_ROCM=0",
                       build="pre-stamping-dir")
        # and the second run stays quiet too: no marker was ever written
        self.assert_ok("USE_NCCL", False, "-DUSE_CUDA=0", "-DUSE_ROCM=0",
                       build="pre-stamping-dir")

    def test_poisoned_default_cache_falls_back(self) -> None:
        self.skip_if_cuda_installed()
        # the state a configure that died before its fallback leaves: the
        # declaration default ON in the cache under the declaration's own
        # help string, plus the _LAST_SEEN stamp a real declaration pass
        # writes, and no request marker. a plain rerun must fall back,
        # not read the leftover default as a request
        build = self.tmp / "poisoned-cache"
        build.mkdir()
        (build / "CMakeCache.txt").write_text(
            "//Use CUDA\n"
            "USE_CUDA:BOOL=ON\n"
            "\n"
            "_USE_CUDA_LAST_SEEN:INTERNAL=ON\n")
        self.assert_ok("USE_CUDA", False, build="poisoned-cache")


if __name__ == "__main__":
    unittest.main()
