import dataclasses
import functools
import os
import platform

import re
import subprocess
import sys
from typing import Any, Callable, Dict, List

import torch
from torch._inductor import config

_IS_WINDOWS = sys.platform == "win32"


def get_compiler_version_info(compiler):
    SUBPROCESS_DECODE_ARGS = ("oem",) if _IS_WINDOWS else ()
    env = os.environ.copy()
    env["LC_ALL"] = "C"  # Don't localize output
    try:
        version_string = subprocess.check_output(
            [compiler, "-v"], stderr=subprocess.STDOUT, env=env
        ).decode(*SUBPROCESS_DECODE_ARGS)
    except Exception as e:
        try:
            version_string = subprocess.check_output(
                [compiler, "--version"], stderr=subprocess.STDOUT, env=env
            ).decode(*SUBPROCESS_DECODE_ARGS)
        except Exception as e:
            return ""
    # Mutiple lines to one line string.
    version_string = version_string.replace("\r", "_")
    version_string = version_string.replace("\n", "_")
    return version_string


def _get_isa_dry_compile_fingerprint(isa_flags: str) -> str:
    # ISA dry compile will cost about 1 sec time each startup time.
    # Please check the issue: https://github.com/pytorch/pytorch/issues/100378
    # Actually, dry compile is checking compile capability for ISA.
    # We just record the compiler version, isa options and pytorch version info,
    # and generated them to output binary hash path.
    # It would optimize and skip compile existing binary.
    from torch._inductor.cpp_builder import _get_cpp_compiler

    compiler_info = get_compiler_version_info(_get_cpp_compiler())
    torch_version = torch.__version__
    fingerprint = f"{compiler_info}={isa_flags}={torch_version}"
    return fingerprint


class VecISA:
    _bit_width: int
    _macro: List[str]
    _arch_flags: str
    _dtype_nelements: Dict[torch.dtype, int]

    # Note [Checking for Vectorized Support in Inductor]
    # TorchInductor CPU vectorization reuses PyTorch vectorization utility functions
    # Hence, TorchInductor would depend on Sleef* to accelerate mathematical functions
    # like exp, pow, sin, cos and etc.
    # But PyTorch and TorchInductor might use different compilers to build code. If
    # PyTorch uses gcc-7/g++-7 to build the release package, the libtorch_cpu.so
    # will not expose the Sleef* AVX512 symbols since gcc-7/g++-7 cannot pass
    # avx512 check in CMake - FindAVX.cmake. But TorchInductor install the latest
    # gcc/g++ compiler by default while it could support the AVX512 compilation.
    # Therefore, there would be a conflict sleef version between PyTorch and
    # TorchInductor. Hence, we dry-compile the following code to check whether current
    # HW platform and PyTorch both could support AVX512 or AVX2. And suppose ARM
    # also needs the logic
    # In fbcode however, we are using the same compiler for pytorch and for inductor codegen,
    # making the runtime check unnecessary.
    _avx_code = """
#if defined(CPU_CAPABILITY_AVX512) || defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_ZVECTOR) || defined(CPU_CAPABILITY_NEON)
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#endif

#if defined(__GNUC__)
#define __at_align__ __attribute__((aligned(64)))
#elif defined(_WIN32)
#define __at_align__ __declspec(align(64))
#else
#define __at_align__
#endif
__at_align__ float in_out_ptr0[16] = {0.0};

extern "C" void __avx_chk_kernel() {
    auto tmp0 = at::vec::Vectorized<float>(1);
    auto tmp1 = tmp0.exp();
    tmp1.store(in_out_ptr0);
}
"""  # noqa: B950

    _avx_py_load = """
import torch
from ctypes import cdll
cdll.LoadLibrary("__lib_path__")
"""

    def bit_width(self) -> int:
        return self._bit_width

    def nelements(self, dtype: torch.dtype = torch.float) -> int:
        return self._dtype_nelements[dtype]

    def build_macro(self) -> List[str]:
        return self._macro

    def build_arch_flags(self) -> str:
        return self._arch_flags

    def __hash__(self) -> int:
        return hash(str(self))

    @functools.lru_cache(None)  # noqa: B019
    def __bool__(self) -> bool:
        from torch._inductor.codecache import get_lock_dir, LOCK_TIMEOUT, write
        from torch._inductor.cpp_builder import CppBuilder, CppTorchOptions

        if config.cpp.vec_isa_ok is not None:
            return config.cpp.vec_isa_ok

        if config.is_fbcode():
            return True

        key, input_path = write(
            VecISA._avx_code,
            "cpp",
            extra=_get_isa_dry_compile_fingerprint(self._arch_flags),
        )
        from filelock import FileLock

        lock_dir = get_lock_dir()
        lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
        with lock:
            output_dir = os.path.dirname(input_path)
            buid_options = CppTorchOptions(vec_isa=self, warning_all=False)
            x86_isa_help_builder = CppBuilder(
                key,
                [input_path],
                buid_options,
                output_dir,
            )
            try:
                # Check if the output file exist, and compile when not.
                output_path = x86_isa_help_builder.get_target_file_path()
                output_path = output_path.replace("\\", "/")
                if not os.path.isfile(output_path):
                    status, target_file = x86_isa_help_builder.build()

                # Check build result
                subprocess.check_call(
                    [
                        sys.executable,
                        "-c",
                        VecISA._avx_py_load.replace("__lib_path__", output_path),
                    ],
                    stderr=subprocess.DEVNULL,
                    env={**os.environ, "PYTHONPATH": ":".join(sys.path)},
                )
            except Exception as e:
                print("!!!!!!! 2")
                return False
            
            print("!!!!!!! 3")

            return True


@dataclasses.dataclass
class VecNEON(VecISA):
    _bit_width = 256  # This is required to leverage the compute implemented in aten/src/ATen/cpu/vec/vec256/vec256_float_neon.h
    _macro = ["CPU_CAPABILITY_NEON"]
    if sys.platform == "darwin" and platform.processor() == "arm":
        _macro.append("AT_BUILD_ARM_VEC256_WITH_SLEEF")
    _arch_flags = ""  # Unused
    _dtype_nelements = {torch.float: 8, torch.bfloat16: 16, torch.float16: 16}

    def __str__(self) -> str:
        return "asimd"  # detects the presence of advanced SIMD on armv8-a kernels

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__


@dataclasses.dataclass
class VecAVX512(VecISA):
    _bit_width = 512
    _macro = ["CPU_CAPABILITY_AVX512"]
    _arch_flags = (
        "-mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma"
        if not _IS_WINDOWS
        else "/arch:AVX512"
    )  # TODO: use cflags
    _dtype_nelements = {torch.float: 16, torch.bfloat16: 32, torch.float16: 32}

    def __str__(self) -> str:
        return "avx512"

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__


@dataclasses.dataclass
class VecAVX2(VecISA):
    _bit_width = 256
    _macro = ["CPU_CAPABILITY_AVX2"]
    _arch_flags = (
        "-mavx2 -mfma" if not _IS_WINDOWS else "/arch:AVX2"
    )  # TODO: use cflags
    _dtype_nelements = {torch.float: 8, torch.bfloat16: 16, torch.float16: 16}

    def __str__(self) -> str:
        return "avx2"

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__


@dataclasses.dataclass
class VecZVECTOR(VecISA):
    _bit_width = 256
    _macro = [
        "CPU_CAPABILITY_ZVECTOR",
        "CPU_CAPABILITY=ZVECTOR",
        "HAVE_ZVECTOR_CPU_DEFINITION",
    ]
    _arch_flags = "-mvx -mzvector"
    _dtype_nelements = {torch.float: 8, torch.bfloat16: 16, torch.float16: 16}

    def __str__(self) -> str:
        return "zvector"

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__


class InvalidVecISA(VecISA):
    _bit_width = 0
    _macro = [""]
    _arch_flags = ""
    _dtype_nelements = {}

    def __str__(self) -> str:
        return "INVALID_VEC_ISA"

    def __bool__(self) -> bool:  # type: ignore[override]
        return False

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__


def x86_isa_checker() -> List[str]:
    supported_isa: List[str] = []

    def _check_and_append_supported_isa(
        dest: List[str], isa_supported: bool, isa_name: str
    ):
        if isa_supported:
            dest.append(isa_name)

    Arch = platform.machine()
    """
    Arch value is x86_64 on Linux, and the value is AMD64 on Windows.
    """
    if Arch != "x86_64" and Arch != "AMD64":
        return supported_isa

    avx2 = torch.cpu._is_cpu_support_avx2()
    avx512 = torch.cpu._is_cpu_support_avx512()

    _check_and_append_supported_isa(supported_isa, avx2, "avx2")
    _check_and_append_supported_isa(supported_isa, avx512, "avx512")

    return supported_isa


invalid_vec_isa = InvalidVecISA()
supported_vec_isa_list = [VecAVX512(), VecAVX2(), VecNEON()]


# Cache the cpuinfo to avoid I/O overhead. Meanwhile, the cpuinfo content
# might have too much redundant content that is useless for ISA check. Hence,
# we only cache some key isa information.
@functools.lru_cache(None)
def valid_vec_isa_list() -> List[VecISA]:
    if sys.platform == "darwin" and platform.processor() == "arm":
        return [VecNEON()]

    cur_os = sys.platform
    if cur_os != "linux" and cur_os != "win32":
        return []

    if platform.machine() == "s390x":
        with open("/proc/cpuinfo") as _cpu_info:
            while True:
                line = _cpu_info.readline()
                if not line:
                    break
                # process line
                featuresmatch = re.match(r"^features\s*:\s*(.*)$", line)
                if featuresmatch:
                    for group in featuresmatch.groups():
                        if re.search(r"[\^ ]+vxe[\$ ]+", group):
                            return [VecZVECTOR()]
        return []

    isa_list = []
    _cpu_supported_isa = x86_isa_checker()
    for isa in supported_vec_isa_list:
        if str(isa) in _cpu_supported_isa and isa:
            isa_list.append(isa)
    return isa_list


def pick_vec_isa() -> VecISA:
    if config.is_fbcode():
        return VecAVX2()

    _valid_vec_isa_list: List[VecISA] = valid_vec_isa_list()
    if not _valid_vec_isa_list:
        return invalid_vec_isa

    # If the simdlen is None, it indicates determine the vectorization length automatically
    if config.cpp.simdlen is None:
        assert _valid_vec_isa_list
        return _valid_vec_isa_list[0]

    for isa in _valid_vec_isa_list:
        if config.cpp.simdlen == isa.bit_width():
            return isa

    return invalid_vec_isa
