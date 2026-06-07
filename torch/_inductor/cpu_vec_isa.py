# mypy: allow-untyped-defs
import dataclasses
import functools
import os
import platform
import re
import subprocess
import sys
import warnings
from collections.abc import Callable
from typing import Any

import torch
from torch._inductor import config
from torch._inductor.utils import python_subprocess_env


_IS_WINDOWS = sys.platform == "win32"


def _get_isa_dry_compile_fingerprint(isa_flags: str) -> str:
    # ISA dry compile will cost about 1 sec time each startup time.
    # Please check the issue: https://github.com/pytorch/pytorch/issues/100378
    # Actually, dry compile is checking compile capability for ISA.
    # We just record the compiler version, isa options and pytorch version info,
    # and generated them to output binary hash path.
    # It would optimize and skip compile existing binary.
    from torch._inductor.cpp_builder import get_compiler_version_info, get_cpp_compiler

    compiler_info = get_compiler_version_info(get_cpp_compiler())
    torch_version = torch.__version__
    fingerprint = f"{compiler_info}={isa_flags}={torch_version}"
    return fingerprint


class VecISA:
    """Describes a CPU SIMD ISA (AVX2, AVX512, NEON, SVE, VSX, ZVECTOR) for
    inductor's CPU vectorization.

    Carries the bit width, ``CPU_CAPABILITY_*`` build macros, compiler arch
    flags, and the per-dtype lane count used by the codegen. Subclasses (one
    per ISA) override the class-level fields; instances are hashable by their
    string representation and usable as a dict key. ``bool(vec_isa)`` runs the
    dry-compile + dlopen probe described in
    `Note [Checking for Vectorized Support in Inductor]`_ to verify the host
    toolchain actually produces a working ``.so`` for this ISA.
    """

    _bit_width: int
    _macro: list[str]
    _arch_flags: str
    _dtype_nelements: dict[torch.dtype, int]

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
#if defined(CPU_CAPABILITY_AVX512) || defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_ZVECTOR) || defined(CPU_CAPABILITY_NEON) || defined(CPU_CAPABILITY_VSX) || defined(CPU_CAPABILITY_SVE)
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#endif

alignas(64) float in_out_ptr0[16] = {0.0};

extern "C" void __avx_chk_kernel() {
    auto tmp0 = at::vec::Vectorized<float>(1);
    auto tmp1 = tmp0.exp();
    tmp1.store(in_out_ptr0);
}
"""

    # Skip the slow `import torch` (which has hung under load in CI) on Linux
    # only: the parent puts torch's lib dir on LD_LIBRARY_PATH so the linker
    # resolves libtorch_cpu for the test .so on its own. macOS can't do this --
    # SIP strips DYLD_LIBRARY_PATH from the child -- so it falls back to import.
    # TODO: extend the no-import path to Windows once its CI build is green
    # enough to validate it.
    _avx_py_load = """
import sys
if sys.platform != "linux":
    import torch  # noqa: F401
from ctypes import cdll
cdll.LoadLibrary("__lib_path__")
"""

    def bit_width(self) -> int:
        return self._bit_width

    def nelements(self, dtype: torch.dtype = torch.float) -> int:
        return self._dtype_nelements[dtype]

    def build_macro(self) -> list[str]:
        return self._macro

    def build_arch_flags(self) -> str:
        return self._arch_flags

    def __hash__(self) -> int:
        return hash(str(self))

    @staticmethod
    def _build_probe_env() -> dict[str, str]:
        """Construct the child env for the dlopen probe.

        Make libtorch_cpu (and other torch shlibs) findable by the Linux
        dynamic linker via LD_LIBRARY_PATH so the probe child does not need to
        ``import torch`` to bring them into its address space. Prepend rather
        than append so a successful probe is guaranteed to bind against the
        torch we are currently running, not an older install on the user's
        loader path. macOS and Windows do not use this fast path; they
        ``import torch`` in the child instead (see ``_avx_py_load``).
        """
        lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
        env = python_subprocess_env()
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            os.pathsep.join([lib_dir, existing]) if existing else lib_dir
        )
        return env

    def _probe_load(self, output_path: str) -> bool:
        """Spawn a child Python that ``dlopen``s ``output_path``.

        Bound the dlopen probe so a stuck child cannot hang the parent for
        the whole 30-minute outer timeout (observed in CI). ``subprocess.run``
        kills the child on ``TimeoutExpired`` (``check_call`` leaks it).
        Returns ``False`` on timeout or load failure, ``True`` on success.
        """
        probe_cmd = [
            sys.executable,
            "-c",
            VecISA._avx_py_load.replace("__lib_path__", output_path),
        ]
        try:
            subprocess.run(
                probe_cmd,
                stderr=subprocess.DEVNULL,
                env=self._build_probe_env(),
                timeout=60,
                check=True,
            )
        except subprocess.TimeoutExpired:
            warnings.warn(
                f"VecISA dlopen probe for {self} hung after 60s",
                stacklevel=2,
            )
            return False
        except subprocess.CalledProcessError:
            return False
        return True

    def check_build(self, code: str) -> bool:
        """Dry-compile ``code`` with this ISA's flags and verify the resulting
        shared library can be loaded.

        Compiles into the inductor codecache, then spawns a short-lived
        subprocess to ``dlopen`` the ``.so``. Returns ``True`` if both build
        and load succeed, ``False`` on compile error, load failure, or dlopen
        probe timeout.
        """
        from torch._inductor.codecache import get_lock_dir, LOCK_TIMEOUT, write
        from torch._inductor.cpp_builder import (
            CppBuilder,
            CppTorchOptions,
            normalize_path_separator,
        )

        key, input_path = write(
            code,
            "cpp",
            extra=_get_isa_dry_compile_fingerprint(self.build_arch_flags()),
        )
        from torch.utils._filelock import FileLock

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
                output_path = normalize_path_separator(
                    x86_isa_help_builder.get_target_file_path()
                )
                if not os.path.isfile(output_path):
                    x86_isa_help_builder.build()
            except Exception:
                return False

            return self._probe_load(output_path)

    def __bool__(self) -> bool:
        return self.__bool__impl(config.cpp.vec_isa_ok)

    @functools.cache  # noqa: B019
    def __bool__impl(self, vec_isa_ok) -> bool:
        if vec_isa_ok is not None:
            return vec_isa_ok

        if config.is_fbcode():
            return True

        return self.check_build(VecISA._avx_code)


@dataclasses.dataclass
class VecNEON(VecISA):
    _bit_width = 128  # This is required to leverage the compute implemented in aten/src/ATen/cpu/vec/vec128/vec128_float_neon.h
    _macro = ["CPU_CAPABILITY_NEON", "AT_BUILD_ARM_VEC256_WITH_SLEEF"]
    _arch_flags = ""  # Unused
    _dtype_nelements = {torch.float: 4, torch.bfloat16: 8, torch.float16: 8}

    def __str__(self) -> str:
        if config.is_fbcode():
            return "neon"
        return "asimd"  # detects the presence of advanced SIMD on armv8-a kernels

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__  # type: ignore[assignment]


@dataclasses.dataclass
class VecSVE(VecISA):
    _bit_width: int = 256
    _armv9a_supported: bool | None = dataclasses.field(
        default=None, init=False, compare=False
    )

    def __post_init__(self) -> None:
        assert self._bit_width in (128, 256)
        nelements = self._bit_width // 32
        self._macro = [
            "CPU_CAPABILITY_SVE",
            f"CPU_CAPABILITY_SVE{self._bit_width}",
            "AT_BUILD_ARM_VEC256_WITH_SLEEF",
            "__ARM_FEATURE_BF16",
        ]
        self._arch_flags = (
            f"-march=armv8-a+sve+bf16 -msve-vector-bits={self._bit_width}"
        )
        self._armv9a_arch_flags = (
            "-march=armv9-a+sve2+fp16fml+sha3+bf16+i8mm "
            f"-msve-vector-bits={self._bit_width}"
        )
        self._dtype_nelements = {
            torch.float: nelements,
            torch.bfloat16: nelements * 2,
            torch.float16: nelements * 2,
        }

    def __str__(self) -> str:
        if config.is_fbcode():
            return "neon"
        return "asimd"

    def _has_armv9a(self) -> bool:
        if self._armv9a_supported is None:
            try:
                self._armv9a_supported = bool(
                    torch.cpu.get_capabilities().get("sve2", False)
                )
            except Exception:
                self._armv9a_supported = False
        return self._armv9a_supported

    def build_arch_flags(self) -> str:
        if self._has_armv9a():
            return self._armv9a_arch_flags
        return self._arch_flags

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__  # type: ignore[assignment]


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
    _is_avx512_bf16_supported = False

    def __str__(self) -> str:
        return "avx512"

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__  # type: ignore[assignment]

    _avx512_bf16_code = """
#include <cstdint>
#include <immintrin.h>

extern "C" __m512bh __avx512_bf16_chk_kernel(__m512 a, __m512 b) {
    return _mm512_cvtne2ps_pbh(a, b);
}
"""

    @functools.cache  # noqa: B019
    # pyrefly: ignore [bad-override]
    def __bool__(self) -> bool:
        if super().__bool__():
            if config.is_fbcode():
                return False
            # check avx512_bf16
            if torch.cpu._is_avx512_bf16_supported() and not _IS_WINDOWS:
                # save _arch_flags
                base_flags = self._arch_flags
                # temporarily change _arch_flags for avx512_bf16 check_build
                self._arch_flags += " -mavx512bf16"
                if self.check_build(self._avx512_bf16_code):
                    self._is_avx512_bf16_supported = True
                # restore _arch_flags
                self._arch_flags = base_flags

            return True
        return False

    @functools.lru_cache(None)  # noqa: B019
    def is_avx512_bf16_supported(self) -> bool:
        return self._is_avx512_bf16_supported

    def build_arch_flags(self) -> str:
        if self._is_avx512_bf16_supported:
            return self._arch_flags + " -mavx512bf16"
        else:
            return self._arch_flags


@dataclasses.dataclass
class VecAVX512VNNI(VecAVX512):
    _bit_width = 512
    _arch_flags = VecAVX512._arch_flags + " -mavx512vnni -mavx512vl"
    _dtype_nelements = {
        torch.float: 16,
        torch.bfloat16: 32,
        torch.float16: 32,
        torch.int8: 64,
        torch.uint8: 64,
    }

    def __str__(self) -> str:
        return super().__str__() + " avx512_vnni"

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__  # type: ignore[assignment]

    _avx512_vnni_code = """
#include <cstdint>
#include <immintrin.h>

extern "C" __m256i __avx512_vnni_chk_kernel_1(__m256i src, __m256i a, __m256i b) {
    return _mm256_dpbusd_epi32(src, a, b);
}

extern "C" __m512i __avx512_vnni_chk_kernel_2(__m512i src, __m512i a, __m512i b) {
    return _mm512_dpbusd_epi32(src, a, b);
}
"""

    @functools.cache  # noqa: B019
    def __bool__(self) -> bool:
        if super().__bool__():
            if config.is_fbcode():
                return False
            if (
                torch.cpu._is_vnni_supported()
                and not _IS_WINDOWS
                and self.check_build(self._avx512_vnni_code)
            ):
                return True
        return False

    def build_arch_flags(self) -> str:
        return self._arch_flags


@dataclasses.dataclass
class VecAMX(VecAVX512VNNI):
    _arch_flags = VecAVX512VNNI._arch_flags + " -mamx-tile -mamx-bf16 -mamx-int8"
    # check amx_fp16 separately since it is not always supported when amx is supported
    # amx_fp16 intrinsic compilation need gcc >=13 on platforms which support amx_fp16
    _is_amx_fp16_supported = False

    def __str__(self) -> str:
        return super().__str__() + " amx_tile"

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__

    _amx_code = """
#include <cstdint>
#include <immintrin.h>

struct amx_tilecfg {
  uint8_t palette_id;
  uint8_t start_row;
  uint8_t reserved_0[14];
  uint16_t colsb[16];
  uint8_t rows[16];
};

extern "C" void __amx_chk_kernel() {
  amx_tilecfg cfg = {0};
  _tile_loadconfig(&cfg);
  _tile_zero(0);
  _tile_dpbf16ps(0, 1, 2);
  _tile_dpbusd(0, 1, 2);
}
"""

    _amx_fp16_code = _amx_code.replace("_tile_dpbf16ps", "_tile_dpfp16ps")

    @functools.cache  # noqa: B019
    def __bool__(self) -> bool:
        if super().__bool__():
            if config.is_fbcode():
                return False
            if self.check_build(VecAMX._amx_code) and torch.cpu._init_amx():
                # check amx-fp16 as well when check amx
                if torch.cpu._is_amx_fp16_supported():
                    # save _arch_flags
                    base_flags = self._arch_flags
                    # temporarily change _arch_flags for amx-fp16 check_build
                    self._arch_flags += " -mamx-fp16"
                    if self.check_build(VecAMX._amx_fp16_code):
                        self._is_amx_fp16_supported = True
                    # restore _arch_flags
                    self._arch_flags = base_flags

                return True
        return False

    @functools.lru_cache(None)  # noqa: B019
    def is_amx_fp16_supported(self) -> bool:
        return self._is_amx_fp16_supported

    def build_arch_flags(self) -> str:
        extra_flags = ""
        if self._is_avx512_bf16_supported:
            # avx512_bf16 is not among the base flags, so we need to check and add it here
            # And we need this flag in the WOQ case for dequantization
            extra_flags += " -mavx512bf16"
        if self._is_amx_fp16_supported:
            extra_flags += " -mamx-fp16"
        return self._arch_flags + extra_flags


@dataclasses.dataclass
class VecAVX2(VecISA):
    _bit_width = 256
    _macro = ["CPU_CAPABILITY_AVX2"]
    _arch_flags = (
        "-mavx2 -mfma -mf16c" if not _IS_WINDOWS else "/arch:AVX2"
    )  # TODO: use cflags
    _dtype_nelements = {torch.float: 8, torch.bfloat16: 16, torch.float16: 16}

    def __str__(self) -> str:
        return "avx2"

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__  # type: ignore[assignment]


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

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__  # type: ignore[assignment]


@dataclasses.dataclass
class VecVSX(VecISA):
    _bit_width = 256  # VSX simd supports 128 bit_width, but aten is emulating it as 256
    _macro = ["CPU_CAPABILITY_VSX"]
    _arch_flags = "-mvsx"
    _dtype_nelements = {torch.float: 8, torch.bfloat16: 16, torch.float16: 16}

    def __str__(self) -> str:
        return "vsx"

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__  # type: ignore[assignment]


class InvalidVecISA(VecISA):
    _bit_width = 0
    _macro = [""]
    _arch_flags = ""
    _dtype_nelements = {}

    def __str__(self) -> str:
        return "INVALID_VEC_ISA"

    def __bool__(self) -> bool:  # type: ignore[override]
        return False

    __hash__: Callable[[VecISA], Any] = VecISA.__hash__  # type: ignore[assignment]


def x86_isa_checker() -> list[str]:
    supported_isa: list[str] = []

    def _check_and_append_supported_isa(
        dest: list[str], isa_supported: bool, isa_name: str
    ) -> None:
        if isa_supported:
            dest.append(isa_name)

    Arch = platform.machine()
    """
    Arch value is x86_64 on Linux, and the value is AMD64 on Windows.
    """
    if Arch != "x86_64" and Arch != "AMD64":
        return supported_isa

    avx2 = torch.cpu._is_avx2_supported()
    avx512 = torch.cpu._is_avx512_supported()
    avx512_vnni = avx512 and torch.cpu._is_vnni_supported()
    amx_tile = torch.cpu._is_amx_tile_supported()

    _check_and_append_supported_isa(supported_isa, avx2, "avx2")
    _check_and_append_supported_isa(supported_isa, avx512, "avx512")
    _check_and_append_supported_isa(supported_isa, avx512_vnni, "avx512_vnni")
    _check_and_append_supported_isa(supported_isa, amx_tile, "amx_tile")

    return supported_isa


invalid_vec_isa = InvalidVecISA()
supported_vec_isa_list = [
    VecAMX(),
    VecAVX512VNNI(),
    VecAVX512(),
    VecAVX2(),
    VecNEON(),
    VecSVE(256),
]


def get_isa_from_cpu_capability(
    capability: str | None,
    vec_isa_list: list[VecISA],
    invalid_vec_isa: InvalidVecISA,
):
    # AMX setting is not supported in eager
    # VecAMX will be prioritized for selection when setting ATEN_CPU_CAPABILITY to avx512
    # TODO add sve256 support
    capability_to_isa_str = {
        "default": "INVALID_VEC_ISA",
        "zvector": "zvector",
        "vsx": "vsx",
        "avx2": "avx2",
        "avx512": "avx512",
    }
    if capability in capability_to_isa_str:
        # pyrefly: ignore [bad-index, index-error]
        isa_str = capability_to_isa_str[capability]
        if isa_str == "INVALID_VEC_ISA":
            return invalid_vec_isa
        for vec_isa in vec_isa_list:
            if isa_str in str(vec_isa):
                return vec_isa

    if capability:
        warnings.warn(f"ignoring invalid value for ATEN_CPU_CAPABILITY {capability}")

    return vec_isa_list[0]


# Cache the cpuinfo to avoid I/O overhead. Meanwhile, the cpuinfo content
# might have too much redundant content that is useless for ISA check. Hence,
# we only cache some key isa information.
@functools.cache
def valid_vec_isa_list() -> list[VecISA]:
    isa_list: list[VecISA] = []
    if sys.platform == "darwin" and platform.processor() == "arm":
        isa_list.append(VecNEON())

    if sys.platform not in ["linux", "win32"]:
        return isa_list

    arch = platform.machine()
    if arch == "s390x":
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
                            isa_list.append(VecZVECTOR())
                            break
    elif arch == "ppc64le":
        isa_list.append(VecVSX())
    elif arch == "aarch64":
        caps = torch.cpu.get_capabilities()
        if (caps.get("sve2", False) or caps.get("sve", False)) and caps.get(
            "bf16", False
        ):
            if caps.get("sve_max_length") == 128:
                isa_list.append(VecSVE(128))
            else:
                isa_list.append(VecSVE(256))
        else:
            isa_list.append(VecNEON())

    elif arch in ["x86_64", "AMD64"]:
        """
        arch value is x86_64 on Linux, and the value is AMD64 on Windows.
        """
        _cpu_supported_x86_isa = x86_isa_checker()
        isa_list.extend(
            isa
            for isa in supported_vec_isa_list
            if all(flag in _cpu_supported_x86_isa for flag in str(isa).split()) and isa
        )

    return isa_list


def pick_vec_isa() -> VecISA:
    if config.is_fbcode() and (platform.machine() in ["x86_64", "AMD64"]):
        return VecAVX2()

    _valid_vec_isa_list: list[VecISA] = valid_vec_isa_list()
    if not _valid_vec_isa_list:
        return invalid_vec_isa

    # If the simdlen is None, set simdlen based on the environment ATEN_CPU_CAPABILITY
    # to control CPU vec ISA
    if config.cpp.simdlen is None:
        return get_isa_from_cpu_capability(
            os.getenv("ATEN_CPU_CAPABILITY"), _valid_vec_isa_list, invalid_vec_isa
        )

    for isa in _valid_vec_isa_list:
        if config.cpp.simdlen == isa.bit_width():
            return isa

    return invalid_vec_isa
