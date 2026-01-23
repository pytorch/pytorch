#!/usr/bin/env python3
import concurrent.futures
import distutils.sysconfig
import functools
import itertools
import os
import re
from pathlib import Path
from typing import Any


# We also check that there are [not] cxx11 symbols in libtorch
#
# To check whether it is using cxx11 ABI, check non-existence of symbol:
PRE_CXX11_SYMBOLS = (
    "std::basic_string<",
    "std::list",
)
# To check whether it is using pre-cxx11 ABI, check non-existence of symbol:
CXX11_SYMBOLS = (
    "std::__cxx11::basic_string",
    "std::__cxx11::list",
)
# NOTE: Checking the above symbols in all namespaces doesn't work, because
# devtoolset7 always produces some cxx11 symbols even if we build with old ABI,
# and CuDNN always has pre-cxx11 symbols even if we build with new ABI using gcc 5.4.
# Instead, we *only* check the above symbols in the following namespaces:
LIBTORCH_NAMESPACE_LIST = (
    "c10::",
    "at::",
    "caffe2::",
    "torch::",
)

# Patterns for detecting statically linked libstdc++ symbols
STATICALLY_LINKED_CXX11_ABI = [re.compile(r".*recursive_directory_iterator.*")]


def _apply_libtorch_symbols(symbols):
    return [
        re.compile(f"{x}.*{y}")
        for (x, y) in itertools.product(LIBTORCH_NAMESPACE_LIST, symbols)
    ]


LIBTORCH_CXX11_PATTERNS = _apply_libtorch_symbols(CXX11_SYMBOLS)

LIBTORCH_PRE_CXX11_PATTERNS = _apply_libtorch_symbols(PRE_CXX11_SYMBOLS)


@functools.lru_cache(100)
def get_symbols(lib: str) -> list[tuple[str, str, str]]:
    from subprocess import check_output

    lines = check_output(f'nm "{lib}"|c++filt', shell=True)
    return [x.split(" ", 2) for x in lines.decode("latin1").split("\n")[:-1]]


def grep_symbols(
    lib: str, patterns: list[Any], symbol_type: str | None = None
) -> list[str]:
    def _grep_symbols(
        symbols: list[tuple[str, str, str]], patterns: list[Any]
    ) -> list[str]:
        rc = []
        for _s_addr, _s_type, s_name in symbols:
            # Filter by symbol type if specified
            if symbol_type and _s_type != symbol_type:
                continue
            for pattern in patterns:
                if pattern.match(s_name):
                    rc.append(s_name)
                    continue
        return rc

    all_symbols = get_symbols(lib)
    num_workers = 32
    chunk_size = (len(all_symbols) + num_workers - 1) // num_workers

    def _get_symbols_chunk(i):
        return all_symbols[i * chunk_size : (i + 1) * chunk_size]

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        tasks = [
            executor.submit(_grep_symbols, _get_symbols_chunk(i), patterns)
            for i in range(num_workers)
        ]
        return functools.reduce(list.__add__, (x.result() for x in tasks), [])


def check_lib_statically_linked_libstdc_cxx_abi_symbols(lib: str) -> None:
    cxx11_statically_linked_symbols = grep_symbols(
        lib, STATICALLY_LINKED_CXX11_ABI, symbol_type="T"
    )
    num_statically_linked_symbols = len(cxx11_statically_linked_symbols)
    print(f"num_statically_linked_symbols (T): {num_statically_linked_symbols}")
    if num_statically_linked_symbols > 0:
        raise RuntimeError(
            f"Found statically linked libstdc++ symbols (recursive_directory_iterator): {cxx11_statically_linked_symbols[:100]}"
        )


def _compile_and_extract_symbols(
    cpp_content: str, compile_flags: list[str], exclude_list: list[str] | None = None
) -> list[str]:
    """
    Helper to compile a C++ file and extract all symbols.

    Args:
        cpp_content: C++ source code to compile
        compile_flags: Compilation flags
        exclude_list: List of symbol names to exclude. Defaults to ["main"].

    Returns:
        List of all symbols found in the object file (excluding those in exclude_list).
    """
    import subprocess
    import tempfile

    if exclude_list is None:
        exclude_list = ["main"]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        cpp_file = tmppath / "test.cpp"
        obj_file = tmppath / "test.o"

        cpp_file.write_text(cpp_content)

        result = subprocess.run(
            compile_flags + [str(cpp_file), "-o", str(obj_file)],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Compilation failed: {result.stderr}")

        symbols = get_symbols(str(obj_file))

        # Return all symbol names, excluding those in the exclude list
        return [name for _addr, _stype, name in symbols if name not in exclude_list]


def check_stable_only_symbols(install_root: Path) -> None:
    """
    Test TORCH_STABLE_ONLY and TORCH_TARGET_VERSION by compiling test code.

    This approach tests:
    1. WITHOUT macros -> many torch symbols exposed (compilation succeeds)
    2. WITH TORCH_STABLE_ONLY -> compilation fails with #error directive
    3. WITH TORCH_TARGET_VERSION -> compilation fails with #error directive
    4. WITH both macros -> compilation fails with #error directive
    """
    import subprocess
    import tempfile

    include_dir = install_root / "include"
    if not include_dir.exists():
        raise AssertionError(f"Expected {include_dir} to be present")

    test_cpp_content = """
// Main torch C++ API headers
#include <torch/torch.h>
#include <torch/all.h>

// ATen tensor library
#include <ATen/ATen.h>

// Core c10 headers (commonly used)
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Optional.h>

int main() { return 0; }
"""

    base_compile_flags = [
        "g++",
        "-std=c++17",
        f"-I{include_dir}",
        f"-I{include_dir}/torch/csrc/api/include",
        "-c",  # Compile only, don't link
    ]

    # Compile WITHOUT any macros - should succeed
    symbols_without = _compile_and_extract_symbols(
        cpp_content=test_cpp_content,
        compile_flags=base_compile_flags,
    )

    # We expect constexpr symbols, inline functions used by other headers etc.
    # to produce symbols
    num_symbols_without = len(symbols_without)
    print(f"Found {num_symbols_without} symbols without any macros defined")
    if num_symbols_without == 0:
        raise AssertionError("Expected a non-zero number of symbols without any macros")

    # Helper to verify compilation fails with expected error
    def _expect_compilation_failure(compile_flags: list[str], macro_name: str) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            cpp_file = tmppath / "test.cpp"
            obj_file = tmppath / "test.o"

            cpp_file.write_text(test_cpp_content)

            result = subprocess.run(
                compile_flags + [str(cpp_file), "-o", str(obj_file)],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                raise RuntimeError(
                    f"Expected compilation to fail with {macro_name} defined, but it succeeded"
                )

            stderr = result.stderr
            expected_error_msg = (
                "This file should not be included when either TORCH_STABLE_ONLY "
                "or TORCH_TARGET_VERSION is defined."
            )

            if expected_error_msg not in stderr:
                raise RuntimeError(
                    f"Expected error message to contain:\n  '{expected_error_msg}'\n"
                    f"but got:\n{stderr[:1000]}"
                )

            print(f"Compilation correctly failed with {macro_name} defined")

    compile_flags_with_stable_only = base_compile_flags + ["-DTORCH_STABLE_ONLY"]
    _expect_compilation_failure(compile_flags_with_stable_only, "TORCH_STABLE_ONLY")

    compile_flags_with_target_version = base_compile_flags + [
        "-DTORCH_TARGET_VERSION=1"
    ]
    _expect_compilation_failure(
        compile_flags_with_target_version, "TORCH_TARGET_VERSION"
    )

    compile_flags_with_both = base_compile_flags + [
        "-DTORCH_STABLE_ONLY",
        "-DTORCH_TARGET_VERSION=1",
    ]
    _expect_compilation_failure(compile_flags_with_both, "both macros")


def check_stable_api_symbols(install_root: Path) -> None:
    """
    Test that stable API headers still expose symbols with TORCH_STABLE_ONLY.
    The torch/csrc/stable/c/shim.h header is tested in check_stable_c_shim_symbols
    """
    include_dir = install_root / "include"
    if not include_dir.exists():
        raise AssertionError(f"Expected {include_dir} to be present")

    stable_dir = include_dir / "torch" / "csrc" / "stable"
    if not stable_dir.exists():
        raise AssertionError(f"Expected {stable_dir} to be present")

    stable_headers = list(stable_dir.rglob("*.h"))
    if not stable_headers:
        raise RuntimeError("Could not find any stable headers")

    includes = []
    for header in stable_headers:
        rel_path = header.relative_to(include_dir)
        includes.append(f"#include <{rel_path.as_posix()}>")

    includes_str = "\n".join(includes)
    test_stable_content = f"""
{includes_str}
int main() {{ return 0; }}
"""

    compile_flags = [
        "g++",
        "-std=c++17",
        f"-I{include_dir}",
        f"-I{include_dir}/torch/csrc/api/include",
        "-c",
        "-DTORCH_STABLE_ONLY",
    ]

    symbols_stable = _compile_and_extract_symbols(
        cpp_content=test_stable_content,
        compile_flags=compile_flags,
    )
    num_symbols_stable = len(symbols_stable)
    print(f"Found {num_symbols_stable} symbols in torch/csrc/stable")
    if num_symbols_stable <= 0:
        raise AssertionError(
            f"Expected stable headers to expose symbols with TORCH_STABLE_ONLY, "
            f"but found {num_symbols_stable} symbols"
        )


def check_headeronly_symbols(install_root: Path) -> None:
    """
    Test that header-only utility headers still expose symbols with TORCH_STABLE_ONLY.
    """
    include_dir = install_root / "include"
    if not include_dir.exists():
        raise AssertionError(f"Expected {include_dir} to be present")

    # Find all headers in torch/headeronly
    headeronly_dir = include_dir / "torch" / "headeronly"
    if not headeronly_dir.exists():
        raise AssertionError(f"Expected {headeronly_dir} to be present")
    headeronly_headers = list(headeronly_dir.rglob("*.h"))
    if not headeronly_headers:
        raise RuntimeError("Could not find any headeronly headers")

    # Filter out platform-specific headers that may not compile everywhere
    platform_specific_keywords = [
        "cpu/vec",
    ]

    filtered_headers = []
    for header in headeronly_headers:
        rel_path = header.relative_to(include_dir).as_posix()
        if not any(
            keyword in rel_path.lower() for keyword in platform_specific_keywords
        ):
            filtered_headers.append(header)

    includes = []
    for header in filtered_headers:
        rel_path = header.relative_to(include_dir)
        includes.append(f"#include <{rel_path.as_posix()}>")

    includes_str = "\n".join(includes)
    test_headeronly_content = f"""
{includes_str}
int main() {{ return 0; }}
"""

    compile_flags = [
        "g++",
        "-std=c++17",
        f"-I{include_dir}",
        f"-I{include_dir}/torch/csrc/api/include",
        "-c",
        "-DTORCH_STABLE_ONLY",
    ]

    symbols_headeronly = _compile_and_extract_symbols(
        cpp_content=test_headeronly_content,
        compile_flags=compile_flags,
    )
    num_symbols_headeronly = len(symbols_headeronly)
    print(f"Found {num_symbols_headeronly} symbols in torch/headeronly")
    if num_symbols_headeronly <= 0:
        raise AssertionError(
            f"Expected headeronly headers to expose symbols with TORCH_STABLE_ONLY, "
            f"but found {num_symbols_headeronly} symbols"
        )


def check_aoti_shim_symbols(install_root: Path) -> None:
    """
    Test that AOTI shim headers still expose symbols with TORCH_STABLE_ONLY.
    """
    include_dir = install_root / "include"
    if not include_dir.exists():
        raise AssertionError(f"Expected {include_dir} to be present")

    # There are no constexpr symbols etc., so we need to actually use functions
    # so that some symbols are found.
    test_shim_content = """
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
int main() {
    int32_t (*fp1)() = &aoti_torch_device_type_cpu;
    int32_t (*fp2)() = &aoti_torch_dtype_float32;
    (void)fp1; (void)fp2;
    return 0;
}
"""

    compile_flags = [
        "g++",
        "-std=c++17",
        f"-I{include_dir}",
        f"-I{include_dir}/torch/csrc/api/include",
        "-c",
        "-DTORCH_STABLE_ONLY",
    ]

    symbols_shim = _compile_and_extract_symbols(
        cpp_content=test_shim_content,
        compile_flags=compile_flags,
    )
    num_symbols_shim = len(symbols_shim)
    if num_symbols_shim <= 0:
        raise AssertionError(
            f"Expected shim headers to expose symbols with TORCH_STABLE_ONLY, "
            f"but found {num_symbols_shim} symbols"
        )


def check_stable_c_shim_symbols(install_root: Path) -> None:
    """
    Test that stable C shim headers still expose symbols with TORCH_STABLE_ONLY.
    """
    include_dir = install_root / "include"
    if not include_dir.exists():
        raise AssertionError(f"Expected {include_dir} to be present")

    # Check if the stable C shim exists
    stable_shim = include_dir / "torch" / "csrc" / "stable" / "c" / "shim.h"
    if not stable_shim.exists():
        raise RuntimeError("Could not find stable c shim")

    # There are no constexpr symbols etc., so we need to actually use functions
    # so that some symbols are found.
    test_stable_shim_content = """
#include <torch/csrc/stable/c/shim.h>
int main() {
    // Reference stable C API functions to create undefined symbols
    AOTITorchError (*fp1)(const char*, uint32_t*, int32_t*) = &torch_parse_device_string;
    AOTITorchError (*fp2)(uint32_t*) = &torch_get_num_threads;
    (void)fp1; (void)fp2;
    return 0;
}
"""

    compile_flags = [
        "g++",
        "-std=c++17",
        f"-I{include_dir}",
        f"-I{include_dir}/torch/csrc/api/include",
        "-c",
        "-DTORCH_STABLE_ONLY",
    ]

    symbols_stable_shim = _compile_and_extract_symbols(
        cpp_content=test_stable_shim_content,
        compile_flags=compile_flags,
    )
    num_symbols_stable_shim = len(symbols_stable_shim)
    if num_symbols_stable_shim <= 0:
        raise AssertionError(
            f"Expected stable C shim headers to expose symbols with TORCH_STABLE_ONLY, "
            f"but found {num_symbols_stable_shim} symbols"
        )


def check_lib_symbols_for_abi_correctness(lib: str) -> None:
    print(f"lib: {lib}")
    cxx11_symbols = grep_symbols(lib, LIBTORCH_CXX11_PATTERNS)
    pre_cxx11_symbols = grep_symbols(lib, LIBTORCH_PRE_CXX11_PATTERNS)
    num_cxx11_symbols = len(cxx11_symbols)
    num_pre_cxx11_symbols = len(pre_cxx11_symbols)
    print(f"num_cxx11_symbols: {num_cxx11_symbols}")
    print(f"num_pre_cxx11_symbols: {num_pre_cxx11_symbols}")
    if num_pre_cxx11_symbols > 0:
        raise RuntimeError(
            f"Found pre-cxx11 symbols, but there shouldn't be any, see: {pre_cxx11_symbols[:100]}"
        )
    if num_cxx11_symbols < 100:
        raise RuntimeError("Didn't find enough cxx11 symbols")


def main() -> None:
    if "install_root" in os.environ:
        install_root = Path(os.getenv("install_root"))  # noqa: SIM112
    else:
        if os.getenv("PACKAGE_TYPE") == "libtorch":
            install_root = Path(os.getcwd())
        else:
            install_root = Path(distutils.sysconfig.get_python_lib()) / "torch"

    libtorch_cpu_path = str(install_root / "lib" / "libtorch_cpu.so")
    check_lib_symbols_for_abi_correctness(libtorch_cpu_path)
    check_lib_statically_linked_libstdc_cxx_abi_symbols(libtorch_cpu_path)

    # Check symbols when TORCH_STABLE_ONLY is defined
    check_stable_only_symbols(install_root)
    check_stable_api_symbols(install_root)
    check_headeronly_symbols(install_root)
    check_aoti_shim_symbols(install_root)
    check_stable_c_shim_symbols(install_root)


if __name__ == "__main__":
    main()
