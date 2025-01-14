#!/usr/bin/env python3
import concurrent.futures
import distutils.sysconfig
import functools
import itertools
import os
import re
from pathlib import Path
from typing import Any, List, Tuple


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


def _apply_libtorch_symbols(symbols):
    return [
        re.compile(f"{x}.*{y}")
        for (x, y) in itertools.product(LIBTORCH_NAMESPACE_LIST, symbols)
    ]


LIBTORCH_CXX11_PATTERNS = _apply_libtorch_symbols(CXX11_SYMBOLS)

LIBTORCH_PRE_CXX11_PATTERNS = _apply_libtorch_symbols(PRE_CXX11_SYMBOLS)


@functools.lru_cache(100)
def get_symbols(lib: str) -> List[Tuple[str, str, str]]:
    from subprocess import check_output

    lines = check_output(f'nm "{lib}"|c++filt', shell=True)
    return [x.split(" ", 2) for x in lines.decode("latin1").split("\n")[:-1]]


def grep_symbols(lib: str, patterns: List[Any]) -> List[str]:
    def _grep_symbols(
        symbols: List[Tuple[str, str, str]], patterns: List[Any]
    ) -> List[str]:
        rc = []
        for _s_addr, _s_type, s_name in symbols:
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


def check_lib_symbols_for_abi_correctness(lib: str, pre_cxx11_abi: bool = True) -> None:
    print(f"lib: {lib}")
    cxx11_symbols = grep_symbols(lib, LIBTORCH_CXX11_PATTERNS)
    pre_cxx11_symbols = grep_symbols(lib, LIBTORCH_PRE_CXX11_PATTERNS)
    num_cxx11_symbols = len(cxx11_symbols)
    num_pre_cxx11_symbols = len(pre_cxx11_symbols)
    print(f"num_cxx11_symbols: {num_cxx11_symbols}")
    print(f"num_pre_cxx11_symbols: {num_pre_cxx11_symbols}")
    if pre_cxx11_abi:
        if num_cxx11_symbols > 0:
            raise RuntimeError(
                f"Found cxx11 symbols, but there shouldn't be any, see: {cxx11_symbols[:100]}"
            )
        if num_pre_cxx11_symbols < 1000:
            raise RuntimeError("Didn't find enough pre-cxx11 symbols.")
        # Check for no recursive iterators, regression test for https://github.com/pytorch/pytorch/issues/133437
        rec_iter_symbols = grep_symbols(
            lib, [re.compile("std::filesystem::recursive_directory_iterator.*")]
        )
        if len(rec_iter_symbols) > 0:
            raise RuntimeError(
                f"recursive_directory_iterator in used pre-CXX11 binaries, see; {rec_iter_symbols}"
            )
    else:
        if num_pre_cxx11_symbols > 0:
            raise RuntimeError(
                f"Found pre-cxx11 symbols, but there shouldn't be any, see: {pre_cxx11_symbols[:100]}"
            )
        if num_cxx11_symbols < 100:
            raise RuntimeError("Didn't find enought cxx11 symbols")


def main() -> None:
    if "install_root" in os.environ:
        install_root = Path(os.getenv("install_root"))  # noqa: SIM112
    else:
        if os.getenv("PACKAGE_TYPE") == "libtorch":
            install_root = Path(os.getcwd())
        else:
            install_root = Path(distutils.sysconfig.get_python_lib()) / "torch"

    libtorch_cpu_path = install_root / "lib" / "libtorch_cpu.so"
    pre_cxx11_abi = "cxx11-abi" not in os.getenv("DESIRED_DEVTOOLSET", "")
    check_lib_symbols_for_abi_correctness(libtorch_cpu_path, pre_cxx11_abi)


if __name__ == "__main__":
    main()
