# This file is extracted from triton's setup.py file.
# It allows us to download triton's dependencies without running the full build
import os
import platform
import shutil
import tarfile
import urllib.request

from pathlib import Path
from subprocess import check_call
from tempfile import TemporaryDirectory
from typing import Any, List, NamedTuple

SCRIPT_DIR = Path(__file__).parent
REPO_DIR = SCRIPT_DIR.parent.parent

class Package(NamedTuple):
    package: str
    name: str
    url: str
    include_flag: str
    lib_flag: str
    syspath_var_name: str


# pybind11


def get_pybind11_package_info() -> Package:
    name = "pybind11-2.11.1"
    url = "https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.tar.gz"
    return Package(
        "pybind11", name, url, "PYBIND11_INCLUDE_DIR", "", "PYBIND11_SYSPATH"
    )


# llvm


def get_llvm_package_info() -> Package:
    # added statement for Apple Silicon
    system = platform.system()
    arch = platform.machine()
    if arch == "aarch64":
        arch = "arm64"
    if system == "Darwin":
        arch = platform.machine()
        if arch == "x86_64":
            arch = "x64"
        system_suffix = f"macos-{arch}"
    elif system == "Linux":
        if arch == "arm64":
            system_suffix = "ubuntu-arm64"
        else:
            vglibc = tuple(map(int, platform.libc_ver()[1].split(".")))
            vglibc = vglibc[0] * 100 + vglibc[1]
            system_suffix = "ubuntu-x64" if vglibc > 217 else "centos-x64"
    else:
        return Package(
            "llvm",
            "LLVM-C.lib",
            "",
            "LLVM_INCLUDE_DIRS",
            "LLVM_LIBRARY_DIR",
            "LLVM_SYSPATH",
        )
    # use_assert_enabled_llvm = check_env_flag("TRITON_USE_ASSERT_ENABLED_LLVM", "False")
    # release_suffix = "assert" if use_assert_enabled_llvm else "release"
    llvm_hash_file = open("../cmake/llvm-hash.txt")
    rev = llvm_hash_file.read(8)
    name = f"llvm-{rev}-{system_suffix}"
    url = f"https://tritonlang.blob.core.windows.net/llvm-builds/{name}.tar.gz"
    return Package(
        "llvm", name, url, "LLVM_INCLUDE_DIRS", "LLVM_LIBRARY_DIR", "LLVM_SYSPATH"
    )


def open_url(url: str) -> Any:
    user_agent = (
        "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0"
    )
    headers = {
        "User-Agent": user_agent,
    }
    request = urllib.request.Request(url, None, headers)
    return urllib.request.urlopen(request)


# ---- package data ---


def get_triton_cache_path() -> str:
    user_home = (
        os.getenv("HOME") or os.getenv("USERPROFILE") or os.getenv("HOMEPATH") or None
    )
    if not user_home:
        raise RuntimeError("Could not find user home directory")
    return os.path.join(user_home, ".triton")


def get_thirdparty_packages() -> List[str]:
    triton_cache_path = get_triton_cache_path()
    packages = [get_pybind11_package_info(), get_llvm_package_info()]
    thirdparty_cmake_args = []
    for p in packages:
        package_root_dir = os.path.join(triton_cache_path, p.package)
        package_dir = os.path.join(package_root_dir, p.name)
        if p.syspath_var_name in os.environ:
            package_dir = os.environ[p.syspath_var_name]
        version_file_path = os.path.join(package_dir, "version.txt")
        if p.syspath_var_name not in os.environ and (
            not os.path.exists(version_file_path)
            or Path(version_file_path).read_text() != p.url
        ):
            try:
                shutil.rmtree(package_root_dir)
            except Exception:
                pass
            os.makedirs(package_root_dir, exist_ok=True)
            print(f"downloading and extracting {p.url} ...")
            file = tarfile.open(fileobj=open_url(p.url), mode="r|*")
            file.extractall(path=package_root_dir)
            # write version url to package_dir
            with open(os.path.join(package_dir, "version.txt"), "w") as f:
                f.write(p.url)
        if p.include_flag:
            thirdparty_cmake_args.append(f"-D{p.include_flag}={package_dir}/include")
        if p.lib_flag:
            thirdparty_cmake_args.append(f"-D{p.lib_flag}={package_dir}/lib")
    return thirdparty_cmake_args


def read_triton_pin(rocm_hash: bool = False) -> str:
    triton_file = "triton.txt" if not rocm_hash else "triton-rocm.txt"
    with open(REPO_DIR / ".ci" / "docker" / "ci_commit_pins" / triton_file) as f:
        return f.read().strip()


def main() -> None:
    from argparse import ArgumentParser

    parser = ArgumentParser("Download Triton Dependencies")
    parser.add_argument("--build-rocm", action="store_true")
    parser.add_argument("--commit-hash", type=str)
    args = parser.parse_args()
    build_rocm = args.build_rocm
    commit_hash = args.commit_hash if args.commit_hash else read_triton_pin(build_rocm)

    with TemporaryDirectory() as tmpdir:
        triton_basedir = Path(tmpdir) / "triton"
        triton_pythondir = triton_basedir / "python"
        if build_rocm:
            triton_repo = "https://github.com/ROCmSoftwarePlatform/triton"
            triton_pkg_name = "pytorch-triton-rocm"
        else:
            triton_repo = "https://github.com/openai/triton"
            triton_pkg_name = "pytorch-triton"
        check_call(["git", "clone", triton_repo], cwd=tmpdir)
        check_call(["git", "checkout", commit_hash], cwd=triton_basedir)

        os.chdir(triton_basedir / "python")
        check_call(["pwd"])
        check_call(["ls"])
        get_thirdparty_packages()

if __name__ == "__main__":
    main()
