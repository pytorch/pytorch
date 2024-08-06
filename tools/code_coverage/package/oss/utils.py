from __future__ import annotations

import os
import subprocess

from ..util.setting import CompilerType, TestType, TOOLS_FOLDER
from ..util.utils import print_error, remove_file


def get_oss_binary_folder(test_type: TestType) -> str:
    assert test_type in {TestType.CPP, TestType.PY}
    # TODO: change the way we get binary file -- binary may not in build/bin ?
    return os.path.join(
        get_pytorch_folder(), "build/bin" if test_type == TestType.CPP else "test"
    )


def get_oss_shared_library() -> list[str]:
    lib_dir = os.path.join(get_pytorch_folder(), "build", "lib")
    return [
        os.path.join(lib_dir, lib)
        for lib in os.listdir(lib_dir)
        if lib.endswith(".dylib")
    ]


def get_oss_binary_file(test_name: str, test_type: TestType) -> str:
    assert test_type in {TestType.CPP, TestType.PY}
    binary_folder = get_oss_binary_folder(test_type)
    binary_file = os.path.join(binary_folder, test_name)
    if test_type == TestType.PY:
        # add python to the command so we can directly run the script by using binary_file variable
        binary_file = "python " + binary_file
    return binary_file


def get_llvm_tool_path() -> str:
    return os.environ.get(
        "LLVM_TOOL_PATH", "/usr/local/opt/llvm/bin"
    )  # set default as llvm path in dev server, on mac the default may be /usr/local/opt/llvm/bin


def get_pytorch_folder() -> str:
    # TOOLS_FOLDER in oss: pytorch/tools/code_coverage
    return os.path.abspath(
        os.environ.get(
            "PYTORCH_FOLDER", os.path.join(TOOLS_FOLDER, os.path.pardir, os.path.pardir)
        )
    )


def detect_compiler_type() -> CompilerType | None:
    # check if user specifies the compiler type
    user_specify = os.environ.get("CXX", None)
    if user_specify:
        if user_specify in ["clang", "clang++"]:
            return CompilerType.CLANG
        elif user_specify in ["gcc", "g++"]:
            return CompilerType.GCC

        raise RuntimeError(f"User specified compiler is not valid {user_specify}")

    # auto detect
    auto_detect_result = subprocess.check_output(
        ["cc", "-v"], stderr=subprocess.STDOUT
    ).decode("utf-8")
    if "clang" in auto_detect_result:
        return CompilerType.CLANG
    elif "gcc" in auto_detect_result:
        return CompilerType.GCC
    raise RuntimeError(f"Auto detected compiler is not valid {auto_detect_result}")


def clean_up_gcda() -> None:
    gcda_files = get_gcda_files()
    for item in gcda_files:
        remove_file(item)


def get_gcda_files() -> list[str]:
    folder_has_gcda = os.path.join(get_pytorch_folder(), "build")
    if os.path.isdir(folder_has_gcda):
        # TODO use glob
        # output = glob.glob(f"{folder_has_gcda}/**/*.gcda")
        output = subprocess.check_output(["find", folder_has_gcda, "-iname", "*.gcda"])
        return output.decode("utf-8").split("\n")
    else:
        return []


def run_oss_python_test(binary_file: str) -> None:
    # python test script
    try:
        subprocess.check_call(
            binary_file, shell=True, cwd=get_oss_binary_folder(TestType.PY)
        )
    except subprocess.CalledProcessError:
        print_error(f"Binary failed to run: {binary_file}")
