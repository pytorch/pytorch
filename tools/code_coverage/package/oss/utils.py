import os
import subprocess
from typing import List

from ..util.setting import SCRIPT_FOLDER, TestType
from ..util.utils import print_error, remove_file


def get_oss_binary_folder(test_type: TestType) -> str:
    assert test_type in {TestType.CPP, TestType.PY}
    # TODO: change the way we get binary file -- binary may not in build/bin ?
    return os.path.join(
        get_pytorch_folder(), "build/bin" if test_type == TestType.CPP else "test"
    )


def get_oss_shared_library() -> List[str]:
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
    return os.environ.get("PYTORCH_FOLDER", SCRIPT_FOLDER)


def clean_up_gcda() -> None:
    gcda_files = get_gcda_files()
    for item in gcda_files:
        remove_file(item)


def get_gcda_files() -> List[str]:
    folder_has_gcda = os.path.join(get_pytorch_folder(), "build")
    if os.path.isdir(folder_has_gcda):
        # TODO use glob
        # output = glob.glob(f"{folder_has_gcda}/**/*.gcda")
        output = subprocess.check_output(["find", folder_has_gcda, "-iname", "*.gcda"])
        output = output.decode("utf-8").split("\n")
        return output
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
