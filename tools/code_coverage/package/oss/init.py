import argparse
import os
from typing import List, Optional, Tuple

from ..util.setting import (
    JSON_FOLDER_BASE_DIR,
    LOG_DIR,
    CompilerType,
    Option,
    Test,
    TestList,
    TestType,
)
from ..util.utils import (
    clean_up,
    create_folder,
    print_log,
    raise_no_test_found_exception,
    remove_file,
    remove_folder,
)
from ..util.utils_init import add_arguments_utils, create_folders, get_options
from .utils import (
    clean_up_gcda,
    detect_compiler_type,
    get_llvm_tool_path,
    get_oss_binary_folder,
    get_pytorch_folder,
)


BLOCKED_PYTHON_TESTS = {
    "run_test.py",
    "test_dataloader.py",
    "test_multiprocessing.py",
    "test_multiprocessing_spawn.py",
    "test_utils.py",
}


def initialization() -> Tuple[Option, TestList, List[str]]:
    # create folder if not exists
    create_folders()
    # add arguments
    parser = argparse.ArgumentParser()
    parser = add_arguments_utils(parser)
    parser = add_arguments_oss(parser)
    # parse arguments
    (options, args_interested_folder, args_run_only, arg_clean) = parse_arguments(
        parser
    )
    # clean up
    if arg_clean:
        clean_up_gcda()
        clean_up()
    # get test lists
    test_list = get_test_list(args_run_only)
    # get interested folder -- final report will only over these folders
    interested_folders = empty_list_if_none(args_interested_folder)
    # print initialization information
    print_init_info()
    # remove last time's log
    remove_file(os.path.join(LOG_DIR, "log.txt"))
    return (options, test_list, interested_folders)


def add_arguments_oss(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--run-only",
        help="only run certain test(s), for example: atest test_nn.py.",
        nargs="*",
        default=None,
    )

    return parser


def parse_arguments(
    parser: argparse.ArgumentParser,
) -> Tuple[Option, Optional[List[str]], Optional[List[str]], Optional[bool]]:
    # parse args
    args = parser.parse_args()
    # get option
    options = get_options(args)
    return (options, args.interest_only, args.run_only, args.clean)


def get_test_list_by_type(
    run_only: Optional[List[str]], test_type: TestType
) -> TestList:
    test_list: TestList = []
    binary_folder = get_oss_binary_folder(test_type)
    g = os.walk(binary_folder)
    for _, _, file_list in g:
        for file_name in file_list:
            if run_only is not None and file_name not in run_only:
                continue
            # target pattern in oss is used in printing report -- which tests we have run
            test: Test = Test(
                name=file_name,
                target_pattern=file_name,
                test_set="",
                test_type=test_type,
            )
            test_list.append(test)
    return test_list


def get_test_list(run_only: Optional[List[str]]) -> TestList:
    test_list: TestList = []
    # add c++ test list
    test_list.extend(get_test_list_by_type(run_only, TestType.CPP))
    # add python test list
    py_run_only = get_python_run_only(run_only)
    test_list.extend(get_test_list_by_type(py_run_only, TestType.PY))

    # not find any test to run
    if not test_list:
        raise_no_test_found_exception(
            get_oss_binary_folder(TestType.CPP), get_oss_binary_folder(TestType.PY)
        )
    return test_list


def empty_list_if_none(arg_interested_folder: Optional[List[str]]) -> List[str]:
    if arg_interested_folder is None:
        return []
    # if this argument is specified, just return itself
    return arg_interested_folder


def gcc_export_init():
    remove_folder(JSON_FOLDER_BASE_DIR)
    create_folder(JSON_FOLDER_BASE_DIR)


def get_python_run_only(args_run_only: Optional[List[str]]) -> List[str]:
    # if user specifies run-only option
    if args_run_only:
        return args_run_only

    # if not specified, use default setting, different for gcc and clang
    if detect_compiler_type() == CompilerType.GCC:
        return ["run_test.py"]
    else:
        # for clang, some tests will result in too large intermidiate files that can't be merged by llvm, we need to skip them
        run_only: List[str] = []
        binary_folder = get_oss_binary_folder(TestType.PY)
        g = os.walk(binary_folder)
        for _, _, file_list in g:
            for file_name in file_list:
                if file_name in BLOCKED_PYTHON_TESTS or not file_name.endswith(".py"):
                    continue
                run_only.append(file_name)
            # only run tests in the first-level folder in test/
            break
        return run_only


def print_init_info() -> None:
    print_log("pytorch folder: ", get_pytorch_folder())
    print_log("cpp test binaries folder: ", get_oss_binary_folder(TestType.CPP))
    print_log("python test scripts folder: ", get_oss_binary_folder(TestType.PY))
    print_log("compiler type: ", detect_compiler_type().value)
    print_log(
        "llvm tool folder (only for clang, if you are using gcov please ignore it): ",
        get_llvm_tool_path(),
    )
