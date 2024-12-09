from __future__ import annotations

import os
import subprocess
import time

from ..util.setting import (
    JSON_FOLDER_BASE_DIR,
    MERGED_FOLDER_BASE_DIR,
    TestList,
    TestPlatform,
    TestType,
)
from ..util.utils import (
    check_platform_type,
    convert_to_relative_path,
    create_folder,
    get_raw_profiles_folder,
    get_test_name_from_whole_path,
    print_log,
    print_time,
    related_to_test_list,
    replace_extension,
)
from .utils import get_tool_path_by_platform, run_cpp_test


def create_corresponding_folder(
    cur_path: str, prefix_cur_path: str, dir_list: list[str], new_base_folder: str
) -> None:
    for dir_name in dir_list:
        relative_path = convert_to_relative_path(
            cur_path, prefix_cur_path
        )  # get folder name like 'aten'
        new_folder_path = os.path.join(new_base_folder, relative_path, dir_name)
        create_folder(new_folder_path)


def run_target(
    binary_file: str, raw_file: str, test_type: TestType, platform_type: TestPlatform
) -> None:
    print_log("start run: ", binary_file)
    # set environment variable -- raw profile output path of the binary run
    os.environ["LLVM_PROFILE_FILE"] = raw_file
    # run binary
    if test_type == TestType.PY and platform_type == TestPlatform.OSS:
        from ..oss.utils import run_oss_python_test

        run_oss_python_test(binary_file)
    else:
        run_cpp_test(binary_file)


def merge_target(raw_file: str, merged_file: str, platform_type: TestPlatform) -> None:
    print_log("start to merge target: ", raw_file)
    # run command
    llvm_tool_path = get_tool_path_by_platform(platform_type)
    subprocess.check_call(
        [
            f"{llvm_tool_path}/llvm-profdata",
            "merge",
            "-sparse",
            raw_file,
            "-o",
            merged_file,
        ]
    )


def export_target(
    merged_file: str,
    json_file: str,
    binary_file: str,
    shared_library_list: list[str],
    platform_type: TestPlatform,
) -> None:
    if binary_file is None:
        raise Exception(  # noqa: TRY002
            f"{merged_file} doesn't have corresponding binary!"
        )  # noqa: TRY002
    print_log("start to export: ", merged_file)
    # run export
    cmd_shared_library = (
        ""
        if not shared_library_list
        else f" -object  {' -object '.join(shared_library_list)}"
    )
    # if binary_file = "", then no need to add it (python test)
    cmd_binary = "" if not binary_file else f" -object {binary_file} "
    llvm_tool_path = get_tool_path_by_platform(platform_type)

    cmd = f"{llvm_tool_path}/llvm-cov export {cmd_binary} {cmd_shared_library}  -instr-profile={merged_file} > {json_file}"
    os.system(cmd)


def merge(test_list: TestList, platform_type: TestPlatform) -> None:
    print("start merge")
    start_time = time.time()
    # find all raw profile under raw_folder and sub-folders
    raw_folder_path = get_raw_profiles_folder()
    g = os.walk(raw_folder_path)
    for path, dir_list, file_list in g:
        # if there is a folder raw/aten/, create corresponding merged folder profile/merged/aten/ if not exists yet
        create_corresponding_folder(
            path, raw_folder_path, dir_list, MERGED_FOLDER_BASE_DIR
        )
        # check if we can find raw profile under this path's folder
        for file_name in file_list:
            if file_name.endswith(".profraw"):
                if not related_to_test_list(file_name, test_list):
                    continue
                print(f"start merge {file_name}")
                raw_file = os.path.join(path, file_name)
                merged_file_name = replace_extension(file_name, ".merged")
                merged_file = os.path.join(
                    MERGED_FOLDER_BASE_DIR,
                    convert_to_relative_path(path, raw_folder_path),
                    merged_file_name,
                )
                merge_target(raw_file, merged_file, platform_type)
    print_time("merge take time: ", start_time, summary_time=True)


def export(test_list: TestList, platform_type: TestPlatform) -> None:
    print("start export")
    start_time = time.time()
    # find all merged profile under merged_folder and sub-folders
    g = os.walk(MERGED_FOLDER_BASE_DIR)
    for path, dir_list, file_list in g:
        # create corresponding merged folder in [json folder] if not exists yet
        create_corresponding_folder(
            path, MERGED_FOLDER_BASE_DIR, dir_list, JSON_FOLDER_BASE_DIR
        )
        # check if we can find merged profile under this path's folder
        for file_name in file_list:
            if file_name.endswith(".merged"):
                if not related_to_test_list(file_name, test_list):
                    continue
                print(f"start export {file_name}")
                # merged file
                merged_file = os.path.join(path, file_name)
                # json file
                json_file_name = replace_extension(file_name, ".json")
                json_file = os.path.join(
                    JSON_FOLDER_BASE_DIR,
                    convert_to_relative_path(path, MERGED_FOLDER_BASE_DIR),
                    json_file_name,
                )
                check_platform_type(platform_type)
                # binary file and shared library
                binary_file = ""
                shared_library_list = []
                if platform_type == TestPlatform.FBCODE:
                    from caffe2.fb.code_coverage.tool.package.fbcode.utils import (  # type: ignore[import]
                        get_fbcode_binary_folder,
                    )

                    binary_file = os.path.join(
                        get_fbcode_binary_folder(path),
                        get_test_name_from_whole_path(merged_file),
                    )
                elif platform_type == TestPlatform.OSS:
                    from ..oss.utils import get_oss_binary_file, get_oss_shared_library

                    test_name = get_test_name_from_whole_path(merged_file)
                    # if it is python test, no need to provide binary, shared library is enough
                    binary_file = (
                        ""
                        if test_name.endswith(".py")
                        else get_oss_binary_file(test_name, TestType.CPP)
                    )
                    shared_library_list = get_oss_shared_library()
                export_target(
                    merged_file,
                    json_file,
                    binary_file,
                    shared_library_list,
                    platform_type,
                )
    print_time("export take time: ", start_time, summary_time=True)
