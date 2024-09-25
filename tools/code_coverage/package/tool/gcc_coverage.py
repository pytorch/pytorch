from __future__ import annotations

import os
import subprocess
import time

# gcc is only used in oss
from ..oss.utils import get_gcda_files, run_oss_python_test
from ..util.setting import JSON_FOLDER_BASE_DIR, TestType
from ..util.utils import print_log, print_time
from .utils import run_cpp_test


def update_gzip_dict(gzip_dict: dict[str, int], file_name: str) -> str:
    file_name = file_name.lower()
    gzip_dict[file_name] = gzip_dict.get(file_name, 0) + 1
    num = gzip_dict[file_name]
    return str(num) + "_" + file_name


def run_target(binary_file: str, test_type: TestType) -> None:
    print_log("start run", test_type.value, "test: ", binary_file)
    start_time = time.time()
    assert test_type in {TestType.CPP, TestType.PY}
    if test_type == TestType.CPP:
        run_cpp_test(binary_file)
    else:
        run_oss_python_test(binary_file)

    print_time(" time: ", start_time)


def export() -> None:
    start_time = time.time()
    # collect .gcda files
    gcda_files = get_gcda_files()
    # file name like utils.cpp may have same name in different folder
    gzip_dict: dict[str, int] = {}
    for gcda_item in gcda_files:
        # generate json.gz
        subprocess.check_call(["gcov", "-i", gcda_item])
        # cp json.gz to profile/json folder
        gz_file_name = os.path.basename(gcda_item) + ".gcov.json.gz"
        new_file_path = os.path.join(
            JSON_FOLDER_BASE_DIR, update_gzip_dict(gzip_dict, gz_file_name)
        )
        os.rename(gz_file_name, new_file_path)
        #  unzip json.gz to json
        subprocess.check_output(["gzip", "-d", new_file_path])
    print_time("export take time: ", start_time, summary_time=True)
