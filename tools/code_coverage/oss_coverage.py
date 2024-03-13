#!/usr/bin/env python3
import time

from package.oss.cov_json import get_json_report  # type: ignore[import]
from package.oss.init import initialization  # type: ignore[import]
from package.tool.summarize_jsons import summarize_jsons  # type: ignore[import]
from package.util.setting import TestPlatform  # type: ignore[import]
from package.util.utils import print_time  # type: ignore[import]


def report_coverage() -> None:
    start_time = time.time()
    (options, test_list, interested_folders) = initialization()
    # run cpp tests
    get_json_report(test_list, options)
    # collect coverage data from json profiles
    if options.need_summary:
        summarize_jsons(test_list, interested_folders, [""], TestPlatform.OSS)
    # print program running time
    print_time("Program Total Time: ", start_time)


if __name__ == "__main__":
    report_coverage()
