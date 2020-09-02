import time

from ..tool import clang_coverage, gcc_coverage
from ..util.setting import Option, TestList, TestPlatform
from ..util.utils import check_compiler_type, get_cov_type, print_time
from .init import gcc_export_init
from .run import clang_run, gcc_run


def get_json_report(test_list: TestList, options: Option):
    start_time = time.time()
    cov_type = get_cov_type()
    # TODO change to enum
    check_compiler_type(cov_type)
    if cov_type == "CLANG":
        # run
        if options.need_run:
            clang_run(test_list)
        # merge && export
        if options.need_merge:
            clang_coverage.merge(test_list, TestPlatform.OSS)
        if options.need_export:
            clang_coverage.export(test_list, TestPlatform.OSS)
    elif cov_type == "GCC":
        # run
        if options.need_run:
            gcc_run(test_list)
        # export
        if options.need_export:
            gcc_export_init()
            gcc_coverage.export()

    print_time(
        "collect coverage for cpp tests take time: ", start_time, summary_time=True
    )
