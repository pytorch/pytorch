import math
import statistics
from collections import defaultdict
from typing import DefaultDict, Dict, List, Set

from typing_extensions import TypedDict

Commit = str  # 40-digit SHA-1 hex string


class Case(TypedDict):
    name: str
    seconds: float
    errored: bool
    failed: bool
    skipped: bool


class Suite(TypedDict):
    total_seconds: float
    cases: List[Case]


class ReportMeta(TypedDict, total=False):
    build_pr: str
    build_tag: str
    build_sha1: Commit
    build_branch: str
    build_job: str
    build_workflow_id: str


class Report(ReportMeta):
    total_seconds: float
    suites: Dict[str, Suite]


def regression_info(
    head_sha: Commit,
    head_report: Report,
    base_reports: Dict[Commit, List[Report]],
    *,
    stdev_threshold: int,
) -> str:
    """
    Return a human-readable report describing any test time regressions.

    The head_sha and head_report args give info about the current commit
    and its test times. Since Python dicts maintain insertion order
    (guaranteed as part of the language spec since 3.7), the
    base_reports argument must list the head's several most recent
    master commits, from newest to oldest (so the merge-base is
    list(base_reports)[0]).
    """

    lines = []

    lines.append('Following output is to check this commit for test time regressions:')
    lines.append(f'    {head_sha}')

    lines.append('')

    lines.append(f'Comparing test times against base commit and its {len(base_reports)-1} most recent ancestors:')
    for sha, runs in base_reports.items():
        num_runs = len(runs)
        prefix = str(num_runs).rjust(3)
        plural = ' ' if num_runs == 1 else 's'
        times = [o['total_seconds'] for o in runs]
        t = ''
        if num_runs > 0:
            t += f', total time {statistics.mean(times):8.2f}s'
            if num_runs > 1:
                t += f' ± {statistics.stdev(times):7.2f}s'
        lines.append(f'    {sha} {prefix} run{plural} found in S3{t}')

    lines.append('')

    times = [o['total_seconds'] for runs in base_reports.values() for o in runs]
    total_mean = statistics.mean(times)
    total_stdev = statistics.stdev(times)
    lines.append(f'Prior average total time: {total_mean:8.2f}s ± {total_stdev:.2f}s')
    lines.append(f'Current       total time: {head_report["total_seconds"]:8.2f}s')
    stdevs_bigger = (head_report['total_seconds'] - total_mean) / total_stdev
    stdevs_abs = abs(stdevs_bigger)
    stdevs_floor = math.floor(stdevs_abs)
    stdevs_ceil = math.ceil(stdevs_abs)
    if stdevs_abs < stdev_threshold:
        icon, verb, prep, amount = '🟢', 'maintains', 'within', stdevs_ceil
    else:
        prep, amount = 'by at least', stdevs_floor
        if stdevs_bigger < 0:
            icon, verb = '🟣', 'reduces'
        else:
            icon, verb = '🔴', 'increases'
    plural = '' if amount == 1 else 's'
    lines.append(f'{icon} this commit {verb} total test job time {prep} {amount} standard deviation{plural}')

    all_runs = [head_report] + [run for runs in base_reports.values() for run in runs]
    all_tests: DefaultDict[str, Set[str]] = defaultdict(set)
    for run in all_runs:
        for name, suite in run['suites'].items():
            all_tests[name] |= {case['name'] for case in suite['cases']}

    lines.append('')
    lines.append('------ tests added/removed ------')

    for suite_name, cases in all_tests.items():
        missing_suite = []
        missing_cases = defaultdict(list)
        for commit, runs in base_reports.items():
            for run in runs:
                suite_dict = run['suites'].get(suite_name)
                if suite_dict:
                    run_cases = {case['name'] for case in suite_dict['cases']}
                    for case in cases - run_cases:
                        missing_cases[case].append(commit)
                else:
                    missing_suite.append(commit)
        if missing_suite or missing_cases:
            lines.append('')
            lines.append(f'test suite {suite_name}:')
            if missing_suite:
                lines.append('    missing in these commits:')
                for missing_commit in missing_suite:
                    lines.append(f'        {missing_commit}')
            for case, missing_commits in missing_cases.items():
                lines.append(f'    test case {case} missing in these commits:')
                for missing_commit in missing_commits:
                    lines.append(f'        {missing_commit}')

    lines.append('')
    lines.append('--- tests whose times changed ---')

    for suite_name, cases in all_tests.items():
        curr_suite_dict = head_report['suites'].get(suite_name)
        if not curr_suite_dict:
            for commit, runs in base_reports.items():
                for run in runs:
                    if suite_name in run['suites']:
                        pass
        for commit, runs in base_reports.items():
            for run in runs:
                suite_dict = run['suites'].get(suite_name)

    return ''.join(f'{line}\n' for line in lines)
