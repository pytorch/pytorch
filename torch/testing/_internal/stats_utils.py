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


def plural(n: int) -> str:
    return '' if n == 1 else 's'


def format_seconds(seconds: List[float]) -> str:
    s = ''
    if len(seconds) > 0:
        s += f'total time {statistics.mean(seconds):8.2f}s'
        if len(seconds) > 1:
            s += f' ± {statistics.stdev(seconds):7.2f}s'
    return s


def show_ancestors(num_commits: int) -> str:
    return f'    | : ({num_commits} commit{plural(num_commits)})'


def unlines(lines: List[str]) -> str:
    return ''.join(f'{line}\n' for line in lines)


def graph(
    *,
    head_sha: Commit,
    head_seconds: float,
    base_seconds: Dict[Commit, List[float]],
    on_master: bool,
    ancestry_path: int = 0,
    other_ancestors: int = 0,
) -> str:
    lines = [
        'Commit graph (base is most recent master ancestor with at least one S3 report):',
        '',
        '    : (master)',
        '    |',
    ]

    head_time_str = f'           {format_seconds([head_seconds])}'
    if on_master:
        lines.append(f'    * {head_sha[:10]} (HEAD)   {head_time_str}')
    else:
        lines.append(f'    | * {head_sha[:10]} (HEAD) {head_time_str}')

        if ancestry_path > 0:
            lines += [
                '    | |',
                show_ancestors(ancestry_path),
            ]

        if other_ancestors > 0:
            lines += [
                '    |/|',
                show_ancestors(other_ancestors),
                '    |',
            ]
        else:
            lines.append('    |/')

    is_first = True
    for sha, seconds in base_seconds.items():
        num_runs = len(seconds)
        prefix = str(num_runs).rjust(3)
        base = '(base)' if is_first and num_runs > 0 else '      '
        if num_runs > 0:
            is_first = False
        t = format_seconds(seconds)
        p = plural(num_runs)
        if t:
            p = f'{p}, '.ljust(3)
        lines.append(f'    * {sha[:10]} {base} {prefix} report{p}{t}')

    lines.extend(['    |', '    :'])

    return unlines(lines)


def regression_info(
    *,
    head_sha: Commit,
    head_report: Report,
    base_reports: Dict[Commit, List[Report]],
    stdev_threshold: int,
    job_name: str,
    on_master: bool,
    ancestry_path: int,
    other_ancestors: int,
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

    sections = [
        unlines([
            'Following output is to check this commit for test time regressions:',
            f'    {head_sha}',
        ]),
        graph(
            head_sha=head_sha,
            head_seconds=head_report['total_seconds'],
            base_seconds={
                c: [r['total_seconds'] for r in rs]
                for c, rs in base_reports.items()
            },
            on_master=on_master,
            ancestry_path=ancestry_path,
            other_ancestors=other_ancestors,
        ),
    ]

    lines = []

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

    sections.append(unlines(lines))
    lines = []

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

    sections.append(unlines(lines))
    lines = []

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

    sections.append(unlines(lines))
    lines = []

    return '\n'.join(sections)
