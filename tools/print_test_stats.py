#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import bz2
import datetime
import json
import math
import os
import re
import statistics
import subprocess
import time
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import (Any, DefaultDict, Dict, Iterable, Iterator, List, Optional,
                    Set, Tuple, cast)
from xml.dom import minidom  # type: ignore[import]

import requests
from typing_extensions import TypedDict
from tools.stats_utils.s3_stat_parser import (newify_case, get_S3_object_from_bucket, get_test_stats_summaries_for_job,
                                              Report, Status, Commit, HAVE_BOTO3, Version2Case, VersionedReport,
                                              Version1Report, Version2Report, ReportMetaMeta)



SimplerSuite = Dict[str, Version2Case]
SimplerFile = Dict[str, SimplerSuite]
SimplerReport = Dict[str, SimplerFile]


class Stat(TypedDict):
    center: float
    spread: Optional[float]


class CaseDiff(TypedDict):
    margin: str
    name: str
    was: Optional[Tuple[Stat, Status]]
    now: Optional[Version2Case]


class SuiteDiff(TypedDict):
    margin: str
    name: str
    was: Optional[Stat]
    now: Optional[float]
    cases: List[CaseDiff]


# TODO: consolidate this with the get_cases function from
# tools/test_history.py

# Here we translate to a three-layer format (file -> suite -> case)
# rather than a two-layer format (suite -> case) because as mentioned in
# a comment in the body of this function, if we consolidate suites that
# share a name, there will be test case name collisions, and once we
# have those, there's no clean way to deal with it in the diffing logic.
# It's not great to have to add a dummy empty string for the filename
# for version 1 reports, but it's better than either losing cases that
# share a name (for version 2 reports) or using a list of cases rather
# than a dict.
def simplify(report: Report) -> SimplerReport:
    if 'format_version' not in report:  # version 1 implicitly
        v1report = cast(Version1Report, report)
        return {
            # we just don't have test filename information sadly, so we
            # just make one fake filename that is the empty string
            '': {
                suite_name: {
                    # This clobbers some cases that have duplicate names
                    # because in version 1, we would merge together all
                    # the suites with a given name (even if they came
                    # from different files), so there were actually
                    # situations in which two cases in the same suite
                    # shared a name (because they actually originally
                    # came from two suites that were then merged). It
                    # would probably be better to warn about the cases
                    # that we're silently discarding here, but since
                    # we're only uploading in the new format (where
                    # everything is also keyed by filename) going
                    # forward, it shouldn't matter too much.
                    case['name']: newify_case(case)
                    for case in suite['cases']
                }
                for suite_name, suite in v1report['suites'].items()
            }
        }
    else:
        v_report = cast(VersionedReport, report)
        version = v_report['format_version']
        if version == 2:
            v2report = cast(Version2Report, v_report)
            return {
                filename: {
                    suite_name: suite['cases']
                    for suite_name, suite in file_data['suites'].items()
                }
                for filename, file_data in v2report['files'].items()
            }
        else:
            raise RuntimeError(f'Unknown format version: {version}')


def plural(n: int) -> str:
    return '' if n == 1 else 's'


def display_stat(
    x: Stat,
    format: Tuple[Tuple[int, int], Tuple[int, int]],
) -> str:
    spread_len = format[1][0] + 1 + format[1][1]
    spread = x['spread']
    if spread is not None:
        spread_str = f' ± {spread:{spread_len}.{format[1][1]}f}s'
    else:
        spread_str = ' ' * (3 + spread_len + 1)
    mean_len = format[0][0] + 1 + format[0][1]
    return f'{x["center"]:{mean_len}.{format[0][1]}f}s{spread_str}'


def list_stat(l: List[float]) -> Stat:
    return {
        'center': statistics.mean(l),
        'spread': statistics.stdev(l) if len(l) > 1 else None
    }


def zero_stat() -> Stat:
    return {'center': 0, 'spread': None}


def recenter(was: Stat, now: float) -> Stat:
    return {'center': now - was['center'], 'spread': was['spread']}


def sum_normals(stats: Iterable[Stat]) -> Stat:
    """
    Returns a stat corresponding to the sum of the given stats.

    Assumes that the center and spread for each of the given stats are
    mean and stdev, respectively.
    """
    l = list(stats)
    spread: Optional[float]
    if any(stat['spread'] is not None for stat in l):
        spread = math.sqrt(sum((stat['spread'] or 0)**2 for stat in l))
    else:
        spread = None
    return {
        'center': sum(stat['center'] for stat in l),
        'spread': spread,
    }


def format_seconds(seconds: List[float]) -> str:
    if len(seconds) > 0:
        x = list_stat(seconds)
        return f'total time {display_stat(x, ((5, 2), (4, 2)))}'.strip()
    return ''


def show_ancestors(num_commits: int) -> str:
    return f'    | : ({num_commits} commit{plural(num_commits)})'


def unlines(lines: List[str]) -> str:
    return ''.join(f'{line}\n' for line in lines)


def matching_test_times(
    *,
    base_reports: Dict[Commit, List[SimplerReport]],
    filename: str,
    suite_name: str,
    case_name: str,
    status: Status,
) -> List[float]:
    times: List[float] = []
    for reports in base_reports.values():
        for report in reports:
            file_data = report.get(filename)
            if file_data:
                suite = file_data.get(suite_name)
                if suite:
                    case = suite.get(case_name)
                    if case:
                        t = case['seconds']
                        s = case['status']
                        if s == status:
                            times.append(t)
    return times


def analyze(
    *,
    head_report: SimplerReport,
    base_reports: Dict[Commit, List[SimplerReport]],
) -> List[SuiteDiff]:
    nonempty_shas = [sha for sha, reports in base_reports.items() if reports]
    # most recent master ancestor with at least one S3 report,
    # or empty list if there are none (will show all tests as added)
    base_report = base_reports[nonempty_shas[0]] if nonempty_shas else []

    # find all relevant suites (those in either base or head or both)
    all_reports = [head_report] + base_report
    all_suites: Set[Tuple[str, str]] = {
        (filename, suite_name)
        for r in all_reports
        for filename, file_data in r.items()
        for suite_name in file_data.keys()
    }

    removed_suites: List[SuiteDiff] = []
    modified_suites: List[SuiteDiff] = []
    added_suites: List[SuiteDiff] = []

    for filename, suite_name in sorted(all_suites):
        case_diffs: List[CaseDiff] = []
        head_suite = head_report.get(filename, {}).get(suite_name)
        base_cases: Dict[str, Status] = dict(sorted(set.intersection(*[
            {
                (n, case['status'])
                for n, case
                in report.get(filename, {}).get(suite_name, {}).items()
            }
            for report in base_report
        ] or [set()])))
        case_stats: Dict[str, Stat] = {}
        if head_suite:
            now = sum(case['seconds'] for case in head_suite.values())
            if any(
                filename in report and suite_name in report[filename]
                for report in base_report
            ):
                removed_cases: List[CaseDiff] = []
                for case_name, case_status in base_cases.items():
                    case_stats[case_name] = list_stat(matching_test_times(
                        base_reports=base_reports,
                        filename=filename,
                        suite_name=suite_name,
                        case_name=case_name,
                        status=case_status,
                    ))
                    if case_name not in head_suite:
                        removed_cases.append({
                            'margin': '-',
                            'name': case_name,
                            'was': (case_stats[case_name], case_status),
                            'now': None,
                        })
                modified_cases: List[CaseDiff] = []
                added_cases: List[CaseDiff] = []
                for head_case_name in sorted(head_suite):
                    head_case = head_suite[head_case_name]
                    if head_case_name in base_cases:
                        stat = case_stats[head_case_name]
                        base_status = base_cases[head_case_name]
                        if head_case['status'] != base_status:
                            modified_cases.append({
                                'margin': '!',
                                'name': head_case_name,
                                'was': (stat, base_status),
                                'now': head_case,
                            })
                    else:
                        added_cases.append({
                            'margin': '+',
                            'name': head_case_name,
                            'was': None,
                            'now': head_case,
                        })
                # there might be a bug calculating this stdev, not sure
                was = sum_normals(case_stats.values())
                case_diffs = removed_cases + modified_cases + added_cases
                if case_diffs:
                    modified_suites.append({
                        'margin': ' ',
                        'name': suite_name,
                        'was': was,
                        'now': now,
                        'cases': case_diffs,
                    })
            else:
                for head_case_name in sorted(head_suite):
                    head_case = head_suite[head_case_name]
                    case_diffs.append({
                        'margin': ' ',
                        'name': head_case_name,
                        'was': None,
                        'now': head_case,
                    })
                added_suites.append({
                    'margin': '+',
                    'name': suite_name,
                    'was': None,
                    'now': now,
                    'cases': case_diffs,
                })
        else:
            for case_name, case_status in base_cases.items():
                case_stats[case_name] = list_stat(matching_test_times(
                    base_reports=base_reports,
                    filename=filename,
                    suite_name=suite_name,
                    case_name=case_name,
                    status=case_status,
                ))
                case_diffs.append({
                    'margin': ' ',
                    'name': case_name,
                    'was': (case_stats[case_name], case_status),
                    'now': None,
                })
            removed_suites.append({
                'margin': '-',
                'name': suite_name,
                # there might be a bug calculating this stdev, not sure
                'was': sum_normals(case_stats.values()),
                'now': None,
                'cases': case_diffs,
            })

    return removed_suites + modified_suites + added_suites


def case_diff_lines(diff: CaseDiff) -> List[str]:
    lines = [f'def {diff["name"]}: ...']

    case_fmt = ((3, 3), (2, 3))

    was = diff['was']
    if was:
        was_line = f'    # was {display_stat(was[0], case_fmt)}'
        was_status = was[1]
        if was_status:
            was_line += f' ({was_status})'
        lines.append(was_line)

    now = diff['now']
    if now:
        now_stat: Stat = {'center': now['seconds'], 'spread': None}
        now_line = f'    # now {display_stat(now_stat, case_fmt)}'
        now_status = now['status']
        if now_status:
            now_line += f' ({now_status})'
        lines.append(now_line)

    return [''] + [f'{diff["margin"]} {l}' for l in lines]


def display_suite_diff(diff: SuiteDiff) -> str:
    lines = [f'class {diff["name"]}:']

    suite_fmt = ((4, 2), (3, 2))

    was = diff['was']
    if was:
        lines.append(f'    # was {display_stat(was, suite_fmt)}')

    now = diff['now']
    if now is not None:
        now_stat: Stat = {'center': now, 'spread': None}
        lines.append(f'    # now {display_stat(now_stat, suite_fmt)}')

    for case_diff in diff['cases']:
        lines.extend([f'  {l}' for l in case_diff_lines(case_diff)])

    return unlines([''] + [f'{diff["margin"]} {l}'.rstrip() for l in lines] + [''])


def anomalies(diffs: List[SuiteDiff]) -> str:
    return ''.join(map(display_suite_diff, diffs))


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


def case_delta(case: CaseDiff) -> Stat:
    was = case['was']
    now = case['now']
    return recenter(
        was[0] if was else zero_stat(),
        now['seconds'] if now else 0,
    )


def display_final_stat(stat: Stat) -> str:
    center = stat['center']
    spread = stat['spread']
    displayed = display_stat(
        {'center': abs(center), 'spread': spread},
        ((4, 2), (3, 2)),
    )
    if center < 0:
        sign = '-'
    elif center > 0:
        sign = '+'
    else:
        sign = ' '
    return f'{sign}{displayed}'.rstrip()


def summary_line(message: str, d: DefaultDict[str, List[CaseDiff]]) -> str:
    all_cases = [c for cs in d.values() for c in cs]
    tests = len(all_cases)
    suites = len(d)
    sp = f'{plural(suites)})'.ljust(2)
    tp = f'{plural(tests)},'.ljust(2)
    # there might be a bug calculating this stdev, not sure
    stat = sum_normals(case_delta(c) for c in all_cases)
    return ''.join([
        f'{message} (across {suites:>4} suite{sp}',
        f'{tests:>6} test{tp}',
        f' totaling {display_final_stat(stat)}',
    ])


def summary(analysis: List[SuiteDiff]) -> str:
    removed_tests: DefaultDict[str, List[CaseDiff]] = defaultdict(list)
    modified_tests: DefaultDict[str, List[CaseDiff]] = defaultdict(list)
    added_tests: DefaultDict[str, List[CaseDiff]] = defaultdict(list)

    for diff in analysis:
        # the use of 'margin' here is not the most elegant
        name = diff['name']
        margin = diff['margin']
        cases = diff['cases']
        if margin == '-':
            removed_tests[name] += cases
        elif margin == '+':
            added_tests[name] += cases
        else:
            removed = list(filter(lambda c: c['margin'] == '-', cases))
            added = list(filter(lambda c: c['margin'] == '+', cases))
            modified = list(filter(lambda c: c['margin'] == '!', cases))
            if removed:
                removed_tests[name] += removed
            if added:
                added_tests[name] += added
            if modified:
                modified_tests[name] += modified

    return unlines([
        summary_line('Removed ', removed_tests),
        summary_line('Modified', modified_tests),
        summary_line('Added   ', added_tests),
    ])


def regression_info(
    *,
    head_sha: Commit,
    head_report: Report,
    base_reports: Dict[Commit, List[Report]],
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
    simpler_head = simplify(head_report)
    simpler_base: Dict[Commit, List[SimplerReport]] = {}
    for commit, reports in base_reports.items():
        simpler_base[commit] = [simplify(r) for r in reports]
    analysis = analyze(
        head_report=simpler_head,
        base_reports=simpler_base,
    )

    return '\n'.join([
        unlines([
            '----- Historic stats comparison result ------',
            '',
            f'    job: {job_name}',
            f'    commit: {head_sha}',
        ]),

        # don't print anomalies, because sometimes due to sharding, the
        # output from this would be very long and obscure better signal

        # anomalies(analysis),

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
        summary(analysis),
    ])


class TestCase:
    def __init__(self, dom: Any) -> None:
        self.class_name = str(dom.attributes['classname'].value)
        self.name = str(dom.attributes['name'].value)
        self.time = float(dom.attributes['time'].value)
        self.errored = len(dom.getElementsByTagName('error')) > 0
        self.failed = len(dom.getElementsByTagName('failure')) > 0
        self.skipped = len(dom.getElementsByTagName('skipped')) > 0


class TestSuite:
    def __init__(self, name: str) -> None:
        self.name = name
        self.test_cases: Dict[str, TestCase] = dict()
        self.failed_count = 0
        self.skipped_count = 0
        self.errored_count = 0
        self.total_time = 0.0

    def __repr__(self) -> str:
        rc = f'{self.name} run_time: {self.total_time:.2f} tests: {len(self.test_cases)}'
        if self.skipped_count > 0:
            rc += f' skipped: {self.skipped_count}'
        return f'TestSuite({rc})'

    def append(self, test_case: TestCase) -> None:
        self.test_cases[test_case.name] = test_case
        self.total_time += test_case.time
        self.failed_count += 1 if test_case.failed else 0
        self.skipped_count += 1 if test_case.skipped else 0
        self.errored_count += 1 if test_case.errored else 0

    def replace(self, test_case: TestCase) -> float:
        name = test_case.name
        assert name in self.test_cases, f'Error: attempting to replace nonexistent test case {name}'
        old_time = self.test_cases[name].time
        # We don't replace anything if the old test case was not shorter.
        if old_time >= test_case.time:
            return 0.0
        self.total_time = self.total_time + test_case.time - old_time
        self.test_cases[name] = test_case
        return test_case.time - old_time

    def print_report(self, num_longest: int = 3) -> None:
        sorted_tests = sorted(self.test_cases.values(), key=lambda x: x.time)
        test_count = len(sorted_tests)
        print(f"class {self.name}:")
        print(f"    tests: {test_count} failed: {self.failed_count} skipped: {self.skipped_count} errored: {self.errored_count}")
        print(f"    run_time: {self.total_time:.2f} seconds")
        print(f"    avg_time: {self.total_time/test_count:.2f} seconds")
        if test_count >= 2:
            print(f"    median_time: {statistics.median(x.time for x in sorted_tests):.2f} seconds")
        sorted_tests = sorted_tests[-num_longest:]
        print(f"    {len(sorted_tests)} longest tests:")
        for test in reversed(sorted_tests):
            print(f"        {test.name} time: {test.time:.2f} seconds")
        print("")


class TestFile:
    def __init__(self, name: str) -> None:
        self.name = name
        self.total_time = 0.0
        self.test_suites: Dict[str, TestSuite] = dict()

    def append(self, test_case: TestCase) -> None:
        suite_name = test_case.class_name
        if suite_name not in self.test_suites:
            self.test_suites[suite_name] = TestSuite(suite_name)
        if test_case.name in self.test_suites[suite_name].test_cases:
            # We expect duplicate tests for test_cpp_extensions_aot, distributed/test_distributed_fork,
            # and distributed/test_distributed_spawn. In these cases, we store the test case that took the longest,
            # as in these jobs, the duplicate tests are run in parallel.
            # For other unexpected cases, we should raise a warning.
            if self.name == 'test_cpp_extensions_aot' or \
               self.name == 'distributed/test_distributed_fork' or \
               self.name == 'distributed/test_distributed_spawn' or \
               self.name == 'cpp':  # The caffe2 cpp tests spawn duplicate test cases as well.
                time_difference = self.test_suites[suite_name].replace(test_case)
                self.total_time += time_difference
            else:
                raise RuntimeWarning(f'Duplicate test case {test_case.name} in suite {suite_name} called from {self.name}')
        else:
            self.test_suites[suite_name].append(test_case)
            self.total_time += test_case.time


def parse_report(path: str) -> Iterator[TestCase]:
    dom = minidom.parse(path)
    for test_case in dom.getElementsByTagName('testcase'):
        yield TestCase(test_case)


def parse_reports(folder: str) -> Dict[str, TestFile]:
    reports = glob(os.path.join(folder, '**', '*.xml'), recursive=True)
    tests_by_file = dict()
    for report in reports:
        test_filename = re.sub(r'\.', '/', os.path.basename(os.path.dirname(report)))
        if test_filename not in tests_by_file:
            tests_by_file[test_filename] = TestFile(test_filename)
        for test_case in parse_report(report):
            tests_by_file[test_filename].append(test_case)
    return tests_by_file

def build_info() -> ReportMetaMeta:
    return {
        "build_pr": os.environ.get("CIRCLE_PR_NUMBER", ""),
        "build_tag": os.environ.get("CIRCLE_TAG", ""),
        "build_sha1": os.environ.get("CIRCLE_SHA1", ""),
        "build_branch": os.environ.get("CIRCLE_BRANCH", ""),
        "build_job": os.environ.get("CIRCLE_JOB", ""),
        "build_workflow_id": os.environ.get("CIRCLE_WORKFLOW_ID", ""),
    }


def build_message(test_case: TestCase) -> Dict[str, Dict[str, Any]]:
    return {
        "normal": {
            **build_info(),
            "test_suite_name": test_case.class_name,
            "test_case_name": test_case.name,
        },
        "int": {
            "time": int(time.time()),
            "test_total_count": 1,
            "test_total_time": int(test_case.time * 1000),
            "test_failed_count": 1 if test_case.failed > 0 else 0,
            "test_skipped_count": 1 if test_case.skipped > 0 else 0,
            "test_errored_count": 1 if test_case.errored > 0 else 0,
        },
    }


def send_report_to_scribe(reports: Dict[str, TestFile]) -> None:
    access_token = os.environ.get("SCRIBE_GRAPHQL_ACCESS_TOKEN")

    if not access_token:
        print("No scribe access token provided, skip sending report!")
        return
    print("Scribe access token provided, sending report...")
    url = "https://graph.facebook.com/scribe_logs"
    r = requests.post(
        url,
        data={
            "access_token": access_token,
            "logs": json.dumps(
                [
                    {
                        "category": "perfpipe_pytorch_test_times",
                        "message": json.dumps(build_message(test_case)),
                        "line_escape": False,
                    }
                    for test_file in reports.values()
                    for test_suite in test_file.test_suites.values()
                    for test_case in test_suite.test_cases.values()
                ]
            ),
        },
    )
    r.raise_for_status()


def assemble_s3_object(
    reports: Dict[str, TestFile],
    *,
    total_seconds: float,
) -> Version2Report:
    return {
        **build_info(),  # type: ignore[misc]
        'total_seconds': total_seconds,
        'format_version': 2,
        'files' : {
            name: {
                'total_seconds': test_file.total_time,
                'suites': {
                    name: {
                        'total_seconds': suite.total_time,
                        'cases': {
                            name: {
                                'seconds': case.time,
                                'status': 'skipped' if case.skipped else
                                          'errored' if case.errored else
                                          'failed' if case.failed else None
                            }
                            for name, case in suite.test_cases.items()
                        },
                    }
                    for name, suite in test_file.test_suites.items()
                }
            }
            for name, test_file in reports.items()
        }
    }


def send_report_to_s3(head_report: Version2Report) -> None:
    job = os.environ.get('CIRCLE_JOB')
    sha1 = os.environ.get('CIRCLE_SHA1')
    branch = os.environ.get('CIRCLE_BRANCH', '')
    if branch not in ['master', 'nightly'] and not branch.startswith("release/"):
        print("S3 upload only enabled on master, nightly and release branches.")
        print(f"skipping test report on branch: {branch}")
        return
    now = datetime.datetime.utcnow().isoformat()
    key = f'test_time/{sha1}/{job}/{now}Z.json.bz2'  # Z meaning UTC
    obj = get_S3_object_from_bucket('ossci-metrics', key)
    # use bz2 because the results are smaller than gzip, and the
    # compression time penalty we pay is only about half a second for
    # input files of a few megabytes in size like these JSON files, and
    # because for some reason zlib doesn't seem to play nice with the
    # gunzip command whereas Python's bz2 does work with bzip2
    obj.put(Body=bz2.compress(json.dumps(head_report).encode()))


def print_regressions(head_report: Report, *, num_prev_commits: int) -> None:
    sha1 = os.environ.get("CIRCLE_SHA1", "HEAD")

    base = subprocess.check_output(
        ["git", "merge-base", sha1, "origin/master"],
        encoding="ascii",
    ).strip()

    count_spec = f"{base}..{sha1}"
    intermediate_commits = int(subprocess.check_output(
        ["git", "rev-list", "--count", count_spec],
        encoding="ascii"
    ))
    ancestry_path = int(subprocess.check_output(
        ["git", "rev-list", "--ancestry-path", "--count", count_spec],
        encoding="ascii",
    ))

    # if current commit is already on master, we need to exclude it from
    # this history; otherwise we include the merge-base
    commits = subprocess.check_output(
        ["git", "rev-list", f"--max-count={num_prev_commits+1}", base],
        encoding="ascii",
    ).splitlines()
    on_master = False
    if base == sha1:
        on_master = True
        commits = commits[1:]
    else:
        commits = commits[:-1]

    job = os.environ.get("CIRCLE_JOB", "")
    objects: Dict[Commit, List[Report]] = defaultdict(list)

    for commit in commits:
        objects[commit]
        summaries = get_test_stats_summaries_for_job(sha=commit, job_prefix=job)
        for _, summary in summaries.items():
            objects[commit].extend(summary)

    print()
    print(regression_info(
        head_sha=sha1,
        head_report=head_report,
        base_reports=objects,
        job_name=job,
        on_master=on_master,
        ancestry_path=ancestry_path - 1,
        other_ancestors=intermediate_commits - ancestry_path,
    ), end="")


def positive_integer(value: str) -> float:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"{value} is not a natural number")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive rational number")
    return parsed


def reports_has_no_tests(reports: Dict[str, TestFile]) -> bool:
    for test_file in reports.values():
        for test_suite in test_file.test_suites.values():
            if len(test_suite.test_cases) > 0:
                return False
    return True

if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser(
        "Print statistics from test XML output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--longest-of-class",
        type=positive_integer,
        default=3,
        metavar="N",
        help="how many longest tests to show for each class",
    )
    parser.add_argument(
        "--class-print-threshold",
        type=positive_float,
        default=1.0,
        metavar="N",
        help="Minimal total time to warrant class report",
    )
    parser.add_argument(
        "--longest-of-run",
        type=positive_integer,
        default=10,
        metavar="N",
        help="how many longest tests to show from the entire run",
    )
    if HAVE_BOTO3:
        parser.add_argument(
            "--upload-to-s3",
            action="store_true",
            help="upload test time to S3 bucket",
        )
        parser.add_argument(
            "--compare-with-s3",
            action="store_true",
            help="download test times for base commits and compare",
        )
    parser.add_argument(
        "--num-prev-commits",
        type=positive_integer,
        default=10,
        metavar="N",
        help="how many previous commits to compare test times with",
    )
    parser.add_argument(
        "--use-json",
        metavar="FILE.json",
        help="compare S3 with JSON file, instead of the test report folder",
    )
    parser.add_argument(
        "folder",
        help="test report folder",
    )
    args = parser.parse_args()

    reports_by_file = parse_reports(args.folder)
    if reports_has_no_tests(reports_by_file):
        print(f"No tests in reports found in {args.folder}")
        sys.exit(0)

    try:
        send_report_to_scribe(reports_by_file)
    except Exception as e:
        print(f"error encountered when uploading to scribe: {e}")

    # longest_tests can contain duplicates as the same tests can be spawned from different files
    longest_tests : List[TestCase] = []
    total_time = 0.0
    for filename, test_filename in reports_by_file.items():
        for suite_name, test_suite in test_filename.test_suites.items():
            if test_suite.total_time >= args.class_print_threshold:
                test_suite.print_report(args.longest_of_class)
                total_time += test_suite.total_time
                longest_tests.extend(test_suite.test_cases.values())
    longest_tests = sorted(longest_tests, key=lambda x: x.time)[-args.longest_of_run:]

    obj = assemble_s3_object(reports_by_file, total_seconds=total_time)

    if args.upload_to_s3:
        try:
            send_report_to_s3(obj)
        except Exception as e:
            print(f"error encountered when uploading to s3: {e}")

    print(f"Total runtime is {datetime.timedelta(seconds=int(total_time))}")
    print(f"{len(longest_tests)} longest tests of entire run:")
    for test_case in reversed(longest_tests):
        print(f"    {test_case.class_name}.{test_case.name}  time: {test_case.time:.2f} seconds")

    if args.compare_with_s3:
        head_json = obj
        if args.use_json:
            head_json = json.loads(Path(args.use_json).read_text())
        print_regressions(head_json, num_prev_commits=args.num_prev_commits)
