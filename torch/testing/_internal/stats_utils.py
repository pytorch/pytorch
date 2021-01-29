import math
import statistics
from collections import defaultdict
from typing import (DefaultDict, Dict, Iterable, List, Optional, Set, Tuple,
                    TypeVar)

from typing_extensions import TypedDict

Commit = str  # 40-digit SHA-1 hex string
Status = Optional[str]  # errored, failed, skipped, or None

# represent suite as dict because indexing is useful
SimplerCase = Tuple[float, Status]
SimplerSuite = Dict[str, SimplerCase]
SimplerReport = Dict[str, SimplerSuite]


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


class Stat(TypedDict):
    center: float
    spread: Optional[float]


class CaseDiff(TypedDict):
    margin: str
    name: str
    was: Optional[Tuple[Stat, Status]]
    now: Optional[SimplerCase]


class SuiteDiff(TypedDict):
    margin: str
    name: str
    was: Optional[Stat]
    now: Optional[float]
    cases: List[CaseDiff]


def case_status(case: Case) -> Status:
    for k in {'errored', 'failed', 'skipped'}:
        if case[k]:  # type: ignore
            return k
    return None


def simplify(report: Report) -> SimplerReport:
    return {
        suite_name: {
            case['name']: (case['seconds'], case_status(case))
            for case in suite['cases']
        }
        for suite_name, suite in report['suites'].items()
    }


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
    base_reports: Dict[Commit, List[SimplerReport]],
    suite_name: str,
    case_name: str,
    status: Status,
) -> List[float]:
    times: List[float] = []
    for reports in base_reports.values():
        for report in reports:
            suite = report.get(suite_name)
            if suite:
                case = suite.get(case_name)
                if case:
                    t, s = case
                    if s == status:
                        times.append(t)
    return times


T = TypeVar('T')


def safe_union(sets: List[Set[T]]) -> Set[T]:
    """
    Return the union of sets.
    """
    return set.union(*(sets or [set()]))


def safe_intersection(sets: List[Set[T]]) -> Set[T]:
    """
    Return the intersection of sets, or the empty set if no sets.
    """
    return set.intersection(*(sets or [set()]))


def is_anomaly(was: Stat, now: float, threshold: int) -> bool:
    center = was['center']
    spread = was['spread'] or 0
    distance = abs(now - center)
    # guard against tiny values because they introduce a lot of noise
    cutoff = max(0.5 * center, threshold * spread)
    return bool(spread > 0.01 and distance > cutoff)


def find_anomalies(
    *,
    head_report: SimplerReport,
    base_reports: Dict[Commit, List[SimplerReport]],
    stdev_threshold: int,
) -> List[SuiteDiff]:
    # most recent master ancestor with at least one S3 report
    base_sha = next(sha for sha, reports in base_reports.items() if reports)

    # find all relevant suites (those in either base or head or both)
    all_reports = [head_report] + base_reports[base_sha]
    all_suites = {k for r in all_reports for k in r.keys()}

    removed_suites: List[SuiteDiff] = []
    modified_suites: List[SuiteDiff] = []
    added_suites: List[SuiteDiff] = []

    for suite_name in sorted(all_suites):
        case_diffs: List[CaseDiff] = []
        head_suite = head_report.get(suite_name)
        base_cases = dict(sorted(safe_intersection([
            {(n, s) for n, (_, s) in report.get(suite_name, {}).items()}
            for report in base_reports[base_sha]
        ])))
        case_stats: Dict[str, Stat] = {}
        if head_suite:
            now = sum(case[0] for case in head_suite.values())
            if any(suite_name in report for report in base_reports[base_sha]):
                removed_cases: List[CaseDiff] = []
                for case_name, case_status in base_cases.items():
                    case_stats[case_name] = list_stat(matching_test_times(
                        base_reports,
                        suite_name,
                        case_name,
                        case_status,
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
                        head_seconds, head_status = head_case
                        base_status = base_cases[head_case_name]
                        if head_status != base_status or is_anomaly(stat, head_seconds, stdev_threshold):
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
                    base_reports,
                    suite_name,
                    case_name,
                    case_status,
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
        now_stat: Stat = {'center': now[0], 'spread': None}
        now_line = f'    # now {display_stat(now_stat, case_fmt)}'
        now_status = now[1]
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


def anomalies(
    head_report: Report,
    base_reports: Dict[Commit, List[Report]],
    stdev_threshold: int,
) -> str:
    simpler_head = simplify(head_report)
    simpler_base: Dict[Commit, List[SimplerReport]] = {}
    for commit, reports in base_reports.items():
        simpler_base[commit] = [simplify(r) for r in reports]
    return ''.join(map(display_suite_diff, find_anomalies(
        head_report=simpler_head,
        base_reports=simpler_base,
        stdev_threshold=stdev_threshold,
    )))


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


def summary(
    head_report: Report,
    base_reports: Dict[Commit, List[Report]],
    stdev_threshold: int,
) -> str:
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
    return '\n'.join([
        unlines([
            'Following output is to check for test time regressions:',
            f'    job: {job_name}',
            f'    commit: {head_sha}',
        ]),
        anomalies(
            head_report=head_report,
            base_reports=base_reports,
            stdev_threshold=stdev_threshold,
        ),
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
        summary(
            head_report=head_report,
            base_reports=base_reports,
            stdev_threshold=stdev_threshold,
        ),
    ])
