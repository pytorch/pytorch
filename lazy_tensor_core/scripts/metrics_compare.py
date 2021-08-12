#!/usr/bin/env python3
"""
Compares metric reports contained in two separate files, and prints a summary
of differences, sorted by the percent change.
"""

import argparse
import collections

import lazy_tensor_core.debug.metrics_compare_utils as mcu

TITLE = ['KEY', 'Val1', 'Val2', 'PCT_CHANGE']
REPORT_FIRST_LINE = 'Metric: CompileTime'
COUNTERS = 'Counters'
PERCENTILES = 'Percentiles'
HIGH_PRI_KEYS = ['CompileTime', 'ExecuteTime']


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('filepath1', help='File 1, containing lazy tensor metrics')
    parser.add_argument('filepath2', help='File 2, containing lazy tensor metrics')
    parser.add_argument('--topn-counters', '-c', type=int, default=10)
    parser.add_argument('--topn-percentiles', '-p', type=int, default=10)
    parser.add_argument('--skip-1', type=int, default=0)
    parser.add_argument('--skip-2', type=int, default=0)
    parser.add_argument('--threshold', '-t', type=float, default=50.0)
    parser.add_argument('--no-humanize', '-r', action='store_true')
    parser.add_argument(
        '--show',
        '-s',
        nargs='+',
        type=str,
        help='Metrics to always show',
        default=None)
    return parser.parse_args()


def extract_report(filename, n_reports_to_skip):
    skipcount = 0
    with open(filename, 'r') as f:
        while skipcount <= n_reports_to_skip:
            line = f.readline()
            skipcount += line.startswith(REPORT_FIRST_LINE)
            continue
        lines = [line]
        for line in f:
            if line == '':
                break
            lines.append(line)
    return '\n'.join(lines)


def print_absent_key_info(report1, report2):
    delta_1_2 = set(report1.keys()).difference(set(report2.keys()))
    delta_2_1 = set(report2.keys()).difference(set(report1.keys()))
    if delta_2_1:
        print('\tKeys missing from report 1:')
        for key in delta_2_1:
            print('\t\t{} - {} in report 2'.format(key, report2.pop(key)))
    if delta_1_2:
        print('Keys missing from report 2:')
        for key in delta_1_2:
            print('\t\t{} - {} in report 1'.format(key, report1.pop(key)))
    if not delta_1_2 and not delta_2_1:
        print('Keys are exactly the same in reports.')


def print_separator():
    print('\n' + '-' * 80 + '\n')


def sort_counters(report1, report2):
    delta = {}
    for key, val1 in report1.items():
        if isinstance(val1, tuple):
            v1, v2 = val1[0], report2[key][0]
        else:
            v1, v2 = val1, report2[key]
        delta[key] = (-v1 + v2) / v1 * 100
    delta = sorted(delta.items(), key=lambda item: abs(item[1]), reverse=True)
    delta = [(key, report1[key], report2[key], pct_change)
             for key, pct_change in delta
             if abs(pct_change) > args.threshold]
    return delta


def percentile_priority(key, priorities):
    metric, p = key.split('__')
    p = int(p.split('_')[1])
    return (priorities[metric], metric, -p)


def sort_percentiles(report1, report2):
    delta = {
        key: (-val1 + report2[key][0]) / val1 * 100
        for key, (val1, val2) in report1.items()
    }
    priorities = collections.defaultdict(list)
    hipri = args.show or HIGH_PRI_KEYS
    for key, d in delta.items():
        m = key.split('__')[0]
        try:
            p = 2**20 - hipri.index(m)
        except ValueError:
            p = abs(d)
        priorities[key.split('__')[0]].append(p)
    else:
        for key, ps in priorities.items():
            priorities[key] = sum(ps) / float(len(ps))
    delta = sorted(
        delta.items(),
        key=lambda item: percentile_priority(item[0], priorities),
        reverse=True)
    delta = [(key, report1[key], report2[key], pct_change)
             for key, pct_change in delta
             if percentile_priority(key, priorities)[0] > args.threshold]
    return delta


def sort_metrics(report1, report2, descr):
    f = sort_counters if descr == COUNTERS else sort_percentiles
    return f(report1, report2)


def get_pretty_row_format(formatted_rows):
    rows = [TITLE] + formatted_rows
    lens = [2 + max(len(str(_)) for _ in col) for col in map(list, zip(*rows))]
    pretty_format = ['{' + ':>{}'.format(length) + '}' for length in lens]
    return ''.join(pretty_format)


def split_counters_percentiles(report):
    counters, percentiles = {}, {}
    for key, val in report.items():
        # counters, metrics totals and acucmulators
        if key.endswith('__Value') or key.endswith(
                '__TotalSamples') or '__Accumulator' in key:
            counters[key] = val
        else:
            percentiles[key] = val
    return counters, percentiles


def format_row(k, v1, v2, p):
    # v1, v2 are of the form:
    #    (parsed_int, humanized_str)
    if k.endswith('__Value') or k.endswith('__TotalSamples'):
        k = k.replace('__Value', '').replace('__TotalSamples', '.Count')
        if isinstance(v1, tuple):
            v1, v2 = int(v1[0]), int(v2[0])
        v1, v2 = int(v1), int(v2)
    elif '__Percentile_' in k or '__Accumulator' in k:
        k = k.replace('__Percentile_', '.P').replace('__Accumulator', '.Total')
        if not args.no_humanize:
            v1 = v1[1]
            v2 = v2[1]
        else:
            v1 = '{:.4f}'.format(v1[0])
            v2 = '{:.4f}'.format(v2[0])
    p = '{:.1f}'.format(p)
    return k, v1, v2, p


def format_dat(dat):
    return list(format_row(k, v1, v2, p) for k, v1, v2, p in dat)


def print_pct_changes(args, report1, report2, descr):
    sorted_dat = sort_metrics(report1, report2, descr)
    formatted_dat = format_dat(sorted_dat)
    topn = getattr(args, 'topn_{}'.format(descr).lower())
    print(
        'Changes sorted by pct -- {} -- min threshold {} -- max rows {}\n'.format(
            descr, args.threshold, topn))
    pretty_format = get_pretty_row_format(formatted_dat)
    print(pretty_format.format(*TITLE))
    for i, row_args in enumerate(formatted_dat):
        if i == topn:
            break
        print(pretty_format.format(*row_args))


def print_comparison_summary(args, report1, report2):
    counters1, percentiles1 = split_counters_percentiles(report1)
    counters2, percentiles2 = split_counters_percentiles(report2)
    print_absent_key_info(report1, report2)  # also equalizes the keys
    print_separator()
    print_pct_changes(args, counters1, counters2, descr=COUNTERS)
    print_separator()
    print_pct_changes(args, percentiles1, percentiles2, descr=PERCENTILES)


def main(args):
    report1 = extract_report(args.filepath1, args.skip_1)
    report2 = extract_report(args.filepath2, args.skip_2)
    report1 = mcu.parse_metrics_report(report1, dehumanize=False)
    report2 = mcu.parse_metrics_report(report2, dehumanize=False)
    print_comparison_summary(args, report1, report2)


if __name__ == '__main__':
    args = parse_args()
    main(args)
