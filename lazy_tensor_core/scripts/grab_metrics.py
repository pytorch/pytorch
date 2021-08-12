#!/usr/bin/env python3
# Given a log file in which the metrics report has been dumped, extracts the
# different metrics across multiple points and produces data in a format which
# can be graphed.
# Can also produce data which is a combination of other metric, using the
# --synth parameters:
#
#   --synth 'LiveDataHandles:CreateDataHandles - DestroyDataHandles'
#

from __future__ import print_function

import argparse
import collections
import fileinput
import os
import re
import sys

try:
    import matplotlib
    # Force matplotlib to use the AGG backend and hence prevent missing DISPLAY
    # errors when run on terminals.
    matplotlib.use('AGG')
    import matplotlib.pyplot as plt
except ImportError:
    pass

Graph = collections.namedtuple('Graph', 'name points')


def make_file_name(name):
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)


def save_graph_image(graph, path, format='png', dpi=100):
    plt.rcParams['figure.dpi'] = dpi

    x = []
    y = []
    for p in graph.points:
        x.append(p[0])
        y.append(p[1])
    plt.plot(x, y)
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title(graph.name)
    plt.savefig(path, format=format)
    plt.cla()
    plt.clf()


def create_graph_images(graphs, path, format='png', dpi=100):
    if 'matplotlib.pyplot' not in sys.modules:
        print(
            'Missing matplotlib package: `pip install matplotlib`', file=sys.stderr)
        return
    for g in graphs:
        save_graph_image(
            g,
            os.path.join(path,
                         make_file_name(g.name) + '.' + format),
            format=format,
            dpi=dpi)


def print_graphs(graphs, fd):
    for g in graphs:
        print('[{}]'.format(g.name), file=fd)
        for p in g.points:
            print('{}\t{}'.format(p[0], p[1]), file=fd)
        print('', file=fd)


def parse_metrics(lines):
    # Counter: CreateCompileHandles
    #  Value: 1631
    # Metric: CompileTime
    #  TotalSamples: 2
    metrics = collections.defaultdict(list)
    counter, metric = None, None
    for line in lines:
        if counter is not None:
            m = re.match(r'\s*Value: ([^\s]+)', line)
            if m:
                metrics[counter].append(float(m.group(1)))
                counter = None
                continue
        if metric is not None:
            # Here parsing Accumulator is better
            m = re.match(r'\s*TotalSamples: ([^\s]+)', line)
            if m:
                metrics[metric].append(float(m.group(1)))
                metric = None
                continue
        m = re.match(r'Counter: ([^\s]+)', line)
        if m:
            counter, metric = m.group(1), None
            continue
        m = re.match(r'Metric: ([^\s]+)', line)
        if m:
            counter, metric = None, m.group(1)
            continue
    return metrics


def create_metric_graph(args, metric, metric_data):
    points = []
    for i, v in enumerate(metric_data):
        points.append((i, v))
    return Graph(name=metric, points=tuple(points))


def create_synth_graph(args, synth, metrics):
    name, expr = synth.split(':', 1)
    xvars = set()
    for m in re.finditer(r'[a-zA-Z_][a-zA-Z_0-9]*', expr):
        xvars.add(m.group(0))
    xvars = list(xvars)
    xmetrics = []
    for v in xvars:
        metric_data = metrics.get(v, None)
        if metric_data is None:
            raise RuntimeError('Unknown metric: {}'.format(v))
        xmetrics.append(metric_data)
    points = []
    x = 0
    while True:
        env = {}
        for i, v in enumerate(xvars):
            metric_data = xmetrics[i]
            if x >= len(metric_data):
                break
            env[v] = float(metric_data[x])
        if len(env) < len(xvars):
            break
        y = eval(expr, env)
        points.append((x, y))
        x += 1
    return Graph(name=name, points=tuple(points))


def match_metric(name, metrics):
    for m in metrics:
        if re.match(m, name):
            return True
    return False


def create_report(args, metrics):
    graphs = []
    for metric in metrics.keys():
        if not args.metrics or match_metric(metric, args.metrics):
            graphs.append(create_metric_graph(args, metric, metrics[metric]))
    for synth in (args.synth or []):
        graphs.append(create_synth_graph(args, synth, metrics))
    return graphs


def process_metrics(args, files):
    metrics = parse_metrics(fileinput.input(files))
    graphs = create_report(args, metrics)
    print_graphs(graphs, sys.stdout)
    if args.image_path:
        if not os.path.isdir(args.image_path):
            os.mkdir(args.image_path)
        create_graph_images(graphs, args.image_path)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--metrics', action='append', type=str)
    arg_parser.add_argument('--synth', action='append', type=str)
    arg_parser.add_argument('--image_path', type=str)
    args, files = arg_parser.parse_known_args()
    process_metrics(args, files)
