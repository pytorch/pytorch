#!/usr/bin/env python3
# Parses the output of LTC_SAVE_TENSORS_FILE and produces statistics about graph
# types and Python frames.

from __future__ import print_function

import argparse
import collections
import difflib
import os
import re
import sys

GraphInfo = collections.namedtuple('GraphInfo',
                                   'id, graph, ngraph, frame, key, hashes')


def save_graph(graph, path):
    with open(path, 'w') as fd:
        fd.write('\n'.join(graph))


def normalize(graph):
    ngraph = []
    for line in graph:
        line = re.sub(r'([a-z][a-z0-9_-]*)\.[0-9.]+', r'\1', line)
        m = re.match(r'(\s*)%\d+\s*=\s*(.*::[^(]+\()([^)]*)(.*)', line)
        if m:
            args = re.sub(r'%\d+', r'?', m.group(3))
            line = m.group(1) + m.group(2) + args + m.group(4)
        ngraph.append(line)
    return ngraph


def prase_graphs(gfile, dest_dir, graphs=None):
    if dest_dir:
        if os.path.isdir(dest_dir):
            raise RuntimeError('Folder already exists: {}'.format(dest_dir))
        os.mkdir(dest_dir)

    if graphs is None:
        graphs = []
    graph, frame, last_frame, hashes = None, None, None, None
    for line in gfile:
        line = line.rstrip('\n')
        if frame is not None:
            if re.match(r'\s*$', line):
                last_frame = frame
                frame = None
            else:
                frame.append(line)
            continue
        if graph is not None:
            m = re.match(r'## END_GRAPH$', line)
            if m:
                if dest_dir:
                    save_graph(graph,
                               os.path.join(dest_dir, 'graph_{:04d}'.format(len(graphs))))
                graphs.append(
                    GraphInfo(
                        id=len(graphs),
                        graph=graph,
                        ngraph=normalize(graph),
                        frame=last_frame,
                        key='\n'.join(graph),
                        hashes=hashes))
                graph, last_frame, hashes = None, None, None
            else:
                graph.append(line)
            continue
        m = re.match(r'TensorsGraphInfo:', line)
        if m:
            frame = []
            continue
        m = re.match(r'## BEGIN_GRAPH$', line)
        if m:
            graph = []
            continue
        m = re.match(r'Hashes: \(([^)]*)\)', line)
        if m:
            hashes = m.group(1)
            continue
    return graphs


def group_by_frame(graphs):
    fgroup = collections.defaultdict(list)
    for graph in graphs:
        fgroup['\n'.join(graph.frame)].append(graph)
    return fgroup


def group_by_hashes(graphs):
    hgroup = collections.defaultdict(list)
    for graph in graphs:
        hgroup[graph.hashes].append(graph)
    return hgroup


def check_collisions(graphs):
    hgroup = group_by_hashes(graphs)
    for h in hgroup.keys():
        hgraphs = hgroup[h]
        hdict = collections.defaultdict(list)
        for graph in hgraphs:
            hdict[graph.key].append(graph)
        if len(hdict) > 1:
            print('Collision on hash: {}\nGroups:'.format(h), file=sys.stderr)
            for glist in hdict.values():
                print('  Ids: {}'.format([g.id for g in glist]), file=sys.stderr)


def dict_add_instance(gmap, i):
    if i in gmap:
        gmap[i] += 1
        return False
    else:
        gmap[i] = 1
        return True


def diff_graphs(g1, g2, name1, name2, prefix=''):
    diff = difflib.unified_diff(g1.ngraph, g2.ngraph, name1, name2)
    result = ''
    for line in diff:
        if line[-1] != '\n':
            result += '{}{}\n'.format(prefix, line)
        else:
            result += '{}{}'.format(prefix, line)
    return result


def process_graphs(args):
    if not args.files:
        graphs = prase_graphs(sys.stdin, args.graphdir)
    else:
        graphs = []
        for fname in args.files:
            with open(fname, 'r') as fd:
                prase_graphs(fd, args.graphdir, graphs=graphs)
    print('Parsed {} graph(s)'.format(len(graphs)))
    fgroup = group_by_frame(graphs)
    print('{} frame group(s)'.format(len(fgroup)))
    for f in fgroup.keys():
        fgraphs = fgroup[f]
        gmap = dict()
        uniq_graphs = []
        for graph in fgraphs:
            if dict_add_instance(gmap, graph.key):
                uniq_graphs.append(graph)
        print('Frame has {} graph(s) ({} unique):\n{}\n'.format(
            len(fgraphs), len(uniq_graphs), f))
        for i in range(len(uniq_graphs) - 1, 0, -1):
            count = gmap[uniq_graphs[i].key]
            prev_count = gmap[uniq_graphs[i - 1].key]
            print(
                '  Frame {} (len={}, count={}, id={}, h=({})) vs {} (len={}, count={}, id={} h=({}))'
                .format(i - 1, len(uniq_graphs[i - 1].graph), prev_count,
                        uniq_graphs[i - 1].id, uniq_graphs[i - 1].hashes, i,
                        len(uniq_graphs[i].graph), count, uniq_graphs[i].id,
                        uniq_graphs[i].hashes))
            print(
                diff_graphs(
                    uniq_graphs[i - 1],
                    uniq_graphs[i],
                    'frame-{}'.format(i - 1),
                    'frame-{}'.format(i),
                    prefix='  '))
    if args.collisions_check:
        check_collisions(graphs)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--graphdir', type=str)
    arg_parser.add_argument('--collisions_check', action='store_true')
    args, files = arg_parser.parse_known_args()
    args.files = files
    process_graphs(args)
