#!/usr/bin/env python3

import collections
import logging
import os
import re

logging.basicConfig(format='%(name)s: %(message)s')
logger = logging.getLogger('pt-ltc-profiler')


def parse_frame_content(line):
    # Python Frames:
    m = re.match(r'Python Frames:', line)
    if m:
        return line
    # train_loop_fn (test/test_train_mp_imagenet.py:216)
    m = re.match(r'.*\s\(.*:\d*\)', line)
    if m:
        return line
    # [Unlowered Op _local_scalar_dense from Device TPU:0]
    m = re.match(r'\[TAG\s(.*)\sFrom Thread\s\d*\]', line)
    if m:
        return f'Unlowered Op: "{m.group(1)}"\n'


def create_report(frames):
    mrkeys = sorted(frames.keys(), key=lambda x: frames[x], reverse=True)
    report = []
    report.append('=' * 80)
    report.append(
        'Unlowered Op usage summary (more of these ops, lower performance)')
    report.append(
        'Note: _local_scalar_dense typically indicates CPU context access')
    report.append('-' * 80)
    for key in mrkeys:
        report.append('FRAME (count={}):'.format(frames[key]))
        report.append(f'{key}')
        report.append('')
    report.append('=' * 80)

    if os.environ.get('PT_LTC_DEBUG_FILE'):
        with open(os.environ.get('PT_LTC_DEBUG_FILE'), 'a') as f:
            f.write('\n'.join(report))
    else:
        for line in '\n'.join(report).split('\n'):
            logger.warning(line)


def parse_frames(lines):
    frames = collections.defaultdict(int)
    frame, skip_frames = [], False
    for line in lines:
        if re.match(r'C\+\+ Frames:', line):
            skip_frames = True
            continue
        elif re.match(r'\*{3}\sEnd stack trace\s\*{3}', line):
            skip_frames = False
            continue
        if skip_frames:
            continue

        content = parse_frame_content(line)
        if content:
            frame.append(content)
            continue
        if frame:
            frames[''.join(frame)] += 1
            frame = []

    return frames


def process_frames(fname):
    frames = parse_frames(open(fname, 'r'))
    create_report(frames)
