#!/usr/bin/env python3

from __future__ import division
from __future__ import print_function

import argparse
import os
import re
import sys
import torch
import traceback

_SAVE_DIR = None
_TENSOR_IDS = {}
_STEP = None


def _is_master_ordinal():
    try:
        import lazy_tensor_core.core.lazy_model as ltm
        return ltm.is_master_ordinal()
    except ImportError:
        return True


def _index_of(sizes, lindex):
    index = []
    for size in reversed(sizes):
        index.append(lindex % size)
        lindex = lindex // size
    return list(reversed(index))


def _get_save_dir():
    if _SAVE_DIR is not None:
        return _SAVE_DIR
    return os.environ.get('MODELCMP_SAVEDIR', None)


def _get_tensor_name(name):
    if name is None:
        name = 'noname'
        # Go fetch first frame which is not this module.
        self_fname = os.path.basename(__file__)
        for st in traceback.extract_stack(limit=8):
            # st = tuple(filename, lineno, function, text)
            fname = os.path.basename(st[0])
            if fname != self_fname:
                return '{}.l{}'.format(fname, st[1])
    return name


def compare_tensors(tensor1, tensor2, rtol=1e-05, atol=1e-08, max_diffs=25):
    sizes1 = list(tensor1.size())
    sizes2 = list(tensor2.size())
    if sizes1 != sizes2:
        return 'Tensors have different shape: {} vs. {}\n'.format(sizes1, sizes2)

    values1 = tensor1.flatten().tolist()
    values2 = tensor2.flatten().tolist()
    diffs = []
    for i in range(0, len(values1)):
        v1 = values1[i]
        v2 = values2[i]
        r = max(abs(v1), abs(v2)) * rtol
        error = abs(v1 - v2)
        if error > max(r, atol):
            diffs.append((error, i, v1, v2))

    top_diffs = sorted(diffs, key=lambda x: x[0], reverse=True)[:max_diffs]
    report = ''
    for error, i, v1, v2 in top_diffs:
        report += '{}: {} vs. {}\terror={}\n'.format(
            _index_of(sizes1, i), v1, v2, error)
    if len(diffs) > max_diffs:
        report += '... aborting after {} differences\n'.format(max_diffs)
    return report


def _collect_saved_tensors(path):
    files = []
    for root, dirnames, filenames in os.walk(path):
        for fname in filenames:
            if re.match(r'.*\.\d+$', fname):
                files.append(fname)
    return set(files)


def configure(save_dir):
    global _SAVE_DIR, _TENSOR_IDS, _STEP
    _SAVE_DIR = save_dir
    _TENSOR_IDS = {}
    _STEP = None


def save(name, tensor, step=None):
    global _TENSOR_IDS, _STEP
    # Allow the model compare save API to be left in place and being a noop if
    # configured with a None _SAVE_DIR.
    save_dir = _get_save_dir()
    if save_dir is not None:
        tensor_data = tensor.cpu()
        if _is_master_ordinal():
            name = _get_tensor_name(name)
            if step is not None:
                path = os.path.join(save_dir, 'step-{}'.format(step))
                if not os.path.isdir(path):
                    os.mkdir(path)
                if step != _STEP:
                    _STEP = step
                    _TENSOR_IDS = {}
            else:
                path = save_dir
            id = _TENSOR_IDS.get(name, 0)
            _TENSOR_IDS[name] = id + 1
            path = os.path.join(path, '{}.{}'.format(name, id))
            torch.save(tensor_data, path)
    return tensor


def _parse_path(path):
    fname = os.path.basename(path)
    rpath = os.path.dirname(path)
    stepname = os.path.basename(rpath)
    step = None
    m = re.match(r'step-(\d+)$', stepname)
    if m:
        step = int(m.group(1))
        rpath = os.path.dirname(rpath)
    id = None
    m = re.match(r'(.*)\.(\d+)$', fname)
    assert m, fname
    return m.group(1), int(m.group(2)), step, rpath


def tensor_file_compare(path1, path2, rtol=1e-05, atol=1e-08, max_diffs=25):
    tensor1 = torch.load(path1)
    tensor2 = torch.load(path2)
    report = compare_tensors(
        tensor1, tensor2, rtol=rtol, atol=atol, max_diffs=max_diffs)
    if report:
        name, id, step, _ = _parse_path(path1)
        lines = report.split('\n')
        report = 'Changes in saved tensor "{}" with id={}{}\n'.format(
            name, id, ' at step={}'.format(step) if step else '')
        for line in lines:
            report += '  {}\n'.format(line)
    return report


def compare(save_dir1, save_dir2, rtol=1e-05, atol=1e-08, max_diffs=25):
    files1 = _collect_saved_tensors(save_dir1)
    files2 = _collect_saved_tensors(save_dir2)
    report = ''
    for path1 in files1:
        if path1 not in files2:
            report += 'Mismatch: {} not in {}\n'.format(path1, save_dir2)
        else:
            report += tensor_file_compare(
                os.path.join(save_dir1, path1),
                os.path.join(save_dir2, path1),
                rtol=rtol,
                atol=atol,
                max_diffs=max_diffs)
    for path2 in files2:
        if path2 not in files1:
            report += 'Mismatch: {} not in {}\n'.format(path2, save_dir1)
    return report


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--rtol', type=float, default=1e-05)
    arg_parser.add_argument('--atol', type=float, default=1e-08)
    arg_parser.add_argument('--max_diffs', type=int, default=25)
    arg_parser.add_argument(
        'savedir1',
        type=str,
        metavar='SAVEDIR1',
        help='The path to the folder containing the first model savedir')
    arg_parser.add_argument(
        'savedir2',
        type=str,
        metavar='SAVEDIR2',
        help='The path to the folder containing the second model savedir')
    args = arg_parser.parse_args()
    report = compare(
        args.savedir1,
        args.savedir2,
        rtol=args.rtol,
        atol=args.atol,
        max_diffs=args.max_diffs)
    if report:
        print(report, file=sys.stderr)
        sys.exit(1)
