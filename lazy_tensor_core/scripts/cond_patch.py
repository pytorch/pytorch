#!/usr/bin/env python3

from __future__ import print_function

import argparse
import glob
import os
import re
import subprocess
import sys


def get_log(repo_folder, depth):
    return subprocess.check_output(
        ['git', '-C', repo_folder, 'log', '-{}'.format(depth)]).decode('utf-8')


def is_applied(log, revno):
    revrx = 'Pull Request resolved: .*[/#]{}'.format(revno)
    return re.search(revrx, log)


def select_patches(patch_folder, repo_folder, depth):
    log = get_log(repo_folder, depth)
    files = sorted(glob.glob(os.path.join(patch_folder, '*.diff')))
    selected = []
    for ppath in files:
        revno = os.path.splitext(os.path.basename(ppath))[0]
        # Patches which are not all digits (PR numbers) are always applied.
        if not re.match(r'\d+$', revno) or not is_applied(log, revno):
            selected.append(ppath)
    return selected


def apply_patch(ppath, repo_folder, level):
    return subprocess.call([
        'patch', '-d', repo_folder, '-p{}'.format(level), '-i', ppath, '-E', '-l',
        '-r', '-', '-s', '--no-backup-if-mismatch'
    ])


def patch_repo(args):
    patches = select_patches(
        os.path.normpath(args.patch_folder), os.path.normpath(args.repo_folder),
        args.log_depth)
    for ppath in patches:
        print('Applying patch file: {}'.format(ppath), file=sys.stderr)
        if apply_patch(ppath, os.path.normpath(args.repo_folder), args.level):
            raise RuntimeError('Failed to apply patch: {}'.format(ppath))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--level', type=int, default=1)
    arg_parser.add_argument('--log_depth', type=int, default=1000)
    arg_parser.add_argument(
        'patch_folder',
        type=str,
        metavar='PATCH_FOLDER',
        help='The path to the folder containing the patches')
    arg_parser.add_argument(
        'repo_folder',
        type=str,
        metavar='REPO_FOLDER',
        help='The path to the root folder of the repo to be patched')
    args, files = arg_parser.parse_known_args()
    patch_repo(args)
