#!/usr/bin/env python3

from __future__ import print_function

import argparse
import glob
import os
import site
import subprocess


def find_torch_ltc_site(site_paths):
    for site_path in site_paths:
        # If there is one named 'lazy_tensor_core', this is what we pick.
        path = os.path.join(site_path, 'lazy_tensor_core', 'lib')
        if os.path.isdir(path):
            return [site_path, path]
        dirs = glob.glob(os.path.join(site_path, 'lazy_tensor_core*'))
        # Get the most recent one.
        for xpath in sorted(dirs, key=os.path.getmtime):
            path = os.path.join(xpath, 'lib')
            if os.path.isdir(path):
                return [site_path, path]
            if os.path.isfile(os.path.join(xpath, 'libptltc.so')):
                return [site_path, xpath, os.path.join(xpath, 'lazy_tensor_core', 'lib')]
    raise RuntimeError('Unable to find lazy_tensor_core package in {}'.format(site_path))


def find_torch_site(site_paths):
    for site_path in site_paths:
        path = os.path.join(site_path, 'torch', 'lib')
        if os.path.isdir(path):
            return [path]
    raise RuntimeError('Unable to find torch package in {}'.format(site_path))


def list_rpaths(path):
    if subprocess.call(['patchelf', '--shrink-rpath', path]) != 0:
        raise RuntimeError('Failed to shrink RPATH folders: {}'.format(path))
    return subprocess.check_output(['patchelf', '--print-rpath',
                                    path]).decode('utf-8').strip('\n').split(':')


def set_rpaths(path, rpaths):
    if subprocess.call(['patchelf', '--set-rpath', ':'.join(rpaths), path]) != 0:
        raise RuntimeError('Failed to set RPATH folders {}: {}'.format(
            rpaths, path))


def fixup_binary(args):
    site_paths = site.getsitepackages()
    ltc_rpaths = find_torch_ltc_site(site_paths)
    torch_rpaths = find_torch_site(site_paths)
    rpaths = list_rpaths(args.binary)
    rpaths = ltc_rpaths + torch_rpaths + rpaths
    set_rpaths(args.binary, rpaths)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        'binary',
        type=str,
        metavar='BINARY',
        help='The path to the binary to be patched')
    args, files = arg_parser.parse_known_args()
    fixup_binary(args)
