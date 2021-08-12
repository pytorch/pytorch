#!/usr/bin/env python3

from __future__ import print_function

import argparse
import concurrent.futures
import time
import lazy_tensor_core.utils.gcsfs as gs


def run_benchmark(args):
    gblob = gs.stat(args.gsfile)
    print('GCS file {} is {} bytes long'.format(args.gsfile, gblob.size))

    def thread_fn():
        ts = time.time()
        for n in range(0, args.test_count):
            with gs.open(args.gsfile) as fd:
                assert len(fd.read()) == gblob.size
        return time.time() - ts

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.num_threads) as executor:
        fobjs = []
        for _ in range(0, args.num_threads):
            fobjs.append(executor.submit(thread_fn))

        total_time = 0.0
        for future in concurrent.futures.as_completed(fobjs):
            total_time = max(total_time, future.result())

    bytes_sec = gblob.size * args.test_count * args.num_threads / total_time
    print('{:.2f}MB/s'.format(bytes_sec / (1024 * 1024)))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--test_count', type=int, default=5)
    arg_parser.add_argument('--num_threads', type=int, default=1)
    arg_parser.add_argument(
        'gsfile',
        type=str,
        metavar='GSFILE',
        help='The path to the GCS file to be used for benchmark')
    args = arg_parser.parse_args()
    run_benchmark(args)
