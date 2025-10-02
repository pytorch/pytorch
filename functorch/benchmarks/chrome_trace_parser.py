#!/usr/bin/env python3
import argparse
import logging
import os

import pandas as pd

from torch._functorch.benchmark_utils import compute_utilization


log = logging.getLogger(__name__)

# process the chrome traces output by the pytorch profiler
# require the json input file's name to be in format {model_name}_chrome_trace_*.json
# the runtimes file should have format (model_name, runtime)


def get_model_name(filename):
    """
    Get model name from a file in format {model_name}_chrome_trace_*.json
    """
    _, tail = os.path.split(filename)
    modelname = tail[: tail.find("_chrome_trace")]
    return modelname


def get_total_length(run_times_df, modelname):
    return float(run_times_df[run_times_df["name"] == modelname]["runtime"])


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument(
        "--runtime", "-runf", help="file name of the runtime file", required=True
    )
    group.add_argument(
        "--filename",
        "-f",
        action="append",
        help="a filename of the json file to process",
    )
    group.add_argument("--folder", "-fd", help="a folder of the json files to process")
    args = parser.parse_args()

    if args.filename:
        filenames = args.filename
    elif args.folder:
        filenames = []
        directory = args.folder
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            if os.path.isfile(f) and f.endswith(".json"):
                filenames.append(f)
    else:
        print("Please provide a filename or a folder name")

    print("modelname, GPU Utilization, MM and Conv time")

    run_times_df = pd.read_csv(args.runtime)
    for filename in filenames:
        try:
            modelname = get_model_name(filename)
            total_length = get_total_length(run_times_df, modelname) * 1e6
            utilization, mm_conv_utilization = compute_utilization(
                filenames, total_length
            )
            print(f"{modelname}, {utilization}, {mm_conv_utilization}")
        except BaseException:  # noqa: B036
            log.exception("%s, ERROR", filename)
            print(f"{filename}, ERROR")


if __name__ == "__main__":
    main()
