import argparse
import csv
import dataclasses
import gc
import itertools
import json
import os

from common import Experiment
from generate import test_configs
from model_zoo import models


DEFAULT_OUTPUT_FILE = "gpt_fast_benchmark.csv"


def output_csv(output_file, headers, row):
    if os.path.exists(output_file):
        with open(output_file) as fd:
            lines = list(csv.reader(fd)) or [[]]
            if headers and len(headers) > len(lines[0]):
                # if prior results failed the header might not be filled in yet
                lines[0] = headers
            else:
                headers = lines[0]
    else:
        lines = [headers]

    if output_file != DEFAULT_OUTPUT_FILE:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    lines.append([(f"{x:.6f}" if isinstance(x, float) else x) for x in row])
    with open(output_file, "w") as fd:
        writer = csv.writer(fd, lineterminator="\n")
        for line in lines:
            writer.writerow(list(line) + ["0"] * (len(headers) - len(line)))


def output_json(output_file, headers, row):
    """
    Write the result into JSON format, so that it can be uploaded to the benchmark database
    to be displayed on OSS dashboard. The JSON format is defined at
    https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database
    """
    mapping_headers = {headers[i]: v for i, v in enumerate(row)}
    record = {
        "benchmark": {
            "name": "PyTorch LLM benchmark",
            "mode": "inference",
            "dtype": mapping_headers["dtype"],
            "extra_info": {
                "device": mapping_headers["device"],
                "arch": mapping_headers["arch"],
            },
        },
        "model": {
            "name": mapping_headers["name"],
            "origins": ["pytorch"],
        },
        "metric": {
            "compilation_time": mapping_headers["compilation_time"],
            "tokens_per_second": [mapping_headers["tokens_per_second"]],
            "memory_bandwidth": mapping_headers["memory_bandwidth"],
        },
    }

    with open(f"{os.path.splitext(output_file)[0]}.json", "a") as f:
        print(json.dumps(record), file=f)


def main(args):
    results = []

    if args.only:
        if args.only not in models:
            raise ValueError(f"Unknown model: {args.only}")
        experiments = [(args.only, models[args.only])]
    else:
        experiments = [
            (model_name, benchmark_class)
            for model_name, benchmark_class in models.items()
        ]

    configs_to_test = [args.test_config] if args.test_config else list(test_configs)

    devices = [args.device] if args.device else ["cuda", "cpu"]

    results = []
    for model_name, benchmark_class in experiments:
        for test_config, device in itertools.product(configs_to_test, devices):
            print("Processing:", model_name, test_config, device)
            benchmark = benchmark_class(model_name, device, test_config)
            res = benchmark.run_inference()
            print("Results:", dataclasses.astuple(res))
            results.append(dataclasses.astuple(res))

            # Clean up the memory to avoid OOM
            del benchmark
            gc.collect()

    headers = [field.name for field in dataclasses.fields(Experiment)]

    for row in results:
        output_csv(args.output_file, headers, row)
        output_json(args.output_file, headers, row)

    print(f"Results saved to {args.output_file}")
    print(f"Results saved to {os.path.splitext(output_file)[0]}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help="Set the output CSV file to save the benchmark results",
    )
    parser.add_argument(
        "--only",
        help="Specify a model to run exclusively",
    )
    parser.add_argument(
        "--device", help="Specify the device to use", choices=["cuda", "cpu"]
    )
    parser.add_argument(
        "--test-config",
        help="Specify the test config to use",
        choices=test_configs,
    )
    args = parser.parse_args()

    main(args)
