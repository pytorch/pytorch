#!/usr/bin/env python3

"""
A wrapper over the benchmark infrastructure to generate commonly used commands,
parse results and generate csv/graphs.

The script works on manually written TABLE (see below). We can add more commands
in the future.

One example usage is
-> python benchmarks/runner.py --suites=torchbench --inference
This command will generate the commands for the default compilers (see DEFAULTS
below) for inference, run them and visualize the logs.

If you want to just print the commands, you could use the following command
-> python benchmarks/runner.py --print_run_commands --suites=torchbench --inference

Similarly, if you want to just visualize the already finished logs
-> python benchmarks/runner.py --visualize_logs --suites=torchbench --inference

If you want to test float16
-> python benchmarks/runner.py --suites=torchbench --inference --dtypes=float16

"""


import argparse
import dataclasses
import functools
import glob
import importlib
import io
import itertools
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime
from os.path import abspath, exists
from random import randint

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch

import torch._dynamo
from matplotlib import rcParams
from scipy.stats import gmean
from tabulate import tabulate

rcParams.update({"figure.autolayout": True})
plt.rc("axes", axisbelow=True)

DEFAULT_OUTPUT_DIR = "benchmark_logs"


log = logging.getLogger(__name__)

TABLE = {
    "training": {
        "ts_nnc": "--training --speedup-ts ",
        "ts_nvfuser": "--training --nvfuser --speedup-dynamo-ts ",
        "eager": "--training --backend=eager ",
        "aot_eager": "--training --backend=aot_eager ",
        "aot_cudagraphs": "--training --backend=aot_cudagraphs ",
        "aot_nvfuser": "--training --nvfuser --backend=aot_ts_nvfuser ",
        "nvprims_nvfuser": "--training --backend=nvprims_nvfuser ",
        "inductor": "--training --inductor ",
        "inductor_no_cudagraphs": "--training --inductor --disable-cudagraphs ",
    },
    "inference": {
        "ts_nnc": "--speedup-ts",
        "ts_nvfuser": "-n100 --speedup-ts --nvfuser",
        "trt": "-n100 --speedup-trt",
        "ts_nvfuser_cudagraphs": "--backend=cudagraphs_ts",
        "inductor": "-n50 --inductor",
    },
}

INFERENCE_COMPILERS = tuple(TABLE["inference"].keys())
TRAINING_COMPILERS = tuple(TABLE["training"].keys())

DEFAULTS = {
    "training": [
        "eager",
        "aot_eager",
        "aot_cudagraphs",
        "nvprims_nvfuser",
        "inductor",
        "inductor_no_cudagraphs",
    ],
    "inference": ["ts_nvfuser_cudagraphs", "inductor"],
    "flag_compilers": {
        "training": ["inductor", "inductor_no_cudagraphs"],
        "inference": ["inductor"],
    },
    "dtypes": [
        "float32",
    ],
    "suites": ["torchbench", "huggingface", "timm_models"],
    "devices": [
        "cuda",
    ],
    "quick": {
        "torchbench": '-k "resnet..$"',
        "huggingface": "-k Albert",
        "timm_models": ' -k "^resnet" -k "^inception"',
    },
}


DASHBOARD_DEFAULTS = {
    "dashboard_image_uploader": "/fsx/users/anijain/bin/imgur.sh",
    "dashboard_archive_path": "/data/home/anijain/cluster/cron_logs",
    "dashboard_gh_cli_path": "/data/home/anijain/miniconda/bin/gh",
}


def flag_speedup(x):
    return x < 0.95


def flag_compilation_latency(x):
    return x > 120


def flag_compression_ratio(x):
    return x < 0.9


FLAG_FNS = {
    "speedup": flag_speedup,
    "compilation_latency": flag_compilation_latency,
    "compression_ratio": flag_compression_ratio,
}


def percentage(part, whole, decimals=2):
    if whole == 0:
        return 0
    return round(100 * float(part) / float(whole), decimals)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", action="append", help="cpu or cuda")
    parser.add_argument("--dtypes", action="append", help="float16/float32/amp")
    parser.add_argument("--suites", action="append", help="huggingface/torchbench/timm")
    parser.add_argument(
        "--compilers",
        action="append",
        help=f"For --inference, options are {INFERENCE_COMPILERS}. For --training, options are {TRAINING_COMPILERS}",
    )

    parser.add_argument(
        "--flag-compilers",
        action="append",
        help="List of compilers to flag issues. Same format as --compilers.",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Just runs one model. Helps in debugging"
    )
    parser.add_argument(
        "--output-dir",
        help="Choose the output directory to save the logs",
        default=DEFAULT_OUTPUT_DIR,
    )

    # Choose either generation of commands, pretty parsing or e2e runs
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--print_run_commands",
        action="store_true",
        help="Generate commands and saves them to run.sh",
    )
    group.add_argument(
        "--visualize_logs",
        action="store_true",
        help="Pretty print the log files and draw graphs",
    )
    group.add_argument(
        "--run",
        action="store_true",
        default=True,
        help="Generate commands, run and parses the files",
    )

    parser.add_argument(
        "--log-operator-inputs",
        action="store_true",
        default=False,
        help="Log operator inputs",
    )

    parser.add_argument(
        "--extra-args", default="", help="Append commandline with these args"
    )

    # Choose either inference or training
    group_mode = parser.add_mutually_exclusive_group(required=True)
    group_mode.add_argument(
        "--inference", action="store_true", help="Only run inference related tasks"
    )
    group_mode.add_argument(
        "--training", action="store_true", help="Only run training related tasks"
    )

    parser.add_argument(
        "--update-dashboard",
        action="store_true",
        default=False,
        help="Updates to dashboard",
    )
    parser.add_argument(
        "--dashboard-image-uploader",
        default=DASHBOARD_DEFAULTS["dashboard_image_uploader"],
        help="Image uploader command",
    )
    parser.add_argument(
        "--dashboard-archive-path",
        default=DASHBOARD_DEFAULTS["dashboard_archive_path"],
        help="Archived directory path",
    )
    parser.add_argument(
        "--archive-name",
        help="Directory name under dashboard-archive-path to copy output-dir to. "
        "If not provided, a generated name is used.",
    )
    parser.add_argument(
        "--dashboard-gh-cli-path",
        default=DASHBOARD_DEFAULTS["dashboard_gh_cli_path"],
        help="Github CLI path",
    )
    args = parser.parse_args()
    return args


def get_mode(args):
    if args.inference:
        return "inference"
    return "training"


def get_skip_tests(suite):
    """
    Generate -x seperated string to skip the unusual setup training tests
    """
    skip_tests = set()
    original_dir = abspath(os.getcwd())
    module = importlib.import_module(suite)
    os.chdir(original_dir)

    if hasattr(module, "SKIP"):
        skip_tests.update(module.SKIP)
    if hasattr(module, "SKIP_TRAIN"):
        skip_tests.update(module.SKIP_TRAIN)

    skip_tests = map(lambda name: f"-x {name}", skip_tests)
    skip_str = " ".join(skip_tests)
    return skip_str


def generate_csv_name(args, dtype, suite, device, compiler, testing):
    mode = get_mode(args)
    return f"{compiler}_{suite}_{dtype}_{mode}_{device}_{testing}.csv"


def generate_commands(args, dtypes, suites, devices, compilers, output_dir):
    mode = get_mode(args)
    with open("run.sh", "w") as runfile:
        lines = []

        lines.append("# Setup the output directory")
        lines.append(f"rm -rf {output_dir}")
        lines.append(f"mkdir {output_dir}")
        lines.append("")

        for testing in ["performance", "accuracy"]:
            for iter in itertools.product(suites, devices, dtypes):
                suite, device, dtype = iter
                lines.append(
                    f"# Commands for {suite} for device={device}, dtype={dtype} for {mode} and for {testing} testing"
                )
                info = TABLE[mode]
                for compiler in compilers:
                    base_cmd = info[compiler]
                    output_filename = f"{output_dir}/{generate_csv_name(args, dtype, suite, device, compiler, testing)}"
                    cmd = f"python benchmarks/dynamo/{suite}.py --{testing} --{dtype} -d{device} --output={output_filename}"
                    cmd = f"{cmd} {base_cmd} {args.extra_args} --no-skip --dashboard"

                    skip_tests_str = get_skip_tests(suite)
                    cmd = f"{cmd} {skip_tests_str}"

                    if args.log_operator_inputs:
                        cmd = f"{cmd} --log-operator-inputs"

                    if args.quick:
                        filters = DEFAULTS["quick"][suite]
                        cmd = f"{cmd} {filters}"

                    if testing == "performance" and compiler in (
                        "inductor",
                        "inductor_no_cudagraphs",
                    ):
                        cmd = f"{cmd} --cold_start_latency"
                    lines.append(cmd)
                lines.append("")
        runfile.writelines([line + "\n" for line in lines])


def generate_dropdown_comment(title, body):
    str_io = io.StringIO()
    str_io.write(f"{title}\n")
    str_io.write("<details>\n")
    str_io.write("<summary>see more</summary>\n")
    str_io.write(f"{body}")
    str_io.write("\n")
    str_io.write("</details>\n\n")
    return str_io.getvalue()


def build_summary():
    import git

    out_io = io.StringIO()

    def print_commit_hash(path, name):
        if exists(path):
            repo = git.Repo(path, search_parent_directories=True)
            sha = repo.head.object.hexsha
            out_io.write(f"{name} commit: {sha}\n")
        else:
            out_io.write(f"{name} Absent\n")

    def env_var(name):
        out_io.write(f"{name} = {os.environ[name]}\n")

    out_io.write("## Commit hashes ##\n")
    print_commit_hash(".", "torch._dynamo")
    print_commit_hash("../pytorch", "pytorch")
    print_commit_hash("../functorch", "functorch")
    print_commit_hash("../torchbenchmark", "torchbench")

    out_io.write("\n")
    out_io.write("## TorchDynamo config flags ##\n")
    for key in dir(torch._dynamo.config):
        val = getattr(torch._dynamo.config, key)
        if not key.startswith("__") and isinstance(val, bool):
            out_io.write(f"torch._dynamo.config.{key} = {val}\n")

    out_io.write("\n")
    out_io.write("## Torch version ##\n")
    out_io.write(f"torch: {torch.__version__}\n")

    out_io.write("\n")
    out_io.write("## Environment variables ##\n")
    env_var("TORCH_CUDA_ARCH_LIST")
    env_var("CUDA_HOME")
    env_var("USE_LLVM")

    out_io.write("\n")
    out_io.write("## GPU details ##\n")
    out_io.write(f"CUDNN VERSION: {torch.backends.cudnn.version()}\n")
    out_io.write(f"Number CUDA Devices: {torch.cuda.device_count()}\n")
    out_io.write(f"Device Name: {torch.cuda.get_device_name(0)}\n")
    out_io.write(
        f"Device Memory [GB]: {torch.cuda.get_device_properties(0).total_memory/1e9}\n"
    )

    title = "## Build Summary"
    comment = generate_dropdown_comment(title, out_io.getvalue())
    with open(f"{output_dir}/gh_build_summary.txt", "w") as gh_fh:
        gh_fh.write(comment)


@functools.lru_cache(None)
def archive_data(archive_name):
    if archive_name is not None:
        prefix_match = re.search(r"\w+(?=_performance)", archive_name)
        if prefix_match is not None:
            prefix = prefix_match.group(0)
        else:
            prefix = ""
        day_match = re.search(r"day_(\d+)_", archive_name)
        if day_match is not None:
            day = day_match.group(1)
        else:
            day = "000"
    else:
        day = datetime.today().strftime("%j")
        prefix = datetime.today().strftime(f"day_{day}_%d_%m_%y")
    return day, prefix


@functools.lru_cache(None)
def default_archive_name(dtype):
    _, prefix = archive_data(None)
    return f"{prefix}_performance_{dtype}_{randint(100, 999)}"


def archive(src_dir, dest_dir_prefix, archive_name, dtype):
    if archive_name is None:
        archive_name = default_archive_name(dtype)
    # Copy the folder to archived location
    dest = os.path.join(dest_dir_prefix, archive_name)
    shutil.copytree(src_dir, dest, dirs_exist_ok=True)
    print(f"copied contents of {src_dir} to {dest}")


class Parser:
    def __init__(
        self, suites, devices, dtypes, compilers, flag_compilers, mode, output_dir
    ):
        self.suites = suites
        self.devices = devices
        self.dtypes = dtypes
        self.compilers = compilers
        self.flag_compilers = flag_compilers
        self.output_dir = output_dir
        self.mode = mode

    def has_header(self, output_filename):
        header_present = False
        with open(output_filename, "r") as f:
            line = f.readline()
            if "dev" in line:
                header_present = True
        return header_present


class ParsePerformanceLogs(Parser):
    def __init__(
        self, suites, devices, dtypes, compilers, flag_compilers, mode, output_dir
    ):
        super().__init__(
            suites, devices, dtypes, compilers, flag_compilers, mode, output_dir
        )
        self.parsed_frames = defaultdict(lambda: defaultdict(None))
        self.untouched_parsed_frames = defaultdict(lambda: defaultdict(None))
        self.metrics = [
            "speedup",
            "abs_latency",
            "compilation_latency",
            "compression_ratio",
        ]
        self.bottom_k = 50
        self.parse()

    def plot_graph(self, df, title):
        labels = df.columns.values.tolist()
        labels = labels[3:]
        df.plot(
            x="name",
            y=labels,
            kind="bar",
            width=0.65,
            title=title,
            ylabel="Speedup over eager",
            xlabel="",
            grid=True,
            figsize=(max(len(df.index) / 4, 5), 10),
            edgecolor="black",
        )
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{title}.png")

    def read_csv(self, output_filename):
        if self.has_header(output_filename):
            return pd.read_csv(output_filename)
        else:
            return pd.read_csv(
                output_filename,
                names=[
                    "dev",
                    "name",
                    "batch_size",
                    "speedup",
                    "abs_latency",
                    "compilation_latency",
                    "compression_ratio",
                ],
                header=None,
                engine="python",
            )

    def parse(self):
        self.extract_df("accuracy", "accuracy")
        for metric in self.metrics:
            self.extract_df(metric, "performance")

    def clean_batch_sizes(self, frames):
        # Clean up batch sizes when its 0
        if len(frames) == 1:
            return frames
        batch_sizes = frames[0]["batch_size"].to_list()
        for frame in frames[1:]:
            frame_batch_sizes = frame["batch_size"].to_list()
            for idx, (batch_a, batch_b) in enumerate(
                zip(batch_sizes, frame_batch_sizes)
            ):
                assert batch_a == batch_b or batch_a == 0 or batch_b == 0, print(
                    f"a={batch_a}, b={batch_b}"
                )
                batch_sizes[idx] = max(batch_a, batch_b)
        for frame in frames:
            frame["batch_size"] = batch_sizes
        return frames

    def extract_df(self, metric, testing):
        for iter in itertools.product(self.suites, self.devices, self.dtypes):
            suite, device, dtype = iter
            frames = []
            for compiler in self.compilers:
                output_filename = f"{self.output_dir}/{compiler}_{suite}_{dtype}_{self.mode}_{device}_{testing}.csv"
                df = self.read_csv(output_filename)
                if metric not in df:
                    df.insert(len(df.columns), metric, np.nan)
                df = df[["dev", "name", "batch_size", metric]]
                df.rename(columns={metric: compiler}, inplace=True)
                df["batch_size"] = df["batch_size"].astype(int)
                frames.append(df)

            # Merge the results
            frames = self.clean_batch_sizes(frames)
            if len(self.compilers) == 1:
                df = frames[0]
            else:
                # Merge data frames
                df = pd.merge(frames[0], frames[1], on=["dev", "name", "batch_size"])
                for idx in range(2, len(frames)):
                    df = pd.merge(df, frames[idx], on=["dev", "name", "batch_size"])

            df_copy = df.copy()
            df_copy = df_copy.sort_values(
                by=list(reversed(self.compilers)), ascending=False
            )
            if "inductor" in self.compilers:
                df_copy = df_copy.sort_values(by="inductor", ascending=False)
            self.untouched_parsed_frames[suite][metric] = df_copy

            if testing == "performance":
                df_accuracy = self.parsed_frames[suite]["accuracy"]
                perf_rows = []
                for model_name in df["name"]:
                    perf_row = df[df["name"] == model_name].copy()
                    acc_row = df_accuracy[df_accuracy["name"] == model_name]
                    for compiler in self.compilers:
                        if not perf_row.empty:
                            if acc_row.empty:
                                perf_row[compiler] = 0.0
                            elif acc_row[compiler].iloc[0] not in (
                                "pass",
                                "pass_due_to_skip",
                            ):
                                perf_row[compiler] = 0.0
                    perf_rows.append(perf_row)
                df = pd.concat(perf_rows)
            df = df.sort_values(by=list(reversed(self.compilers)), ascending=False)

            if "inductor" in self.compilers:
                df = df.sort_values(by="inductor", ascending=False)
            self.parsed_frames[suite][metric] = df

    def get_passing_entries(self, compiler, df):
        return df[compiler][df[compiler] > 0]

    def comp_time(self, compiler, df):
        df = self.get_passing_entries(compiler, df)
        # df = df.sort_values(by=compiler, ascending=False)[compiler][: self.bottom_k]
        if df.empty:
            return "0.0"

        return f"{df.mean():.2f}"

    def geomean(self, compiler, df):
        cleaned_df = self.get_passing_entries(compiler, df).clip(1)
        if cleaned_df.empty:
            return "0.0x"
        return f"{gmean(cleaned_df):.2f}x"

    def passrate(self, compiler, df):
        total = len(df.index)
        passing = df[df[compiler] > 0.0][compiler].count()
        perc = int(percentage(passing, total, decimals=0))
        return f"{perc}%, {passing}/{total}"

    def memory(self, compiler, df):
        df = self.get_passing_entries(compiler, df)
        df = df.fillna(0)
        df = df[df > 0]
        if df.empty:
            return "0.0x"
        return f"{df.mean():.2f}x"

    def exec_summary_df(self, fn, metric):
        """
        Generate a table with passrate and geomean perf
        """
        cols = {}
        cols["Compiler"] = self.compilers
        for suite in self.suites:
            df = self.parsed_frames[suite][metric]
            # speedups = [self.geomean(compiler, df) for compiler in self.compilers]
            speedups = [fn(compiler, df) for compiler in self.compilers]
            col = pd.Series(data=speedups, index=self.compilers)
            cols[suite] = col
        df = pd.DataFrame(cols)
        df = df.fillna(0)
        df.to_csv(os.path.join(self.output_dir, f"{fn.__name__}.csv"))
        return df

    def exec_summary_text(self, caption, fn, metric):
        df = self.exec_summary_df(fn, metric)
        tabform = tabulate(df, headers="keys", tablefmt="pretty", showindex="never")

        str_io = io.StringIO()
        str_io.write(f"{caption}")
        str_io.write("~~~\n")
        str_io.write(f"{tabform}\n")
        str_io.write("~~~\n")
        return str_io.getvalue()

    def generate_executive_summary(self):
        description = (
            "We evaluate different backends "
            "across three benchmark suites - torchbench, huggingface and timm. We run "
            "these experiments on A100 GPUs. Each experiment runs one iteration of forward "
            "and backward pass. For accuracy, we check the numerical correctness of forward "
            "pass outputs and gradients by comparing with native pytorch. We measure speedup "
            "by normalizing against the performance of native pytorch. We report mean "
            "compilation latency numbers and peak memory footprint reduction ratio. \n\n"
            "Caveats\n"
            "1) Batch size has been reduced to workaround OOM errors. Work is in progress to "
            "reduce peak memory footprint.\n"
            "2) Experiments do not cover dynamic shapes.\n"
            "3) Experimental setup does not have optimizer.\n\n"
        )

        comment = generate_dropdown_comment("", description)
        str_io = io.StringIO()
        str_io.write("\n")
        str_io.write("## Executive Summary ##\n")
        str_io.write(comment)

        speedup_caption = "Geometric mean speedup \n"
        speedup_summary = self.exec_summary_text(
            speedup_caption, self.geomean, "speedup"
        )

        passrate_caption = "Passrate\n"
        passrate_summary = self.exec_summary_text(
            passrate_caption, self.passrate, "speedup"
        )

        comp_time_caption = "Mean compilation time (seconds)\n"
        comp_time_summary = self.exec_summary_text(
            comp_time_caption, self.comp_time, "compilation_latency"
        )

        peak_memory_caption = (
            "Peak memory footprint compression ratio (higher is better)\n"
        )
        peak_memory_summary = self.exec_summary_text(
            peak_memory_caption, self.memory, "compression_ratio"
        )

        str_io.write(
            "To measure performance, compilation latency and memory footprint reduction, "
            "we remove the models that fail accuracy checks.\n\n"
        )
        str_io.write(passrate_summary)
        str_io.write(speedup_summary)
        str_io.write(comp_time_summary)
        str_io.write(peak_memory_summary)
        self.executive_summary = str_io.getvalue()

    def flag_bad_entries(self, suite, metric, flag_fn):
        df = self.untouched_parsed_frames[suite][metric]
        df = df.drop("dev", axis=1)
        df = df.rename(columns={"batch_size": "bs"})
        # apply flag_fn elementwise to flag_compilers columns,
        # if one element fails, the entire row is flagged
        flag = np.logical_or.reduce(
            df[self.flag_compilers].applymap(flag_fn),
            axis=1,
        )
        df = df[flag]
        df = df.assign(suite=suite)
        return df.reindex(columns=["suite", "name"] + self.flag_compilers)

    def get_metric_title(self, metric):
        if metric == "speedup":
            return "Performance speedup"
        elif metric == "accuracy":
            return "Accuracy"
        elif metric == "compilation_latency":
            return "Compilation latency (sec)"
        elif metric == "compression_ratio":
            return "Peak Memory Compression Ratio"
        elif metric == "abs_latency":
            return "Absolute latency (ms)"
        raise RuntimeError("unknown metric")

    def generate_warnings(self):
        title = "## Warnings ##"
        body = (
            "We flag models where:\n\n"
            " - speedup < 0.95x\n"
            " - compilation latency > 120 sec.\n"
            " - compression ratio < 0.9\n\n"
        )
        for metric in [
            "speedup",
            "compilation_latency",
            "compression_ratio",
        ]:
            dfs = []
            for suite in self.suites:
                dfs.append(self.flag_bad_entries(suite, metric, FLAG_FNS[metric]))
            df = pd.concat(dfs, axis=0)
            if df.empty:
                continue
            tabform = tabulate(df, headers="keys", tablefmt="pretty", showindex="never")
            str_io = io.StringIO()
            str_io.write("\n")
            str_io.write(self.get_metric_title(metric) + " warnings\n")
            str_io.write("~~~\n")
            str_io.write(f"{tabform}\n")
            str_io.write("~~~\n")
            body += str_io.getvalue()

        comment = generate_dropdown_comment(title, body)
        return comment

    def prepare_message(self, suite):
        title = f"## {suite} suite with {self.dtypes[0]} precision ##"
        body = ""
        for metric in [
            "speedup",
            "accuracy",
            "compilation_latency",
            "compression_ratio",
            "abs_latency",
        ]:
            df = self.untouched_parsed_frames[suite][metric]
            df = df.drop("dev", axis=1)
            df = df.rename(columns={"batch_size": "bs"})
            tabform = tabulate(df, headers="keys", tablefmt="pretty", showindex="never")
            str_io = io.StringIO()
            str_io.write("\n")
            str_io.write(self.get_metric_title(metric) + "\n")
            str_io.write("~~~\n")
            str_io.write(f"{tabform}\n")
            str_io.write("~~~\n")
            body += str_io.getvalue()

        comment = generate_dropdown_comment(title, body)
        return comment

    def gen_summary_files(self):
        self.generate_executive_summary()
        for suite in self.suites:
            self.plot_graph(
                self.untouched_parsed_frames[suite]["speedup"],
                f"{suite}_{self.dtypes[0]}",
            )

        with open(f"{self.output_dir}/gh_title.txt", "w") as gh_fh:
            str_io = io.StringIO()
            str_io.write("\n")
            str_io.write(f"# Performance Dashboard for {self.dtypes[0]} precision ##\n")
            str_io.write("\n")
            gh_fh.write(str_io.getvalue())

        with open(f"{self.output_dir}/gh_executive_summary.txt", "w") as gh_fh:
            gh_fh.write(self.executive_summary)
        print(self.executive_summary)

        with open(f"{self.output_dir}/gh_warnings.txt", "w") as gh_fh:
            warnings_body = self.generate_warnings()
            gh_fh.write(warnings_body)
            print(warnings_body)

        str_io = io.StringIO()
        for suite in self.suites:
            str_io.write(self.prepare_message(suite))
        str_io.write("\n")
        print(str_io.getvalue())
        with open(f"{self.output_dir}/gh_{self.mode}.txt", "w") as gh_fh:
            gh_fh.write(str_io.getvalue())


def parse_logs(args, dtypes, suites, devices, compilers, flag_compilers, output_dir):
    mode = get_mode(args)
    build_summary()

    parser_class = ParsePerformanceLogs
    parser = parser_class(
        suites, devices, dtypes, compilers, flag_compilers, mode, output_dir
    )
    parser.gen_summary_files()
    return


@dataclasses.dataclass
class LogInfo:
    # Day of the year this log was generated
    day: str

    # Directory path where all logs are present
    dir_path: str


def get_date(log_info):
    return datetime.strptime(f"{log_info.day}", "%j").strftime("%m-%d")


class AccuracyRegressionTracker:
    """
    Compares the most recent 2 accuracy benchmarks to find previously
    passing models that now fail.
    """

    def __init__(self, args):
        self.args = args
        self.lookup_file = os.path.join(self.args.dashboard_archive_path, "lookup.csv")
        assert os.path.exists(self.lookup_file)

    def find_last_2(self, suite, device, dtype, compiler):
        df = pd.read_csv(self.lookup_file, names=("day", "mode", "prec", "path"))
        df = df[df["mode"] == "performance"]
        df = df[df["prec"] == dtype]
        df = df[::-1]
        parsers = []
        for path in df["path"]:
            output_dir = os.path.join(self.args.dashboard_archive_path, path)
            if os.path.exists(
                os.path.join(
                    output_dir,
                    generate_csv_name(
                        self.args, dtype, suite, device, compiler, "accuracy"
                    ),
                )
            ):
                parsers.append(
                    ParsePerformanceLogs(
                        [suite],
                        [device],
                        [dtype],
                        [compiler],
                        [compiler],
                        get_mode(self.args),
                        output_dir,
                    )
                )
            if len(parsers) >= 2:
                return parsers
        return None

    def generate_comment(self):
        title = "## Accuracy Regressions ##\n"
        body = (
            "For each relevant compiler, we compare the most recent 2 reports "
            "(that run actually the compiler) to find models where previously "
            "successful accuracy tests now fail.\n\n"
        )
        dtype = self.args.dtypes[0]
        device = self.args.devices[0]
        regressions_present = False
        for suite in self.args.suites:
            dfs = []
            for compiler in self.args.flag_compilers:
                last2 = self.find_last_2(suite, device, dtype, compiler)
                if last2 is None:
                    continue

                df_cur, df_prev = [
                    last2[i].untouched_parsed_frames[suite]["accuracy"] for i in (0, 1)
                ]
                df_merge = df_cur.merge(df_prev, on="name", suffixes=("_cur", "_prev"))
                flag = np.logical_and(
                    df_merge[compiler + "_prev"].apply(lambda x: "pass" in x),
                    df_merge[compiler + "_cur"].apply(lambda x: "pass" not in x),
                )
                df_bad = df_merge[flag]
                dfs.append(
                    pd.DataFrame(
                        data={
                            "compiler": compiler,
                            "name": df_bad["name"],
                            "prev_status": df_bad[compiler + "_prev"],
                            "cur_status": df_bad[compiler + "_cur"],
                        }
                    )
                )

            if not dfs:
                continue
            df = pd.concat(dfs, axis=0)
            if df.empty:
                continue
            regressions_present = True
            tabform = tabulate(df, headers="keys", tablefmt="pretty", showindex="never")
            str_io = io.StringIO()
            str_io.write("\n")
            str_io.write(f"Accuracy regressions for {suite}\n")
            str_io.write("~~~\n")
            str_io.write(f"{tabform}\n")
            str_io.write("~~~\n")
            body += str_io.getvalue()

        if not regressions_present:
            body += "No accuracy regressions found.\n"

        comment = generate_dropdown_comment(title, body)

        with open(f"{self.args.output_dir}/gh_accuracy_regression.txt", "w") as gh_fh:
            gh_fh.write(comment)
            print(comment)


class RegressionTracker:
    """
    Plots progress of different metrics over time to detect regressions.
    """

    def __init__(self, args):
        self.args = args
        self.suites = self.args.suites
        self.lookup_file = os.path.join(self.args.dashboard_archive_path, "lookup.csv")
        assert os.path.exists(self.lookup_file)
        self.k = 10

    def find_last_k(self):
        """
        Find the last k pairs of (day number, log_path)
        """
        dtype = self.args.dtypes[0]
        df = pd.read_csv(self.lookup_file, names=("day", "mode", "prec", "path"))
        df = df[df["mode"] == "performance"]
        df = df[df["prec"] == dtype]
        log_infos = []
        for day, path in zip(df["day"], df["path"]):
            log_infos.append(LogInfo(day, path))

        assert len(log_infos) >= self.k
        log_infos = log_infos[len(log_infos) - self.k :]
        return log_infos

    def generate_comment(self):
        title = "## Metrics over time ##\n"
        str_io = io.StringIO()
        for name in glob.glob(self.args.output_dir + "/*over_time.png"):
            output = (
                subprocess.check_output([self.args.dashboard_image_uploader, name])
                .decode("ascii")
                .rstrip()
            )
            str_io.write(f"\n{name} : ![]({output})\n")
        comment = generate_dropdown_comment(title, str_io.getvalue())

        with open(f"{self.args.output_dir}/gh_regression.txt", "w") as gh_fh:
            gh_fh.write(comment)

    def diff(self):
        log_infos = self.find_last_k()

        for metric in ["geomean", "passrate"]:
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
            for idx, suite in enumerate(self.suites):
                dfs = []
                for log_info in log_infos:
                    dir_path = os.path.join(
                        self.args.dashboard_archive_path, log_info.dir_path
                    )
                    assert os.path.exists(dir_path)
                    gmean_filename = os.path.join(dir_path, f"{metric}.csv")
                    if not os.path.exists(gmean_filename):
                        continue
                    df = pd.read_csv(gmean_filename)
                    if suite not in df:
                        continue
                    if metric == "geomean":
                        df[suite] = df[suite].str.replace("x", "").astype(float)
                    elif metric == "passrate":
                        df[suite] = df[suite].str.split("%").str[0].astype(float)
                    df.insert(0, "day", get_date(log_info))
                    df = df.pivot(index="day", columns="Compiler", values=suite)

                    # Interim stage when both inductor_cudagraphs and inductor exist
                    df = df.rename(columns={"inductor_cudagraphs": "inductor"})
                    for col_name in df.columns:
                        if col_name not in self.args.compilers:
                            df = df.drop(columns=[col_name])
                    dfs.append(df)

                df = pd.concat(dfs)
                df = df.interpolate(method="linear")
                ax = df.plot(
                    ax=axes[idx],
                    kind="line",
                    ylabel=metric,
                    xlabel="Date",
                    grid=True,
                    ylim=0 if metric == "passrate" else 0.8,
                    title=suite,
                    style=".-",
                    legend=False,
                )
                ax.legend(loc="lower right", ncol=2)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{metric}_over_time.png"))

        self.generate_comment()


class DashboardUpdater:
    """
    Aggregates the information and makes a comment to Performance Dashboard.
    https://github.com/pytorch/torchdynamo/issues/681
    """

    def __init__(self, args):
        self.args = args
        self.output_dir = args.output_dir
        self.lookup_file = os.path.join(self.args.dashboard_archive_path, "lookup.csv")
        assert os.path.exists(self.lookup_file)

    def archive(self):
        dtype = self.args.dtypes[0]
        # Copy the folder to archived location
        archive(
            self.output_dir,
            self.args.dashboard_archive_path,
            self.args.archive_name,
            dtype,
        )
        day, _ = archive_data(self.args.archive_name)
        target_dir = (
            default_archive_name(dtype)
            if self.args.archive_name is None
            else self.args.archive_name
        )

        # Update lookup csv the folder to arhived logs
        subprocess.check_call(
            f'echo "{day},performance,{dtype},{target_dir}" >> {self.lookup_file}',
            shell=True,
        )

    def upload_graphs(self):
        title = "## Performance graphs ##\n"
        str_io = io.StringIO()
        for name in glob.glob(self.output_dir + "/*png"):
            if "over_time" not in name:
                output = (
                    subprocess.check_output([self.args.dashboard_image_uploader, name])
                    .decode("ascii")
                    .rstrip()
                )
                str_io.write(f"\n{name} : ![]({output})\n")
        comment = generate_dropdown_comment(title, str_io.getvalue())

        with open(f"{self.output_dir}/gh_graphs.txt", "w") as gh_fh:
            gh_fh.write(comment)

    def gen_comment(self):
        files = [
            "gh_title.txt",
            "gh_executive_summary.txt",
            "gh_warnings.txt",
            "gh_regression.txt",
            "gh_accuracy_regression.txt",
            "gh_training.txt",
            "gh_graphs.txt",
        ]
        all_lines = []
        for f in files:
            try:
                with open(os.path.join(self.output_dir, f), "r") as fh:
                    all_lines.extend(fh.readlines())
            except FileNotFoundError:
                pass

        return "\n".join([x.rstrip() for x in all_lines])

    def comment_on_gh(self, comment):
        """
        Send a commment to dashboard
        """
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(comment)
            filename = f.name

        subprocess.check_call(
            [
                self.args.dashboard_gh_cli_path,
                "issue",
                "comment",
                "--repo=https://github.com/pytorch/torchdynamo.git",
                "681",
                "-F",
                filename,
            ]
        )

        os.remove(filename)

    def update(self):
        self.upload_graphs()
        AccuracyRegressionTracker(self.args).generate_comment()
        try:
            RegressionTracker(self.args).diff()
        except Exception as e:
            logging.exception(e)
            with open(f"{self.args.output_dir}/gh_regression.txt", "w") as gh_fh:
                gh_fh.write("")

        comment = self.gen_comment()
        self.comment_on_gh(comment)

        self.archive()


if __name__ == "__main__":
    args = parse_args()

    def extract(key):
        return DEFAULTS[key] if getattr(args, key, None) is None else getattr(args, key)

    dtypes = extract("dtypes")
    suites = extract("suites")
    devices = extract("devices")

    if args.inference:
        compilers = DEFAULTS["inference"] if args.compilers is None else args.compilers
        flag_compilers = (
            DEFAULTS["flag_compilers"]["inference"]
            if args.flag_compilers is None
            else args.flag_compilers
        )
    else:
        assert args.training
        compilers = DEFAULTS["training"] if args.compilers is None else args.compilers
        flag_compilers = (
            DEFAULTS["flag_compilers"]["training"]
            if args.flag_compilers is None
            else args.flag_compilers
        )

    output_dir = args.output_dir
    args.compilers = compilers
    args.devices = devices
    args.dtypes = dtypes
    args.flag_compilers = flag_compilers
    args.suites = suites

    if args.print_run_commands:
        generate_commands(args, dtypes, suites, devices, compilers, output_dir)
    elif args.visualize_logs:
        parse_logs(args, dtypes, suites, devices, compilers, flag_compilers, output_dir)
    elif args.run:
        generate_commands(args, dtypes, suites, devices, compilers, output_dir)
        # TODO - Do we need to worry about segfaults
        try:
            os.system("bash run.sh")
        except Exception as e:
            print(
                "Running commands failed. Please run manually (bash run.sh) and inspect the errors."
            )
            raise e
        if not args.log_operator_inputs:
            archive(
                output_dir, args.dashboard_archive_path, args.archive_name, dtypes[0]
            )
            parse_logs(
                args, dtypes, suites, devices, compilers, flag_compilers, output_dir
            )

    if args.update_dashboard:
        DashboardUpdater(args).update()
