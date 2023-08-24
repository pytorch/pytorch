import argparse
import os
import csv
import subprocess
from os import listdir
from os.path import isdir, join

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, default="./data_logs")
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument(
    "--benchmark",
    type=str,
    help="torchbench Python script path, e.g. benchmark/dynamo/huggingface.py",
    required=True,
)
parser.add_argument(
    "--model-csv",
    type=str,
    help="""
        the path to csv file to determine the models in the benchmark to collect kernels
        e.g. benchmarks/dynamo/ci_expected_accuracy/inductor_huggingface_training.csv
    """,
    required=True,
)
parser.add_argument("--dtype", type=str, help="delimited list input", default="amp")
parser.add_argument(
    "--mode", type=str, help="delimited list input", default="training,inference"
)


def main(args):
    log_dir = args.log_dir
    data_dir = args.data_dir
    dtype_list = args.dtype.split(",")
    mode_list = args.mode.split(",")
    benchmark_py = args.benchmark
    model_list_path = args.model_csv

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    for dtype in dtype_list:
        for mode in mode_list:
            model_list_path = model_list_path
            model_names = []
            with open(model_list_path) as f:
                csv_reader = csv.reader(f, delimiter=",")
                for row in csv_reader:
                    model_names.append(row[0])
            model_names = model_names[1:]
            print(model_names)

            for model_name in model_names:
                cache_dir = os.path.join(data_dir, f"{dtype}_{mode}_{model_name}")
                log_file = os.path.join(
                    log_dir, f"{dtype}_{mode}_{model_name}.kernels.log"
                )

                # if cache_dir exists, remove .pkl (raw data) and .best_config (best config) in it
                if isdir(cache_dir):
                    for kernel in sorted(listdir(cache_dir)):
                        kernel_path = join(cache_dir, kernel)
                        if not isdir(kernel_path):
                            continue
                        for file in listdir(kernel_path):
                            file_path = join(kernel_path, file)
                            if file.endswith((".pkl", ".best_config")):
                                cmd = "rm -rf " + file_path
                                print(cmd)
                                subprocess.call(cmd, shell=True)

                # run benchmark
                my_env = os.environ.copy()
                my_env["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
                my_env["TORCHINDUCTOR_DUMP_AUTOTUNER_DATA"] = "1"
                my_env["TORCH_LOGS"] = "+inductor"
                my_env["TORCHINDUCTOR_BENCHMARK_KERNEL"] = "1"
                cmd = (
                    f"""python3 {benchmark_py} --{dtype} --performance --{mode} --inductor -d cuda"""
                    f""" --only {model_name} > {log_file} 2>&1"""
                )
                print(cmd)
                subprocess.check_call(cmd, env=my_env, shell=True)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
