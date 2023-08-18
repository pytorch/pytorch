import argparse
import subprocess
import os
from os import listdir
from os.path import join, isdir

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
    LOG_DIR = args.log_dir
    DATA_DIR = args.cache_dir
    DTYPE_LIST = args.dtype.split(",")
    MODE_LIST = args.mode.split(",")
    BENCHMARK_PY = args.benchmark
    MODEL_LIST_PATH = args.model_csv

    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    for DTYPE in DTYPE_LIST:
        for MODE in MODE_LIST:
            model_list_path = MODEL_LIST_PATH
            model_names = []
            with open(model_list_path, "r") as f:
                for line in f.readlines()[1:]:
                    line = line.split(",")[0]
                    model_names.append(line)
            print(model_names)

            for model_name in model_names:
                cache_dir = os.path.join(DATA_DIR, f"{DTYPE}_{MODE}_{model_name}")
                log_file = os.path.join(
                    LOG_DIR, f"{DTYPE}_{MODE}_{model_name}.kernels.log"
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
                                os.system(cmd)

                # run benchmark
                my_env = os.environ.copy()
                my_env["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
                my_env["TORCH_LOGS"] = "+inductor"
                my_env["TORCHINDUCTOR_BENCHMARK_KERNEL"] = "1"
                cmd = f"""python3 {BENCHMARK_PY} --{DTYPE} --performance --{MODE} --inductor -d cuda --only {model_name} > {log_file} 2>&1"""
                print(cmd)
                try:
                    pro = subprocess.Popen(
                        cmd, env=my_env, shell=True, preexec_fn=os.setsid
                    )
                    pro.wait(timeout=None)
                except subprocess.TimeoutExpired as exc:
                    print(exc)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
