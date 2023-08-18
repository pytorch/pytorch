import argparse
import subprocess, os, signal
from os import listdir
from os.path import join, isdir

parser = argparse.ArgumentParser()
parser.add_argument("--kernel_dir", type=str, default="./data_hf")
parser.add_argument("--radius", type=int, default=0)
parser.add_argument("--timeout", type=int, default=90)


def main(args):
    KERNEL_DIR = args.kernel_dir

    seen_kernels = set()

    for model in sorted(listdir(KERNEL_DIR)):
        model_path = join(KERNEL_DIR, model)
        if not isdir(model_path):
            continue

        for kernel in sorted(listdir(model_path)):
            kernel_path = join(model_path, kernel)
            if not isdir(kernel_path):
                continue

            # remove best config file
            for py in listdir(kernel_path):
                py_path = join(kernel_path, py)
                if py.endswith((".best_config")):
                    cmd = "rm -rf " + py_path
                    print(cmd)
                    os.system(cmd)

            # run kernel
            for py in listdir(kernel_path):
                py_path = join(kernel_path, py)
                if not py.endswith(".py"):
                    continue
                kernel_name = py[:-3]

                # skip graph python file
                with open(py_path, "r") as file:
                    content = file.read()
                    if "AsyncCompile()" in content:
                        print("Skip " + py_path + " GRAPH")
                        continue

                # skip seen kernels
                if kernel_name in seen_kernels:
                    print("Skip " + py_path + " <<<<<< " + kernel_name + " seen before")
                    continue

                cache_dir = kernel_path
                log_path = join(kernel_path, kernel_name + ".log")
                all_config_path = join(kernel_path, kernel_name + ".all_config")

                seen_kernels.add(kernel_name)
                if os.path.exists(log_path) and os.path.exists(all_config_path):
                    # already benchmarked
                    continue
                assert not os.path.exists(log_path) and not os.path.exists(
                    all_config_path
                )

                my_env = os.environ.copy()
                my_env["TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE"] = "1"
                if args.radius > 0:
                    my_env["TORCHINDUCTOR_COORDINATE_DESCENT_TUNING"] = "1"
                    my_env["TORCHINDUCTOR_COORDINATE_DESCENT_RADIUS"] = str(args.radius)
                    my_env[
                        "TORCHINDUCTOR_COORDINATE_DESCENT_CHECK_ALL_DIRECTIONS"
                    ] = "1"
                my_env["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
                my_env["TORCH_LOGS"] = "+inductor"
                my_env["TORCHINDUCTOR_BENCHMARK_KERNEL"] = "1"
                cmd = f"""python3 {py_path} > {log_path} 2>&1"""
                print(cmd)
                try:
                    pro = subprocess.Popen(
                        cmd, env=my_env, shell=True, preexec_fn=os.setsid
                    )
                    pro.wait(timeout=args.time_out)
                except subprocess.TimeoutExpired as exc:
                    print(exc)
                    os.killpg(os.getpgid(pro.pid), signal.SIGTERM)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
