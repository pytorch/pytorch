import os
from os.path import exists, abspath
import importlib
import pickle
from pyexpat import model
import sys
import logging
import argparse
import subprocess
import warnings

import torch
from functorch._src.benchmark_remat_utils import get_test_cases, get_skip_cases, check_remat_info_gm, profile_module, get_non_zero_mincut_memory_graphs


"""
Benchmark rematerialization algorithm on forward and backward graphs of torchbench models.
Need to have "../torch_bench_graphs" folder with dumped graphs

Example:

python functorch/benchmarks/benchmark_remat_torchbench.py --isolate --devices='cuda' 

This command will benchmark on graphs that have rematerialization opportunities.

If you only want to see how much memory will be reduced by the mincut optimization, WITHOUT actually benchmarking the performance, 
you can use the --info flag. With --info flag, all graphs are benchmarked. 
"""

current_dir = os.getcwd()
torch.backends.cuda.matmul.allow_tf32 = True

os.environ["KALDI_ROOT"] = "/tmp"  # avoids some spam
for torchbench_dir in (
    "../torch_bench_graphs",
):
    if exists(torchbench_dir):
        break
assert exists(torchbench_dir), "../torch_bench_graphs does not exist"
torchbench_dir = abspath("../")
os.chdir(torchbench_dir)
sys.path.append(torchbench_dir)
log = logging.getLogger(__name__)


test_cases = get_test_cases()
SKIP_CASES = get_skip_cases()
non_zero_mincut_memory_graphs = get_non_zero_mincut_memory_graphs()

test_cases = [ 'torch_bench_graphs/hf_T5/hf_T5_joint_8',
 'torch_bench_graphs/mobilenet_v3_large/mobilenet_v3_large_joint_0']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", "-d", action="append", help="cpu or cuda")
    parser.add_argument("--only", help="used by --isolate to run just one model")
    parser.add_argument(
        "--isolate", action="store_true", help="run each model in its own process"
    )

    parser.add_argument(
        "--info", action="store_true", help="only print out info without benchmarking"
    )

    args = parser.parse_args()
    args.devices = args.devices or ["cpu"]

    # nvfuser:
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(True)

    if args.only:
        # for device in args.devices:
        torch.manual_seed(1337)

        dir = args.only
        path = dir.split('/')
        model_name = path[-1]
        module_path = '.'.join(path)
        input_data_path = f'{dir}/{model_name}.input'
        if model_name in SKIP_CASES:
            return

        if (not args.info) and (model_name not in non_zero_mincut_memory_graphs): #
            return

        module = importlib.import_module(module_path)

        m = module.FxModule()

        inputs = []
        with (open(input_data_path, 'rb')) as f:
            
            inputs_meta = pickle.load(f)
            for meta in inputs_meta:
                if(len(meta)==4):
                    type, shape, stride, dtype= meta
                    device = 'cuda'
                else:
                    type, shape, stride, dtype, device = meta
                if dtype in {torch.int, torch.int32, torch.int64, torch.bool, torch.int, torch.uint8}:
                    input = torch.randint(0, 1, shape, dtype=dtype, device=device)
                else:
                    input = torch.rand(shape, dtype=dtype, device=device)
                inputs.append(input)
        m.to(device)

        if args.info:
            check_remat_info_gm(model_name, m, inputs)
        else:
            profile_module(model_name, m, inputs)

    elif args.isolate:
        if args.info:
            print("name, num_fusion_group, num_remat_group, memory_reduced, num_node_pairs", flush=True)
        else:
            print("name, eager_time, scripted_cuda_time, fused_cuda_time, remat_cuda_time, num_fusion_group, num_remat_group, memory_reduced, num_node_pairs_touching", flush=True)
        os.chdir(current_dir)
        for name in test_cases:
            try:
                subprocess.check_call([sys.executable] + sys.argv + [f"--only={name}"])
            except subprocess.SubprocessError:
                print(name,"ERROR")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main()
