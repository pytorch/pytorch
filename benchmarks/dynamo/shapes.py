from torch._inductor.select_algorithm import add_feedback_saver, clear_feedback_saver
import torch
from microbenchmarks.operator_inp_utils import OperatorInputsLoader
from torch.utils._ordered_set import OrderedSet
aten = torch.ops.aten
loader = OperatorInputsLoader.get_huggingface_loader()
from triton.testing import do_bench
import csv

from torch._inductor import config
torch.set_grad_enabled(False)
config.fx_graph_cache = False
config.force_disable_caches = True

def zip_dicts(dict1, dict2, d1_default, d2_default):
    """
    Zip two dictionaries together, replacing missing keys with default values.

    Args:
        dict1 (dict): The first dictionary.
        dict2 (dict): The second dictionary.
        d1_default (Any): the default value for the first dictionary
        d2_default (Any): the default value for the second dictionary

    Yields:
        tuple: A tuple containing the key, the value from dict1 (or d1_default if missing),
               and the value from dict2 (or d2_default if missing).
    """
    # Find the union of all keys
    all_keys = OrderedSet(dict1.keys()) | OrderedSet(dict2.keys())

    # Iterate over all keys
    for key in all_keys:
        # Get the values from both dictionaries, or default if missing
        value1 = dict1.get(key)
        value2 = dict2.get(key)

        yield (
            key,
            value1 if value1 is not None else d1_default,
            value2 if value2 is not None else d2_default,
        )


def compare_op():
    for op in [aten.mm.default, aten.addmm.default]:
        for dtype in [torch.bfloat16, torch.float16]:
            with open(f"{op}_{dtype}_benchmark_results.csv", "w", newline='') as file2:
                file2.write("M,K,N,BLOCK_M,BLOCK_K,BLOCK_N,NUM_STAGES,NUM_WARPS,GROUP_M,do_bench_time\n")

            with open(f"new_old_config_compare_{op}_{dtype}.csv", 'w', newline='') as file:
                def feedback_saver(timings, name, input_nodes, choices, profiled_time):
                    with open(f"{op}_{dtype}_benchmark_results.csv", "a", newline='') as file2:
                        if name == "addmm":
                            M, K, N = input_nodes[1].layout.size[0], input_nodes[1].layout.size[1], input_nodes[2].layout.size[1]
                        elif name == "mm":
                            M, K, N = input_nodes[0].layout.size[0], input_nodes[0].layout.size[1], input_nodes[1].layout.size[1]
                        else:
                            raise Exception(f"Unknown op {name}")
                        
                        file2.write("--------------------\n")
                        for choice, db_time in timings.items():
                            if not isinstance(choice, torch._inductor.select_algorithm.TritonTemplateCaller):
                                continue
                            BLOCK_M, BLOCK_K, BLOCK_N = tuple(map(int, choice.log_info['tile_shape'].strip('()').split(',')))
                            line = ",".join(map(str, [M, K, N, BLOCK_M, BLOCK_K, BLOCK_N, choice.log_info['num_stages'], choice.log_info['num_warps'], choice.log_info['GROUP_M'], db_time]))
                            file2.write(line + "\n")
                            file2.flush()
                add_feedback_saver(feedback_saver)
                # writer = csv.writer(file)
                # writer.writerow(['M', 'K', 'N', 'Old_Time', 'New_Time'])
                for i, (args, kwargs) in enumerate(loader.get_inputs_for_operator(op, dtype=dtype, device="cuda")):
                    torch._dynamo.reset()

                    try:
                        inp_t = args[1]
                        weight_t = args[2]
                    except:
                        inp_t = args[0]
                        weight_t = args[1]


                    if len(inp_t.shape) != 2:
                        continue

                    # dont know why we have these
                    if inp_t.numel() == 0:
                        continue

                    print(f"{inp_t.shape[0]}_{inp_t.shape[1]}_{weight_t.shape[1]}")
                    speeds = []
                    M, K, N  = inp_t.shape[0], inp_t.shape[1], weight_t.shape[1]

                    for benchmarking_space in ["SAME", 1]:
                        with open(f"{op}_{dtype}_benchmark_results.csv", "a") as file2:
                            if benchmarking_space == "SAME":
                                file2.write("SAME\n")
                            else:
                                file2.write("Top 1\n")
                        torch._dynamo.reset()

                        context = config.patch({
                            "fx_graph_cache":  False,
                            "force_disable_caches": True,
                            "max_autotune_gemm_backends": "TRITON",
                            # TODO
                            #"max_autotune_gemm_search_space": "EXHAUSTIVE",
                            "max_autotune_gemm_search_space": "DEFAULT",
                            "matmul_gemm_autotune_benchmark_space": benchmarking_space
                        })

                        with context:
                            #in1 = torch.zeros((M, K)).cuda().to(dtype=dtype)
                            in2 = torch.zeros((K, N)).cuda().to(dtype=dtype)
                            if op == aten.addmm.default:
                                in3 = torch.zeros((M, N)).cuda().to(dtype=dtype)
                                def fn(inp):
                                    return inp @ in2 + in3
                            else:
                                def fn(inp):
                                    return inp @ in2

                            mod = torch.compile(fn, mode="max-autotune-no-cudagraphs", fullgraph=True, dynamic=False)
                            # run a few times to make sure it happens
                            mod(inp_t)
                            mod(inp_t)
                            # speeds.append(do_bench(lambda: mod(inp_t)))

                    # writer.writerow([M, K, N, speeds[0], speeds[1]])
                    file.flush()
                clear_feedback_saver()

# compare_op("new_old_config_compare_addmm_float16.csv", aten.addmm.default, dtype=torch.float16)
# compare_op("new_old_config_compare_addmm_bfloat16.csv", aten.addmm.default, dtype=torch.bfloat16)
# compare_op("new_old_config_compare_addmm_float32.csv", aten.addmm.default, dtype=torch.float32)
compare_op()
