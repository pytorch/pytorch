
from torch._inductor.select_algorithm import add_feedback_saver
import torch
from microbenchmarks.operator_inp_utils import OperatorInputsLoader
from torch.utils._ordered_set import OrderedSet
aten = torch.ops.aten
loader = OperatorInputsLoader.get_huggingface_loader()
from triton.testing import do_bench
from torch._inductor import utils
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


def compare_op(filename, op, dtype=torch.bfloat16, device="cuda"):
    with open(filename, 'w', newline='') as file, open(f"{op}_{dtype}_benchmark_results.csv", "a", newline='') as file2:
        file2.write("BLOCK_M,BLOCK_N,BLOCK_K,NUM_STAGES,NUM_WARPS,GROUP_M,do_bench_time,profile_time")
        def feedback_saver(timings, name, input_nodes, choices, profiled_time):
            pt = profiled_time()
            if name == "addmm":
                M, K, N = input_nodes[1].layout.size[0], input_nodes[1].layout.size[1], input_nodes[2].layout.size[1]
            elif name == "mm":
                M, K, N = input_nodes[0].layout.size[0], input_nodes[0].layout.size[1], input_nodes[1].layout.size[1]
            else:
                raise Exception(f"Unknown op {name}")
            
            file2.write("--------------------\n")
            file2.write(f"{M},{K},{N}\n")
            for choice, db_time, profile_time in zip_dicts(timings, pt, None, None):
                if not isinstance(choice, torch._inductor.select_algorithm.TritonTemplateCaller):
                    continue
                BLOCK_M, BLOCK_N, BLOCK_K = tuple(map(int, choice.log_info['tile_shape'].strip('()').split(',')))
                line = ",".join(map(str, [BLOCK_M, BLOCK_N, BLOCK_K, choice.log_info['num_stages'], choice.log_info['num_warps'], choice.log_info['GROUP_M'], db_time, profile_time]))
                file2.write(line + "\n")
                file2.flush()
        add_feedback_saver(feedback_saver)
        writer = csv.writer(file)
        writer.writerow(['Inp_Shapes', 'Old_Time', 'New_Time'])
        for i, (args, kwargs) in enumerate(loader.get_inputs_for_operator(op, dtype=torch.bfloat16, device="cuda")):
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

            print([f"{inp_t.shape[0]}_{inp_t.shape[1]}_{weight_t.shape[1]}"])
            speeds = []
            M, K, N  = inp_t.shape[0], inp_t.shape[1], weight_t.shape[1]

            for new_configs in [False, True]:
                torch._dynamo.reset()

                context = config.patch({
                    "fx_graph_cache":  False,
                    "force_disable_caches": True,
                    "new_configs": new_configs,
                })

                with context:
                    mod = torch.nn.Linear(
                        in_features=inp_t.shape[1],
                        out_features=weight_t.shape[0], 
                        bias=True,
                    ).cuda().bfloat16()
                    
                    mod = torch.compile(mod, mode="max-autotune-no-cudagraphs", fullgraph=True, dynamic=False)
                    speeds.append(do_bench(lambda: mod(inp_t)))

            writer.writerow([f"{inp_t.shape[0], inp_t.shape[1], weight_t.shape[1]}", speeds[0], speeds[1]])
            print([f"{inp_t.shape[0]}_{inp_t.shape[1]}_{weight_t.shape[1]}", speeds[0], speeds[1]])
            file.flush()

# compare_op("new_old_config_compare_addmm_float16.csv", aten.addmm.default, dtype=torch.float16)
# compare_op("new_old_config_compare_addmm_bfloat16.csv", aten.addmm.default, dtype=torch.bfloat16)
# compare_op("new_old_config_compare_addmm_float32.csv", aten.addmm.default, dtype=torch.float32)
compare_op("new_old_config_compare_mm_float16.csv", aten.mm.default, dtype=torch.float16)
compare_op("new_old_config_compare_mm_bfloat16.csv", aten.mm.default, dtype=torch.bfloat16)
compare_op("new_old_config_compare_mm_float32.csv", aten.mm.default, dtype=torch.float32)
