# TORCH_NCCL_AVOID_RECORD_STREAMS=1 CUDA_VISIBLE_DEVICES=0,1 tlp python -m torch.distributed.run --standalone --nproc_per_node=2 test_aot_graph_inductor_multigpu.py

# AOT ID: ['13_forward']
import logging
import os
import sys

import torch
torch_log = logging.getLogger("torch")
import logging
from datetime import datetime
import time

from torch._inductor.async_compile import AsyncCompile
log = logging.getLogger(__name__)

# Keep a max of 100,000 alloc/free events in the recorded history
# leading up to the memory snapshot.
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000

def start_record_memory_history() -> None:
    if not torch.cuda.is_available():
        log.info("CUDA unavailable. Not recording memory history")
        return

    log.info("Starting memory snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(
        max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
    )

def stop_record_memory_history() -> None:
    if not torch.cuda.is_available():
        log.info("CUDA unavailable. Not recording memory history")
        return

    log.info("Stopping memory snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(enabled=None)

def export_memory_snapshot(filepath_prefix) -> None:
    if not torch.cuda.is_available():
        log.info("CUDA unavailable. Not exporting memory snapshot")
        return

    try:
        log.info(f"Saving memory snapshot to local file: {filepath_prefix}.pickle")
        torch.cuda.memory._dump_snapshot(f"{filepath_prefix}.pickle")
    except Exception as e:
        log.info(f"Failed to capture memory snapshot {e}")
        return


def create_example_inputs(device, times=10, repeat=10):
    with open(os.path.join(os.path.dirname(__file__), 'aot_backward_graph_inputs.txt'), 'r') as file:
        graph_inputs_str = file.read()
    
    global_dict = {
        'torch': torch,
        'device': torch.device,
        '__builtins__': __builtins__
    }
    
    exec(graph_inputs_str, global_dict)

    example_inputs = global_dict['benchmark_compiled_module'](device)
    
    return example_inputs


if __name__ == "__main__":
    import torch.distributed as dist

    rank = int(os.getenv("RANK"))
    local_rank = int(os.getenv("LOCAL_RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")

    import textwrap

    import torch

    # NOTE: you usually find the FX graph from this Inductor generated code file's corresponding
    # aot_forward / aot_backward / aot_inference graph (you can find it in tlparse)
    with open(
        os.path.join(os.path.dirname(__file__), "aot_backward_graph.txt"), "r"
    ) as file:
        module_string = file.read()

    def convert_module_string_to_object(module_string):
        try:
            # Clean up the string and fix indentation
            module_string = textwrap.dedent(module_string)

            module_string = module_string.replace(
                "device(type='cuda', index=0)",
                f"device(type='cuda', index={local_rank})",
            )

            # Create the necessary global context for eval
            global_dict = {
                "torch": torch,
                "device": torch.device,
                "__builtins__": __builtins__,
            }

            # First, evaluate the class definition
            exec(module_string, global_dict)

            # Get the lambda class from the global dictionary
            GraphModule = global_dict["GraphModule"]

            # Instantiate the module
            module = GraphModule()

            return module

        except Exception as e:
            raise Exception(f"Error converting module string: {str(e)}")

    # Hack to make tlparse work
    from torch._logging import structured

    convert_frame_intern = structured.intern_string(__file__)
    torch._logging.trace_structured(
        "dynamo_start",
        lambda: {"stack": []},
    )

    if local_rank == 0:
        import time

        from torch.profiler import profile, ProfilerActivity

        unix_timestamp = time.time()

        def trace_handler(p):
            trace_file_path = f"gpu_traces/{int(unix_timestamp)}/trace_{str(p.step_num)}_rank{local_rank}_{int(unix_timestamp)}.json"
            os.makedirs(os.path.dirname(trace_file_path), exist_ok=True)
            print(trace_file_path)
            p.export_chrome_trace(trace_file_path)

        my_schedule = torch.profiler.schedule(wait=5, warmup=20, active=3)
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=my_schedule,
            on_trace_ready=trace_handler,
            with_stack=False,
        )

    from torch._inductor.compile_fx import compile_fx

    gm = convert_module_string_to_object(module_string)
    # NOTE: make sure to have all the necessary Inductor configs
    config_patches = {
        "inplace_buffers": False,
        "allow_buffer_reuse": False,
        "force_disable_caches": True,
        "reorder_for_locality": False,
        "shape_padding": False,
        "comprehensive_padding": False,
        "reorder_for_compute_comm_overlap": True,
        "reorder_for_peak_memory": True,
    }
    import gc

    num_iters = 30
    with torch.no_grad():
        example_inputs = create_example_inputs(device, times=1, repeat=1)
        inductor_compiled_fn = compile_fx(
            gm, example_inputs, config_patches=config_patches
        )
        if local_rank == 0:
            start_record_memory_history()
        iteration_times = []
        for _ in range(num_iters):
            start_time = time.time()
            inductor_compiled_fn(*example_inputs)
            iteration_time = time.time() - start_time
            iteration_times.append(iteration_time)
            if local_rank == 0:
                prof.step()
        if local_rank == 0:
            os.makedirs("memory_snapshots", exist_ok=True)
            memory_snapshot_file_path = f"memory_snapshots/memory_snapshot_{int(datetime.now().timestamp())}"
            export_memory_snapshot(memory_snapshot_file_path)
            print(f"{memory_snapshot_file_path}.pickle")
            print(f"iteration_times: {iteration_times}")
            print(f"Min iteration time: {min(iteration_times):.4f} seconds")
            print(f"Max iteration time: {max(iteration_times):.4f} seconds")
            print(f"Median iteration time: {sorted(iteration_times)[len(iteration_times)//2]:.4f} seconds")
