"""
Adapted from fsdp.py in https://github.com/pytorch/pytorch/pull/110609.
"""

"""
rm artifacts/* && TORCH_NCCL_AVOID_RECORD_STREAMS=1 CUDA_VISIBLE_DEVICES=4,5 TORCH_LOGS_RANKS=0 TORCH_COMPILE_DEBUG=1 torchrun --standalone --nproc_per_node=2 test_dynamo_fsdp.py >artifacts/run_output.txt 2>&1

# 2-gpu
rm artifacts/* && TORCH_NCCL_AVOID_RECORD_STREAMS=1 TORCH_LOGS="aot_graphs,output_code,compiled_autograd,recompiles,+dynamo,aot,inductor" TORCH_COMPILE_DEBUG=1 CUDA_VISIBLE_DEVICES=4,5 TORCH_LOGS_RANKS=0 TORCH_COMPILE_DEBUG=1 torchrun --standalone --nproc_per_node=2 test_dynamo_fsdp.py >artifacts/run_output.txt 2>&1

# 8-gpu
rm artifacts/* && TORCH_NCCL_AVOID_RECORD_STREAMS=1 TORCH_LOGS="aot_graphs,output_code,compiled_autograd,recompiles,+dynamo,aot,inductor" TORCH_COMPILE_DEBUG=1 TORCH_LOGS_RANKS=0 TORCH_COMPILE_DEBUG=1 torchrun --standalone --nproc_per_node=8 test_dynamo_fsdp.py >artifacts/run_output.txt 2>&1

TORCH_NCCL_AVOID_RECORD_STREAMS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=6,7 TORCH_LOGS_RANKS=0 TORCH_COMPILE_DEBUG=1 torchrun --standalone --nproc_per_node=2 test_dynamo_fsdp.py >artifacts/run_output.txt 2>&1

TORCH_NCCL_AVOID_RECORD_STREAMS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=6 TORCH_LOGS_RANKS=0 TORCH_COMPILE_DEBUG=1 torchrun --standalone --nproc_per_node=1 test_dynamo_fsdp.py >artifacts/run_output.txt 2>&1

TORCH_NCCL_AVOID_RECORD_STREAMS=1 CUDA_VISIBLE_DEVICES=6,7 TORCH_LOGS_RANKS=0 torchrun --standalone --nproc_per_node=2 test_dynamo_fsdp.py >artifacts/run_output.txt 2>&1

TORCH_NCCL_AVOID_RECORD_STREAMS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=6 TORCH_LOGS_RANKS=0 torchrun --standalone --nproc_per_node=1 test_dynamo_fsdp.py >artifacts/run_output.txt 2>&1

TORCH_NCCL_AVOID_RECORD_STREAMS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=6 TORCH_LOGS_RANKS=0 TORCH_COMPILE_DEBUG=1 torchrun --standalone --nproc_per_node=1 test_dynamo_fsdp.py >artifacts/run_output.txt 2>&1

TORCH_NCCL_AVOID_RECORD_STREAMS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=6 TORCH_LOGS_RANKS=0 gdb --args python3 /data/users/willfeng/miniconda3/bin/torchrun --standalone --nproc_per_node=1 test_dynamo_fsdp.py
"""
import contextlib
import logging
import os
import sys
import traceback
import gc
import itertools

import torch
import torch._dynamo
import torch.distributed as dist
import torch.nn as nn
from torch._dynamo import compiled_autograd
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed._composable.fsdp._fsdp_init import (
    _get_managed_modules,
    _get_managed_states,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed._tensor import init_device_mesh
from llama_model_toy import ToyTransformer, ModelArgs
# from llama_model import Transformer, ModelArgs

# from torchviz import make_dot

torch_log = logging.getLogger("torch")

device_type = "cuda"

# ======== REMOVE WHEN READY TO MERGE ========
import argparse
import os
import subprocess
import sys
import urllib
import urllib.parse
import uuid

from typing import Optional

PERFETTO_UI_ROOT_URL = (
    "https://interncache-all.fbcdn.net/manifold/perfetto-artifacts/tree/ui/index.html"
)
MANIFOLD_FOLDER = "perfetto_internal_traces/tree/shared_trace"
DEFAULT_TTL_SEC = 28 * 24 * 60 * 60


def upload_trace_file(local_path: str, overwrite: bool = False) -> Optional[str]:
    file_name = os.path.basename(local_path)
    manifold_path = os.path.join(
        MANIFOLD_FOLDER, f"{os.getlogin()}_{str(uuid.uuid4())}_{file_name}"
    )
    cmd = [
        "manifold",
        "put",
        local_path,
        manifold_path,
        "--ttl",
        str(DEFAULT_TTL_SEC),
        "--userData",
        "false",
    ]
    ret = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    if ret.returncode == 0:
        print("Uploaded trace successfully.")
        return manifold_path
    else:
        print("[ERROR] Upload failed, maybe the trace file exists.")
        return None


def print_perfetto_ui_url(manifold_path: str) -> None:
    url = (
        PERFETTO_UI_ROOT_URL
        + "#!/?url=https://interncache-all.fbcdn.net/manifold/"
        + urllib.parse.quote_plus(manifold_path)
    )
    print(f"The trace is accessible at:\n{url}")


import socket
from datetime import datetime, timedelta

# Keep a max of 100,000 alloc/free events in the recorded history
# leading up to the memory snapshot.
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000

def start_record_memory_history() -> None:
   if not torch.cuda.is_available():
       print("CUDA unavailable. Not recording memory history")
       return

   print("Starting memory snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(
       max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
   )

def stop_record_memory_history() -> None:
   if not torch.cuda.is_available():
       print("CUDA unavailable. Not recording memory history")
       return

   print("Stopping memory snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(enabled=None)

def export_memory_snapshot(file_prefix) -> None:
   if not torch.cuda.is_available():
       print("CUDA unavailable. Not exporting memory snapshot")
       return

   try:
       print(f"Saving memory snapshot to local file: artifacts/{file_prefix}.pickle")
       torch.cuda.memory._dump_snapshot(f"artifacts/{file_prefix}.pickle")
   except Exception as e:
       print(f"Failed to capture memory snapshot {e}")
       raise
# ======== REMOVE WHEN READY TO MERGE ========



def count_ops(graph, freqs=None, freqs_ge=None, ops=None):
    def match_rng_op(node, op):
        if isinstance(node.target, torch._ops.HigherOrderOperator):
            if node.name == "run_and_save_rng_state":
                return node.args[0] == op
            elif node.name == "run_with_rng_state":
                return node.args[1] == op
        return False

    if freqs:
        for op, freq in zip(ops, freqs):
            actual_count = 0
            for node in graph.nodes:
                if match_rng_op(node, op) or node.target == op:
                    actual_count += 1
            err_msg = f"In graph {graph} \n\n Expected {op} to have occurred {freq} times in the graph, but got {actual_count}."
            assert actual_count == freq, err_msg
    else:
        assert freqs_ge is not None
        for op, freq_ge in zip(ops, freqs_ge):
            actual_count = 0
            for node in graph.nodes:
                if match_rng_op(node, op) or node.target == op:
                    actual_count += 1
            assert (
                actual_count >= freq_ge
            ), f"In graph {graph}  \n\n Expected {op} to have occurred at least {freq_ge} times in the graph, but got {actual_count}."
    return graph



def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    torch_log.error(
        "Uncaught exception\n%s",
        "".join(traceback.format_exception(exc_type, exc_value, exc_traceback)),
    )


sys.excepthook = handle_exception

# NOTE: copied from TorchTrain
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper as ptd_checkpoint_wrapper
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.utils.checkpoint import checkpoint, _pt2_selective_checkpoint_context_fn_gen

class ACConfigClass:
    mode: str = "selective"
    selective_ac_option: str = "2"

def checkpoint_wrapper(module, config):
    if config.mode == "selective" and config.selective_ac_option == "op":

        def _get_custom_policy(meta):
            no_recompute_list = [
                torch.ops.aten.mm.default,
            ]
            def _custom_policy(mode, func, *args, **kwargs):
                mm_count_key = f"{mode}_mm_count"
                if func == torch.ops.aten.mm.default:
                    if mm_count_key not in meta:
                        meta[mm_count_key] = 0
                    meta[mm_count_key] += 1
                # Saves output of all compute ops, except every second mm
                return func in no_recompute_list and not (
                    func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0
                )

            return _custom_policy

        def selective_checkpointing_context_fn():
            meta = {}
            return _pt2_selective_checkpoint_context_fn_gen(_get_custom_policy(meta))

        return ptd_checkpoint_wrapper(
            module,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            checkpoint_fn=checkpoint,
            context_fn=selective_checkpointing_context_fn,
            use_reentrant=False,
            preserve_rng_state=False,
        )
    elif config.mode == "full":
        return ptd_checkpoint_wrapper(
            module,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            checkpoint_fn=checkpoint,
            use_reentrant=False,
            preserve_rng_state=False,
        )

    elif config.mode == "selective" and config.selective_ac_option.isdigit():
        """enables selective checkpointing of candidate layers.
        Usage:
        'selective_ac_option' with a positive 'int' value in config controls which layers to checkpoint.
        1 == checkpointing every one (all).
        2 == checkpoint every 2nd one
        """
        every_x_layer = int(config.selective_ac_option)
        assert (
            every_x_layer >= 0
        ), f"selective layer AC policy (every_x_layer) expects a positive integer, received {every_x_layer}"

        checkpoint_wrapper.__dict__.setdefault("_count", 0)

        checkpoint_wrapper._count += 1
        if not every_x_layer or checkpoint_wrapper._count % every_x_layer == 0:
            torch_log.warning(f"Applying SAC to layer: {checkpoint_wrapper._count}")
            return ptd_checkpoint_wrapper(
                module,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                checkpoint_fn=checkpoint,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        # skip activation checkpointing and store activations for this layer
        else:
            return module
    else:
        raise NotImplementedError(
            "Unknown AC type or AC config. Only selective op and selective layer ac implemented currently."
        )


test_case = "toy_transformer"  # "simple_mlp" / "simple_seq_module" / "nested_fully_shard" / "toy_transformer"
balanced = True
mixed_precision = False  # TODO(yf225): when True, fails accuracy test, needs debugging
apply_fsdp = True

def create_input(hidden_dim):
    torch.manual_seed(0)
    # inp = torch.randn((2, hidden_dim), device=device_type, requires_grad=False)
    inp = torch.zeros((2, hidden_dim), device=device_type, requires_grad=False, dtype=torch.long)
    return inp


def init(activation_checkpoint):
    from torch.testing._internal.common_fsdp import MLP
    # simple_mlp + balanced -> works
    # simple_mlp + unbalanced -> works
    # nested_fully_shard + balanced -> works
    # nested_fully_shard + unbalanced -> works
    if balanced:
        hidden_dim = 5120
    else:
        hidden_dim = 5121
    if mixed_precision:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
        fsdp_config = {"mp_policy": mp_policy}
    else:
        fsdp_config = {}
    if activation_checkpoint:
        ac_config = ACConfigClass()  # use default in this class
        torch._dynamo.config._experimental_support_context_fn_in_torch_utils_checkpoint = True
    mesh = init_device_mesh("cuda", (world_size,))

    torch.manual_seed(0)
    if test_case == "simple_mlp":
        model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, device=device_type),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, device=device_type),  # FC->RELU->FC is a good test
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, device=device_type),
        )
        if activation_checkpoint:
            model = checkpoint_wrapper(model, ac_config)
        if apply_fsdp:
            fully_shard(model, reshard_after_forward=True, _reshard_after_forward_root=True, **fsdp_config)
    elif test_case == "simple_seq_module":  # this causes `len(splits) == 1` which is an interesting case for `replace_foreach_all_gather_copy_out_pattern` FX pass.
        class SimpleModule(nn.Module):
            def __init__(self, device: torch.device):
                super().__init__()
                self.param = nn.Parameter(torch.randn(hidden_dim, hidden_dim, device=device))

            def forward(self, x):
                return torch.matmul(x, self.param)

        model = nn.Sequential(*[SimpleModule(torch.device("cuda")) for _ in range(1)])
        if activation_checkpoint:
            model = checkpoint_wrapper(model, ac_config)
        if apply_fsdp:
            for mod in model:
                fully_shard(mod, mesh=mesh, reshard_after_forward=True, _reshard_after_forward_root=True, **fsdp_config)
            fully_shard(model, mesh=mesh, reshard_after_forward=True, _reshard_after_forward_root=True, **fsdp_config)
    elif test_case == "nested_fully_shard":
        class TestModule(nn.Module):
            def __init__(self, n_layers):
                super().__init__()
                self.layers = torch.nn.ModuleList()
                for layer_id in range(n_layers):
                    self.layers.append(MLP(hidden_dim))

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        model = TestModule(n_layers=8)
        assert apply_fsdp
        assert mesh is not None
        for layer_id, mod in enumerate(model.layers):
            if activation_checkpoint:
                mod = checkpoint_wrapper(mod, ac_config)
            fully_shard(mod, mesh=mesh, reshard_after_forward=True, _reshard_after_forward_root=True, **fsdp_config)
            model.layers[layer_id] = mod
        model = fully_shard(model, mesh=mesh, reshard_after_forward=True, _reshard_after_forward_root=True, **fsdp_config)
        # if apply_fsdp:
        #     for mod in model:
        #         fully_shard(mod, mesh=mesh, reshard_after_forward=True, _reshard_after_forward_root=True, **fsdp_config)
        #     fully_shard(model, mesh=mesh, reshard_after_forward=True, _reshard_after_forward_root=True, **fsdp_config)
    elif test_case == "toy_transformer":
        torch._functorch.config.aggressive_recomputation = False
        torch._inductor.config.allow_buffer_reuse = False
        torch._inductor.config.inplace_buffers = False
        torch._inductor.config.reorder_for_compute_comm_overlap = True
        torch._inductor.config.reorder_for_compute_comm_overlap_passes = [
            "sink_waits",
            "raise_comms",
        ]
        torch._inductor.config.raise_last_usage = True
        model_args = ModelArgs(
            dim=hidden_dim,
            n_layers=3,
            n_heads=1,
            vocab_size=1024,
        )
        transformer_class = ToyTransformer  # makes comm-induced peak memory issue more prominent
        # transformer_class = Transformer
        model = transformer_class(model_args)
        for layer_id, mod in enumerate(model.layers):
            if activation_checkpoint:
                mod = checkpoint_wrapper(mod, ac_config)
            if apply_fsdp:
                fully_shard(mod, mesh=mesh, reshard_after_forward=True, _reshard_after_forward_root=True, **fsdp_config)
            model.layers[layer_id] = mod
        if apply_fsdp:
            model = fully_shard(model, mesh=mesh, reshard_after_forward=True, _reshard_after_forward_root=True, **fsdp_config)
        else:
            model = model.cuda()
    # elif test_case == "test_tags_function":
    #     class TestModule(torch.nn.Module):
    #         def __init__(self, device):
    #             super().__init__()
    #             self.param = torch.nn.Parameter(torch.randn(hidden_dim, hidden_dim, device=device))

    #         def gn(self, x, y):
    #             return torch.sigmoid(torch.matmul(x, y))

    #         def forward(self, x):
    #             return self.gn(x, self.param)

    #     model = TestModule("cuda")
    #     if activation_checkpoint:
    #         ac_config = ACConfigClass()
    #         ac_config.mode = "full"
    #         model = checkpoint_wrapper(model, ac_config)
    #     if apply_fsdp:
    #         fully_shard(model, mesh=mesh, reshard_after_forward=True, _reshard_after_forward_root=True, **fsdp_config)

    #     # graph_count = 0
    #     # def count_ops_pass(graph):
    #     #     nonlocal graph_count
    #     #     if graph_count == 0:  # assume the first graph is FWD graph, second is BWD graph
    #     #         count_ops(graph, freqs=[1], ops=[torch.ops.aten.mm.default])
    #     #     elif graph_count == 1:
    #     #         count_ops(graph, freqs=[3], ops=[torch.ops.aten.mm.default])
    #     #     else:
    #     #         raise RuntimeError("Unexpected graph_count")
    #     #     graph_count += 1
    #     #     return graph

    #     # torch._inductor.config.post_grad_custom_post_pass = count_ops_pass

    #     # fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
    #     # bw_compiler = functools.partial(
    #     #     count_ops, freq=3, op=torch.ops.aten.mm.default
    #     # )  # mm recomputed in the bwd
    #     # backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
    #     # self._validate(fn, backend, x, y)
    # else:
    #     # FSDP1
    #     fsdp_kwargs = {
    #         "use_orig_params": True,
    #         "auto_wrap_policy": ModuleWrapPolicy({nn.Linear}),
    #         # "limit_all_gathers": False,
    #     }
    #     model = FSDP(
    #         model,
    #         **fsdp_kwargs,
    #     )
    optim = torch.optim.SGD(model.parameters(), lr=1e-6)
    return model, optim, hidden_dim


def printing_eager(gm, inputs):
    gm.graph.print_tabular()
    return gm.forward


local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])


def run(model, optim, n_iter, hidden_dim):
    torch.manual_seed(42)
    losses = []
    for i in range(n_iter):
        torch_log.warning(f"Starting iteration: {i}")
        optim.zero_grad(set_to_none=True)
        inp = create_input(hidden_dim)
        torch_log.warning("FORWARD")
        out = model(inp)
        torch_log.warning("END FORWARD")
        loss = out.sum()
        losses.append(loss.item())
        torch_log.warning("BACKWARD")
        # torch_log.warning("OUT GRAPH\n%s", make_dot(loss))
        loss.backward()
        torch_log.warning("END BACKWARD")
        optim.step()
        del loss
        del out
        del inp
        torch.cuda.synchronize()
        torch_log.warning(f"Done iteration: {i}")
    print(f"losses: {losses}")
    return losses


def main_compiled(n_iter, activation_checkpoint, backend):
    model, optim, hidden_dim = init(activation_checkpoint=activation_checkpoint)
    # # per-param FSDP does lazy init using 1st run, so run it once to init using eager mode
    # run(model, optim, 1, hidden_dim)
    # print("done eager 1st run for compiled!")

    def compiler_fn(gm):
        torch_log.warning("Compiling autograd?")
        return torch.compile(gm, backend=backend, fullgraph=False)

    if apply_fsdp:
        torch._dynamo.config.trace_distributed = True
        torch._functorch.config.move_view_chain_to_bwd_graph = True

    torch._inductor.config.triton.unique_kernel_names = True

    # if dist.get_rank() == 0:
    #     # HACK: delay rank 0 by X seconds, so that rank 1 will always fail first.
    #     import time
    #     time.sleep(600)
    model_compiled = torch.compile(model, backend=backend, fullgraph=False)
    with compiled_autograd.enable(compiler_fn):
        res = run(model_compiled, optim, n_iter, hidden_dim)
    print(f"res: {res}")
    torch._dynamo.reset()
    for state in torch.distributed._composable_state._module_state_mapping.values():
        if hasattr(state._fsdp_param_group, "fsdp_params"):
            for fsdp_param in state._fsdp_param_group.fsdp_params:
                fsdp_param._sharded_param_data.untyped_storage().resize_(0)
    managed_modules = _get_managed_modules(model)
    params, buffers = _get_managed_states(managed_modules)
    for tensor in itertools.chain(params, buffers):
        try:
            tensor.untyped_storage().resize_(0)
        except:
            pass
    torch.distributed._composable_state._module_state_mapping = {}
    del model_compiled
    del model
    optim.zero_grad(set_to_none=True)
    del optim
    return res


def main_eager(n_iter, activation_checkpoint):
    model, optim, hidden_dim = init(activation_checkpoint=activation_checkpoint)
    # per-param FSDP does lazy init using 1st run, so run it once to init using eager mode
    run(model, optim, 1, hidden_dim)
    print("done eager 1st run for eager!")

    res = run(model, optim, n_iter, hidden_dim)
    for state in torch.distributed._composable_state._module_state_mapping.values():
        if hasattr(state._fsdp_param_group, "fsdp_params"):
            for fsdp_param in state._fsdp_param_group.fsdp_params:
                fsdp_param._sharded_param_data.untyped_storage().resize_(0)
    managed_modules = _get_managed_modules(model)
    params, buffers = _get_managed_states(managed_modules)
    for tensor in itertools.chain(params, buffers):
        try:
            tensor.untyped_storage().resize_(0)
        except:
            pass
    torch.distributed._composable_state._module_state_mapping = {}
    del model
    optim.zero_grad(set_to_none=True)
    del optim
    return res


def execute_and_profile(callable, profiler_trace_path, memory_snapshot_file_prefix):
    from torch.profiler import profile, ProfilerActivity
    # if dist.get_rank() == 0:
    #     start_record_memory_history()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        ret = callable()
    if dist.get_rank() == 0:
        prof.export_chrome_trace(profiler_trace_path)
        if not os.path.exists(profiler_trace_path):
            raise Exception(f"[ERROR] The trace file doesn't exist: {profiler_trace_path}")
        manifold_path = upload_trace_file(profiler_trace_path)
        if manifold_path:
            print_perfetto_ui_url(manifold_path)
        # export_memory_snapshot(memory_snapshot_file_prefix)
        # stop_record_memory_history()
    return ret


if __name__ == "__main__":
    n_iter = 3
    assert device_type == "cuda"
    device = f"{device_type}:{local_rank}"
    if device_type == "cuda":
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(device)
    else:
        dist.init_process_group(backend="gloo")
        # torch.set_device(device)

    if dist.get_rank() == 0:
        start_record_memory_history()
    ac_test_order = [True]
    backends = ["inductor"]

    def test_eager(activation_checkpoint):
        losses_eager = execute_and_profile(
            lambda: main_eager(n_iter=n_iter, activation_checkpoint=activation_checkpoint),
            f"artifacts/eager_ac{activation_checkpoint}_trace.json",
            f"eager_ac{activation_checkpoint}_memory_snapshot",
        )
        print(f"losses_eager: {losses_eager}")
        gc.collect()
        gc.collect()
        return losses_eager

    def test_compile(activation_checkpoint, backend):
        losses_compiled = execute_and_profile(
            lambda: main_compiled(n_iter=n_iter, activation_checkpoint=activation_checkpoint, backend=backend),
            f"artifacts/compiled_{backend}_ac{activation_checkpoint}_trace.json",
            f"compiled_{backend}_ac{activation_checkpoint}_memory_snapshot",
        )
        print(f"losses_compiled: {losses_compiled}")
        gc.collect()
        gc.collect()
        return losses_compiled

    for activation_checkpoint in ac_test_order:
        for backend in backends:
            losses_compiled = test_compile(activation_checkpoint, backend)
        # losses_eager = test_eager(activation_checkpoint)
        # for loss_compiled, loss_eager in zip(losses_compiled, losses_eager):
        #     assert torch.allclose(torch.tensor(loss_compiled), torch.tensor(loss_eager), rtol=1e-3), f"{loss_compiled} vs {loss_eager}"

    if dist.get_rank() == 0:
        export_memory_snapshot(f"combined_fsdp{apply_fsdp}_ac{ac_test_order}_backend{backends}_memory_snapshot")
        stop_record_memory_history()
