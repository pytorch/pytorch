import torch.fx as fx
import copy
import torch
import math
import sys
from typing import Callable, List
from functools import wraps, partial
from dataclasses import dataclass
from .compile_utils import get_placeholders, get_outputs
from torch.utils._content_store import ContentStoreWriter
from torch.hub import tqdm
from torch.multiprocessing.reductions import StorageWeakRef
import os.path

is_tuple = object()

@dataclass
class LoadTensorMeta:
    size: List[int]
    stride: List[int]
    dtype: torch.dtype
    device: torch.device

class ConcreteProp(torch.fx.Interpreter):
    def __init__(self, mod, *, writer=None, skip_offload=False):
        super().__init__(mod)
        self.writer = writer
        self.skip_offload = skip_offload
        self.seen_storages = set()

    def run_node(self, n):
        self.pbar.update(1)
        r = super().run_node(n)
        name = n.name

        if isinstance(r, torch.Tensor):
            if self.writer is None:
                n.meta['concrete_value'] = r
            else:
                if StorageWeakRef(r.untyped_storage()) in self.seen_storages:
                    # Refuse to offload tensors which alias other live
                    # tensors, because this will violate operator contracts
                    n.meta['concrete_value'] = None
                else:
                    if not self.skip_offload:
                        self.writer.write_tensor(os.path.join("eager", name), r)
                    n.meta['concrete_value'] = LoadTensorMeta(
                        r.size(),
                        r.stride(),
                        r.dtype,
                        r.device
                    )
                    self.seen_storages.add(StorageWeakRef(r.untyped_storage()))
        else:
            n.meta['concrete_value'] = is_tuple

        return r

    def propagate(self, *args):
        with tqdm(
            desc="Saving intermediates for delta debugging",
            total=len(self.module.graph.nodes),
            disable=self.writer is None
        ) as pbar:
            self.pbar = pbar
            r = super().run(*args)
            if not self.skip_offload:
                pbar.set_description("Saved!  To skip next time, run with --skip-saving-eager-intermediates")
            return r

def is_load_tensor_node(node):
    return node.op == 'call_function' and node.target is torch.ops.debugprims.load_tensor.default


# inplace modifies node/inps
def _convert_node_to_placeholder(graph, node, inps):
    if node.op == 'output' or node.op == "placeholder":
        return False

    if is_load_tensor_node(node):
        return False

    concrete_val = node.meta.get('concrete_value', None)

    if isinstance(concrete_val, torch.Tensor):
        node.op = 'placeholder'
        node.target = node.name
        node.args = ()
        node.kwargs = {}

        inps.append(concrete_val)
        return True

    elif concrete_val is None:
        return False

    elif concrete_val is is_tuple:
        r = False
        for tuple_user in list(node.users):
            r = _convert_node_to_placeholder(graph, tuple_user, inps) or r
        # NB: We must not erase the node at this point, because
        # we are iterating over the nodes and this would change
        # the iteration order
        # graph.erase_node(node)
        return r

    elif isinstance(concrete_val, LoadTensorMeta):
        node.op = 'call_function'
        node.target = torch.ops.debugprims.load_tensor.default
        node.args = (os.path.join("eager", node.name), concrete_val.size, concrete_val.stride)
        node.kwargs = {
            'device': concrete_val.device,
            'dtype': concrete_val.dtype,
        }
        return True

    return False


def dump_state(fx_g, inps):
    print(f"""
# Working Repro with {len(fx_g.graph.nodes)} nodes
inps = {[(i.shape, i.dtype, i.device.type) for i in inps]}
inps = [torch.zeros(())] + [torch.ones(shape, dtype=dtype, device=device) for (shape, dtype, device) in inps]
{fx_g.code}
""")

def is_power_of_two(n):
    if n == 0:
        return False
    return (n & (n - 1)) == 0

@dataclass
class ReproState:
    graph: fx.Graph
    inps: List[torch.Tensor]

    def __post_init__(self):
        ph_nodes = get_placeholders(self.graph)
        assert len(ph_nodes) == len(self.inps)

def minifier(
    fail_f: fx.GraphModule, inps, module_fails, dump_state: Callable = dump_state, *,
    save_dir=None, offload_to_disk=False, skip_offload=False, skip_sanity=False,
    max_granularity=None
):
    """
    Minimizes a FX graph with given inputs, such that the resulting FX graph still returns True for module_fails.

    Does 2 main strategies:
    1. Truncates suffix: Removes some suffix from the graph and sets a new output.
    2. Delta Debugging: Tries replacing half of the graph with inputs. If fails,
        tries replacing quarter of the graph, etc.

    >>> # xdoctest: +SKIP(failing)
    >>> failing_function = fx.symbolic_trace(f)
    >>> minimize(failing_function, [torch.randn(5)], lambda fx_g, inps: fx_g(*inps))

    note: module_fails returns True if it fails.
    """
    assert isinstance(inps, (tuple, list))

    failing_graph = fail_f.graph
    cur_size = len(failing_graph.nodes)

    if max_granularity is not None and not is_power_of_two(max_granularity):
        raise RuntimeError(f"max_granularity {max_granularity} not power of two")

    num_queries = 0

    def deepcopy_fx_graph(fx_graph):
        return fx.GraphModule(fail_f, copy.deepcopy(fx_graph)).graph


    def graph_fails(graph, inps):
        nonlocal num_queries
        graph = copy.deepcopy(graph)
        num_queries += 1
        mod = fx.GraphModule(fail_f, graph)
        mod.graph.lint()
        return module_fails(mod, inps)

    writer = None
    if offload_to_disk:
        writer = ContentStoreWriter(save_dir)

    ConcreteProp(fail_f, writer=writer, skip_offload=skip_offload).propagate(*inps)
    if not skip_sanity and not graph_fails(failing_graph, inps):
        raise RuntimeError("Input graph did not fail the tester")
    print(f"Started off with {cur_size} nodes", file=sys.stderr)

    def _register_strategy(strategy: Callable, name: str):
        @wraps(strategy)
        def new_func(old_state: ReproState, granularity=1):
            print(file=sys.stderr)
            print(
                f"Strategy: {name} (G: {granularity}) "
                f"({len(old_state.graph.nodes)} nodes, {len(old_state.inps)} inputs)",
                file=sys.stderr
            )
            new_state = strategy(deepcopy_fx_graph(old_state.graph), list(old_state.inps), granularity)
            if new_state is not None:
                new_nodes = len(new_state.graph.nodes)
                old_nodes = len(old_state.graph.nodes)
                new_inps = len(new_state.inps)
                old_inps = len(old_state.inps)
                new_outs = len(get_outputs(new_state.graph))
                old_outs = len(get_outputs(old_state.graph))
                progress_made = False
                if new_nodes < old_nodes:
                    progress_made = True
                    print(f"SUCCESS: Went from {old_nodes} to {new_nodes} nodes", file=sys.stderr)
                if new_inps > old_inps:
                    progress_made = True
                    print(f"SUCCESS: Went from {old_inps} to {new_inps} inputs", file=sys.stderr)
                if new_outs < old_outs:
                    progress_made = True
                    print(f"SUCCESS: Went from {old_outs} to {new_outs} outputs", file=sys.stderr)

                if not progress_made:
                    raise RuntimeError("Success raised but no progress made?")

                if not graph_fails(new_state.graph, new_state.inps):
                    print("WARNING: Something went wrong, not applying this minification", file=sys.stderr)
                    return None
                return new_state
            else:
                print(f"FAIL: {name}", file=sys.stderr)
            return None

        return new_func

    def register_strategy(name: str):
        return partial(_register_strategy, name=name)

    @register_strategy("Truncate suffix")
    def remove_suffix(cur_graph, cur_inps, granularity):
        tested = set()
        new_graph = fx.Graph()
        env = {}
        for idx, node in enumerate(cur_graph.nodes):
            new_node = new_graph.node_copy(node, lambda x: env[x])
            if node.op not in ['placeholder', 'output']:
                # If idx is divisible by (granularity * 2), it would have been checked already.
                if idx % granularity == 0 and (idx % (granularity * 2) != 0) and idx not in tested:
                    output_node = new_graph.output((new_node,))
                    if len(new_graph.nodes) < len(cur_graph.nodes) and graph_fails(new_graph, cur_inps):
                        return ReproState(new_graph, cur_inps)
                    else:
                        tested.add(idx)
                        new_graph.erase_node(output_node)
            env[node] = new_node
        return None

    @register_strategy("Remove outputs")
    def remove_outputs(cur_graph, cur_inps, granularity):
        granularity = max(1, granularity // 2)
        for idx, node in enumerate(cur_graph.nodes):
            node.idx = idx
            if node.op == 'output':
                output = node
                break

        output_args = sorted(output.args[0], key=lambda x: x.idx if isinstance(x, fx.Node) else int(1e9))
        if len(output_args) == 1:
            return None

        for idx in range(0, len(output_args), granularity):
            output.args = (output_args[:idx] + output_args[idx + granularity:],)
            if graph_fails(cur_graph, cur_inps):
                return ReproState(cur_graph, cur_inps)
        return None


    def remove_unused_inputs_unchecked(cur_state: ReproState):
        cur_graph = cur_state.graph
        cur_inps = cur_state.inps
        ph_nodes = get_placeholders(cur_graph)
        assert len(ph_nodes) == len(cur_inps)

        new_inps = []
        for idx in range(len(ph_nodes)):
            if len(ph_nodes[idx].users) == 0:
                cur_graph.erase_node(ph_nodes[idx])
            else:
                new_inps.append(cur_inps[idx])
        if len(new_inps) < len(cur_inps):
            return ReproState(cur_graph, new_inps)
        return None

    def remove_unused_inputs_checked(cur_state: ReproState):
        new_state = remove_unused_inputs_unchecked(cur_state)
        if new_state is not None and graph_fails(new_state.graph, new_state.inps):
            return new_state
        return None

    def _remove_unused_wrapper(cur_graph, cur_inps, granularity):
        return remove_unused_inputs_checked(ReproState(cur_graph, cur_inps))

    remove_unused_inputs = register_strategy("Remove unused inputs")(_remove_unused_wrapper)

    @register_strategy("Eliminate dead code")
    def eliminate_dead_code(cur_graph, cur_inps, granularity):
        if cur_graph.eliminate_dead_code() and graph_fails(cur_graph, cur_inps):
            return ReproState(cur_graph, cur_inps)
        return None


    def _consolidate_placeholders(cur_graph, inps):
        new_graph = fx.Graph()
        env = {}
        seen_non_placeholder = False

        # Move all placeholders to the front; also, if any load_tensor
        # is at the front, convert it into an input (because it can be live
        # all the time)
        for node in cur_graph.nodes:
            if node.op == 'placeholder':
                new_node = new_graph.node_copy(node, lambda x: env[x])
                env[node] = new_node
            elif not seen_non_placeholder and is_load_tensor_node(node):
                new_node = new_graph.placeholder(node.name)
                env[node] = new_node
                inps.append(torch.ops.debugprims.load_tensor.default(*node.args, **node.kwargs))
            else:
                seen_non_placeholder = True

        # Move everyone else
        for node in cur_graph.nodes:
            if node not in env:
                new_node = new_graph.node_copy(node, lambda x: env[x])
                env[node] = new_node
        return new_graph

    @register_strategy("Delta Debugging")
    def delta_debugging(cur_graph: fx.Graph, cur_inps, granularity):
        num_nodes = len(cur_graph.nodes)
        for start_range in range(0, num_nodes, granularity):
            is_removing = False
            new_graph = deepcopy_fx_graph(cur_graph)
            new_inps = cur_inps[:]
            end_range = min(num_nodes, start_range + granularity)
            for idx in range(start_range, end_range):
                new_node = list(new_graph.nodes)[idx]
                if _convert_node_to_placeholder(new_graph, new_node, new_inps):
                    is_removing = True
            if not is_removing:
                continue
            new_graph.eliminate_dead_code()
            new_graph = _consolidate_placeholders(new_graph, new_inps)
            new_state = remove_unused_inputs_unchecked(ReproState(new_graph, new_inps))
            if new_state is None:
                new_state = ReproState(new_graph, new_inps)
            if graph_fails(new_state.graph, new_state.inps):
                return ReproState(new_state.graph, new_state.inps)

        return None

    @register_strategy("Consolidate Inputs")
    def consolidate_inputs(cur_graph, cur_inps, granularity):
        old_len = len(cur_inps)
        cur_graph = _consolidate_placeholders(cur_graph, cur_inps)
        if len(cur_inps) > old_len and graph_fails(cur_graph, cur_inps):
            return ReproState(cur_graph, cur_inps)
        return None

    failing_state = ReproState(failing_graph, inps)

    def try_granularity(failing_state, granularity, use_non_granular):
        print(f"Trying granularity {granularity}", file=sys.stderr)

        strategies = []
        num_nodes = len(failing_state.graph.nodes)
        num_outputs = len(get_outputs(failing_state.graph))
        if num_outputs > num_nodes // 2:
            strategies += [remove_outputs]

        if use_non_granular:
            strategies += [eliminate_dead_code, remove_unused_inputs, consolidate_inputs]

        strategies += [remove_suffix, delta_debugging]

        for strategy in strategies:
            new_state = strategy(failing_state, granularity)
            if new_state is not None:
                return new_state
        return None

    while True:
        dump_state(fx.GraphModule(fail_f, failing_state.graph), failing_state.inps)
        granularity = int(2**(math.floor(math.log2(len(failing_state.graph.nodes)))))
        if max_granularity is not None:
            granularity = min(max_granularity, granularity)
        new_state = try_granularity(failing_state, granularity, use_non_granular=True)
        if new_state is not None:
            failing_state = new_state
            continue

        granularity //= 2
        has_progress = False
        while granularity >= 1:
            new_state = try_granularity(failing_state, granularity, use_non_granular=False)
            if new_state is not None:
                failing_state = new_state
                has_progress = True
                break
            granularity //= 2
        if has_progress:
            continue

        new_state = remove_outputs(failing_state, 1)
        if new_state is not None:
            failing_state = new_state
            continue

        break

    if not graph_fails(failing_state.graph, failing_state.inps):
        raise RuntimeError("Uh oh, something went wrong :( Final graph is not failing")

    print(f"Made {num_queries} queries", file=sys.stderr)
    failing_fx = fx.GraphModule(fail_f, failing_state.graph)
    dump_state(failing_fx, failing_state.inps)
    print("Wrote minimal repro out to repro.py", file=sys.stderr)
    return failing_fx, failing_state.inps
