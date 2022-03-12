import subprocess
import torch.fx as fx
import copy
import torch
import math


class ConcreteProp(torch.fx.Interpreter):
    def run_node(self, n):
        result = super().run_node(n)

        found_tensor = False

        def extract_tensor_meta(obj):
            if isinstance(obj, torch.Tensor):
                nonlocal found_tensor
                found_tensor = True
                return obj
            else:
                return obj

        from torch.fx.node import map_aggregate
        concrete_value = map_aggregate(result, extract_tensor_meta)
        if found_tensor:
            n.meta['concrete_value'] = concrete_value
        return result

    def propagate(self, *args):
        return super().run(*args)


def _get_placeholders(graph):
    return list(filter(lambda x: x.op == 'placeholder', graph.nodes))

# inplace modifies node/inps


def _convert_node_to_placeholder(node, inps):
    if node.op == 'output':
        return
    node.op = 'placeholder'
    node.args = ()
    node.target = node.name
    concrete_val = node.meta['concrete_value']
    if isinstance(concrete_val, torch.Tensor):
        inps.append(concrete_val)
    else:
        inps.append(torch.zeros(()))
        for tuple_user in list(node.users):
            _convert_node_to_placeholder(tuple_user, inps)


def minifier(fail_f: fx.GraphModule, inps, module_fails):
    """
    Minimizes a FX graph with given inputs, such that the resulting FX graph still returns True for module_fails.

    Does 2 main strategies:
    1. Truncates suffix: Removes some suffix from the graph and sets a new output.
    2. Delta Debugging: Tries replacing half of the graph with inputs. If fails,
        tries replacing quarter of the graph, etc.

    >>> failing_function = fx.symbolic_trace(f)
    >>> minimize(failing_function, [torch.randn(5)], lambda fx_g, inps: fx_g(*inps))

    note: module_fails returns True if it fails.
    """
    failing_graph = fail_f.graph
    cur_size = len(failing_graph.nodes)

    def graph_fails(graph, inps):

        mod = fx.GraphModule(fail_f, graph)
        mod.graph.lint()
        return module_fails(mod, inps)

    ConcreteProp(fail_f).propagate(*inps)
    if not graph_fails(failing_graph, inps):
        raise RuntimeError("Input graph did not fail the tester")
    print(f"Started off with {cur_size} nodes")

    def remove_suffix(cur_graph, cur_inps):
        print("Strategy: Remove suffix")
        assert graph_fails(cur_graph, cur_inps)
        gap = 2**math.floor(math.log2(len(cur_graph.nodes)))
        tested = set()
        while gap >= 1:
            new_graph = fx.Graph()
            env = {}
            for idx, node in enumerate(cur_graph.nodes):
                new_node = new_graph.node_copy(node, lambda x: env[x])
                if node.op not in ['placeholder', 'output']:
                    if idx % gap == 0 and idx not in tested:
                        output_node = new_graph.output((new_node,))
                        if graph_fails(new_graph, cur_inps) and len(new_graph.nodes) < len(cur_graph.nodes):
                            print()
                            print(f"SUCCESS: Removed [{idx}:{len(cur_graph.nodes)})")
                            return (new_graph, cur_inps), True
                        else:
                            tested.add(idx)
                            new_graph.erase_node(output_node)
                env[node] = new_node
            gap //= 2
        print("FAIL: Could not remove suffix")
        return (cur_graph, cur_inps), False

    def remove_unused_inputs(cur_graph, cur_inps):
        assert graph_fails(cur_graph, cur_inps)
        ph_nodes = _get_placeholders(cur_graph)
        if len(ph_nodes) != len(cur_inps):
            print(cur_graph)
            print(len(cur_inps))
        assert len(ph_nodes) == len(cur_inps)

        new_inps = []
        for idx in range(len(ph_nodes)):
            if len(ph_nodes[idx].users) == 0:
                cur_graph.erase_node(ph_nodes[idx])
            else:
                new_inps.append(cur_inps[idx])

        if len(new_inps) < len(cur_inps) and graph_fails(cur_graph, new_inps):
            print("Strategy: Remove unused inputs")
            print(f"SUCCESS: Went from {len(cur_inps)} inputs to {len(new_inps)} inputs")
            return (cur_graph, new_inps), True
        else:
            return (cur_graph, new_inps), False

    def eliminate_dead_code(cur_graph, cur_inps):
        orig_size = len(cur_graph.nodes)
        if cur_graph.eliminate_dead_code() and graph_fails(cur_graph, cur_inps):
            print("Strategy: Eliminate dead code")
            print(f"SUCCESS: Went from {orig_size} nodes to {len(cur_graph.nodes)} nodes")
            return (cur_graph, cur_inps), True
        else:
            return (cur_graph, cur_inps), False

    def consolidate_placeholders(cur_graph):
        new_graph = fx.Graph()
        env = {}
        for node in cur_graph.nodes:
            if node.op == 'placeholder':
                new_node = new_graph.node_copy(node, lambda x: env[x])
                env[node] = new_node

        for node in cur_graph.nodes:
            if node.op != 'placeholder':
                new_node = new_graph.node_copy(node, lambda x: env[x])
                env[node] = new_node
        return new_graph

    def delta_debugging(cur_graph: fx.Graph, cur_inps):
        print("Strategy: Delta Debugging")
        assert graph_fails(cur_graph, cur_inps)
        starting_placeholders = len(_get_placeholders(cur_graph))
        num_nodes = len(cur_graph.nodes)
        gap = int(2**math.floor(math.log2(num_nodes)))
        while gap >= 1:
            for start_range in range(0, num_nodes, gap):
                is_removing = False
                new_graph = copy.deepcopy(cur_graph)
                new_inps = cur_inps[:]
                end_range = min(num_nodes, start_range + gap)
                for idx in range(start_range, end_range):
                    new_node = list(new_graph.nodes)[idx]
                    if new_node.op not in ['placeholder', 'output']:
                        is_removing = True
                        _convert_node_to_placeholder(new_node, new_inps)
                if not is_removing:
                    continue
                new_graph = consolidate_placeholders(new_graph)
                if graph_fails(new_graph, new_inps):
                    print(
                        f"SUCCESS: Removed ({start_range}:{end_range}] - Went from {starting_placeholders} "
                        f"placeholders to {len(_get_placeholders(new_graph))}"
                    )
                    return (new_graph, new_inps), True
            gap //= 2

        print("FAIL: Could not remove prefix")
        return (cur_graph, inps), False

    print("###################")
    print(f"Current size: {len(failing_graph.nodes)}")
    print("###################")
    while True:
        any_succeeded = False
        strategies = [
            remove_suffix, eliminate_dead_code, remove_unused_inputs,
            delta_debugging, eliminate_dead_code, remove_unused_inputs
        ]
        for strategy in strategies:
            out = strategy(copy.deepcopy(failing_graph), inps[:])
            (cur_graph, cur_inps), succeeded = out
            if succeeded:
                print()
                print("###################")
                print(f"Current size: {len(cur_graph.nodes)}")
                print("###################")
                failing_graph = cur_graph
                inps = cur_inps
                any_succeeded = True

        if not any_succeeded:
            break
    failing_fx = fx.GraphModule(fail_f, failing_graph)
    print(f"""
inps = {[(i.shape, i.dtype) for i in inps]}
inps = [torch.zeros(())] + [torch.ones(shape, dtype=dtype, device='cuda') for (shape, dtype) in inps]
{failing_fx.code}
f = torch.jit.script(forward)
with torch.jit.fuser("fuser2"):
  for _ in range(5):
    f(*inps)""")
    return failing_fx, inps


def check_nvfuser_subprocess(f, inps):
    f.to_folder("temp")
    with open("_temp.py", 'w') as fil:
        fil.write(f'''
import torch
from temp import FxModule
f = FxModule().cuda()
inps = {[(i.shape, i.dtype) for i in inps]}
inps = [torch.ones(shape, dtype=dtype, device='cuda') for shape, dtype in inps]
with torch.jit.fuser("fuser2"):
    nf = torch.jit.script(f)
    for _ in range(5):
        nf(*inps)
    ''')
    p = subprocess.Popen(["PYTORCH_NVFUSER_DISABLE_FALLBACK=1 python _temp.py"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    if p.returncode != 0:
        err = err.decode('utf-8')
        print(err)
        return True
    return False


def check_nvfuser_correctness_subprocess(f, inps):
    f.to_folder("temp")
    with open("_temp.py", 'w') as fil:
        fil.write(f'''
import torch
from temp import FxModule
f = FxModule().cuda()
inps = {[(i.shape, i.dtype) for i in inps]}
inps = [torch.randn(shape, dtype=dtype, device='cuda')
        if dtype.is_floating_point else torch.ones(shape, dtype=dtype, device='cuda')
        for shape, dtype in inps]

ref = f(*inps)
nv_f = torch.jit.script(f)
with torch.jit.fuser("fuser2"):
    for _ in range(5):
        res = nv_f(*inps)
for a, b in zip(ref, res):
    if not torch.allclose(a, b, atol=0.1):
        exit(1)
''')
    p = subprocess.Popen(["PYTORCH_NVFUSER_DISABLE_FALLBACK=1 python _temp.py"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    if p.returncode != 0:
        err = err.decode('utf-8')
        print(err)
        return True
    return False
