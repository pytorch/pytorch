import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._C._te as te
import torch.fx as fx
from torch.fx import map_arg
from torch.fx.passes.shape_prop import ShapeProp
import operator
scope = te.KernelScope()

# NNC Lowering Pass

class kernel_arena_scope(object):
    def __enter__(self):
        self.scope = te.KernelScope()

    def __exit__(self, typ, val, traceback):
        self.scope = None

def get_dim_args(dims):
    dim_args = []
    for dim in dims:
        dim_args.append(te.DimArg(te.ExprHandle.int(dim), 'i' + str(len(dim_args))))
    return dim_args

def get_te_shapes(shape):
    return [te.ExprHandle.int(i) for i in shape]

def to_expr(x):
    if isinstance(x, int):
        return te.ExprHandle.int(x)
    elif isinstance(x, float):
        return te.ExprHandle.float(x)
    else:
        raise RuntimeError(f"type {type(x)} not supported")

def get_nnc_type(dtype):
    if dtype == torch.float:
        return te.Dtype.Float
    elif dtype == torch.long:
        return te.Dtype.Long
    else:
        raise RuntimeError("nyi")


lowering_functions = { }
def index_or_broadcast(shape, *args):
    out = []
    for idx, arg in enumerate(args):
        if idx >= len(shape): continue
        if shape[idx] == 1:
            out.append(to_expr(0))
        else:
            out.append(arg)
    return out

def ones_like_lower(name, out_shape, inp_shapes, args):
    def f(*idxs):
        return to_expr(1.0)
    res = te.Compute(name, get_dim_args(out_shape), f)
    return res

def expand_lower(name, out_shape, inp_shapes, args):
    inp_shape = inp_shapes[0][0]
    A = args[0]
    def f(*idxs):
        return A.load(index_or_broadcast(inp_shape, args))
    return te.Compute(name, get_dim_args(out_shape), f)

def mm_lower(name, out_shape, inp_shapes, args):
    M1 = args[0]
    M2 = args[1]
    N, M = inp_shapes[0][0]
    P = inp_shapes[1][0][1]

    def f(n, p, m):
        return M1.load([n, m]) * M2.load([m, p])
    mm = te.Compute('mm', get_dim_args([N,P,M]), f)
    out = te.Reduce(name, get_dim_args([N, P]), te.Sum(), mm, get_dim_args([M]))
    return out.buf(), [mm.stmt(), out.stmt()]
    # C = torch._C._te.BufHandle('C', get_te_shapes([N, P]), get_nnc_type(torch.float))
    # s = torch._C._te.ExternalCall(C, "nnc_aten_matmul", [M1, M2], [])
    # return C, [s]

def transpose_lower(name, out_shape, inp_shapes, args):
    if len(args) == 1:
        idx_1, idx_2 = 0, 1
    else:
        idx_1, idx_2 = args[1], args[2]
    def transpose(shape):
        shape[idx_1], shape[idx_2] = shape[idx_2], shape[idx_1]
        return shape
    def f(*idxs):
        idxs = transpose(list(idxs))
        return args[0].load(idxs)
    return te.Compute(name, get_dim_args(out_shape), f)


lowering_functions[torch.ops.aten.ones_like] = ones_like_lower
lowering_functions[torch.ops.aten.expand] = expand_lower
lowering_functions[torch.ops.aten.mm] = mm_lower
lowering_functions[torch.ops.aten.t] = transpose_lower

func_to_aten = {
    operator.getitem: torch.ops.aten.slice,
    operator.add: torch.ops.aten.add,
    operator.mul: torch.ops.aten.mul,
    torch.mul: torch.ops.aten.mul,
    torch.sin: torch.ops.aten.sin,
    torch.cos: torch.ops.aten.cos,
}


def process_shape(x):
    if len(x) == 0:
        return [1]
    return x
def lower_function(node, op, nnc_args, args):
    inp_shapes = fx.node.map_aggregate(args, lambda arg: (process_shape(arg.meta['tensor_meta'].shape), arg.meta['tensor_meta'].dtype) if isinstance(arg, fx.Node) and 'tensor_meta' in arg.meta else None)
    if op in lowering_functions:
        out = lowering_functions[op](node.name, process_shape(node.meta['tensor_meta'].shape), inp_shapes, nnc_args)
    else:
        if op in func_to_aten:
            op = func_to_aten[op]
        aten_str = f'aten::{op.__name__}'
        print(aten_str, nnc_args)
        out = te.lower(aten_str, list(nnc_args), get_te_shapes(node.meta['tensor_meta'].shape), get_nnc_type(torch.float))
    if isinstance(out, te.Tensor):
        return out.buf(), [out.stmt()]
    else:
        return out[0], out[1]

def nnc_compile(model: torch.nn.Module, example_inputs) -> torch.nn.Module:
    """
    nnc_compile(model, example_inputs) returns a function with the same args
    as `model.forward`, with an extra argument corresponding to where the
    output is stored. This function takes the inputs (which must be PyTorch
    tensors with the same shapes as example_inputs), and passes them to an
    NNC executor.
    """
    fx_model = fx.symbolic_trace(model)
    ShapeProp(fx_model).propagate(*example_inputs)

    # This env maps from nodes to `te.ExprHandle`, which represent the output
    # of an NNC computation.
    env = {}


    def get_te_type(node):
        return get_nnc_type(node.meta['tensor_meta'].dtype)

    def gen_compute(args):
        te_args = [env[arg.name] for arg in args]

    def lookup_env(l):
        res = fx.node.map_aggregate(l, lambda x: env[x.name] if isinstance(x, fx.Node) else x)
        return res

    def fetch_attr(target : str):
        target_atoms = target.split('.')
        attr_itr = fx_model
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    outs = None
    inputs = []
    module_attrs = []
    compute_stmts = []
    for node in fx_model.graph.nodes:
        if node.op == 'placeholder':
            # We simply map the input placeholder to a `te.Placeholder`, which
            # also represents an input to the NNC computation.
            shapes = get_te_shapes(node.meta['tensor_meta'].shape)
            placeholder = te.Placeholder(node.name, get_te_type(node), shapes)
            env[node.name] = placeholder.data()
            inputs.append(placeholder)
        elif node.op == 'call_function':
            # This does the bulk of the work - we call `lower_function`, which
            # returns a `te.ExprHandle` (the output of a NNC computation), and
            # put it in our environment.
            if 'tensor_meta' in node.meta:
                # todo: fix kwargs handling
                if node.kwargs:
                    raise RuntimeError("kwargs nyi")
                buf, stmt = lower_function(node, node.target, lookup_env(node.args), node.args)
                # if isinstance(stmt, list)
                compute_stmts.extend(stmt)
                env[node.name] = buf
            elif node.target == getattr or node.target == operator.getitem:
                # todo: handle non-tensor computations correctly
                continue
        elif node.op == 'output':
            args = node.args
            if not isinstance(args, tuple):
                args = (args,)
            if isinstance(args[0], tuple):
                args = args[0]
            te_args = lookup_env(args)
            outs = (list(te_args), [i.meta['tensor_meta'].shape for i in args])
        elif node.op == 'get_attr':
            # As NNC doesn't have any concept of state, we pull out the module
            # attributes and pass them in as inputs to NNC.
            module_attrs.append(node)
            shapes = get_te_shapes(node.meta['tensor_meta'].shape)
            placeholder = te.Placeholder(node.name, get_te_type(node), shapes)
            env[node.name] = placeholder.data()
        else:
            print(node.op, node.target)
            raise RuntimeError("not yet implemented")


    loopnest = te.LoopNest(te.Stmt(compute_stmts), outs[0])
    # loopnest.inline_intermediate_bufs(True)
    loopnest.simplify()
    # print(loopnest)
    loopnest.prepare_for_codegen()
    stmt = te.simplify(loopnest.root_stmt())
    cg = te.construct_codegen('llvm', stmt, [te.BufferArg(x) for x in [env[i.name] for i in module_attrs] + inputs + outs[0]])
    alloc_results = [torch.empty(i) for i in outs[1]]
    def f(*inps, out_tensors=None):
        # begin = time.time()
        if out_tensors is None:
            results = alloc_results
        else:
            results = out_tensors
        if module_attrs:
            module_stuff = [fetch_attr(i.target).data for i in module_attrs]
        else:
            module_stuff = []
        # begin2 = time.time()
        cg.call(module_stuff + list(inps) + results)
        # print("inner", time.time()-begin2)
        if out_tensors is None:
            # print("outer", time.time()-begin)
            if len(results) == 1:
                return results[0]
            return results
    return f


################################
# Example usage and Benchmarking
################################

def bench(f, warmup=3, iters=1000):
    for _ in range(warmup):
        f()
    begin = time.time()
    for _ in range(iters):
        f()
    print(time.time()-begin)
if __name__ == '__main__':
    def f(a, b):
        return (torch.cos(a)* torch.sin(b))[:2000]

    mod = fx.symbolic_trace(f)
    inps = (torch.randn(5000), torch.randn(5000))
    ShapeProp(mod).propagate(*inps)
    cg = nnc_compile(mod, inps)
    bench(lambda: cg(*inps))
    bench(lambda: f(*inps))
    exit(0)