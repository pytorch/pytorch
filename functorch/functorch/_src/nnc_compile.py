# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._C._te as te
import torch.fx as fx
import torch.utils._pytree as pytree
from torch.fx import map_arg
from torch.fx.passes.shape_prop import ShapeProp, TensorMetadata
import operator
import functools

def truncate(model, k):
    model = fx.symbolic_trace(model)
    new_graph= fx.Graph()
    env = {}

    cnt = 0
    for node in list(model.graph.nodes):
        new_node = new_graph.node_copy(node, lambda x: env[x.name])
        env[node.name] = new_node
        cnt += 1
        if cnt == k:
            new_graph.output(env[node.name])
            break

    return fx.GraphModule(model, new_graph)

# NNC Lowering Pass
def remove_args(model: torch.nn.Module, args):
    fx_model = fx.symbolic_trace(model)
    for node in fx_model.graph.nodes:
        if node.op == 'placeholder' and node.target in args:
            assert(len(node.users) == 0)
            fx_model.graph.erase_node(node)
    fx_model.recompile()
    return fx_model

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
    elif dtype == torch.int32:
        return te.Dtype.Int
    elif dtype == torch.long:
        return te.Dtype.Long
    elif dtype == torch.float64:
        return te.Dtype.Double
    elif dtype == torch.bool:
        return te.Dtype.Bool
    else:
        raise RuntimeError(f"type nyi {dtype}")


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

def zeros_like_lower(name, out_shape, inp_shapes, args):
    def f(*idxs):
        return to_expr(0.0)
    res = te.Compute(name, get_dim_args(out_shape), f)
    return res

def full_like_lower(name, out_shape, inp_shapes, args):
    def f(*idxs):
        return to_expr(args[1])
    res = te.Compute(name, get_dim_args(out_shape), f)
    return res


def prod(x, start=1):
    t = start
    for i in x: t *= i
    return t

def encode_idxs(shape, idxs):
    assert(len(shape) == len(idxs))
    cur = 1
    out = to_expr(0)
    for dim, idx in reversed(list(zip(shape, idxs))):
        out += to_expr(cur) * idx
        cur *= dim
    return out

def reshape_lower(name, out_shape, inp_shapes, args):
    X, shape = args
    start_shape = list(inp_shapes[0][0])
    end_shape = out_shape
    def get_orig_idxs(idxs):
        absolute_new = encode_idxs(end_shape, idxs)
        new_idxs = []
        total_old = prod(start_shape)
        for dim in start_shape:
            total_old //= dim
            new_idxs.append(absolute_new / to_expr(total_old))
            absolute_new %= to_expr(total_old)
        return new_idxs

    def f(*idxs):
        idxs = list(idxs)
        orig_idxs = get_orig_idxs(idxs)
        return X.load(orig_idxs)
    return te.Compute(name, get_dim_args(out_shape), f)

# def select_lower(name, out_shape, inp_shapes, args):
#     A = args[0]
#     dim = args[1]
#     idx = args[2]
#     import pdb; pdb.set_trace()
#     def f(*idxs):
#         # idxs = list(idxs)
#         idxs.insert(dim, to_expr(idx))
#         # idxs = [to_expr(0)]
#         return A.load(idxs)
#     res = te.Compute(name, get_dim_args(out_shape), f)
#     return res

def dot_lower(name, out_shape, inp_shapes, args):
    mul_te = te.lower('aten::mul', list(args), get_te_shapes(inp_shapes[0][0]), get_nnc_type(inp_shapes[0][1]))
    res = te.lower('aten::sum', [mul_te.buf()], get_te_shapes(out_shape), get_nnc_type(inp_shapes[0][1]))
    return (res.buf(), [mul_te.stmt(), res.stmt()])

def mv_lower(name, out_shape, inp_shapes, args):
    A = args[0]
    B = args[1]
    N, M = inp_shapes[0][0]

    # def f(n, m):
    #     return A.load([n, m]) * B.load([m])
    # mm = te.Compute('mm', get_dim_args([N,M]), f)
    # out = te.Reduce(name, get_dim_args([N]), te.Sum(), mm, get_dim_args([M]))
    # return out.buf(), [mm.stmt(), out.stmt()]
    C = torch._C._te.BufHandle('C', get_te_shapes([N]), get_nnc_type(inp_shapes[0][1]))
    s = torch._C._te.ExternalCall(C, "nnc_aten_mv", [A, B], [])
    return C, [s]

def digamma_lower(name, out_shape, inp_shapes, args):
    out = te.BufHandle('out', get_te_shapes(out_shape), get_nnc_type(inp_shapes[0][1]))
    s = te.ExternalCall(out, "nnc_aten_digamma", [args[0]], [])
    return out, [s]

def ger_lower(name, out_shape, inp_shapes, args):
    A = args[0]
    B = args[1]
    A_len = inp_shapes[0][0][0]
    B_len = inp_shapes[1][0][0]
    A_squeeze = te.lower('aten::unsqueeze', [args[0], 1], get_te_shapes([A_len, 1]), get_nnc_type(inp_shapes[0][1]))
    B_squeeze = te.lower('aten::unsqueeze', [args[1], 0], get_te_shapes([1, B_len]), get_nnc_type(inp_shapes[1][1]))
    out = te.lower('aten::mul', [A_squeeze.buf(), B_squeeze.buf()], get_te_shapes(out_shape), get_nnc_type(inp_shapes[0][1]))
    return out.buf(), [A_squeeze.stmt(), B_squeeze.stmt(), out.stmt()]

def triangular_solve_lower(name, out_shape, inp_shapes, args):
    A = args[0]
    B = args[1]
    C = torch._C._te.BufHandle('C', get_te_shapes(out_shape[0]), get_nnc_type(inp_shapes[0][1]))
    s = torch._C._te.ExternalCall(C, "nnc_aten_triangular_solve", [A, B], [to_expr(args[2]), to_expr(args[3]), to_expr(args[4])])
    return (C, None), [s]

def binary_cross_entropy_lower(name, out_shape, inp_shapes, args):
    self_ = args[0]
    target = args[1]
    if args[2] != None or args[3] == 2:
        raise RuntimeError(f"weight={args[2]} and reduction={args[3]} not supported")
    def f(*idxs):
        return to_expr(0.0) - (self_.load(idxs).log() * target.load(idxs) + (to_expr(1.0) - target.load(idxs)) * (to_expr(1.0) - self_.load(idxs)).log())
    val = te.Compute(name, get_dim_args(inp_shapes[0][0]), f)
    if args[3] == 0:
        return val.buf(), [val.stmt()]
    mean_te = te.lower('aten::mean', [val.buf()], get_te_shapes(out_shape), get_nnc_type(inp_shapes[0][1]))
    return mean_te.buf(), [val.stmt(), mean_te.stmt()]


def binary_cross_entropy_with_logits_lower(name, out_shape, inp_shapes, args):
    pred = te.lower('aten::sigmoid', [args[0]], get_te_shapes(inp_shapes[0][0]), get_nnc_type(inp_shapes[0][1]))
    loss_buf, loss_stmts = binary_cross_entropy_lower('binary_cross_entropy', out_shape, list(inp_shapes) + [None], [pred, args[1], args[3], args[4]])
    return loss_buf, [pred.stmt()] + loss_stmts

def detach_lower(name, out_shape, inp_shapes, args):
    return args[0], []

# def clone_lower(name, out_shape, inp_shapes, args):
#     return args[0], []

lowering_functions[torch.ops.aten.full_like] = full_like_lower
lowering_functions[torch.ops.aten.zeros_like] = zeros_like_lower
lowering_functions[torch.ops.aten.ones_like] = ones_like_lower
lowering_functions[torch.ops.aten.dot] = dot_lower
# lowering_functions[torch.ops.aten.select] = select_lower
lowering_functions[torch.ops.aten.digamma] = digamma_lower
lowering_functions[torch.ops.aten.mv] = mv_lower
lowering_functions[torch.ops.aten.ger] = ger_lower
lowering_functions[torch.ops.aten.reshape] = reshape_lower
lowering_functions[torch.ops.aten.view] = reshape_lower
lowering_functions[torch.ops.aten.triangular_solve] = triangular_solve_lower
lowering_functions[torch.ops.aten.binary_cross_entropy] = binary_cross_entropy_lower
lowering_functions[torch.ops.aten.binary_cross_entropy_with_logits] = binary_cross_entropy_with_logits_lower
lowering_functions[torch.ops.aten.detach] = detach_lower
# lowering_functions[torch.ops.aten.clone] = clone_lower



func_to_aten = {
    operator.add: torch.ops.aten.add,
    operator.mul: torch.ops.aten.mul,
    torch.mul: torch.ops.aten.mul,
    torch.sin: torch.ops.aten.sin,
    torch.cos: torch.ops.aten.cos,
}


def process_shape(x):
    if len(x) == 0:
        return torch.Size([1])
    return x

def map_node_meta(f, node_meta):
    if isinstance(node_meta, TensorMetadata):
        return f(node_meta)
    elif isinstance(node_meta, tuple):
        return tuple([map_node_meta(f, i) for i in node_meta])
    elif isinstance(node_meta, list):
        return list([map_node_meta(f, i) for i in node_meta])
    return f(node_meta)

def lower_function(node, op, nnc_args, args):
    inp_shapes = fx.node.map_aggregate(args, lambda arg: (process_shape(arg.meta['tensor_meta'].shape), arg.meta['tensor_meta'].dtype) if isinstance(arg, fx.Node) and 'tensor_meta' in arg.meta else None)
    out_shape = map_node_meta(lambda x: process_shape(x.shape), node.meta['tensor_meta'])
    if op in lowering_functions:
        out = lowering_functions[op](node.name, out_shape, inp_shapes, nnc_args)
    else:
        out_shape = pytree.tree_map(lambda x: get_te_shapes(x), out_shape)
        aten_str = f'aten::{op.__name__}'
        out_type = map_node_meta(lambda x: get_nnc_type(x.dtype), node.meta['tensor_meta'])
        out = te.lower(aten_str, list(nnc_args), out_shape, out_type)
    if isinstance(out, te.Tensor):
        return out.buf(), [out.stmt()]
    else:
        return out[0], out[1]

# This will not work properly in the presence of aliasing/views
def remove_inplace(fx_model: fx.GraphModule) -> torch.nn.Module:
    new_map = {}
    for node in fx_model.graph.nodes:
        node.args = map_arg(node.args, lambda x: new_map[x] if x in new_map else x)
        if node.op == 'call_function' and node.target == torch.ops.aten.mul_:
            node.target = torch.ops.aten.mul
            new_map[node.args[0]] = node
    return fx_model

def nnc_compile(fx_model: fx.GraphModule, example_inputs, get_loopnest = False) -> torch.nn.Module:
    """
    nnc_compile(model, example_inputs) returns a function with the same args
    as `model.forward`, with an extra argument corresponding to where the
    output is stored. This function takes the inputs (which must be PyTorch
    tensors with the same shapes as example_inputs), and passes them to an
    NNC executor.
    """
    t = fx_model.graph.flatten_inps(*example_inputs)
    ShapeProp(fx_model).propagate(*fx_model.graph.flatten_inps(*example_inputs))
    fx_model = remove_inplace(fx_model)

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
    attr_bufs = []
    compute_stmts = []
    inputs_or_attrs = set()
    for node in fx_model.graph.nodes:
        if node.op == 'placeholder':
            # We simply map the input placeholder to a `te.Placeholder`, which
            # also represents an input to the NNC computation.
            if 'tensor_meta' not in node.meta:
                continue
            shapes = get_te_shapes(node.meta['tensor_meta'].shape)
            placeholder = te.BufHandle(node.name, shapes, get_te_type(node))
            env[node.name] = placeholder
            inputs.append(placeholder)
            inputs_or_attrs.add(placeholder)
        elif node.op == 'call_function':
            if node.target == operator.getitem:
                iterable = lookup_env(node.args)[0]
                env[node.name] = iterable[node.args[1]]
                continue
            # This does the bulk of the work - we call `lower_function`, which
            # returns a `te.ExprHandle` (the output of a NNC computation), and
            # put it in our environment.
            if 'tensor_meta' in node.meta:
                # todo: fix kwargs handling
                # if node.kwargs:
                #     raise RuntimeError("kwargs nyi")
                buf, stmt = lower_function(node, node.target, lookup_env(node.args), node.args)
                # if isinstance(stmt, list)
                compute_stmts.extend(stmt)
                env[node.name] = buf
            elif node.target == getattr or node.target == operator.getitem:
                # todo: handle non-tensor computations correctly
                continue
        elif node.op == 'output':
            args = node.args
            args = pytree.tree_map(lambda x: list(x) if isinstance(x, fx.immutable_collections.immutable_list) else x, args)
            flat_args, _ = pytree.tree_flatten(list(args))
            te_args = lookup_env(flat_args)
            outs = (list(te_args), [(i.meta['tensor_meta'].shape, i.meta['tensor_meta'].dtype) for i in flat_args])
        elif node.op == 'get_attr':
            # As NNC doesn't have any concept of state, we pull out the module
            # attributes and pass them in as inputs to NNC.
            module_attrs.append(node)
            shapes = get_te_shapes(process_shape(node.meta['tensor_meta'].shape))
            placeholder = te.BufHandle(node.name, shapes,  get_te_type(node))
            env[node.name] = placeholder
            attr_bufs.append(placeholder)
            inputs_or_attrs.add(placeholder)
        else:
            print(node.op, node.target)
            raise RuntimeError("not yet implemented")


    if len(compute_stmts) == 0:
        raise RuntimeError("Doesn't support compiling empty")

    outs = [list(i) for i in zip(*list(outs))]

    buf_outs = [i for i, _ in outs]
    loopnest = te.LoopNest(te.Stmt(compute_stmts), buf_outs)
    if get_loopnest:
        return loopnest
    # loopnest.inline_intermediate_bufs(True)
    loopnest.simplify()
    loopnest.prepare_for_codegen()
    stmt = te.simplify(loopnest.root_stmt())
    cg = te.construct_codegen('llvm', stmt, [te.BufferArg(x) for x in [env[i.name] for i in module_attrs] + inputs + buf_outs])

    module_stuff = [fetch_attr(i.target).contiguous().data for i in module_attrs]

    ph_to_inp_map = {}
    for idx, _ in enumerate(outs):
        if outs[idx][0] in inputs:
            ph_to_inp_map[outs[idx][0]] = len(module_attrs) + inputs.index(outs[idx][0])
        elif outs[idx][0] in attr_bufs:
            ph_to_inp_map[outs[idx][0]] = attr_bufs.index(outs[idx][0])

    def get_outs(inps):
        return [inps[ph_to_inp_map[buf]] if buf in inputs_or_attrs else torch.empty(shape, dtype=dtype) for buf, (shape,dtype) in outs]

    def f(*inps):
        inps = fx_model.graph.flatten_inps(*inps)
        module_inps = module_stuff + list(inps)
        results = get_outs(module_inps)
        full_inps = module_inps + results
        cg.call(full_inps)
        if len(results) == 1:
            results = fx_model.graph.unflatten_outs(results[0])
        else:
            results = fx_model.graph.unflatten_outs(results)
        return results
    return f

def make_nnc(f):
    @functools.wraps(f)
    def wrapped(*args):
        fx_model = make_fx(f)(*args)  # noqa: F821
        fx_model.graph.lint()
        compiled_f = nnc_compile(fx_model, args, get_loopnest=True)
        return compiled_f

    return wrapped

def get_ops(fx_model: fx.GraphModule):
    vals = set()
    for node in fx_model.graph.nodes:
        if node.op == 'call_function':
            vals.add(node.target.__name__)
    return vals



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
