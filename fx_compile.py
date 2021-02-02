import time
import torch
import torch.nn as nn
import torch._C.te as te
import torch.fx as fx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import operator

class kernel_arena_scope(object):
    def __enter__(self):
        self.scope = te.KernelScope()

    def __exit__(self, typ, val, traceback):
        self.scope = None

class ShapeProp:
    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())
        self.shape_env = {}

    def propagate(self, *args):
        args_iter = iter(args)
        env : Dict[str, Node] = {}

        def load_arg(a):
            return torch.fx.node.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target : str):
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        for node in self.graph.nodes:
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = fetch_attr(node.target)
            elif node.op == 'call_function':
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'output':
                return self.shape_env

            if isinstance(result, torch.Tensor):
                self.shape_env[node.name] = (result.shape, result.dtype)
                node.shape = result.shape
                node.dtype = result.dtype

            env[node.name] = result

        return self.shape_env



############ New Stuff ##############

def binary_mapping(op):
    def f(a, b):
        return op(a, b)
    return f

decomposition_rules = {}
binary_decompositions = [
    (operator.matmul, torch.mm),
    (operator.add, torch.add),
    (operator.mul, torch.mul),
    (operator.sub, torch.sub),
    (operator.truediv, torch.div),
    (operator.eq, torch.eq),
    (operator.gt, torch.gt),
    (operator.ge, torch.ge),
    (operator.lt, torch.lt),
    (operator.le, torch.le),
    (operator.ne, torch.ne),
    (operator.and_, torch.bitwise_and)
]
for old, new in binary_decompositions:
    decomposition_rules[old] = binary_mapping(new)

def addmm_decompose(input, mat1, mat2):
    return torch.add(input , torch.mm(mat1, mat2))

decomposition_rules[torch.addmm] = addmm_decompose

def decompose(model: torch.nn.Module, shape_env) -> torch.nn.Module:
    model = fx.symbolic_trace(model)
    new_graph = fx.Graph()
    env = {}
    for node in model.graph.nodes:
        if node.op == 'call_function' and node.target in decomposition_rules:
            tracer = fx.proxy.GraphAppendingTracer(new_graph)
            proxy_args = [fx.Proxy(env[x.name], tracer=tracer) if isinstance(x, fx.Node) else x for x in node.args]
            new_node = decomposition_rules[node.target](*proxy_args).node
            env[node.name] = new_node
        else:
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node
    return fx.GraphModule(model, new_graph)

def get_dim_args(dims):
    dim_args = []
    for dim in dims:
        dim_args.append(te.DimArg(te.ExprHandle.int(dim), 'i' + str(len(dim_args))))
    return dim_args

def to_expr(x):
    if isinstance(x, int):
        return te.ExprHandle.int(x)
    elif isinstance(x, float):
        return te.ExprHandle.float(x)


lowering_functions = {}

def wrap_compute(f):
    def fn_lower(name, out_shape, inp_shapes, args):
        X = te.Compute(name, get_dim_args(out_shape), f(inp_shapes, args))
        return X
    return fn_lower

def gen_unary_nnc(op):
    def gen_op_nnc(inp_shapes, args):
        def f(*idxs):
            return op(args[0].load(idxs))
        return f
    return gen_op_nnc

unary_lowerings = [
    (torch.sin, lambda x: x.sin()),
    (torch.cos, lambda x: x.cos()),
    (torch.tan, lambda x: x.tan()),
    (torch.asin, lambda x: x.asin()),
    (torch.acos, lambda x: x.acos()),
    (torch.atan, lambda x: x.atan()),
    (torch.sinh, lambda x: x.sinh()),
    (torch.cosh, lambda x: x.cosh()),
    (torch.tanh, lambda x: x.tanh()),
    (torch.sigmoid, lambda x: x.sigmoid()),
    (torch.exp, lambda x: x.exp()),
    (torch.expm1, lambda x: x.expm1()),
    (torch.expm1, lambda x: x.expm1()),
    (torch.abs, lambda x: x.abs()),
    (torch.log, lambda x: x.log()),
    (torch.log2, lambda x: x.log2()),
    (torch.log10, lambda x: x.log10()),
    (torch.log1p, lambda x: x.log1p()),
    (torch.erf, lambda x: x.erf()),
    (torch.erfc, lambda x: x.erfc()),
    (torch.sqrt, lambda x: x.sqrt()),
    (torch.rsqrt, lambda x: x.rsqrt()),
    (torch.ceil, lambda x: x.ceil()),
    (torch.floor, lambda x: x.floor()),
    (torch.round, lambda x: x.round()),
    (torch.trunc, lambda x: x.trunc()),
    (torch.lgamma, lambda x: x.lgamma()),
]

for torch_op, nnc_fn in unary_lowerings:
    lowering_functions[torch_op] = wrap_compute(gen_unary_nnc(nnc_fn))

def gen_binary_nnc(op):
    def is_nnc_obj(x):
        return isinstance(x, te.Placeholder) or isinstance(x, te.Tensor)
    def gen_op_nnc(inp_shapes, args):
        if is_nnc_obj(args[0]) and is_nnc_obj(args[1]):
            A_shape, A_dtype = inp_shapes[0]
            B_shape, B_dtype = inp_shapes[1]
            A, B = args

            def index_or_broadcast(shape, *args):
                out = []
                for idx, arg in enumerate(args):
                    if idx >= len(shape): continue
                    if shape[idx] == 1:
                        out.append(to_expr(0))
                    else:
                        out.append(arg)
                return out

            def f(*idxs):
                return op(A.load(index_or_broadcast(A_shape, *idxs)), B.load(index_or_broadcast(B_shape, *idxs)))
            return f
        else:
            if is_nnc_obj(args[0]):
                def f(*idxs):
                    return op(args[0].load(idxs), to_expr(args[1]))
                return f
            else:
                def f(*idxs):
                    return op(to_expr(args[0]), args[1].load(idxs))
                return f

    return gen_op_nnc


binary_lowerings = [
(torch.add,lambda a, b: a+b),
(torch.mul,lambda a, b: a*b),
(torch.sub,lambda a, b: a-b),
(torch.div,lambda a, b: a/b),
(torch.eq,lambda a, b: a==b),
(torch.gt,lambda a, b: a>b),
(torch.lt,lambda a, b: a<b),
(torch.ge,lambda a, b: a>=b),
(torch.le,lambda a, b: a<=b),
(torch.max,lambda a, b: te.max(a, b)),
(torch.min,lambda a, b: te.min(a, b)),
]
for torch_op, nnc_fn in binary_lowerings:
    lowering_functions[torch_op] = wrap_compute(gen_binary_nnc(nnc_fn))

def clamp_lower(inp_shapes, args):
    def f(*idxs):
        val = args[0].load(idxs)
        return te.ifThenElse(val < to_expr(args[1]), to_expr(args[1]), te.ifThenElse(val > to_expr(args[2]), to_expr(args[2]), val))
    return f

lowering_functions[torch.clamp] = wrap_compute(clamp_lower)

def transpose_lower(name, out_shape, inp_shapes, args):
    idx_1, idx_2 = args[1], args[2]
    def transpose(shape):  # awful - mixes up new indexes and old indexes
        shape[idx_1], shape[idx_2] = shape[idx_2], shape[idx_1]
        return shape
    def f(*idxs):
        idxs = transpose(list(idxs))
        return args[0].load(idxs)
    return te.Compute(name, get_dim_args(out_shape), f)

def flatten_lower(name, out_shape, inp_shapes, args):
    A, start_dim, end_dim = args
    shape = list(inp_shapes[0][0])
    flattened_region = shape[start_dim:end_dim+1]
    def prod(x):
        t = 1
        for i in x:
            t *= i
        return t
    def get_orig_idxs(i):
        idxs = []
        total = prod(flattened_region)
        for dim in flattened_region:
            total //= dim
            idxs.append(i / to_expr(total))
            i = i % to_expr(total)
        return idxs
    def f(*idxs):
        idxs = list(idxs)
        idxs = idxs[:start_dim] + get_orig_idxs(idxs[start_dim]) + idxs[start_dim+1:]
        return A.load(idxs)
    return te.Compute(name, get_dim_args(out_shape), f)

def cat_lower(name, out_shape, inp_shapes, args):
    tensors = args[0]
    dim = args[1]
    lengths = [i[0][dim] for i in inp_shapes[0]]
    def f(*idxs):
        idxs = list(idxs)
        sm = lengths[0]
        load = tensors[0].load(idxs)
        for length, tensor in list(zip(lengths, tensors))[1:]:
            new_idxs = idxs[:]
            new_idxs[dim] -= to_expr(sm)
            load = te.ifThenElse(idxs[dim] < to_expr(sm), load, tensor.load(new_idxs))
        return load
    return te.Compute(name, get_dim_args(out_shape), f)

lowering_functions[torch.transpose] = transpose_lower
lowering_functions[torch.flatten] = flatten_lower
lowering_functions[torch.cat] = cat_lower

def bmm_lower(name, out_shape, inp_shapes, args):
    M1 = args[0]
    M2 = args[1]
    B, N, M = inp_shapes[0][0]
    P = inp_shapes[1][0][2]

    def f(b, n, p, m):
        return M1.load([b, n, m]) * M2.load([b, m, p])
    mm = te.Compute('mm', get_dim_args([B,N,P,M]), f)
    return te.SumReduce(name, get_dim_args([B, N, P]), mm, get_dim_args([M]))


def mm_lower(name, out_shape, inp_shapes, args):
    M1 = args[0]
    M2 = args[1]
    N, M = inp_shapes[0][0]
    P = inp_shapes[1][0][1]

    def f(n, p, m):
        return M1.load([n, m]) * M2.load([m, p])
    mm = te.Compute('mm', get_dim_args([N,P,M]), f)
    return te.SumReduce(name, get_dim_args([N, P]), mm, get_dim_args([M]))

lowering_functions[torch.bmm] = bmm_lower
lowering_functions[torch.mm] = mm_lower


def lower_function(name, op, shape_env, nnc_args, args):
    inp_shapes = fx.node.map_aggregate(args, lambda arg: shape_env[arg.name] if isinstance(arg, fx.Node) else None)
    return lowering_functions[op](name, shape_env[name][0], inp_shapes, nnc_args)

def nnc_compile(model: torch.nn.Module, shape_specialize, shape_env) -> torch.nn.Module:
    fx_model = fx.symbolic_trace(model)
    env = {}
    def get_te_shapes(name):
        return [te.ExprHandle.int(i) for i in shape_env[name][0]]

    def get_nnc_type(dtype):
        if dtype == torch.float:
            return torch._C.te.Dtype.Float
        elif dtype == torch.long:
            return torch._C.te.Dtype.Long
        else:
            raise RuntimeError("nyi")

    def get_te_type(name):
        return get_nnc_type(shape_env[name][1])

    def gen_compute(args):
        te_args = [env[arg.name] for arg in args]

    def lookup_env(l):
        return fx.node.map_aggregate(l, lambda x: env[x.name] if isinstance(x, fx.Node) else x)

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
    for node in fx_model.graph.nodes:
        if node.op == 'placeholder':
            shapes = get_te_shapes(node.name)
            env[node.name] = te.Placeholder(node.name, get_te_type(node.name), shapes)
            inputs.append(env[node.name])
        elif node.op == 'call_function':
            result = lower_function(node.name, node.target, shape_env, lookup_env(node.args), node.args)
            env[node.name] = result
        elif node.op == 'output':
            outs = list(lookup_env(node.args))
        elif node.op == 'get_attr':
            module_attrs.append(node)
            env[node.name] = te.Placeholder(node.name, get_te_type(node.name), shapes)

    loopnest = te.LoopNest(outs)
    loopnest.prepare_for_codegen()
    stmt = te.simplify(loopnest.root_stmt())
    cg = te.construct_codegen('llvm', stmt, [te.BufferArg(x) for x in [env[i.name] for i in module_attrs] + inputs + outs])
    def f(inps):
        module_stuff = [fetch_attr(i.target) for i in module_attrs]
        cg.call(module_stuff + list(inps))
    return f

class DeepAndWide(torch.nn.Module):
    def __init__(self, num_features=50):
        super(DeepAndWide, self).__init__()
        self.mu = torch.nn.Parameter(torch.randn(1, num_features))
        self.sigma = torch.nn.Parameter(torch.randn(1, num_features))
        self.fc_w = torch.nn.Parameter(torch.randn(1, num_features + 1))
        self.fc_b = torch.nn.Parameter(torch.randn(1))

    def forward(self, ad_emb_packed, user_emb, wide):
        wide_offset = wide + self.mu
        wide_normalized = wide_offset * self.sigma
        wide_preproc = torch.clamp(wide_normalized, 0., 10.)
        user_emb_t = torch.transpose(user_emb, 1, 2)
        dp_unflatten = torch.bmm(ad_emb_packed, user_emb_t)
        dp = torch.flatten(dp_unflatten, 1, -1)
        inp = torch.cat([dp, wide_preproc], 1)
        t1 = torch.transpose(self.fc_w, 1, 0)
        fc1 = torch.addmm(self.fc_b, inp, t1)
        return fc1



with kernel_arena_scope():
    with torch.no_grad():
        num_features = 100
        mod = DeepAndWide(num_features)

        # Phabricate sample inputs
        batch_size = 1
        embedding_size = 100
        ad_emb_packed = torch.randn(batch_size, 1, embedding_size)
        user_emb = torch.randn(batch_size, 1, embedding_size)
        wide = torch.randn(batch_size, num_features)
        inps = (ad_emb_packed, user_emb, wide)
        out = torch.empty(batch_size, 1)

        shape_env = ShapeProp(fx.symbolic_trace(mod)).propagate(*inps)
        mod = decompose(mod, shape_env)
        shape_env = ShapeProp(fx.symbolic_trace(mod)).propagate(*inps)
        print(shape_env)
        cg = nnc_compile(mod, True, shape_env)

        iters = 1000

        for _ in range(10):
            cg([ad_emb_packed, user_emb,wide, out])
        begin = time.time()
        for _ in range(iters):
            cg([ad_emb_packed, user_emb,wide, out])
        print("NNC time: ", time.time()-begin)

        mod_jit = torch.jit.script(DeepAndWide(num_features))
        for _ in range(10):
            mod_jit(ad_emb_packed, user_emb,wide)
        begin = time.time()
        for _ in range(iters):
            mod_jit(ad_emb_packed, user_emb,wide)
        print("PyTorch time", time.time()-begin)

        static_runtime = torch._C._jit_to_static_runtime(mod_jit._c)
        for _ in range(10):
            static_runtime.run([ad_emb_packed, user_emb,wide])
        begin = time.time()
        for _ in range(iters):
            static_runtime.run([ad_emb_packed, user_emb,wide])
        print("Static Runtime time", time.time()-begin)

        m = mod
        jax_inps = [m.mu, m.sigma, m.fc_w, m.fc_b, ad_emb_packed, user_emb, wide]
        jax_inps = [jnp.array(i.numpy()) for i in jax_inps]
        jax_fn = jit(jax_fn)
        # mod = mod
        for _ in range(10):
            jax_fn(*jax_inps)
        begin = time.time()
        for _ in range(iters):
            jax_fn(*jax_inps)
        print("Jax Time: ", time.time()-begin)
        print("Sums:", out.sum(), mod(*inps).sum())
