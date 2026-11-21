# torch.vm

```{eval-rst}
.. automodule:: torch.vm
```

## Graphs

The default passes make each tensor operation of a block a direct call of its
kernel. `torch.vm.passes.Graphs()` looks one level up: in every block, once
the other passes are done with it, it finds the runs of such calls together
with what sits between them (reads of globals and attributes, constants,
arithmetic on sizes) and takes each run down as a graph. A run ends in front
of everything else: a call of a Python function, an operation that writes to
one of its arguments or draws random numbers, a number read from a tensor. A
run of fewer than `min_ops` operations (4 unless the pass is made with
another number) is no graph.

With `Graphs()` in the list the function runs as it runs without it. The
engine runs a graph as it runs the rest of the block: the same kernels, on
the same values, in the same order, on the autograd tape of eager mode. The
results are those of eager mode bit for bit, and `torch.vm.is_exact` is true
for such a list. What the pass adds is what `torch.vm.regions` reports:

```python
import torch

P = torch.vm.passes


def f(x, w):
    return torch.tanh(x @ w).sum() + (x * 2).mean()


g = torch.vm(f, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs()))
x, w = torch.randn(4, 8), torch.randn(8, 8, requires_grad=True)
for _ in range(3):
    out = g(x, w)
torch.equal(out, f(x, w))  # True
type(out.grad_fn).__name__  # 'AddBackward0': the last operation, on eager's tape
torch.vm.is_exact((*torch.vm.DEFAULT_PASSES, P.Graphs()))  # True
torch.vm.regions(g)
# [{'where': 'f (block 0)[2]', 'state': 'built', 'ops': 6,
#   'operators': ['aten::matmul', 'aten::tanh', 'aten::sum',
#   'aten::mul.Tensor', 'aten::mean', 'aten::add.Tensor'], 'inputs': 3,
#   'outputs': 1, 'ends': [(0, 11)], 'why': 'the function returns',
#   'variants': 0, 'hits': 0, 'misses': 0, 'exports': 0, 'version': 1,
#   'forms': []}]
```

`where` names the function, the block, and the instruction of the first
operation of the graph (`g.dump()` shows them). The state
`"built"` says that the engine runs the graph itself. `inputs` and `outputs`
are how many values the graph takes from the code in front of it (here `x`,
`w` and the function `torch.tanh`) and how many the code behind it reads.
`ends` is the block and the instruction the function goes on at behind the
graph, and `why` what ends the graph there. What was left out of the graphs
of a block that has tensor work on both sides of it is listed too, with the
state `"vm"` and the reason:

```python
def two(x):
    y = x.sin().cos() * 2 + x
    print(end="")
    y.add_(1)
    return (y.tanh() * y).exp().sum() - y.mean()


g = torch.vm(two, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs()))
for _ in range(3):
    g(torch.randn(6))
for found in torch.vm.regions(g):
    print(found["where"], found["state"], found.get("ops"), found.get("reason"))
# two (block 0)[1] built 4 None
# two (block 0)[15] built 6 None
# two (block 0)[10] vm None a call that runs no kernel
# two (block 0)[13] vm None aten::add_.Tensor writes to an argument
```

A graph is a part of one block of one function. It does not go on through a
call of a Python function, a branch or a loop: the function that is called
has graphs of its own, in its own blocks. A model made of modules therefore
has its operations spread over the `forward` of each module, one or two to a
block, and none of those runs is a graph unless `min_ops` is lowered:

```python
model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))


def infer(model, x):
    return model(x)


g = torch.vm(infer, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(min_ops=1)))
for _ in range(3):
    g(model, torch.randn(4, 8))
sorted((found["where"], found["operators"]) for found in torch.vm.regions(g))
# [('Linear.forward (block 0)[4]', ['aten::linear']), ('relu (block 4)[2]', ['aten::relu'])]
```

### Passes over graphs

A graph leaves the engine only for a pass over graphs: a function of two
arguments, `pass_(gm, input_types) -> gm`, that stands behind `Graphs()` in
the list. (With such a pass in the list and no `Graphs()`, the pass is taken
to be there, in front of the first of them. A pass over graphs in front of
`Graphs()` is a `ValueError`.) The engine then puts an instruction in front
of each graph, which `dump()` shows as `graph f (block 0)[2:11]: 6
operations`, and the graph is `"waiting"`.

The next call that comes to the graph runs its operations as before, and the
engine takes down what they did: the kernel call each operation made, and the
values the operations took from outside the graph. From that one run it
describes the graph to Python, where it becomes a `torch.fx.GraphModule`: a
placeholder per tensor the graph takes, one `call_function` per operation, of
the operator overload the function called (`torch.ops.aten.matmul.default`,
not what a kernel decomposes into) with every argument of its schema, and as
its output a tuple of the values that the code behind the graph reads.
Nothing is traced for this: there are no fake tensors and no symbolic sizes.
`input_types` has a `torch.vm.TensorType` per placeholder, with the dtype,
the device, `requires_grad` and the sizes of the tensors of that call:

```python
shown = []


def show(gm, input_types):
    shown.append((gm.code, input_types))
    return gm


g = torch.vm(f, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), show))
x, w = torch.randn(4, 8), torch.randn(8, 8)
for _ in range(4):
    out = g(x, w)
print(shown[0][0])
# def forward(self, x, w):
#     matmul_default = torch.ops.aten.matmul.default(x, w);  w = None
#     tanh_default = torch.ops.aten.tanh.default(matmul_default);  matmul_default = None
#     sum_default = torch.ops.aten.sum.default(tanh_default, dtype = None);  tanh_default = None
#     mul_tensor = torch.ops.aten.mul.Tensor(x, 2);  x = None
#     mean_default = torch.ops.aten.mean.default(mul_tensor, dtype = None);  mul_tensor = None
#     add_tensor = torch.ops.aten.add.Tensor(sum_default, mean_default, alpha = 1);  sum_default = mean_default = None
#     return (add_tensor,)
shown[0][1]  # [float32[4, 8], float32[8, 8]]
torch.equal(out, f(x, w))  # True: the graph, as it was given, runs eager's kernels
```

The passes are applied in the order of the list. What the last one returns
is what the engine calls in place of the operations from then on, with the
tensors in the order of the placeholders. A pass rewrites the graph and
returns it:

```python
def sigmoid_for_tanh(gm, input_types):
    for node in gm.graph.nodes:
        if node.target is torch.ops.aten.tanh.default:
            node.target = torch.ops.aten.sigmoid.default
    return gm


g = torch.vm(f, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), sigmoid_for_tanh))
for _ in range(4):
    out = g(x, w)
torch.equal(out, torch.sigmoid(x @ w).sum() + (x * 2).mean())  # True
```

What was made of a graph runs for calls like the one it was exported from,
and for no others. Like it means: tensors of the same class, dtype, device,
sizes, strides and `requires_grad`; the same state of the thread (gradients
on or off, inference mode, autocast on or off and the dtype it casts to, the
default dtype, whether algorithms have to be deterministic, the dispatch
keys it adds); and the same
numbers and objects where an operation takes one from outside the graph (the
function a global names, a float; a list or a tuple of numbers by what is in
it at the call, to any depth, not by which list it is: one that was changed
in place since is another value). A call of another kind runs the
operations and is exported in its turn. What was made for a kind of call is
a form of the graph, and a graph has `max_variants` of them at most (8
unless `Graphs` is made with another number); calls of further kinds then
run the operations every time. The forms are tried in the order they last
ran, so the kind of call that keeps coming is found first. A pass states
what else its rewrite depends on with `torch.vm.guard(gm, placeholder,
torch.vm.Dim(0, range(16, 33)))`: a call that fails a guard runs the
operations.

`torch.vm.regions` lists the forms of a graph under `forms`, the one that ran
last first: what a call has to be like to be of the kind (`checks`, as
text), how many calls ran the form (`hits`), and the `reason` if calls of
the kind are not exported. `exports` is how many times the graph was handed
to the passes, and `fn.stats()["graphs"]` has the totals for a function and
what it calls.

```python
(found,) = torch.vm.regions(g)
found["state"], found["variants"], found["hits"], found["misses"]  # ('compiled', 1, 2, 1)
for _ in range(3):
    out = g(torch.randn(6, 8), w)  # another size: exported again
(found,) = torch.vm.regions(g)
found["state"], found["variants"], found["hits"], found["misses"]  # ('compiled', 2, 4, 2)
[(form["checks"][1:], form["hits"]) for form in found["forms"]]
# [(['%x.1: float32[6, 8]', '%w.2: float32[8, 8]'], 2), (['%x.1: float32[4, 8]', '%w.2: float32[8, 8]'], 2)]
g.stats()["graphs"]  # {'built': 1, 'exported': 2, 'hits': 4, 'misses': 2, 'given_up': 0}
```

A number that an operation takes is one of the checks, wherever the function
reads it from: a global, a variable of an enclosing function, an attribute
of a module. The same graph with another number is another form, and the
old number finds the old form again:

```python
SCALE = 0.5


def scaled(x):
    return ((x.sin() * SCALE).cos() + SCALE).tanh() * 2


def keep(gm, input_types):
    return gm


g = torch.vm(scaled, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), keep))
x = torch.randn(5)
for _ in range(3):
    out = g(x)
SCALE = 0.25
for _ in range(3):
    out = g(x)
torch.equal(out, scaled(x))  # True
(found,) = torch.vm.regions(g)
[(form["checks"], form["hits"]) for form in found["forms"]]
# [(['%4 == 0.25', '%8 == 0.25', '%x.1: float32[5]'], 2), (['%4 == 0.5', '%8 == 0.5', '%x.1: float32[5]'], 1)]
```

The states `torch.vm.regions` reports of a graph:

- `"built"`: the engine runs the graph itself. This is the state of every
  graph when the list has no pass over graphs.
- `"waiting"`: the list has a pass over graphs, and the graph is exported at
  the next call that comes to it.
- `"compiled"`: what the passes made runs in place of the operations for
  the kinds of calls that were exported; `variants` is how many forms with
  something to run there are.
  A `reason` next to it says why no further kind of call is exported.
- `"vm"`: the engine runs the operations, for the `reason` given. Either
  the graph was given up: a pass raised (`the pass broken raised
  RuntimeError: not today`) or returned something that is no graph of the
  same inputs, what was made of the graph raised the first time it was
  called (`the compiled graph failed: ...`), in which case the operations
  ran in its place, or something between its operations could run Python
  code (below). Or every kind of call it has seen is one that is not
  exported (`a tensor of the call carries a forward-mode gradient`). The
  entries for what is in no graph have this state too.

`hits` counts the calls that ran what was made of the graph, and `misses`
those that came to the graph and ran its operations, the calls that were
exported among them.

A pass that raises gives the graph up, and the reason says what it raised.
For a pass that raised inside something it calls, the reason ends with the
function and the line it was raised in:

```python
def helper(gm):
    raise IndexError("list index out of range")


def deep(gm, input_types):
    return helper(gm)


def small(x, w):
    return ((x @ w).tanh() * 2).sin().sum()


given_up = torch.vm(small, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), deep))
left, right = torch.randn(4, 8), torch.randn(8, 8)
for _ in range(3):
    fine = torch.equal(given_up(left, right), small(left, right))
said = torch.vm.regions(given_up)[0]
raised, where = said["reason"].split(" (in ")
fine, said["state"], raised, where.split(",")[0]
# (True, 'vm', 'the pass deep raised IndexError: list index out of range', 'helper')
```

What the function does between its operations (it reads a global or an
attribute, takes an item of a list, adds two numbers) is not in the graph.
With something made of the graph in place of the operations, those
instructions run in front of it or behind it, which is not where the function
has them, and a second time when the call turns out to be one that nothing
was exported for. That is right only for what runs no Python code. So the
engine looks at each of them on every call before it runs it: an attribute
that is found without a property, a descriptor or a `__getattr__` written in
Python, an item of a list, a tuple or a dict under a number or a str,
arithmetic on exact numbers. A dict or a set compares the key that is looked
up with every key it holds under the same hash, and a key of a class with an
`__eq__` does that in Python. So a lookup in a dict or a set
(`d[k]`, `k in d`) runs no Python code only while every key in it is
a number, a str or None, or a tuple of them: a dict that has only ever been
given strs for keys is known to hold no other (CPython 3.13 and below say so
of a dict; under 3.14 it counts as any other), any other dict and any set is
looked through where it holds at most 32 keys, and is taken to run Python
code where it holds more. A call in which one of these could run Python
code runs the operations of the function as they are, every read once and
where the function has it, and after three such calls in a row the graph is
left to the engine:

```python
class Box:
    def __init__(self):
        self.scale = 0.5
        self.made = 0

    def __getattr__(self, name):  # asked when the attribute is not there
        self.made += 1
        return 0.5 + self.made


def boxed(x, box):
    return ((x.sin() * box.scale).cos() + box.scale).tanh() * 2


g = torch.vm(boxed, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), keep))
x, box, theirs = torch.randn(5), Box(), Box()
for _ in range(4):
    out = g(x, box)
torch.vm.regions(g)[0]["state"]  # 'compiled'
del box.scale, theirs.scale  # from here on a read of it runs __getattr__
for _ in range(4):
    same = torch.equal(g(x, box), boxed(x, theirs))
same, box.made, theirs.made  # (True, 8, 8)
found = torch.vm.regions(g)[0]
found["state"], found["reason"]  # ('vm', 'load_attr could run Python code')
```

Where the function binds a variable anew between its operations, the variable
lets go of what it held. If that was the last reference to an object whose
end runs Python code (a `__del__`, a weak reference with a callback), the
code runs at the binding, in front of what the function reads behind it.
What was made of the graph binds the variable when it has run, behind every
read. So the engine looks, on every call and before anything of it runs, at
what the call would let go of that way. Each such object has to end without
Python code running: neither it nor anything it refers to has a finalizer,
or a weak reference that is told of its end (the entry of a weak dictionary
or of a weak set, which only takes itself out, is none). Or it has to stay
alive: a parameter of the function or another value of it holds it, the
function keeps it, or it has more references than the call lets go of. A
call for which neither can be shown runs the operations of the function as
they are:

```python
class Ends:
    def __del__(self):
        STATE[0] += 1.0  # what the function reads behind the binding


STATE, HELD = [0.5], []


def lets_go(x):
    held = HELD.pop()
    y = (x.sin() + 1).cos() * STATE[0]
    held = None  # the last reference to what `pop` gave
    return (y + STATE[0]).tanh(), held


def eight_calls(fn):
    STATE[0] = 0.5
    outs = []
    for at in range(8):
        HELD.append(Ends() if at == 6 else 0)
        outs.append(fn(x)[0])
    return outs


g = torch.vm(lets_go, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), keep))
x = torch.randn(5)
got, expected = eight_calls(g), eight_calls(lets_go)
all(torch.equal(a, b) for a, b in zip(got, expected)), STATE[0]  # (True, 1.5)
```

What an object refers to is what it shows the garbage collector, up to 64
objects (with more, the call runs the operations). What a tensor holds
outside of that, in the nodes of its history, is not seen; objects in a
reference cycle end when the collector runs, as they do in eager mode; and
what a pass's own callable does is not looked at.

Some kinds of calls are not exported: one whose operations write to what
they are given, one with a tensor that carries a forward-mode gradient or is
of a subclass of `torch.Tensor`, and one under a `TorchDispatchMode`, a
`TorchFunctionMode` or a transform of `torch.func`, where the operations
have to run as the function wrote them for the mode or the transform to see
them. Such a kind of call is remembered like any other. The calls of the
kind run the operations and are not looked at again, and the other kinds of
calls go on being exported.

```python
from torch.utils._python_dispatch import TorchDispatchMode


class Seen(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.seen = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.seen.append(func.__name__)
        return func(*args, **(kwargs or {}))


g = torch.vm(f, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), sigmoid_for_tanh))
x, w = torch.randn(4, 8), torch.randn(8, 8)
for _ in range(3):
    out = g(x, w)
torch.equal(out, f(x, w))  # False: the pass made a sigmoid of the tanh
with Seen() as mode:
    out = g(x, w)
torch.equal(out, f(x, w))  # True: under the mode the operations of the function ran
mode.seen
# ['mm.default', 'tanh.default', 'sum.default', 'mul.Tensor', 'mean.default', 'add.Tensor']
torch.equal(g(x, w), f(x, w))  # False: without the mode, what was exported runs again
```

The dtype that autocast casts to, and whether algorithms have to be
deterministic, are part of the state of the thread because a compiler builds
both into what it makes of a graph. A call under another dtype is another
kind of call, and the first dtype finds what was made for it again:

```python
def cast(x, w):
    return (torch.mm(torch.mm(x, w), w).relu() * 2 + 1).tanh()


g = torch.vm(cast, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), keep))
kinds = []
for dtype in (torch.bfloat16, torch.float16, torch.bfloat16):
    with torch.autocast("cpu", dtype=dtype):
        for _ in range(3):
            out = g(x, w)
    kinds.append((out.dtype, torch.vm.regions(g)[0]["variants"]))
kinds  # [(torch.bfloat16, 1), (torch.float16, 2), (torch.bfloat16, 2)]
```

### AOTAutograd and Inductor

Two passes over graphs come with `torch.vm`, and they are the ones that
compile a graph:

- `torch.vm.passes.AOTAutograd()` has AOTAutograd trace the graph, and its
  backward when gradients are recorded, ahead of time. The eager kernels run
  both.
- `torch.vm.passes.Inductor()` has Inductor compile the graph, and its
  backward when gradients are recorded.

Each is called as any pass over a graph is, `pass_(gm, input_types)`,
compiles the graph for the tensors of the call it was exported from, and
returns a graph whose forward calls what it compiled. The compilers are
imported when such a pass first runs, not when it is put in a list: a
process that uses `torch.vm` with the default passes, with `Graphs()`, or
with passes over graphs of its own imports neither `torch._inductor` nor
`torch._functorch.aot_autograd` nor `torch._dynamo`.

```python
g = torch.vm(f, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), P.Inductor()))
x, w = torch.randn(4, 8), torch.randn(8, 8, requires_grad=True)
kinds = []
for _ in range(5):
    out = g(x, w)
    kinds.append(type(out.grad_fn).__name__)
kinds
# ['AddBackward0', 'AddBackward0', 'CompiledFunctionBackward', 'CompiledFunctionBackward', 'CompiledFunctionBackward']
torch.allclose(out, f(x, w))  # True
(found,) = torch.vm.regions(g)
found["state"], found["variants"], found["hits"], found["misses"]  # ('compiled', 1, 3, 1)
```

The first call is the one the block is profiled on and the second the one
the graph is exported from, and both run the operations of the function; from
the third call on the compiled graph runs, and with gradients recorded what
it returns has one compiled autograd function as its `grad_fn`. A call under
`torch.no_grad()` is another kind of call and is compiled by itself, without
a backward. What a compiler makes of the operations is not what eager runs,
so the last bits of a result may differ, and `torch.vm.is_exact` is false
for a list with one of these passes in it (and for a list with a pass of
one's own, which is not taken to be exact).

### The order of the list

- `Graphs()` can stand anywhere among the passes over a block. It looks at a
  block when the others are done with it, wherever it stands.
- The passes over a graph stand behind `Graphs()`, in the order they are to
  be applied in. When the list has such a pass and no `Graphs()`, it is taken
  to be there, in front of the first of them.
- At most one pass compiles, and it is the last pass over a graph: what it
  returns is run, not rewritten.

`torch.vm` raises a `ValueError` for a list that breaks one of these when the
function is handed to it:

```python
default = torch.vm.DEFAULT_PASSES
torch.vm(f, passes=(P.Inductor(), *default, P.Graphs()))
# ValueError: passes has torch.vm.passes.Inductor() in front of torch.vm.passes.Graphs(): a pass over a graph
# is given the graphs that Graphs makes, so it goes behind it
torch.vm(f, passes=(*default, P.Graphs(), P.AOTAutograd(), P.Inductor()))
# ValueError: passes has torch.vm.passes.AOTAutograd() and torch.vm.passes.Inductor(): a graph is compiled
# once, by the last pass over it
torch.vm(f, passes=(*default, P.Graphs(), P.Inductor(), keep))
# ValueError: passes has keep behind torch.vm.passes.Inductor(): what a pass that compiles a graph returns is
# run, not rewritten, so it is the last pass over a graph
torch.vm(f, passes=(*default, keep, P.Inductor())).passes
# ['resolve_dispatch', 'tensor_accessors', 'method_calls', 'fold', 'dce', 'graphs']
```

### Limits

What graphs do not do at this stage:

- One block. A graph does not go on through a call of a Python function, a
  branch or a loop, as said above.
- No effects. An operation that writes to one of its arguments is in no
  graph, and a call whose operations turn out to write when they are taken
  down is not exported: a batch norm in training, which updates its running
  statistics, runs its operations on every call, and the same graph is
  exported for the calls in eval mode.
- Sizes are fixed per export. Every size a graph is called with is a kind of
  call of its own, with a graph of its own, so a function that is called
  with many sizes uses up `max_variants`. Sizes of the tensors the graph
  computes are computed in it (`torch.ops.aten.sym_size.int`), but its
  inputs have the sizes of the call it was exported from. With `Inductor()`
  every export is a compilation.
- Under a `TorchDispatchMode`, a `TorchFunctionMode` or a transform of
  `torch.func` the operations of the function run, so that those see them,
  and the calls under one are not exported.

## API Reference

```{eval-rst}
.. autofunction:: torch.vm.vm
```

```{eval-rst}
.. autodata:: torch.vm.DEFAULT_PASSES
```

```{eval-rst}
.. autofunction:: torch.vm.is_exact
```

```{eval-rst}
.. autoclass:: torch.vm.Type
```

```{eval-rst}
.. autoclass:: torch.vm.TensorType
    :members:
```

```{eval-rst}
.. autoclass:: torch.vm.IntType
```

```{eval-rst}
.. autoclass:: torch.vm.Interval
    :members:
```

```{eval-rst}
.. autofunction:: torch.vm.type_of
```

```{eval-rst}
.. autofunction:: torch.vm.materialize
```

```{eval-rst}
.. autofunction:: torch.vm.guard
```

```{eval-rst}
.. autoclass:: torch.vm.Check
```

```{eval-rst}
.. autoclass:: torch.vm.Dtype
```

```{eval-rst}
.. autoclass:: torch.vm.Device
```

```{eval-rst}
.. autoclass:: torch.vm.Dim
```

```{eval-rst}
.. autoclass:: torch.vm.Stride
```

```{eval-rst}
.. autoclass:: torch.vm.Contiguous
```

```{eval-rst}
.. autoclass:: torch.vm.Aligned
```

```{eval-rst}
.. autofunction:: torch.vm.swap
```

```{eval-rst}
.. autofunction:: torch.vm.fuse_checked
```

```{eval-rst}
.. autoexception:: torch.vm.Mismatch
```

```{eval-rst}
.. autofunction:: torch.vm.load_passes
```

```{eval-rst}
.. autofunction:: torch.vm.regions
```

```{eval-rst}
.. autofunction:: torch.vm.memory
```

```{eval-rst}
.. autofunction:: torch.vm.is_available
```

```{eval-rst}
.. autoexception:: torch.vm.Unsupported
```

## torch.vm.passes

```{eval-rst}
.. automodule:: torch.vm.passes
```

```{eval-rst}
.. autoclass:: torch.vm.passes.BuiltinPass
```

```{eval-rst}
.. autoclass:: torch.vm.passes.Inline
```

```{eval-rst}
.. autoclass:: torch.vm.passes.Unroll
```

```{eval-rst}
.. autoclass:: torch.vm.passes.ModuleShortcuts
```

```{eval-rst}
.. autoclass:: torch.vm.passes.Graphs
```

```{eval-rst}
.. autoclass:: torch.vm.passes.AOTAutograd
```

```{eval-rst}
.. autoclass:: torch.vm.passes.Inductor
```

```{eval-rst}
.. autoclass:: torch.vm.passes.ResolveDispatch
```

```{eval-rst}
.. autoclass:: torch.vm.passes.TensorAccessors
```

```{eval-rst}
.. autoclass:: torch.vm.passes.MethodCalls
```

```{eval-rst}
.. autoclass:: torch.vm.passes.NarrowDims
```

```{eval-rst}
.. autoclass:: torch.vm.passes.Fold
```

```{eval-rst}
.. autoclass:: torch.vm.passes.DCE
```
