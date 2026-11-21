# torch.vm

```{eval-rst}
.. automodule:: torch.vm
```

## Graphs

The default passes make each tensor operation of a block a direct call of its
kernel. `torch.vm.passes.Graphs()` looks one level up: in every block, once
the other passes are done with it, it finds the runs of such calls together
with what sits between them (reads of globals and attributes, constants,
arithmetic on sizes). An operation that
the default passes leave as it is can be an operation of a graph as well, if
all it did whenever the block was watched was run kernels that neither write
nor draw random numbers: `torch.cat` and `torch.stack` of a list, a subscript
of a tensor (`x[mask]`, `x[0]`, `x[1:, 2]`), `torch.conv2d` with ints for
stride and padding.

A graph starts at the first operation of such a run, and one starts with the
function. From there the engine takes a path down while a call runs:
the instructions of the function in the order that call runs them, through
the branches as they go and round the loops as often as they go round,
up to the first instruction that can be no part of a graph. A path of fewer
than `min_ops` operations (4 unless the pass is made with another number) is
no graph.

With `Graphs()` in the list the function runs as it runs without it. The
engine runs a graph as it runs the rest of the function: the same kernels, on
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
#   'aten::mul.Tensor', 'aten::mean', 'aten::add.Tensor'], 'inputs': 2,
#   'outputs': 1, 'branches': 0, 'calls': 0, 'paths': 1, 'ends': [(0, 12)],
#   'why': 'the function returns', 'variants': 0, 'hits': 0, 'misses': 1,
#   'exports': 0, 'version': 1, 'forms': []}]
```

`where` names the function, the block, and the instruction of the first
operation of the graph, whether its path starts there or with the function
(`g.dump()` shows them). The state `"built"` says that the engine runs
the graph itself. `ops` and `operators` are the operations of the path that
was taken down last, and `inputs` and `outputs` how many values that path
takes from the code in front of it (here `x` and `w`) and how many the code
behind it reads. `ends` is the block and the instruction the function goes
on at behind the path, and `why` what ends the path there. `misses` counts
the calls that came to the graph and ran its instructions, the one the path
was taken down from among them. What was left out of the graphs of a block
that has tensor work on both sides of it is listed too, with the state
`"vm"` and the reason:

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

### Paths

A path follows the call it is taken down from. Where the function branches
on something that is known without looking into a tensor (a flag, `None`, a
number, a str, a size of a tensor), the path goes the way the call went, and
that way is checked on the calls that come later. A call that goes another
way, to operations that are on no path yet, is taken down in its turn, as
another path of the same graph:

```python
def branchy(x, flag):
    y = x.sin() * 2
    if flag:
        y = y.cos() + 1
    else:
        y = y.tanh() - 1
    return (y * y).exp().sum()


g = torch.vm(branchy, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs()))
x = torch.randn(4)
for flag in (True, True, True, False, False, False):
    same = torch.equal(g(x, flag), branchy(x, flag))
(found,) = torch.vm.regions(g)
found["where"], found["ops"], found["branches"], found["paths"]  # ('branchy (block 0)[1]', 7, 1, 2)
found["operators"][2:4]  # ['aten::tanh', 'aten::sub.Tensor']: of the path that was taken down last
found["ends"], found["why"]  # ([(3, 6)], 'the function returns')
```

`branches` is how many branches the path goes through and `paths` how many
ways through the function were taken down; `ends` says for each path where
the function goes on behind it, once for paths that end at the same place.
The first call that goes a new way runs a block for the first time, which
the engine watches as it watches every block, and the path is taken down at
the call after it.

A loop is its iterations one after the other: a path goes round a loop over
a list, a tuple or a `range` as often as the call does (and round a `while`
loop on a count), with the test of the loop as a branch each time round. The
number of times round is no input of the graph: it is the path. The calls of
`len`, `range`, `enumerate`, `zip` and `isinstance` that such loops are
written with are part of a path.

```python
def loop(x, n):
    y = x.sin()
    for i in range(n):
        y = y * 0.5 + i
    return y.cos().sum()


g = torch.vm(loop, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs()))
for _ in range(3):
    same = torch.equal(g(x, 3), loop(x, 3))
(found,) = torch.vm.regions(g)
found["ops"], found["branches"]  # (9, 4): sin, three times round, cos and sum; four tests of the loop
```

A list that the function makes and fills by appending is the path's own, so
a path that starts with the function goes on through the loop that fills it,
and the operation that is handed the list takes the values that were
appended:

```python
def collect(xs):
    out = []
    for x in xs:
        out.append(x.sin() * 2)
    return torch.cat(out).tanh().sum()


g = torch.vm(collect, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs()))
xs = [torch.randn(3) for _ in range(3)]
for _ in range(3):
    same = torch.equal(g(xs), collect(xs))
(found,) = torch.vm.regions(g)
found["where"], found["ops"], found["inputs"]  # ('collect (block 3)[4]', 9, 1)
found["operators"][-3:]  # ['aten::cat', 'aten::tanh', 'aten::sum']
```

An item that the function reads back out of such a list, at a place that is
a number, is the value that was put there, so an operation can take it. The
place is one of the checks of the path. A list that gets tensors and other
things has its items in the order the function put them where the path
ends:

```python
def routed(xs):
    out = []
    for x in xs:
        out.append(x.sin() * 2)
    return (out[0] + out[-1]).tanh().sum()


g = torch.vm(routed, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs()))
for _ in range(4):
    same = torch.equal(g(xs), routed(xs))
(found,) = torch.vm.regions(g)
found["ops"], found["why"]  # (9, 'the function returns')
```

What ends a path:

- a call that the path does not go into ("Paths across calls" below says
  which), or of something that is no operation, like `print`;
- an operation with several results (`x.chunk(2)`, `torch.max(x, 0)`);
- behind an operation that writes to a tensor from outside the path or draws
  random numbers: whatever a form would have to check behind what was made
  of the graph, and such an operation behind a check of that kind ("Writes"
  below); with `Graphs()` alone, an operation that draws random numbers;
- a store into an object that would run Python code, and a read of a list
  or a dict that a store of the path changed ("Stores into objects");
- a number read from a tensor (`x.item()`, `float(x)`), and with it a branch
  on the data of a tensor;
- an instruction that could run Python code on the values of this call: a
  read of an attribute that is a property, an iterator that is unpacked;
- the end of the function, a block that this call is the first to run (the
  path is taken down again at the next call, up to three times), and 200,000
  instructions.

A branch on the data of a tensor therefore ends a graph, and another starts
behind it:

```python
def on_data(x):
    y = x.sin() * 2 + 1
    if y.sum() > 0:
        y = y.cos()
    return (y * y).tanh().exp().sum()


g = torch.vm(on_data, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs()))
for sign in (1, 1, 1, -1, -1, 1):
    out = g(sign * torch.ones(3))
for found in torch.vm.regions(g):
    print(found["where"], found.get("ops"), found.get("why") or found.get("reason"))
# on_data (block 0)[1] 5 aten::is_nonzero gives a number read from a tensor
# on_data (block 0)[11] None aten::is_nonzero gives a number read from a tensor
# on_data (block 1)[1] 5 the function returns
```

In a function with a `try`, the variables that a handler may read are kept
where the handlers find them. A path goes on across the reads and writes of
such a variable, and what it bound is there when the function goes on behind
it:

```python
def looked_up(x, table):
    y = x.sin() * 2
    try:
        y = y.cos() + table["k"]
    except KeyError:
        y = y - 1
    return (y * 2).tanh().sum()


g = torch.vm(looked_up, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs()))
for table in ({"k": 2.0}, {"k": 2.0}, {"k": 2.0}, {}, {}):
    same = torch.equal(g(x, table), looked_up(x, table))
found = torch.vm.regions(g)[0]
found["where"], found["ops"], found["why"]  # ('looked_up (block 0)[1]', 7, 'the function returns')
```

Graphs start in one's own code and in the code of models that torch has
(`torch.nn`, `torch.optim`, `torch.distributions`, `torch.func`,
`torch.functional`, the methods of `torch.Tensor`). The rest of the library is
code around tensors, and a function of it that is called runs as it is,
without graphs of its own. What prints a tensor is such code:

```python
def printed(x):
    return str(x.sin() * 2)


g = torch.vm(printed, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(min_ops=1)))
for _ in range(3):
    text = g(x)
"_Formatter.__init__" in g.stats()["compiled_callees"]  # True: the engine runs it
sorted({found["where"].split(" (")[0] for found in torch.vm.regions(g)})
# ['Tensor.__format__', 'Tensor.__iter__', 'printed']
```

### Paths across calls

A path goes into the Python functions that its instructions call. The call
is no instruction of the path; the instructions of the function are, and its
parameters are the values the call hands over: positional and keyword
arguments, defaults, `*args` and `**kwargs`. What the function returns is a
value of the path like any other, so the operations in front of a call,
inside it and behind it are one graph, and `min_ops` counts them all:

```python
def scaled(x, scale=2.0, *, shift=1.0):
    return (x * scale + shift).tanh()


def calls(x):
    y = scaled(x.sin())
    y = scaled(y, 3.0)
    return scaled(y, shift=0.5).sum()


g = torch.vm(calls, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs()))
x = torch.randn(4, 4)
for _ in range(4):
    same = torch.equal(g(x), calls(x))
found = torch.vm.regions(g)[0]
found["where"], found["ops"], found["why"]  # ('calls (block 0)[2]', 11, 'the function returns')
```

What a path goes into is what the engine itself would run compiled when the
call is made: a function, a bound method, an object with a `__call__`. A
call of a module comes down to its `forward` while neither the module nor
torch has a hook registered, so a model whose modules call each other is one
graph. A call that the engine has inlined is part of the path like the other
instructions of the function, and so is a function of torch that graphs do
not start in.
The same goes for what the interpreter calls for an object without a call
being written: its `__iter__` where a `for` goes over it (an `nn.Sequential`
or an `nn.ModuleList`, also through `iter`, `enumerate` and `zip`),
its `__len__`, its `__getitem__`, and where an attribute is read, the getter
of a property, the `__get__` of a descriptor whose class is written in
Python, or the `__getattr__` of the class for a name that is not there.
`list(x)` and `tuple(x)` ask the object for its `__len__` or its
`__length_hint__` as well, once they have its iterator, so a path goes into
the `__iter__` for them only where the class has neither, which it checks
at every call: of a module container, or of a tensor, they end the path in
front of the call.

The instruction that such a function runs for gives what the interpreter
makes of what the function returned: `for` and `iter` raise `TypeError` for
what is no iterator, `len` gives the `__index__` of what `__len__` returned
as an `int` and raises for a negative one, and a read whose property or
descriptor raised `AttributeError` asks the `__getattr__` of the class, if
it has one. A path goes on behind the function only where what it returned
is that already (an iterator, an `int` that is not negative), which it
checks at every call. For anything else, and where the path ends inside
the function, which then runs on to its end as instructions (below), the
instruction gives the interpreter's answer and the path ends in front of it.

```python
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(8, 8)
        self.second = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.second(torch.relu(self.first(x))) + x


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = Block()
        self.b = Block()
        self.head = torch.nn.Linear(8, 2)

    def forward(self, x):
        return self.head(self.b(self.a(x)))


net = Net().eval()
g = torch.vm(net, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs()))
v = torch.randn(4, 8)
for _ in range(4):
    same = torch.equal(g(v), net(v))
[(found["where"], found["ops"]) for found in torch.vm.regions(g) if "ops" in found]
# [('Block.forward (block 0)[0]', 4), ('Net.forward (block 0)[0]', 9)]
```

```python
model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))


def infer(model, x):
    return model(x)


g = torch.vm(infer, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(min_ops=3)))
for _ in range(4):
    same = torch.equal(g(model, v), infer(model, v))
[found["operators"] for found in torch.vm.regions(g) if found["where"].startswith("infer")]
# [['aten::linear', 'aten::relu', 'aten::linear']]
```

On every later call the path checks, in front of each function it went into,
that what is called still comes down to that function. A hook that was
registered on a module since, a module of another class in its place, a
function that was swapped: the call is none for this path. The reads of
attributes between the calls are instructions of the path like any other, so
a submodule that was replaced by assignment is read at every call. In the
text of `regions`, `calls` counts these checks, one per call the path goes
into; `branches` counts the branches and the loop tests.

A path does not go into every call. It ends in front of a call

- of a function that the engine does not compile: one it does not support
  (a generator, such as `children()` and `named_modules()` of a module), and
  the places it leaves to the interpreter. A function that it compiled and
  then **left to the interpreter because it measured it to run faster there
  (`demote`, which is on by default) is gone into like any other**: a path
  does not run the function in the VM, it has the instructions of the
  function among its own, so what was measured says nothing about it. The
  small functions of plain Python on the way through a model (an argument
  check, the `__iter__` of a container) are the ones that are demoted, and
  the graph of the model goes through them. A call in which a graph of a
  function waits, or a path of it is taken down, is no sample of that
  measurement: it says nothing about how fast the function is going to be;
- of a function that is on the path eight times already: a function that
  calls itself is gone into that often, and an `nn.Sequential` inside an
  `nn.Sequential` is on a path twice;
- more than 48 calls deep;
- of a `functools.partial`, of a class (constructing an object), of
  `super()`, and of a function that was made on the path (a closure that
  the function defines and calls);
- whose arguments do not fit the function: the call raises the `TypeError`
  where the function has it;
- that hands on `**kwargs` from a dict whose keys are no strs;
- while a trace function is set (a debugger, coverage).

A path starts and ends in one function. Where it would end inside a function
that it went into (at an instruction that can be no part of a path, at a
block that this call is the first to run, at an exception), that function
runs on to its end as instructions, and the path is cut back to in front of
the outermost call it was in, where it ends. `why` then says in which
function, and what it was:

```python
class Notes:
    def __init__(self):
        self.calls = 0

    def noted(self, y):
        print(end="")
        self.calls += 1
        return y.exp()


NOTES = Notes()


def noting(x):
    return NOTES.noted(x.sin().cos() * 2 + 1).tanh().cos().sum()


g = torch.vm(noting, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(min_ops=3)))
for _ in range(4):
    same = torch.equal(g(x), noting(x))
same, NOTES.calls  # (True, 8): the method runs once a call, here and in the function itself
first, second = [found for found in torch.vm.regions(g) if "ops" in found]
first["ops"], first["why"]  # (4, 'in noted: a call that runs no kernel')
second["ops"], second["why"]  # (3, 'the function returns')
```

A function that is called has graphs of its own, as it had before, and they
are met exactly when the function runs as instructions: where it is called
from code that has no path through the call, and where the path of the
caller was not for this call. While a path of a caller holds the call, the
function does not run, and its graphs do nothing: `regions` lists such a
graph with the key `within`, the graph of the caller that holds its
operations and runs them, and `stats()["graphs"]` does not count it. A call
that a path goes through is not among what `regions` lists as left out.
The functions that are called are warm before their callers are, and their
graphs would be exported first, to be of no use a few calls later when the
graph of the caller holds the same operations. So a graph waits, and
`regions` says `"waiting"`, while a function further out is still being
watched by the engine or has a graph that is not done being taken down:

```python
exported = []


def count(gm, input_types):
    exported.append(len([n for n in gm.graph.nodes if n.op.startswith("call")]))
    return gm


g = torch.vm(net, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), count))
for _ in range(8):
    same = torch.equal(g(v), net(v))
same, exported  # (True, [9]): the model, once
handle = net.a.register_forward_hook(lambda module, args, output: output * 2)
for _ in range(4):
    same = torch.equal(g(v), net(v))
same, exported  # (True, [9, 4, 4]): with the hook the path of the model is not for the call
handle.remove()
whole = [found for found in torch.vm.regions(g) if found["where"].startswith("Net.forward")][0]
hits = whole["hits"]
for _ in range(3):
    same = torch.equal(g(v), net(v))
whole = [found for found in torch.vm.regions(g) if found["where"].startswith("Net.forward")][0]
same, whole["hits"] - hits  # (True, 3): and without it, it is again
```

### Passes over graphs

A graph leaves the engine only for a pass over graphs: a function of two
arguments, `pass_(gm, input_types) -> gm`, that stands behind `Graphs()` in
the list. (With such a pass in the list and no `Graphs()`, the pass is taken
to be there, in front of the first of them. A pass over graphs in front of
`Graphs()` is a `ValueError`.) The engine then enters each graph through an
instruction of its own, which `dump()` shows where the graph starts (`graph
f (block 0)[2:11]: 6 operations`), and the graph is `"waiting"`.

The next call that comes to the graph runs its instructions as before, and
the engine takes down the path and what its operations did: the kernel call
each operation made, and the values the operations took from outside the
graph. From that one run it
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

An operation that is no single kernel call is a node of the graph as the
function wrote it, a call of `torch.cat` or a subscript, with the lists,
tuples and slices it is given built of the values of the graph. A list or a
tuple of tensors that the function is handed is taken apart: its tensors are
inputs of the graph, and the form is for a list of that length.

```python
def joined(a, b, mask):
    y = torch.cat([a.sin(), b.cos()], 1).tanh() * 2
    return y[mask].sum() + y[0, 1:].mean()


g = torch.vm(joined, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), show))
a, b = torch.randn(3, 4), torch.randn(3, 4)
mask = torch.cat([a, b], 1) > 0
for _ in range(3):
    out = g(a, b, mask)
torch.equal(out, joined(a, b, mask))  # True
print(shown[-1][0])
# def forward(self, a, b, mask):
#     sin_default = torch.ops.aten.sin.default(a);  a = None
#     cos_default = torch.ops.aten.cos.default(b);  b = None
#     cat = torch.cat([sin_default, cos_default], 1);  sin_default = cos_default = None
#     tanh_default = torch.ops.aten.tanh.default(cat);  cat = None
#     mul_tensor = torch.ops.aten.mul.Tensor(tanh_default, 2);  tanh_default = None
#     getitem = mul_tensor[mask];  mask = None
#     sum_default = torch.ops.aten.sum.default(getitem, dtype = None);  getitem = None
#     getitem_1 = mul_tensor[(0, slice(1, None, None))];  mul_tensor = None
#     mean_default = torch.ops.aten.mean.default(getitem_1, dtype = None);  getitem_1 = None
#     add_tensor = torch.ops.aten.add.Tensor(sum_default, mean_default, alpha = 1);  sum_default = mean_default = None
#     return (add_tensor,)
torch.vm.regions(g)[0]["operators"][5:8]
# ['aten::index.Tensor', 'aten::sum', 'aten::select.int+aten::slice.Tensor']
```

Every path of a graph is exported by itself. The graph a pass is given has
the operations of one way through the function and no branch,
a loop as many times as the call went round,
and it takes as inputs only what its operations take: a flag that a branch
is on is none.
A variable of a loop names one input per time round (`x`, `x_2`, `x_3`).
The operations of the functions that the path went into are in the graph
where they ran, and a tensor that is no variable of a function (a weight
that a module keeps) is an input named by where it was read (`module_weight`
for `module.weight`); the checks of a form name it that way too, and a value
of a called function with the function (`Linear.forward:%5`).
On a later call the engine first runs what the path has in front of its
operations (the reads, the arithmetic on numbers), on registers of the
path's own, and checks that the branches go the way they went. Then what was
made of the graph runs, then what the path has behind its operations, where a
branch on a size that the graph computes is checked, and only then the
function goes on where the path ended. A call that goes another way has
changed nothing by the time a check fails: it tries the next path, and if
none is for it, it runs the instructions of the function from where the
graph starts and is taken down as a path of its own.

```python
g = torch.vm(branchy, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), show))
v = torch.randn(4)
for flag in (True, True, True, False, False, False, True):
    same = torch.equal(g(v, flag), branchy(v, flag))
["cos" in code for code, _ in shown[-2:]]  # [True, False]: a graph for each way
[types for _, types in shown[-2:]]  # [[float32[4]], [float32[4]]]: the flag is no input
(found,) = torch.vm.regions(g)
found["paths"], found["variants"], found["hits"], found["misses"]  # (2, 2, 3, 3)
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

`torch.vm.regions` lists the forms of a graph under `forms`, those of the
path that ran last first: what a call has to be like to be of the kind (`checks`, as
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
  ran in its place (or, for a graph that writes to a tensor from outside,
  the call raised: "Writes"), or something between its operations could run Python
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

A number that is another on every call (a learning rate on a schedule, a
step count) would use up the forms of a graph with kinds of calls that never
come again. When a graph is about to be exported for a fourth number, with
everything else as it was and none of the three forms before it ever run
again, the three are dropped and nothing more is exported for that kind of
call while the number goes on changing. The calls run the operations, and
`regions` gives the reason:

```python
def step(x, rate):
    return ((x + rate).sin() * rate).cos().exp().tanh()


exported = []


def count(gm, input_types):
    exported.append(gm)
    return gm


g = torch.vm(step, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), count))
x = torch.randn(4)
same = [torch.equal(g(x, i / 7), step(x, i / 7)) for i in range(50)]
all(same), len(exported)  # (True, 3)
(found,) = torch.vm.regions(g)
found["state"], found["reason"]  # ('vm', 'its inputs take ever new values')
found["forms"]
# [{'checks': ['%x.1: float32[4]'], 'hits': 0, 'reason': 'its inputs take ever new values'}]
```

Two or three numbers that come in turn keep their forms: by the time a
fourth number comes, one of the forms has run again.

Once such a number has been the same for three calls in a row, the kind of
call is open again: the next call is exported, and the calls behind it run
what was made of the graph. (The calls that run the graph are counted here,
so the function is not measured against the interpreter meanwhile:
`demote=False`.)

```python
exported.clear()
g = torch.vm(step, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), count), demote=False)
for i in range(20):
    out = g(x, i / 7)
same = [torch.equal(g(x, 0.5), step(x, 0.5)) for _ in range(8)]
all(same), len(exported)  # (True, 4)
(found,) = torch.vm.regions(g)
found["state"], found["hits"]  # ('compiled', 3)
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
(`d[k]`, `k in d`, `d.get(k)`) runs no Python code only while every key in it is
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

With `TORCH_VM_GRAPH_TIMES=1` in the environment, `regions` also has `times`
for a graph: the nanoseconds that the calls which ran what was made of it
spent in front of it (the instructions and the checks of the path), in it,
and behind it, and how many such calls there were. The numbers are those of
one run:

```console
$ cat times.py
import torch

P = torch.vm.passes


def f(x, w):
    return torch.tanh(x @ w).sum() + (x * 2).mean()


g = torch.vm(f, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), P.Inductor()))
x, w = torch.randn(4, 8), torch.randn(8, 8)
for _ in range(8):
    out = g(x, w)
print(torch.vm.regions(g)[0]["times"])
$ TORCH_VM_GRAPH_TIMES=1 python times.py
[19891, 1556676, 3290, 6]
```

With `TORCH_VM_GRAPH_DEBUG=1` in the environment the engine says on stderr
what it does with graphs: a line per path it takes down (how long it is,
where it ends and why), per call that is not gone into, and per call that
did not run what was made of a graph, with the check that did not hold:

```console
$ cat debug.py
import torch

P = torch.vm.passes


def branchy(x, flag):
    y = x.sin() * 2
    if flag:
        y = y.cos() + 1
    else:
        y = y.tanh() - 1
    return (y * y).exp().sum()


g = torch.vm(branchy, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), lambda gm, input_types: gm))
x = torch.randn(4)
for flag in (True, True, True, False):
    out = g(x, flag)
$ TORCH_VM_GRAPH_DEBUG=1 python debug.py
torch.vm graph branchy (block 0)[0:4]: a path of 11 instructions, 7 operations, ends at (3, 6): the function returns
torch.vm graph branchy (block 0)[0:4]: a check of the path does not hold: the way of a branch (instruction 4 of 11, in branchy; of: to_bool %5)
torch.vm graph branchy (block 0)[0:4]: a path of 5 instructions, 2 operations, ends at (2, 0): a block that was not profiled yet
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

### Writes

An operation that writes to a tensor is an operation of a graph where the
tensor is one that the path made itself: what an operation of the path
gave, or a view of that. Nothing outside the path has such a tensor before
the path ends, so the write is the graph's own. That holds for the in-place
operators (`y.add_(1)`, `y += z`, `torch.relu_(y)`, `nn.ReLU(inplace=True)`,
`F.leaky_relu(y, 0.1, inplace=True)`), for a write through a view
(`y[:, :2].mul_(3)`), for a store into a subscript (`y[0] = 0.0`), and for
`torch.tensor([...])`, which writes to what it makes:

```python
def writes(x):
    y = x.sin() * 2
    y.add_(1)
    y[0] = 0.0
    half = y[:, :2]
    half.mul_(3)
    return y.cos().sum(), y, half


g = torch.vm(writes, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), show))
x = torch.randn(4, 4)
for _ in range(6):
    got, expected = g(x), writes(x)
[torch.equal(a, b) for a, b in zip(got, expected)]  # [True, True, True]
(found,) = [found for found in torch.vm.regions(g) if "ops" in found]
found["ops"], found["hits"], found["why"]  # (8, 4, 'the function returns')
print(shown[-1][0])
# def forward(self, x):
#     sin_default = torch.ops.aten.sin.default(x);  x = None
#     mul_tensor = torch.ops.aten.mul.Tensor(sin_default, 2);  sin_default = None
#     add__tensor = torch.ops.aten.add_.Tensor(mul_tensor, 1, alpha = 1);  add__tensor = None
#     mul_tensor[0] = 0.0;  setitem = mul_tensor;  setitem = None
#     getitem = mul_tensor[(slice(None, None, None), slice(None, 2, None))]
#     mul__tensor = torch.ops.aten.mul_.Tensor(getitem, 3);  mul__tensor = None
#     cos_default = torch.ops.aten.cos.default(mul_tensor)
#     sum_default = torch.ops.aten.sum.default(cos_default, dtype = None);  cos_default = None
#     return (mul_tensor, getitem, sum_default)
got[2]._base is got[1]  # True
```

The graph a pass is given has the writes as the function wrote them, in
their order. A pass that returns the graph has it run as it is, on the
kernels of eager mode, so what aliases in eager mode aliases here: the view
that the function returns is a view of the tensor it returns, and a write to
one shows in the other. `AOTAutograd()` makes the graph functional before it
traces it and gives the same results bit for bit, with the results aliasing
as before; `Inductor()` compiles that functional graph:

```python
g = torch.vm(writes, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), P.AOTAutograd()))
for _ in range(6):
    got = g(x)
[torch.equal(a, b) for a, b in zip(got, writes(x))], got[2]._base is got[1]  # ([True, True, True], True)
```

Where gradients are recorded, eager mode raises when it goes backward
through an operation whose saved tensor was written to in place afterwards
(`y = x.exp(); y.add_(1)`: the derivative of `exp` is its result). A graph
whose writes a compiler made functional would go backward without a word. So
before a run with a write in place is exported with gradients on, the
functions of autograd that its results came from are asked whether what they
saved is still what it was. Where it is not, calls of that kind are not
exported: the instructions of the function run, and the error is eager's.
The same holds where that cannot be shown: the functions are followed back
through 4096 of them, and a run with a longer history is not exported
(`what autograd saved is not looked at behind 4096 functions`). The same
function without gradients is a graph:

```python
def saves(x):
    y = x.exp()
    y.add_(1)
    return (y * 2).sin().cos().sum()


g = torch.vm(saves, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), P.AOTAutograd()))
for _ in range(6):
    loss = g(torch.randn(4, requires_grad=True))
try:
    loss.backward()
except RuntimeError as error:
    said = str(error)
"modified by an inplace operation" in said  # True
(found,) = [found for found in torch.vm.regions(g) if "ops" in found]
found["state"], found["reason"]  # ('vm', 'a tensor that autograd saved is written in place')
with torch.no_grad():
    for _ in range(6):
        same = torch.equal(g(x), saves(x))
(found,) = [found for found in torch.vm.regions(g) if "ops" in found]
same, found["state"], found["ops"], found["hits"]  # (True, 'compiled', 6, 5)
```

A write to a tensor that comes from outside the path is an operation of the
graph as well: to an argument of the function or a view of one, to a buffer
of a module, to a parameter under `torch.no_grad()`, and what a batch norm
in training mode does to its running statistics. The graph has such a write
as the function wrote it, to one of its inputs:

```python
def outside(x, buffer):
    y = (x.sin() * 2 + 1).cos()
    buffer.add_(y.mean())
    return (y + buffer).exp().tanh().cos().sum()


g = torch.vm(outside, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), show))
mine, theirs = torch.zeros(()), torch.zeros(())
for _ in range(6):
    same = torch.equal(g(x, mine), outside(x, theirs))
same, torch.equal(mine, theirs), mine._version == theirs._version  # (True, True, True)
[(found["ops"], found["why"]) for found in torch.vm.regions(g) if "ops" in found]
# [(11, 'the function returns')]
print(shown[-1][0])
# def forward(self, x, buffer):
#     sin_default = torch.ops.aten.sin.default(x);  x = None
#     mul_tensor = torch.ops.aten.mul.Tensor(sin_default, 2);  sin_default = None
#     add_tensor = torch.ops.aten.add.Tensor(mul_tensor, 1, alpha = 1);  mul_tensor = None
#     cos_default = torch.ops.aten.cos.default(add_tensor);  add_tensor = None
#     mean_default = torch.ops.aten.mean.default(cos_default, dtype = None)
#     add__tensor = torch.ops.aten.add_.Tensor(buffer, mean_default, alpha = 1);  mean_default = add__tensor = None
#     add_tensor_1 = torch.ops.aten.add.Tensor(cos_default, buffer, alpha = 1);  cos_default = buffer = None
#     exp_default = torch.ops.aten.exp.default(add_tensor_1);  add_tensor_1 = None
#     tanh_default = torch.ops.aten.tanh.default(exp_default);  exp_default = None
#     cos_default_1 = torch.ops.aten.cos.default(tanh_default);  tanh_default = None
#     sum_default = torch.ops.aten.sum.default(cos_default_1, dtype = None);  cos_default_1 = None
#     return (sum_default,)
```

A pass that returns the graph has it run on the kernels of eager mode, so
the write is eager's, the version counter of the tensor included. With
`AOTAutograd()` and `Inductor()` the tensor holds what eager mode leaves in
it when the call returns. What they made writes a tensor back once, behind
the operations, and the engine moves its version counter on to where the
operations of the function leave it, so the counter is eager's with all
three. The running statistics of a batch norm are the one difference:
eager's kernel writes them without moving their counters, and what a
compiler made moves them.

What such a graph wrote cannot be taken back, and the graph cannot be run a
second time for the same call. Everything else that a path does is undone by
not being shown: where a check fails behind what was made of the graph, its
results are dropped and the instructions of the function run. A write to an
argument has happened. So nothing may fail behind such a write. A path that
has one ends in front of whatever a form would have to check behind what was
made of the graph, and a path that already has such a check does not take a
write to a tensor from outside, and ends in front of it:

- a branch, the end of a loop or a guard of the code on a value that the
  graph computes;
- an instruction that runs behind the graph and is not known to succeed;
- a store into an object that the graph gives (an attribute of a tensor it
  computed), which is done behind the graph;
- a read of what a write changes of a tensor from in front of the graph: its
  `_version`, its `grad_fn`, `is_leaf`, whatever its class computes for a
  name. Its sizes, dtype, device and `requires_grad` are read in front of
  the graph, since no write of a path changes those (below).

The operations of the graph must not fail either. A form is for exact sizes
and dtypes, so what is left to fail at a later call is an operation that
fails for what a tensor holds: an index that is out of range, a division of
whole numbers by zero. A path that writes to a tensor from outside holds
only operations that cannot: views; the operators that work on each element
by itself and those that reduce, which say so in their tags (on whole
numbers only those of them that cannot fail there); and the kernels of
layers and of putting tensors together that the engine keeps a list of
(convolutions, `linear`, the norms, the pools, `cat`, `softmax`, `where`
and the like). It ends in front of the first operation that is not known to
be one of these, an index by a tensor, `gather`, `embedding` and
`cross_entropy` among them, and a path that already holds such an operation
does not take a write to a tensor from outside. What stands behind the end
of the path runs by itself, or starts the next graph, and raises what eager
mode raises, with everything as eager mode leaves it:

```python
import copy


class Counts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.register_buffer("total", torch.zeros(()))

    def forward(self, x, index):
        y = (x.sin() * 2 + 1).cos()
        self.calls += 1
        self.total.add_(y.mean())
        return y[index].exp().sum() + self.total


def outcome(call, index):
    try:
        return float(call(rows, index))
    except Exception as error:
        return type(error).__name__


model = Counts().eval()
reference = copy.deepcopy(model)
g = torch.vm(model, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), P.AOTAutograd()))
rows, good, bad = torch.randn(4, 4), torch.tensor([0, 2]), torch.tensor([1, 7])
went = []
with torch.no_grad():
    for index in [good] * 6 + [bad, good]:
        went.append((outcome(g, index), outcome(reference, index)))
all(ours == theirs or abs(ours - theirs) < 1e-4 for ours, theirs in went), went[6]
# (True, ('IndexError', 'IndexError'))
model.calls, reference.calls, torch.allclose(model.total, reference.total)  # (8, 8, True)
[(found["ops"], found["why"]) for found in torch.vm.regions(g) if "ops" in found]
# [(6, 'can fail for what its tensors hold, on a path that writes to a tensor from outside')]
```

What is the same in every call that a form is for is no such check. A form
is for exact sizes of what the graph is given, and the sizes of what the
graph computes follow from those, so `y.dim()`, `y.shape[0] > 2`, a loop over
`range(y.shape[1])` and the checks that `nn.BatchNorm2d` makes of its input
stand behind a write. Such a check is decided when the path is taken down
and is not run again with what was made of the graph, so the graph does not
give back the tensor it looks at. Sizes that depend on what the tensors
hold (`y[y > 0]`, `torch.nonzero(y)`) are not of that kind. An operator with
a kernel of its own says in its tags whether its sizes do. One that is made
of other operators (`torch.argwhere`, `torch.where` with one argument,
`repeat_interleave` with a tensor of counts) has no tag that says it: where
graphs are exported, such an operator is run once more when its call is
taken down, on tensors of the same sizes and dtypes that hold nothing, and
its sizes are taken to follow from those of what it is given only if it
runs there and gives the same sizes (and the same dtypes, unless the call
is under autocast, which does not cast tensors that hold nothing). A branch
on a size that does not is a check of every call:

```python
def weighs(x):
    y = (x.sin() * 2 + 1).cos()
    if torch.argwhere(y > 0).shape[0] > 6:
        return (y * 2).exp().tanh().sum()
    return (y * 3).exp().tanh().sum() - 1


g = torch.vm(weighs, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), keep))
all(torch.equal(g(sign * x), weighs(sign * x)) for sign in (1, 1, 1, 1, 1, -1, 3, -1, 1))  # True
```

Behind a write to a tensor from outside there is no such check, and the path
ends in front of the operation, which is one that is not known to be unable
to fail:

```python
def picks(x, buffer):
    y = (x.sin() * 2 + 1).cos()
    buffer.add_(y.mean())
    kept = y[y > 0]
    if kept.shape[0] > 4:
        kept = kept * 2
    return kept.exp().sum() + buffer


g = torch.vm(picks, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), keep))
mine, theirs = torch.zeros(()), torch.zeros(())
for sign in (1, 1, 1, -1, -1, 1, 1, -1):
    same = torch.equal(g(sign * x, mine), picks(sign * x, theirs))
same, torch.equal(mine, theirs)  # (True, True)
[(found["ops"], found["why"]) for found in torch.vm.regions(g) if "ops" in found]
# [(7, 'can fail for what its tensors hold, on a path that writes to a tensor from outside'),
#  (4, 'the function returns')]
```

An operation of a graph may also take such a value where the instructions
compute it behind the graph, from the sizes of a tensor the graph gives:
the graph is made with the value of the call it was exported from, and the
form checks behind the graph that it came out the same. That holds for what
a form can compare (a number, a text, a tuple or a list of them); a tensor
that is made behind the graph is no such value, and an operation that takes
one ends the path.

```python
def reshapes(x):
    y = (x.sin() * 2 + 1).cos()
    rows, columns = y.shape
    return y.view(rows * columns).exp().tanh().sum()


g = torch.vm(reshapes, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), show))
for _ in range(6):
    same = torch.equal(g(x), reshapes(x))
(found,) = [found for found in torch.vm.regions(g) if "ops" in found]
same, found["ops"], found["why"]  # (True, 8, 'the function returns')
"aten.view.default(cos_default, [16])" in shown[-1][0]  # True
```

A graph without such writes whose callable raises the first time it is
called is given up, and the instructions of the function run in its place:
the call returns what eager mode returns. With a write to a tensor from
outside in the graph there is no such second run, since the instructions
would write again what the callable may have written already. What the
callable raises goes out of the call, and where that was its first run the
graph is given up, so that the calls after it run the instructions:

```python
def raising(gm, input_types):
    def fails(tensor):
        raise RuntimeError("what was made of the graph does not run")

    first = next(node for node in gm.graph.nodes if node.op == "placeholder")
    last = next(node for node in gm.graph.nodes if node.op == "output")
    with gm.graph.inserting_before(last):
        gm.graph.call_function(fails, (first,))
    gm.recompile()
    return gm


g = torch.vm(outside, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), raising))
mine, theirs = torch.zeros(()), torch.zeros(())
went = []
for _ in range(5):
    try:
        out = g(x, mine)
        went.append("returned")
    except RuntimeError as error:
        went.append("raised")
    expected = outside(x, theirs)
went  # ['returned', 'returned', 'raised', 'returned', 'returned']
torch.equal(mine, theirs)  # True: the graph had written, once, before it raised
found = torch.vm.regions(g)[0]
found["state"], found["reason"]
# ('vm', 'the compiled graph failed: RuntimeError: what was made of the graph does not run')
```

What a compiler makes of writes to its inputs depends on which of them
share memory, and where: it writes a view of an input back to the place the
view had when it compiled. A form of a path that writes to tensors from
outside is therefore for tensors that lie in memory as those it was
exported for did: which of them have one storage, and where in it each of
those starts (their sizes and strides are the form's anyway). A tensor with
its second row, the same tensor with its third row, and two tensors of
their own are three forms, and `regions` says of each how its tensors lie:

```python
def shares(x, part):
    y = (x.sin() * 2).cos()
    part.mul_(2)
    x.add_(y)
    return (x * part.sum()).tanh().exp() + y


g = torch.vm(shares, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), P.AOTAutograd()))
same = []
for row in (1, 1, 1, 1, 2, 2, 2, None, None, None, 1, 2):
    bases = [torch.arange(12.0).reshape(3, 4) / 7 for _ in range(2)]
    parts = [torch.ones(4) if row is None else base[row] for base in bases]
    got, expected = g(bases[0], parts[0]), shares(bases[1], parts[1])
    same.append(torch.allclose(got, expected) and torch.equal(bases[0], bases[1]))
all(same)  # True
(found,) = [found for found in torch.vm.regions(g) if "ops" in found]
found["ops"], found["exports"]  # (10, 3)
sorted(form["checks"][-1] for form in found["forms"])
# ['no two tensors share memory', 'share memory: %x.1 at 0, %part.2 at 4', 'share memory: %x.1 at 0, %part.2 at 8']
```

A write that changes the sizes of a tensor from outside, or whether it
requires a gradient (`buffer.t_()`, a buffer that is written with a value
that requires one), is no operation of a path: a form is for the tensors as
a call hands them over, and what is made of a graph is compiled for those.
That shows when the operation has run. The path is taken down again and
ends in front of it, the operation runs by itself, and the next graph starts
behind it:

```python
def turns(x, buffer):
    y = (x.sin() * 2 + 1).cos()
    buffer.t_()
    return (y.sum() + buffer.sum()).exp().tanh() * buffer.shape[0]


g = torch.vm(turns, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), keep))
for _ in range(8):
    mine, theirs = torch.arange(6.0).reshape(2, 3), torch.arange(6.0).reshape(2, 3)
    same = torch.equal(g(x, mine), turns(x, theirs))
same, mine.shape == theirs.shape  # (True, True)
[(found["ops"], found["exports"], found["why"]) for found in torch.vm.regions(g) if "ops" in found]
# [(4, 1, 'changes the sizes of a tensor from outside, or whether it requires a gradient'), (6, 1, 'the function returns')]
```

An operation that draws random numbers is under the same rule, since the
state of the generator cannot be taken back either. With `Graphs()` alone
such an operation ends a path, the instructions draw, and every number is
eager's. With a pass over graphs in the list it is an operation of the
graph, in the order of the function. A pass that returns the graph has it
run on the kernels of eager mode, so with the same seed the numbers are
eager's too, and so are those of `AOTAutograd()`, which runs the kernel of
the draw as it is. What `Inductor()` makes of a dropout draws other numbers.
`torch.vm.is_exact` is false for every list with a compiler in it:

```python
def drops(x):
    y = (x.sin() * 2 + 1).cos()
    return (torch.nn.functional.dropout(y, 0.5) + 1).exp().tanh()


def drawn(passes):
    g = torch.vm(drops, passes=passes)
    torch.manual_seed(3)
    ours = [g(x) for _ in range(6)]
    torch.manual_seed(3)
    theirs = [drops(x) for _ in range(6)]
    graphs = [(found["ops"], found["why"]) for found in torch.vm.regions(g) if "ops" in found]
    return [torch.allclose(a, b) for a, b in zip(ours, theirs)], graphs


alone = (*torch.vm.DEFAULT_PASSES, P.Graphs())
drawn(alone)
# ([True, True, True, True, True, True], [(4, 'in dropout: aten::dropout draws random numbers')])
drawn((*alone, keep))  # ([True, True, True, True, True, True], [(8, 'the function returns')])
drawn((*alone, P.AOTAutograd()))  # ([True, True, True, True, True, True], [(8, 'the function returns')])
drawn((*alone, P.Inductor()))  # ([True, True, False, False, False, False], [(8, 'the function returns')])
torch.vm.is_exact(alone), torch.vm.is_exact((*alone, P.AOTAutograd())), torch.vm.is_exact((*alone, P.Inductor()))
# (True, False, False)
```

With these the forward of a model in training mode is a graph: a batch norm
writes its statistics and counts its batches in it, a dropout draws in it.
Going backward and the step of the optimizer are as before; with
`AOTAutograd()` or `Inductor()` what the forward returns has the compiled
backward as its `grad_fn`. A loss that the same function computes is not in
that graph: `cross_entropy` can fail for what its labels hold, so the path
ends in front of it, and the loss is computed behind the graph.

```python
import copy


class Small(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.norm = torch.nn.BatchNorm2d(8)
        self.fc = torch.nn.Linear(8 * 8 * 8, 4)

    def forward(self, x):
        y = torch.relu(self.norm(self.conv(x)))
        return self.fc(torch.flatten(y, 1))


model = Small().train()
reference = copy.deepcopy(model)
g = torch.vm(model, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), keep))
images, labels = torch.randn(8, 3, 8, 8), torch.randint(0, 4, (8,))
losses = []
for net, fn in ((model, g), (reference, reference)):
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    for _ in range(8):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(fn(images), labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
losses[:8] == losses[8:]  # True
state = reference.state_dict()
all(torch.equal(value, state[name]) for name, value in model.state_dict().items())  # True
int(model.norm.num_batches_tracked)  # 8
(found,) = [found for found in torch.vm.regions(g) if "ops" in found]
found["ops"], found["hits"], found["why"]  # (6, 6, 'the function returns')
```

An operation with several results (`x.chunk(2)`, `torch.max(x, 0)`) ends a
path. A write that a path takes as an operation is
not among what `regions` lists as left out.

### Stores into objects

A store into a Python object from outside the path is an effect of the path,
and does not end it: `obj.name = value`, `d[k] = v` on a dict, `l[i] = v`,
`l.append(x)` and `l.extend(xs)` on a list. A store under a key compares it
with the keys the dict holds, as a lookup does, so it is an effect only for
a dict whose keys are compared without Python code ("Passes over graphs");
into any other dict it ends the path. While a path is taken down, the
instructions of the function do the stores where the function has them. Where
what was made of the graph runs in their place, the stores are done when the
path ends, all of them, in the order of the function, and behind every check
of the path and of the form. So nothing is stored for a call that the path
turns out not to be for: that call runs the instructions, which do the
stores themselves. The same holds where what was made of the graph raises:
the instructions run, do the stores in front of what raises, and raise. A
graph that writes to a tensor from outside is the exception ("Writes"): what
was made of it is not run a second time, what it raises goes out, and the
stores of that call are not done.

A read of what a store of the path put there, behind the store, is given the
stored value (`self.n = k` and then `self.n`). What is stored is not one of
the checks: a counter that is another number at every call does not make
another form. A tensor that the graph computed is stored as the object the
graph gave, so two names for it stay one object:

```python
class Keeps(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)
        self.calls = 0
        self.seen = []
        self.last = None

    def forward(self, x, notes):
        y = (self.fc(x).sin() * 2).cos().tanh()
        self.calls += 1
        self.seen.append(y.shape[0])
        notes["calls"] = self.calls
        self.last = y
        return (y + 1).exp().sin().cos().tanh()


model = Keeps().eval()
g = torch.vm(model, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), keep))
v, notes = torch.randn(3, 4), {}
with torch.no_grad():
    for _ in range(8):
        out = g(v, notes)
model.calls, model.seen, notes  # (8, [3, 3, 3, 3, 3, 3, 3, 3], {'calls': 8})
torch.equal(out, (model.last + 1).exp().sin().cos().tanh())  # True
(found,) = [found for found in torch.vm.regions(g) if "ops" in found]
found["ops"], found["hits"], found["exports"]  # (10, 6, 1): one graph across the stores, exported once
```

A store that would run Python code is no effect, and ends the path in front
of itself: into an object whose class has a `__setattr__` of its own, through
a property or another descriptor that takes stores, and on a module where
`nn.Module.__setattr__` does more than one store (a `Parameter` or a module
that is assigned is registered).

A store also lets go of what it replaces: the attribute that was there, the
value under the key, the item at the place. That is looked at as what a
variable lets go of is ("Passes over graphs"), for every call in front of
what was made of the graph and once more when the path ends: where a store
would end an object whose end runs Python code, the path is not for the
call, and the instructions run and end it where the store stands.

```python
class Guarded(Keeps):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)


model = Guarded().eval()
g = torch.vm(model, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), keep))
notes = {}
with torch.no_grad():
    for _ in range(8):
        out = g(v, notes)
model.calls, notes  # (8, {'calls': 8})
[(found["ops"], found["why"]) for found in torch.vm.regions(g) if "ops" in found]
# [(5, 'stores into an object'), (5, 'the function returns')]
```

Since the stores are done when the path ends, what the path reads behind a
store must not be something the store changes. A store changes one dict or
list: the dict an object keeps its attributes in, the dict a module keeps
its buffers in, or the list or dict itself. A read goes through dicts and
lists as well: an attribute through the dict of its object, `len`, a loop or
an item through the list or dict, an iterator or a view through what the
path made it of. A read under a name that the path stored under (an
attribute, an entry of a dict, also by `getattr` or `hasattr`) is the stored
value where it is of the object that was stored into, in the same way. Every
other read behind a store goes past it only where it goes through another
dict or list than the store changes, and the path checks that at every
call: two arguments that are two objects in one call may be one object in
the next, two objects may keep their attributes in one dict, and a dict that
is handed about may be the one an object keeps them in. A call that fails
such a check runs the instructions. Where the read goes through what a store
changed in the very call the path is taken down in (`h.__dict__["n"] = 7`
and then `h.n`; `a.append(1)` and then `len(a)`), the path ends in front of
the read, and so it does for an iterator or a view from outside the path,
of which it is not known what it goes over, and for a read of an attribute
of an object whose class reads its attributes in a function of its own (a
`__getattribute__` written in Python): what that function gives under a
name need not be what was stored under it. A store at a place of a list
(`l[0] = v`) is to a place the list has by then, the items that the stores
of the path in front of it append counted.

```python
class Box:
    def __init__(self):
        self.n = 0
        self.items = []


def two_names(x, a, b):
    y = (x.sin() * 2 + 1).cos()
    a.n = 1
    b.n = 2
    b.items.append(a.n)
    return (y + a.n).exp().tanh().sum()


g = torch.vm(two_names, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), keep))
appended = []
for shared in (False, False, False, False, False, True, False, True):
    a = Box()
    b = a if shared else Box()
    out = g(v, a, b)
    appended.append(b.items[-1])
appended  # [1, 1, 1, 1, 1, 2, 1, 2]: where the two are one object, `a.n` is what `b.n = 2` stored
(found,) = [found for found in torch.vm.regions(g) if "ops" in found]
found["ops"], found["exports"], found["hits"]  # (8, 2, 5)
```

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

- A path starts and ends in one function, and a call that it cannot go
  into ends it ("Paths across calls").
- A path per way. A function whose calls go many ways through it has a path
  and an export for each, and they share the `max_variants` of the graph.
  So does a loop that goes round another number of times on every call.
- Nothing may fail behind a write to a tensor from outside the path
  ("Writes"). A path with such a write, or with a draw of random numbers,
  ends in front of a branch on what a tensor of the graph holds and of a
  size that depends on it, and in front of the first operation that is not
  known to be unable to fail for what its tensors hold (an index by a
  tensor, `gather`, `embedding`, `cross_entropy`): the forward of a model in
  training mode with batch norms and its loss computed in the same function
  is a graph up to the loss. A graph with such a write has no second run
  where what was made of it raises. An operation with several results ends
  a path.
- What a path reads behind a store of its own into an object is checked at
  every call to go through another dict or list than the store changes, a
  check per store ("Stores into objects"); a read through what a store
  changed ends the path.
- In the call in which a path is taken down, a variable that the function
  deletes inside the path (`del name`) is let go of where the path ends and
  not at the `del`. Where it held the last reference to an object whose end
  runs Python code, that code runs behind what the call reads behind the
  `del`. A call that would run what was made of the path is turned down for
  such an object ("Passes over graphs"), and is then one in which a path is
  taken down again, for as long as the graph may get another form.
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
