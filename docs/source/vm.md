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

With `Graphs()` in the list the function computes what it computes without
it. The engine runs a graph as it runs the rest of the function: the same
kernels, on the same values, in the same order. The results are those of
eager mode bit for bit, and so are the gradients: the backward of a graph
whose tensors need a gradient is derived by the engine, from what eager's
own autograd nodes run ("Gradients"). `torch.vm.is_exact` is true for such
a list. What the pass adds is what `torch.vm.regions` reports:

```python
import torch

P = torch.vm.passes


def f(x, w):
    return torch.tanh(x @ w).sum() + (x * 2).mean()


g = torch.vm(f, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs()))
x, w = torch.randn(4, 8), torch.randn(8, 8)
for _ in range(3):
    out = g(x, w)
torch.equal(out, f(x, w))  # True
torch.vm.is_exact((*torch.vm.DEFAULT_PASSES, P.Graphs()))  # True
torch.vm.regions(g)
# [{'where': 'f (block 0)[2]', 'state': 'built', 'ops': 6,
#   'operators': ['aten::matmul', 'aten::tanh', 'aten::sum',
#   'aten::mul.Tensor', 'aten::mean', 'aten::add.Tensor'], 'inputs': 2,
#   'outputs': 1, 'held': 1, 'branches': 0, 'calls': 0, 'paths': 1,
#   'ends': [(0, 12)], 'why': 'the function returns', 'variants': 0,
#   'hits': 0, 'misses': 2, 'exports': 0, 'version': 1, 'forms': []}]
```

`where` names the function, the block, and the instruction of the first
operation of the graph, whether its path starts there or with the function
(`g.dump()` shows them). The state `"built"` says that the engine runs
the graph itself. `ops` and `operators` are the operations of the path that
was taken down last, and `inputs` and `outputs` how many values that path
takes from the code in front of it (here `x` and `w`) and how many the code
behind it reads, `held` how many of those something can still hold when the
function goes on ("Gradients" has what that decides). `ends` is the block
and the instruction the function goes on at behind the path, and `why` what
ends the path there. `misses` counts
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
- behind an operation that writes to a tensor from outside the path or draws
  random numbers: whatever a form would have to check behind what was made
  of the graph, and such an operation behind a check of that kind ("Writes"
  below); with `Graphs()` alone, an operation that draws random numbers (a
  dropout that is told not to drop draws none);
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


# The parameters are made to need no gradient: these are graphs that the
# engine only runs ("Gradients" has what it does for those that need one).
net = Net().eval().requires_grad_(False)
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
- of a `functools.partial`, of a class (constructing an object), and of a
  function that was made on the path (a closure that the function defines
  and calls);
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

That is the graph of a call whose tensors need no gradient. For a call
whose tensors need one the engine derives the backward first, and a pass is
handed two graphs whose tensors need none, the forward and the backward
("Gradients"). With `Graphs(backward=False)` it is handed the one graph of
such a call, with `requires_grad` in its `input_types`.

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
- `torch.vm.passes.Inductor()` has Inductor compile the graph. A graph
  whose tensors need a gradient reaches it as the engine's two graphs, the
  forward and the backward, whose tensors need none ("Gradients"). With
  `AOTAutograd()` in front of it in the list, AOTAutograd derives the
  backward instead, and Inductor compiles what it traced. With neither
  (`Graphs(backward=False)` and `Inductor()` by itself) nothing derives a
  backward: a graph whose tensors need a gradient is not handed to
  Inductor and runs on eager's tape.

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
# ['AddBackward0', 'AddBackward0', 'AddBackward0', 'AddBackward0', 'GraphBackward']
torch.allclose(out, f(x, w))  # True
(found,) = torch.vm.regions(g)
found["state"], found["variants"], found["hits"], found["misses"]  # ('compiled', 1, 3, 1)
```

The first call is the one the block is profiled on and the second the one
the graph is exported from, and both run the operations of the function.
With gradients recorded the next two calls run the graph on eager's tape,
while the engine takes its backward down. From the fifth call on what
Inductor made of the engine's forward runs, and what it returns has the
node of the graph, `GraphBackward`, as its `grad_fn`; the backward is
compiled when gradients first arrive. With `AOTAutograd()` in the list the
compiled graph runs from the third call on, and what it returns has one
compiled autograd function, `CompiledFunctionBackward`, as its `grad_fn`.
A call under `torch.no_grad()` is another kind of call and is compiled by
itself, without a backward. What a compiler makes of the operations is not
what eager runs,
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
state of the generator cannot be taken back either. In a graph that the
engine only runs (`Graphs()` alone, and no call has brought a tensor that
needs a gradient) such an operation ends a path, the instructions draw, and
every number is eager's. Where something stands for the graph, with a pass
over graphs in the list or with the engine's backward ("Gradients"), it is
an operation of the graph, in the order of the function. The engine's own
programs and a pass that returns the graph run the kernels of eager mode,
so with the same seed the numbers are eager's too, and so are those of
`AOTAutograd()`, which runs the kernel of the draw as it is. What
`Inductor()` makes of a draw (of a dropout, of `torch.rand_like`, of
`torch.bernoulli`) draws other numbers. `torch.vm.is_exact` is false for
every list with a compiler in it:

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
Its backward is the engine's ("Gradients"), and the step of the optimizer
is as before; with `AOTAutograd()` in the list what the forward returns has
AOTAutograd's compiled backward as its `grad_fn`. A loss that the same
function computes is not in
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
[form.get("backward") for form in found["forms"]]  # ["the engine's backward"]
```

A write that a path takes as an operation is not among what `regions`
lists as left out.

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
of which it is not known what it goes over. A read of an attribute of an
object whose class reads its attributes in a function of its own (a
`__getattribute__` written in Python) is not given what was stored: the
path goes into the function, and what that reads of the object is looked
at behind the stores ("Classes that read their attributes themselves"). A
store at a place of a list
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

### Operations with several results

An operation that gives several tensors is an operation of a graph:
`x.chunk(2)`, `x.split(3)`, `x.unbind(0)`, `torch.max(x, 1)` and
`torch.min(x, 1)`, `torch.topk`, `torch.sort`. What the function does with
the tuple it gives is computed in the graph: unpacking it
(`a, b = x.chunk(2)`), an item of it (`parts[1]`), an attribute of it
(`torch.max(x, 1).values`).

```python
def several(x):
    y = (x.sin() * 2 + 1).cos()
    a, b = y.chunk(2, 1)
    top, where = torch.max(a, 1)
    lowest = torch.min(b, 1)
    return (a * b).exp() + top.unsqueeze(1) + lowest.values.unsqueeze(1) + y.split(1, 1)[1]


g = torch.vm(several, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), show))
x = torch.randn(4, 4)
for _ in range(6):
    same = torch.equal(g(x), several(x))
(found,) = [found for found in torch.vm.regions(g) if "ops" in found]
same, found["ops"], found["hits"], found["why"]  # (True, 15, 4, 'the function returns')
print(shown[-1][0])
# def forward(self, x):
#     sin_default = torch.ops.aten.sin.default(x);  x = None
#     mul_tensor = torch.ops.aten.mul.Tensor(sin_default, 2);  sin_default = None
#     add_tensor = torch.ops.aten.add.Tensor(mul_tensor, 1, alpha = 1);  mul_tensor = None
#     cos_default = torch.ops.aten.cos.default(add_tensor);  add_tensor = None
#     chunk = cos_default.chunk(2, 1)
#     getitem = chunk[1]
#     getitem_1 = chunk[0];  chunk = None
#     max_1 = torch.max(getitem_1, 1)
#     getitem_2 = max_1[0];  max_1 = None
#     min_1 = torch.min(getitem, 1)
#     mul_tensor_1 = torch.ops.aten.mul.Tensor(getitem_1, getitem);  getitem_1 = getitem = None
#     exp_default = torch.ops.aten.exp.default(mul_tensor_1);  mul_tensor_1 = None
#     unsqueeze_default = torch.ops.aten.unsqueeze.default(getitem_2, 1);  getitem_2 = None
#     add_tensor_1 = torch.ops.aten.add.Tensor(exp_default, unsqueeze_default, alpha = 1);  exp_default = unsqueeze_default = None
#     getattr_1 = min_1.values;  min_1 = None
#     unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(getattr_1, 1);  getattr_1 = None
#     add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, unsqueeze_default_1, alpha = 1);  add_tensor_1 = unsqueeze_default_1 = None
#     split = torch.split(cos_default, 1, 1)
#     getitem_3 = split[1];  split = None
#     add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, getitem_3, alpha = 1);  add_tensor_2 = getitem_3 = None
#     return (cos_default, add_tensor_3)
```

A graph gives back tensors, not the tuple. Where what such an operation
gave is needed whole behind the path (the function returns it, a loop goes
over it), the path is taken down again at the next call and ends in front
of the operation, which the instructions run:

```python
def whole(x):
    y = (x.sin() * 2 + 1).cos().exp().tanh()
    return y.chunk(2, 1)


g = torch.vm(whole, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), keep))
for _ in range(6):
    got, expected = g(x), whole(x)
[torch.equal(a, b) for a, b in zip(got, expected)]  # [True, True]
(found,) = [found for found in torch.vm.regions(g) if "ops" in found]
found["ops"], found["why"]  # (6, 'what an operation with several results gives is used whole')
```

A method of a tensor that is written in Python (`Tensor.split`) is gone
into with the tensor of the graph as its `self`. `isinstance` of a class
whose `__instancecheck__` is a Python function is followed into that
function, and the path checks that what it returned is a bool: for anything
else `isinstance` gives whether it is true, as the interpreter does, and the
path ends in front of the call. A C function
that is called with what the code spreads (`torch.add(*pair)`) is an
operation ("Methods read off a tensor, and what a call spreads" has what may
be spread, "What a call names with a dict" what `**` may hand over):

```python
class Meta(type):
    def __instancecheck__(cls, obj):
        return hasattr(obj, "shape")


class Shaped(metaclass=Meta):
    pass


def spreads(x):
    y = (x.sin() * 2 + 1).cos()
    pair = (y, x)
    if isinstance(y, Shaped):
        y = torch.add(*pair)
    return y.exp().tanh().sum()


g = torch.vm(spreads, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), keep))
for _ in range(6):
    same = torch.equal(g(x), spreads(x))
(found,) = [found for found in torch.vm.regions(g) if "ops" in found]
same, found["ops"], found["why"]  # (True, 8, 'the function returns')
```

### Sizes on a path

The sizes of a tensor are a tuple of numbers, and a path goes through what
a function does with them as it goes through numbers: `y.shape[0] > 2`,
`n, c = y.shape`, and a comparison of the sizes of one tensor with those of
another or with a tuple (`a.shape != b.shape`, `y.shape == (2, 3)`). Such a
comparison is a branch of the path; where the sizes are those of a tensor
the graph computes, they follow from the sizes of what the graph is given,
which a form checks, so other sizes are another form:

```python
def sized(x, y):
    a = (x.sin() + 1).cos()
    if a.shape != y.shape:
        raise ValueError("sizes")
    b = (a * y).tanh()
    if b.shape == (2, 3):
        b = b + 1
    return b.exp()


g = torch.vm(sized, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), keep))
x, y = torch.randn(2, 3), torch.randn(2, 3)
for _ in range(6):
    same = torch.equal(g(x, y), sized(x, y))
(found,) = [found for found in torch.vm.regions(g) if "ops" in found]
same, found["ops"], found["why"]  # (True, 7, 'the function returns')
try:
    g(x, torch.randn(2, 1))
except ValueError as error:
    said = str(error)
said  # 'sizes'
```

Going over the sizes is part of a path as well, which is what
`nn.AdaptiveAvgPool2d((1, 1))` does to work out its output size: the pool,
and whatever stands behind it, is in the graph of the model.

```python
class Pooled(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(8, 4)

    def forward(self, x):
        y = torch.relu(self.conv(x)).sin().cos()
        return self.fc(torch.flatten(self.pool(y), 1)).tanh()


model = Pooled().eval()
g = torch.vm(model, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), keep))
images = torch.randn(2, 3, 8, 8)
with torch.no_grad():
    for _ in range(6):
        same = torch.equal(g(images), model(images))
(found,) = [found for found in torch.vm.regions(g) if "ops" in found]
same, found["ops"], found["why"]  # (True, 8, 'the function returns')
```

### Texts and sequences on a path

What a function computes with its plain values is part of a path too, where
C code alone does it. Turning a value into text is: an f-string, `format`
and `%` on a number, a text, `None`, a dtype or a device, and on a list or a
tuple of those, the sizes of a tensor among them; so are the methods of a
number, a text or a tuple of such on such values (`note.startswith("torch")`).
What a tensor holds is not text a path makes (`f"{y.sum()}"` ends it).
Putting sequences together is on a path as well: `+` on two texts, two lists
or two tuples, `*` with a whole number, and a list that is extended by a list
or a tuple (`(*y.shape[:-1], 2, -1)`, `y.shape[:-1] + (h, w)`, `[1] * n`). A
list, a tuple or a slice made of nothing but sizes of what the graph computes
and of constants is the same in every call that a form is for, as a single
size is, so an operation takes it as a constant of the graph. `a += b` on a
list is no instruction of a path: it changes the list, which may be the
caller's. A layer that checks what it is given with a message, as the norm
layers of vision libraries do, is one graph:

```python
class Guarded(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(channels)
        self.drop = torch.nn.Dropout(0.2)

    def forward(self, x):
        y = self.conv(x)
        torch._assert(y.ndim == 4, f"expected 4D input (got {y.ndim}D input)")
        y = self.drop(torch.relu(self.bn(y)))
        return y.view((*y.shape[:-1], 2, -1)).sum(-1)


model = Guarded(4).eval()
g = torch.vm(model, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs()))
images = torch.randn(2, 4, 8, 8)
with torch.no_grad():
    for _ in range(4):
        same = torch.equal(g(images), model(images))
same, [(found["ops"], found.get("why"), found.get("reason")) for found in torch.vm.regions(g)]
# (True, [(6, 'the function returns', None)])
try:
    g(torch.randn(4, 8, 8))
except AssertionError as error:
    said = str(error)
said  # 'expected 4D input (got 3D input)'
```

The dropout of the layer is among the six operations: a dropout of a model
that is not training drops nothing and draws nothing, and is an operation
like any other, with `Graphs()` alone as well ("Writes" has what one that
drops is).

### Methods read off a tensor, and what a call spreads

`y.to(device)` is a call of a method by its name. Where the code reads the
method first and calls it later, or calls it with `*` (`y.view(*shape)`),
what is called is a method that was bound to the tensor; it is the same
operation, the method of that name of the tensor it was read off. What a
call spreads with `*` is handed over item by item. The items of a tuple or
a list the path built are the values it was built of. Those of any other
list or tuple (the function's own `*sizes`, a `torch.Size`, a list of the
caller) are read off it, with a check at every call that it has that many:

```python
def hands_on(x, pair, *sizes):
    y = (x.sin() + 1).cos()
    shape = y.shape[:-1] + (2, 2)
    y = y.view(*shape).reshape(*sizes)
    return (torch.mul(y, *pair) * 2).tanh().exp()


g = torch.vm(hands_on, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), keep))
x, w = torch.randn(3, 4), torch.randn(4)
for _ in range(6):
    same = torch.equal(g(x, (w,), 3, 4), hands_on(x, (w,), 3, 4))
(found,) = [found for found in torch.vm.regions(g) if "ops" in found]
same, found["ops"], found["why"]  # (True, 9, 'the function returns')
torch.equal(g(x, [w[:1]], 2, 6), hands_on(x, [w[:1]], 2, 6))  # True: other items are another form
```

A call with `*` is an operation only where the operator it runs enters no
Python code. On a GPU that is not so for `bmm`, and for what comes down to
it (`matmul` of two stacks of matrices), whose CUDA kernel is entered
through a Python function (`torch._native`): a path ends in front of such a
call of them, where a call that is written out goes on.

### What a call names with a dict, and what gives back what it was handed

A C function that is called with `**` is an operation where the path built
the dict under names: the call with those names. An operator that draws
random numbers only at a rate (`scaled_dot_product_attention` with its
`dropout_p`) draws nothing at a rate of zero and is an operation like any
other, with the rate checked at every call; at another rate it is one under
the rule of "Writes", as every operation that only draws now is, whether it
is a call site or not. A call with `**` is an operation under the condition
of one with `*`, that the operator it runs enters no Python code: on a GPU
the attention comes down to `bmm` where torch is built without the fused
attention kernels, and a path then ends in front of a call of it with `**`.
`contiguous` of a tensor that is contiguous runs no
kernel and gives the tensor back; it is the operation all the same, which
copies where the tensor of another call is laid out otherwise. A tuple that
a function gives back and its caller unpacks is the values it was made of,
so a tensor that comes back next to something else is still the operation
that gave it, and a method of a tensor that is written in Python
(`Tensor.split`) can be read off such a tensor and called later:

```python
def gives_two(y, **kwargs):
    left, right = y.split(2, dim=-1)
    return (left * right).transpose(0, 1).contiguous(), None


def attends(q, weight):
    y = (q.sin() + 1).cos()
    rate = {"dropout_p": 0.0}
    out = torch.nn.functional.scaled_dot_product_attention(y, y, y, **rate)
    out, nothing = gives_two(out, scale=None)
    return torch.nn.functional.linear(*(out, weight), **{"bias": None}).tanh()


g = torch.vm(attends, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), keep))
q, weight = torch.randn(2, 2, 5, 4), torch.randn(3, 2)
for _ in range(6):
    same = torch.equal(g(q, weight), attends(q, weight))
(found,) = [found for found in torch.vm.regions(g) if "ops" in found]
same, found["ops"], found["why"]  # (True, 10, 'the function returns')
```

A list or a tuple that the path put together of sizes of what the graph
computes and of numbers from in front of it (`(*x.shape[:-1], -1,
self.heads)`) is the same in every call where the path checks those numbers,
which it does when an operation takes such a sequence.

### Classes that read their attributes themselves

A read of an attribute of an object whose class has a `__getattribute__`
written in Python goes into that function, as a read goes into a property
("Paths across calls"), unless the class has a `__getattr__` as well: what
the function raises is then not what the read raises, and such a read is
left to the interpreter. `super()` in a method is an instruction of a path,
and `super().method(...)` is followed like any call of a method.
`super().__getattribute__(key)`, which is the lookup of `object`, is a call
the path makes where that lookup runs no Python code, and goes into the
getter where it finds a property. `getattr` and `hasattr` of such an object
go into the function as a read does: `hasattr` is `True` where the function
returns, whatever it returns, and where it raises `AttributeError`,
`hasattr` and `getattr` with a default make their answer of that, and the
path ends in front of the call. `a in b` goes into a `__contains__` written
in Python and gives whether what that returned is true, `a not in b` whether
it is not; the path goes on behind it where that was a bool, and ends in
front of the instruction for anything else. `hasattr(module, name)` and
`getattr(module, name, default)` for a name that an `nn.Module` does not
have are calls a path makes, at every call, so an attribute that appears
later is seen.

```python
class Settings:
    renamed = {"width": "hidden"}

    def __init__(self):
        self.hidden = 2.0

    def __getattribute__(self, key):
        if key != "renamed" and key in super().__getattribute__("renamed"):
            key = super().__getattribute__("renamed")[key]
        return super().__getattribute__(key)


def reads_settings(x, settings):
    y = (x.sin() + 1).cos()
    if hasattr(settings, "width"):
        y = y * settings.width
    return (y * getattr(settings, "hidden", 1.0)).tanh().exp()


settings = Settings()
g = torch.vm(reads_settings, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), keep))
for _ in range(6):
    same = torch.equal(g(x, settings), reads_settings(x, settings))
(found,) = [found for found in torch.vm.regions(g) if "ops" in found]
same, found["ops"], found["why"]  # (True, 7, 'the function returns')
settings.hidden = 3.0
torch.equal(g(x, settings), reads_settings(x, settings))  # True: what the function reads is read at every call
```

Behind a store of the path such a read goes into the function too. What the
function reads with `super().__getattribute__(key)` is then looked at as a
read of that attribute of the object: it goes on where no store of the path
can have changed it, under the checks a plain read has ("Stores into
objects"), and ends the path where one may have. A decoder that stores into
a cache and then reads its configuration is such code:

```python
class Cache:
    def __init__(self):
        self.keys = None


def keeps_and_reads(x, settings, cache):
    y = (x.sin() + 1).cos()
    cache.keys = y
    return (y * settings.hidden).tanh().exp()


cache = Cache()
g = torch.vm(keeps_and_reads, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), keep))
for _ in range(6):
    same = torch.equal(g(x, settings, cache), keeps_and_reads(x, settings, cache))
(found,) = [found for found in torch.vm.regions(g) if "ops" in found]
same, cache.keys is not None, found["ops"], found["why"]  # (True, True, 6, 'the function returns')
```

### Variables an inner function could take

A function with a variable that an inner function could take (a nested
function, a lambda, a generator expression) keeps it in a cell, which it
makes when it is entered, whether or not the inner function is ever made.
Such a cell is the path's own: making it, storing into it and reading it
are instructions of a path, and a read is what the path put into it last,
so a tensor that goes through a cell is still the operation that gave it.
Making the inner function ends the path in front of it.

```python
def adds_up(y, scale, more):
    total = y * scale
    if more:

        def bump():
            nonlocal total
            total = total + 1

        bump()
    return total + scale


def has_cells(x, more):
    y = (x.sin() + 1).cos()
    return (adds_up(y, 2.0, more) * 2).tanh().exp()


g = torch.vm(has_cells, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), keep))
for _ in range(6):
    same = torch.equal(g(x, False), has_cells(x, False))
same, [(found["ops"], found["why"]) for found in torch.vm.regions(g) if "ops" in found]
# (True, [(8, 'the function returns')])
torch.equal(g(x, True), has_cells(x, True))  # True: where the inner function is made, the instructions make it
```

### Gradients

The engine derives the backward of a graph whose tensors need a gradient
itself, from what eager does. For its first calls such a graph runs on
eager's tape, and each of those calls is taken down: the kernels the
forward ran, and, right behind the forward, the kernels that eager's own
autograd nodes run when they are called one after the other, latest first,
as the autograd engine calls them. What the nodes gave is compared, bit for
bit, with what the autograd engine gives for the same tape. Once two calls
came out as the same operations, the graph runs without a tape: its forward
is one program below autograd, which keeps what the backward reads, and its
results go through one node, `GraphBackward`, whose edges go where the
gradients of the inputs go. Each result has a node of its own in front of
that one, which hands on the gradient that reaches the result: a hook on a
result, and what `retain_grad()` keeps of it, are called where a gradient
reaches that result, and not where the graph's node runs for another.
When gradients arrive, the graph's node runs the backward for the
results that came with one (the backward for all of them is what was taken
down; the one for some of them is made of it) and hands each input its
contributions one by one, in eager's order, so the autograd engine adds
them up as it does for eager:

```python
def two(x, w, b):
    h = (x @ w + b).tanh()
    return (h * h.exp()).sum(), h.relu().mean()


def backward_of(g):
    forms = [form for found in torch.vm.regions(g) for form in found.get("forms", ())]
    return [form.get("backward") for form in forms]


derives = (*torch.vm.DEFAULT_PASSES, P.Graphs())
g = torch.vm(two, passes=derives)
w, b = torch.randn(4, 4, requires_grad=True), torch.randn(4, requires_grad=True)
nodes, says, states = [], [], []
for _ in range(7):
    x = torch.randn(3, 4)
    first, second = g(x, w, b)
    nodes.append(type(first.grad_fn).__name__)
    says.append(backward_of(g))
    states.append(torch.vm.regions(g)[0]["state"])
nodes[:5]  # ['SumBackward0', 'SumBackward0', 'SumBackward0', 'SumBackward0', 'SumBackward0']
nodes[5:]  # ['GraphBackward', 'GraphBackward']
says[2], says[4]  # (['being taken down'], ["the engine's backward"])
states[:3]  # ['built', 'waiting', 'compiled']
node = first.grad_fn.next_functions[0][0]
second.grad_fn.next_functions[0][0] is node, len(node.next_functions)  # (True, 2)
mine = torch.autograd.grad(first * 2 + second, (w, b))
eagers = two(x, w, b)
theirs = torch.autograd.grad(eagers[0] * 2 + eagers[1], (w, b))
[torch.equal(ours, other) for ours, other in zip(mine, theirs)]  # [True, True]
torch.vm.is_exact(derives)  # True
```

The gradients are eager's bit for bit, whichever results come with a
gradient: the kernels are eager's, in eager's order, and `torch.vm.is_exact`
is true for such a list. Two things are not eager's bit for bit, and both
are said where they come up below: what a second backward through a
gradient that was made with `create_graph=True` leaves, and the gradients
that a compiler's kernels compute where `Inductor()` compiled the two
graphs. A second backward of one call raises what eager raises, and so
does a backward behind a write in place to a tensor it needs (an input, a
result, what a result is a view of), with eager's text: the tensor it names
is the first one that eager's nodes would find changed, and the node it
names is the one eager names. Hooks on
results and on inputs run as they do for eager. A result that is a view is
eager's view, made on the tape behind the node.

A graph that gives a tensor and also computes something of it (a hidden
state and the loss computed of it) is cut in front of the first operation
that reads the tensor, and so is one that reads a view which it gives. Each
part has a node of its own, and the part that reads the tensor takes it as
an input like any other, so the gradient of what is computed of the tensor
comes to it through its own gradient function, as in eager mode:
`torch.autograd.grad` at the tensor, `retain_grad()` and a hook on it see
and change what eager's would. That holds for whatever the caller can hold
of the graph when the call goes on: what the function returns, by itself or
in a tuple, a list or a dict, what it stores into an object, and what a
variable holds. A tensor that is only asked for its sizes or its type
behind the graph is gone by then and is no reason to cut (`regions()` has
how many of the tensors a graph gives are `held`), and neither is a read of
a tensor before the graph last wrote to it in place. Such a form is `"in
parts"`; it is taken down twice, and then each part twice:

```python
def keeps(x, w, b):
    h = (x @ w + b).tanh()
    return h, (h * h.exp()).sum()


cut = torch.vm(keeps, passes=derives)
for _ in range(9):
    h, loss = cut(x, w, b)
backward_of(cut)  # ['in parts']
behind = loss.grad_fn.next_functions[0][0]  # the node of the part that reads `h`
h.grad_fn in [node for node, _ in behind.next_functions]  # True
(at_h,) = torch.autograd.grad(loss, [h])
eager_h, eager_loss = keeps(x, w, b)
torch.equal(at_h, torch.autograd.grad(eager_loss, [eager_h])[0])  # True
```

What a graph gives is at the version eager's is at (`_version`), the
statistics of a batch norm included. A backward for some of the inputs
(`torch.autograd.grad(loss, [w])`, `backward(inputs=[w])`) runs the nodes
of eager's that lead to those inputs and no others, as the autograd engine
runs them: the gradients are the same bits, the work for the other inputs
is not done, and a tensor that only the other nodes kept is not looked at.

With `torch.backends.cudnn.benchmark` on, cuDNN picks the algorithm of a
convolution by what it measured and remembers what it picked per thread, so
eager's own bits depend on the thread a node runs on and on what was
measured there. "Bit for bit" then means the same kernels with the same
arguments in the same order: the backward of a graph runs on the autograd
engine's thread, where eager's nodes would, and chooses as the caller set
it. While a graph is taken down and what its nodes gave is compared with
the autograd engine, both sides choose one way: benchmark mode is off and
deterministic mode is on for that long ("Limits").

With no pass over graphs in the list no graph leaves the engine for this:
the engine hands such a graph to itself. A graph starts as one that the
engine only runs (`"built"`). The first call that comes to it with
gradients enabled and a tensor that needs one makes it a graph that is
taken down for its backward: its path is taken down again at the next call
(a path that something stands for goes further than one that is only
reported: through an operation that draws random numbers, say), and from
then on it is `"compiled"`, with a form per kind of call.

What the calls before were does not matter. A graph that has its path from
calls without gradients (a model that is evaluated under `torch.no_grad()`
before it is trained) waits: the first call with gradients enabled takes it
down again, and it is handed over from the next call on if a tensor of that
call needs a gradient. A call with gradients enabled whose tensors need
none cannot be told from one that needs a gradient without taking the graph
down, so behind such calls the graph is taken down again at the second, the
fourth, the eighth such call and so on, and then at every 4096th: a graph
that needed no gradient for n calls with gradients enabled is handed over
at most n calls, and at most 4096, behind the first call that needs one.

```python
late = torch.vm(two, passes=derives)
with torch.no_grad():
    for _ in range(3):
        first, second = late(x, w, b)
torch.vm.regions(late)[0]["state"]  # 'built'
for _ in range(8):
    first, second = late(x, w, b)
type(first.grad_fn).__name__, torch.vm.regions(late)[0]["state"]  # ('GraphBackward', 'compiled')
```

A call of a graph that is taken down for its backward whose tensors need
no gradient has nothing to derive, and its form says so:

```python
with torch.no_grad():
    for _ in range(3):
        first, second = g(x, w, b)
first.requires_grad, [form.get("reason") for form in torch.vm.regions(g)[0]["forms"]]
# (False, ['no tensor of the call needs a gradient: its operations run as they are', None])
```

`Graphs(backward=False)` turns the engine's backward off. A graph whose
tensors need a gradient then runs on eager's tape: with no pass over graphs
its operations run as they are, and each records what eager records; a pass
of one's own is handed the one graph, with the tensors that need a
gradient, and what it returns runs on the tape.

```python
tape = torch.vm(two, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(backward=False)))
for _ in range(7):
    first, second = tape(x, w, b)
type(first.grad_fn).__name__, torch.vm.regions(tape)[0]["state"]  # ('SumBackward0', 'built')
```

`regions()` says for every form of a graph whose tensors need a gradient
what derives its backward, as `backward`: `"being taken down"`, `"the
engine's backward"`, `"in parts"`, or `"eager's tape: "` and why. A graph
keeps eager's tape for good where it cannot be taken down to operations
that are the same at every call:

- an operation of the forward or of the backward looks at what a tensor
  holds, or gives a tensor whose size depends on it (`masked_select`,
  `nonzero`, `item`; the backward of a product, which looks for zeros);
- a node of the backward is written in Python (a custom
  `autograd.Function`, a backward hook of a module, a checkpoint);
- a tensor is made a leaf inside the graph (`requires_grad_()`), an input
  is written to in place with a gradient through the write, or a result is a
  view that the graph cannot make again behind its node;
- an input has a hook or `retain_grad()` while the graph is taken down:
  comparing with the autograd engine would run the hook;
- two calls did not come out as the same operations, or what the nodes gave
  is not what the autograd engine gives.

```python
def picks(x, w, b):
    return (x @ w + b).tanh().masked_select(x > 0).sum(), (x @ w).mean()


g = torch.vm(picks, passes=derives)
for _ in range(5):
    first, second = g(x, w, b)
type(first.grad_fn).__name__  # 'SumBackward0'
backward_of(g)  # ["eager's tape: aten::masked_select looks at what a tensor holds"]
```

A call under anomaly mode, under a transform of `torch.func`, with a
forward-mode level or with saved-tensor hooks runs on eager's tape, that
call only; so does one under `torch.no_grad()`, which needs no backward.

Two backwards are not served by the backward that was taken down: one that
is to be differentiated again (`create_graph=True`), and one for some of
the results of a node of eager's that was handed gradients for all of them
when the graph was taken down (the two halves of a `chunk`), or whose
kernels, which are those of the backward for all the results, do not fit
the gradients that come for those results alone (a `view` in front of which
the gradients of two results meet). For these the
node runs the graph once more, on the tape, from its inputs, and calls
eager's nodes for it. The gradient that a backward with `create_graph=True`
gives is eager's bit for bit. A second backward through that gradient goes
through this copy of the forward where eager's goes through the forward
itself, and the gradients that meet there are not always added in eager's
order: what it leaves is eager's value, and not always its last bit (the
tests hold it to 1e-5). A graph that draws random numbers or writes to a
tensor from outside cannot run once more, and neither can one an input of
which was written to in place since the call: it raises there and says
which of the two backwards it was asked for.

A backward runs under the autocast of the thread that asks for it: none
where it is asked for behind the `with torch.autocast(...)` block that the
forward ran in, which is the usual way, and the forward's where it is asked
for inside that block. The kernels that eager's nodes run are not the same
in the two (an operation that autocast runs in float32 casts what it is
handed). So the backward of a graph whose forward runs under autocast is
taken down twice, once with autocast off and once under the forward's, and
each is compared with the autograd engine under the same state. The node
runs the one that was taken down as its thread runs a backward:

```python
def casts(x, w, b):
    h = (x @ w).tanh()
    return (h * h.exp()).sum() + b.sum(), h.pow(3).mean()


def step(fn, inside):
    w.grad = b.grad = None
    with torch.autocast("cpu", dtype=torch.bfloat16):
        first, second = fn(x, w, b)
        loss = first.float() + second.float() * 2
        if inside:
            loss.backward()
    if not inside:
        loss.backward()
    return first, w.grad.clone(), b.grad.clone()


g = torch.vm(casts, passes=derives)
for _ in range(3):
    behind, within = step(g, False), step(g, True)
type(within[0].grad_fn).__name__, backward_of(g)  # ('GraphBackward', ["the engine's backward"])
[torch.equal(ours, other) for ours, other in zip(behind, step(casts, False))]  # [True, True, True]
[torch.equal(ours, other) for ours, other in zip(within, step(casts, True))]  # [True, True, True]
```

A backward under an autocast that the forward did not run under (autocast
around the backward only, or another dtype) is served as one with
`create_graph=True` is: the forward runs once more on the tape, under its
own autocast, and eager's nodes run under the backward's.

A pass over graphs behind `Graphs()` is handed two plain graphs for such a
form, once it has the engine's backward: the forward, which gives the
results and what it keeps, and the backward for the results that came with
a gradient, which takes what was kept and the gradients (a backward for
other results is another graph, handed over when it is first asked for).
Their tensors need no gradient, so there is nothing left to derive:
`Inductor()` compiles two graphs, and AOTAutograd derives nothing. What a
compiler makes of the kernels is not what eager runs, so with one the
gradients are eager's to the last bits only. A pass that asks for a check
of an input of such a graph is refused, and the form keeps eager's tape
with what the pass raised as the reason. `AOTAutograd()` in the list is the
user asking for its derivation: the graph is then handed to it as it is
with `backward=False`, and `Inductor()` behind it compiles what it traced.
`Inductor()` behind `Graphs(backward=False)`, with no `AOTAutograd()`, is
handed no graph whose tensors need a gradient: nothing in such a list
derives a backward, the graph runs on eager's tape, and `regions()` gives
the reason (the second example below).

```python
taken = []


def count(gm, input_types):
    calls = [node for node in gm.graph.nodes if node.op == "call_function"]
    taken.append((len(input_types), len(calls)))
    return gm


g = torch.vm(two, passes=(*derives, count))
for _ in range(5):
    first, second = g(torch.randn(3, 4), w, b)
(first + second).backward()
taken  # [(3, 8), (4, 17)]: the forward, and the backward for both results
type(first.grad_fn).__name__  # 'GraphBackward'
g = torch.vm(two, passes=(*derives, P.AOTAutograd()))
for _ in range(5):
    first, second = g(torch.randn(3, 4), w, b)
type(first.grad_fn).__name__  # 'CompiledFunctionBackward'
```

```python
def node_of(*passes):
    g = torch.vm(two, passes=(*torch.vm.DEFAULT_PASSES, *passes))
    for _ in range(6):
        first, second = g(torch.randn(3, 4), w, b)
    return type(first.grad_fn).__name__, torch.vm.regions(g)[0].get("reason")


node_of(P.Graphs(), P.Inductor())  # ('GraphBackward', None)
node_of(P.Graphs(), P.AOTAutograd(), P.Inductor())  # ('CompiledFunctionBackward', None)
node_of(P.Graphs(backward=False), P.Inductor())
# ('SumBackward0', 'Inductor() compiles graphs whose tensors need no gradient: the engine derives the backward
# of a graph (Graphs()) and hands it two such graphs, or AOTAutograd() in front of it derives one')
```

### The order of the list

- `Graphs()` can stand anywhere among the passes over a block. It looks at a
  block when the others are done with it, wherever it stands.
- The passes over a graph stand behind `Graphs()`, in the order they are to
  be applied in. When the list has such a pass and no `Graphs()`, it is taken
  to be there, in front of the first of them.
- At most one pass compiles, and it is the last pass over a graph: what it
  returns is run, not rewritten. `AOTAutograd()` directly in front of
  `Inductor()` is the one list with both: AOTAutograd derives the backward
  of a graph and Inductor compiles both, which is one compilation.

`torch.vm` raises a `ValueError` for a list that breaks one of these when the
function is handed to it:

```python
default = torch.vm.DEFAULT_PASSES
torch.vm(f, passes=(P.Inductor(), *default, P.Graphs()))
# ValueError: passes has torch.vm.passes.Inductor() in front of torch.vm.passes.Graphs(): a pass over a graph
# is given the graphs that Graphs makes, so it goes behind it
torch.vm(f, passes=(*default, P.Graphs(), P.Inductor(), P.AOTAutograd()))
# ValueError: passes has torch.vm.passes.Inductor() and torch.vm.passes.AOTAutograd(): a graph is compiled
# once, by the last pass over it
torch.vm(f, passes=(*default, P.Graphs(), P.Inductor(), keep))
# ValueError: passes has keep behind torch.vm.passes.Inductor(): what a pass that compiles a graph returns is
# run, not rewritten, so it is the last pass over a graph
torch.vm(f, passes=(*default, keep, P.Inductor())).passes
# ['resolve_dispatch', 'tensor_accessors', 'method_calls', 'fold', 'dce', 'graphs']
```

### Models

With the default settings and `Graphs()` and `Inductor()` in the list, the
forward of a convolutional model is one graph. Measured on one H100, in
images per second against eager mode (the model in eval mode under
`torch.no_grad()`, the best of three rounds of 60 batches; two runs with
`Inductor()`, one with `Graphs()` alone; eager mode itself differs by up to
3 percent between two runs):

| model | the graph that runs | `Graphs()` | `Graphs(), Inductor()` |
|---|---|---|---|
| yolov3 (TorchBench), bfloat16, batch 8 of 3x384x512 | 328 operations | 0.97x | 2.32x, 2.30x |
| resnet18 (torchvision), float32, batch 16 of 3x224x224 | 69 operations | 0.97x | 1.92x, 1.82x |
| resnet18, bfloat16 | 69 operations | 1.03x | 2.26x, 2.29x |
| resnet50, float32 | 175 operations | 1.02x | 1.74x, 1.75x |
| resnet50, bfloat16 | 175 operations | 1.01x | 2.25x, 2.17x |

With `Graphs()` alone the outputs are those of eager mode bit for bit, the
operations in place included. With `Inductor()` they differ by its
arithmetic: by 2e-4 of the largest value in float32, and by up to 1.4e-2 in
bfloat16. Each graph was exported once. With `Inductor()` at most four small
functions on the way were left to the interpreter by measurement
(`Module.__setattr__`, `ContextProp.__get__`), and the graphs go through
them. The graph of a resnet is its whole forward, the pool, the flatten and
the linear layer at the end included. The graph of yolov3 is the model with
its three detection layers, the stores they make into themselves and the
`torch.meshgrid` of each included; it ends where the model takes the outputs
of the layers apart with `zip(*outputs)`, and the one `torch.cat` behind
that runs by itself.

The forward of a training step is one graph as well: 89 operations for
resnet18 and 228 for resnet50, the batch norms writing their statistics and
counting their batches in it. In steps per second against eager mode
(batch 16 of 3x224x224, 100 classes, cross entropy and SGD with momentum; a
step is forward, loss, backward and update, the best of three runs of 30
steps once warm). These steps were measured without the engine's backward:
one run on eager's tape, with `Graphs(backward=False)`, and three with
AOTAutograd deriving the backward and Inductor compiling it, which is the
list `Graphs(), AOTAutograd(), Inductor()`. The engine's backward has not
been timed on these models:

| model | `Graphs(backward=False)` | `Graphs(), AOTAutograd(), Inductor()` |
|---|---|---|
| resnet18 (eager 125 steps a second) | 1.00x | 1.28x, 1.30x, 1.50x |
| resnet50 (eager 53 to 54 steps a second) | 1.01x | 1.46x, 1.30x, 1.28x |

The figure with `Inductor()` differs from one process to the next by that
much while eager's does not. With `Graphs(backward=False)` alone the losses
of 20 steps and the parameters and statistics after them are eager's bit
for bit. With
`Inductor()` the losses are equal for the two steps that the instructions
run and then go apart, by 4e-5 at the third step and by up to 2e-2 within
the first ten for resnet18 (6e-4 and up to 1e-1 for resnet50): the gradients
of one step from the same state differ by 7e-2 of the largest for resnet18
and 4e-1 for resnet50, with the loss of that step equal to 1e-5 of itself,
and 20 steps of SGD on a freshly initialised model carry that on. On the CPU
the gradient of every parameter of a small model is within 1e-4 of eager's
at each of 30 steps with `AOTAutograd()` and with `Inductor()`, which is
what the tests hold.

The models of timm and the transformer models have other code around their
operations than the convolutional models above, and the graphs they get are
counted here without times: on the CPU, in float32 with a batch of 2, the
models loaded and called as the benchmark runner of `benchmarks/dynamo` does,
with a pass that returns the graph. The operations of a call are the
operators that Python calls in one call of the model in eager mode, as the
profiler counts them (a figure can come out a few over that count, where the
profiler counts an operation of a graph as less than one):

| model | graphs that run | runs of a graph in a call | operations of a call in them |
|---|---|---|---|
| mobilenetv2_100 (timm) | 1 | 1 | 152 of 152 |
| adv_inception_v3 (timm) | 1 | 1 | 314 of 314 |
| deit_base_distilled_patch16_224 (timm) | 1 | 1 | 233 of 233 |
| DistillGPT2 (HuggingFace) | 4 | 9 | 249 of 247 |
| hf_DistilBert (TorchBench) | 3 | 13 | 132 of 136 |
| MobileBertForMaskedLM (HuggingFace) | 3 | 3 | 1295 of 1250 |
| whisper-tiny (HuggingFace) | 5 | 22 | 342 of 385 |

The outputs are those of eager mode bit for bit. A timm model and the vision
transformer are one graph, the whole forward. A transformer language model
has all or nearly all of its operations in graphs, but in several per call:
one per layer or per stack of layers, and a few around them. What their
paths end at is `getattr(config, name, default)` where the configuration
raises for a name it does not have (at the start of every model, where the
attention mask is made), a branch on what the mask holds, the objects the
models construct for their outputs, and library code such as
`inspect.signature` ("Limits"). A training step has fewer of its operations
in graphs (counted on eager's tape, with `Graphs(backward=False)`): for
DistillGPT2, 5 graphs that run 15 times in a step, with 189
of the about 820 operators that the profiler counts in one step of eager
mode (forward, loss, backward and update together), since a dropout that
drops and an attention with a dropout rate are operations only where
nothing that can fail has been taken down in front of them ("Writes").

With the models on a GPU (one H100, counted the same way) the timm models,
the vision transformer and hf_DistilBert have the graphs of the table above.
The models that call `scaled_dot_product_attention` with `**` have fewer of
their operations in graphs there, in the build of torch they were counted
with, which has no fused attention kernels: the attention comes down to
`bmm`, whose CUDA kernel is entered through a Python function, so the call
is no operation ("What a call names with a dict"), and the graph of a layer
ends in front of it and another starts behind it:

| model, on a GPU | graphs that run | runs of a graph in a call | operations of a call in them |
|---|---|---|---|
| DistillGPT2 | 5 | 15 | 189 of 253 |
| MobileBertForMaskedLM | 7 | 170 | 1127 of 1274 |
| whisper-tiny | 6 | 30 | 173 of 389 |
| Qwen3-0.6B (HuggingFace) | 4 | 31 | 2097 of 2217 |

The outputs are eager's bit for bit there as well. With a GPU present
`torch.cuda.is_current_stream_capturing()`, which the mask code of a
transformer asks on its way, is a call that a path does not go through,
where on a machine without one it raises: the path ends there either way.
No times are given for these models: they have not been timed on a GPU.

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
  where what was made of it raises. A path ends in front of an operation
  with several results whose value is needed whole behind it ("Operations
  with several results").
- What a path reads behind a store of its own into an object is checked at
  every call to go through another dict or list than the store changes, a
  check per store ("Stores into objects"); a read through what a store
  changed ends the path.
- A raise that is caught ends a path: `getattr(obj, name, default)` and
  `hasattr` where a `__getattribute__` written in Python raises
  `AttributeError`, a `try` around a call that raises, `Mapping.get` of a key
  that is not there. So do constructing an object (a dataclass of outputs,
  a named tuple, `functools.partial`), making an inner function or a
  lambda, a `with` block, and library code a path is not known to be able
  to call (`inspect.signature`). These are what the paths of transformer
  models end at ("Models").
- A call with `*` or `**` is an operation only where the operator it runs
  enters no Python code. On a GPU the kernel of `bmm` is entered through a
  Python function, so such a call of `bmm`, or of what comes down to it,
  ends a path ("Methods read off a tensor, and what a call spreads";
  "Models" has what that costs a transformer).
- An operation that draws random numbers is one of a path only where
  nothing that can fail has been taken down in front of it ("Writes"). In
  the layers of a transformer in training mode that is seldom so, and their
  paths end at a dropout that drops or at an attention with a dropout rate
  ("Models" has what a training step of one looks like).
- In the call in which a path is taken down, a variable that the function
  deletes inside the path (`del name`) is let go of where the path ends and
  not at the `del`, and so is a value that is in no variable and ends inside
  the path (what a call gave, put into a tuple that the path builds and
  drops). Where one of them held the last reference to an object whose end
  runs Python code, that code runs behind what the call reads behind it. A
  call that would run what was made of the path is turned down for such an
  object ("Passes over graphs"), and is then one in which a path is taken
  down again, for as long as the graph may get another form.
- Sizes are fixed per export. Every size a graph is called with is a kind of
  call of its own, with a graph of its own, so a function that is called
  with many sizes uses up `max_variants`. Sizes of the tensors the graph
  computes are computed in it (`torch.ops.aten.sym_size.int`), but its
  inputs have the sizes of the call it was exported from. With `Inductor()`
  every export is a compilation.
- Under a `TorchDispatchMode`, a `TorchFunctionMode` or a transform of
  `torch.func` the operations of the function run, so that those see them,
  and the calls under one are not exported.
- With the engine's backward ("Gradients") the state behind a backward that
  raises is not eager's. Eager raises at the node that needs the tensor that
  was written to in place, and has by then added up the gradient of every
  leaf whose nodes all ran before that one; the node of a graph raises
  before it gives anything, so no input of the graph has a gradient of that
  backward. The error and its text are eager's.
- A backward that is to be differentiated again, or one for some results of
  a node of eager's with several, runs the graph once more on the tape: a
  graph that draws random numbers or writes to a tensor from outside raises
  for it, where eager computes it, and so does one an input of which was
  written to in place since the call (eager kept what it needs of it).
- While the backward of a graph is taken down and compared with the
  autograd engine, cuDNN's benchmark mode is off and its deterministic mode
  is on. The two settings are the process's: a convolution that another
  thread runs in that time runs that way too. They are as the caller set
  them again when the call returns.
- Where AOTAutograd derives the backward of a graph (with `AOTAutograd()` in
  the list), the gradient at a tensor that the graph gives and also computes
  something of is what `torch.compile` gives and not eager's: one compiled
  function has the gradient of both, so nothing of it arrives at the tensor.
  `torch.autograd.grad(loss, [hidden])` says the tensor was not used,
  `hidden.retain_grad()` keeps nothing, and a hook on `hidden` is handed
  None and changes nothing of what reaches the inputs. The engine's own
  backward cuts such a graph at the tensor ("Gradients").
- A warning that a kernel gives is given at every call, as in eager. One
  that the Python side of an operation gives (`x.T` of a tensor with three
  dimensions, `torch.tensor` of a tensor), or an operator that is made of
  other operators (an attention on a GPU, where torch is built without the
  fused attention kernels), is given by the calls that run the function's
  instructions: a call that the programs of the engine's backward or the
  callable of a pass run gives none, and the call in which a graph is taken
  down a second time or exported gives the first kind twice.
- A loss behind a write is no part of the graph with the engine's backward
  either: a training graph still ends in front of a loss that can fail
  behind a batch norm's write to its statistics ("Writes"), and the nodes of
  the loss stand in front of `GraphBackward` on eager's tape.

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
