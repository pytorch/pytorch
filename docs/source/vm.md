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
#   'outputs': 1, 'branches': 0, 'paths': 1, 'ends': [(0, 12)],
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

What runs of a path, once a pass has made something of its graph, is what
the function ran on that way, less the operations: the reads, the tests and
the other instructions that take nothing the graph gives run in front of
it, at every call, and those that do run behind it. An instruction whose
value nothing uses is left out only if it cannot fail whatever it is given
(a tuple that is built, a constant, a test for `None`; the `append` that a
loop reads off a list the function made, each time round). An attribute or an
item that is read for nothing, a call whose answer is dropped, a division
stay: they are what shows that the call gets as far as the path does. In a
call in which one of them raises, the path is not taken: the instructions
run, and raise where they stand. What changes something the path made
itself (an append to a list it built, the next item of an iterator it
made) is left out only together with the thing ("What a function makes"),
and only where it cannot fail either: an entry that is put into a dict under
a key that is no constant of the function runs in every run, used or not, so
that a call in which the key cannot be hashed runs the instructions, which
raise.

```python
class Settings:
    pass


def checked(settings, x):
    y = x.sin() * 2
    settings.ready  # read for nothing: raises where there is none
    return (y.cos() + 1).exp().sum()


def as_it_is(gm, input_types):  # a pass over graphs: what it returns runs
    return gm


g = torch.vm(checked, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), as_it_is))
settings = Settings()
settings.ready = True
for _ in range(4):
    out = g(settings, x)
del settings.ready
try:
    g(settings, x)
except AttributeError as error:
    print(error)  # 'Settings' object has no attribute 'ready'
```

A loop is its iterations one after the other: a path goes round a loop over
a list, a tuple or a `range` as often as the call does (and round a `while`
loop on a count), with the test of the loop as a branch each time round. The
number of times round is no input of the graph: it is the path. The calls of
`len`, `range`, `enumerate`, `zip` and `isinstance` that such loops are
written with are part of a path.

Going over an iterator uses it up, so a path goes over an iterator only if
it made it itself, of a list, a tuple, a dict or a `range` (`enumerate`,
`zip`, the loop's own). Where the value that was a list when
the path was taken down is an iterator of the caller's at a later call,
that call runs the instructions, which use the iterator up once.

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

What ends a path:

- a call of a Python function, which has graphs of its own, or of
  something else that is no operation, like `print`;
- an operation that writes to one of its arguments or draws random numbers,
  one with several results (`x.split(2)`, `torch.max(x, 0)`), a store into
  an object or a subscript;
- a number read from a tensor (`x.item()`, `float(x)`), and with it a branch
  on the data of a tensor;
- an instruction that could run Python code on the values of this call: a
  read of an attribute that is a property, an iterator that is unpacked;
- in a function with a `try`, a read or a write of a variable that one of
  its handlers may read: such a function keeps them where the handlers find
  them;
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

A graph does not go on through a call of a Python function: the function
that is called has graphs of its own. A model made of modules therefore has
its operations spread over the `forward` of each module, one or two to a
function, and none of those is a graph unless `min_ops` is lowered:

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

### What a function makes

A list, a dict, an iterator: what the instructions of a path make between
its operations is the path's own. One model says what a path does with such
a thing.

**Which values are the thing.** A thing is known by the instruction that
made it. Other values of the function may be that thing, or hold it: an item
or an attribute that is taken out of something that holds it, what an
iterator over such a holder gives, a method that is bound to it, a tuple, a
list or a dict that is built of it. Such a value counts as the thing
wherever it may be it at some call. What the function had before the path,
and what it reads of objects from outside, is never one: nothing of the
path's is stored there before the path ends. So what an iterator of the path
gives is from outside where what it goes over is: `for i, layer in
enumerate(self.layers)` gives the layers.

**What runs.** What changes a thing (an append, `[*a, b]`, an entry of a
dict it builds, the next item of an iterator) runs in a run of the path
whenever anything that runs takes the thing, or what may be it or hold it,
in front of the change or behind it. It is left out only together with the
thing.

**On which side of the graph.** The instructions that take nothing the graph
gives run in front of what stands for the graph, the others behind it
("Paths"). A thing begins in front. It moves behind the graph at the first
instruction that takes it, or what may be it or hold it, and has to run
there: a result of the graph is appended to it. From that instruction on
everything that takes the thing runs behind the graph, also what takes
nothing else of the graph's; everything in front of it ran in front. So what
reads and what changes one thing runs in the order the function has it, on
both sides: a list that is counted, sliced or put into a tuple before a
result is appended to it is counted, sliced and held as it was then, and an
item that is appended behind a read is not seen by the read.

```python
def kept(x, n):
    y = x.sin() * 2
    made = [n]
    holder = (made, len(made))  # in front of the graph: the list has one item
    made.append(y)  # behind the graph: it takes what the graph gives
    made.append(n + 1)  # behind that append in the function, so behind the graph too
    return (y.cos() + 1).exp().sum(), holder


g = torch.vm(kept, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), as_it_is))
for _ in range(4):
    out, holder = g(x, 3)
len(holder[0]), holder[1], holder[0][2]  # (3, 1, 4)
found = torch.vm.regions(g)[0]
found["ops"], found["hits"]  # (6, 2)
```

**What ends a path instead.** A path ends in front of an operation that is
handed something of a thing that has moved behind the graph, other than
tensors and sizes of the graph's own (a list of those is handed over item by
item).

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
it at the call, up to 64 lists, tuples and slices inside one another, not by
which list it is: one that was changed in place since is another value, and
one that goes deeper, or that holds itself, is never the value a form was
made for). A call of another kind runs the
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
what the call would let go of that way, as it is at that call: nothing is
remembered from one call to the next. An object that has a reference
besides those the call lets go of stays alive, whatever it is and whatever
it refers to, and so does one that the function still holds when the graph
has run. The others end, and each has to end without Python code running:
its class has no finalizer, no weak reference is told of its end, and what
it refers to is looked at in the same way, as a garbage collector looks at
a set of objects. So a list that a slice made of a list of layers ends
quietly, since the layers stay. The entry of a weak container, which only
takes itself out, is no such weak reference where the container is exactly
a `WeakKeyDictionary`, a `WeakValueDictionary` or a `WeakSet` of `weakref`,
torch's `WeakIdKeyDictionary`, or the dictionary in which dynamo notes
every module that is made once `torch.compile` has been used; a container
of a class derived from these, or written like them, is taken to run code.
What goes with the entry (the value under a key, the key of a value) is
looked at as well, and the key of a `WeakValueDictionary` has to be one
that is hashed without Python code and is alone under its hash, since
taking the entry out hashes and compares it. A tensor that ends has to be
of the class `torch.Tensor` itself, require no gradient (what a history
holds, a hook or what a `torch.autograd.Function` kept for its backward, is
not to be seen), have memory that is torch's own (a tensor over a numpy
array, a buffer or a DLPack capsule keeps their owner alive), and be one
dense block (not sparse, quantized or nested, and no wrapper around another
tensor); its attributes, its `.grad`, the Python object of its storage
(which ends with the storage: what it has of its own, a weak reference on
it that is told, a `weakref.finalize`) and, for a view, the tensor it is a
view of are looked at, and while a level of forward-mode differentiation is
open, in which a tensor may have a tangent, no tensor ends quietly. An object that shows the
collector nothing (a capsule, a numpy array, a storage) is taken to end
with Python code unless its class is known to hold nothing. A call for
which that cannot be shown runs the operations of the function as they are:

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

What an object refers to is what it shows the garbage collector, and no
more than 4,096 objects that end are looked at (with more, the call runs
the operations). Objects in a reference cycle end when the collector runs,
as they do in eager mode. What a pass's own callable does is not looked at
("Limits").

A call in which a path is taken down holds on to what the path reads until
the run has been described. Where the function lets go, in such a call, of
its last reference to an object whose end may run Python code, the engine
lets go of all it holds right there, so that the object ends where the
instruction is, and makes nothing of that call. A graph that could not take
three calls in a row for what it would let go of (its path was turned down,
or could not be taken down to its end) gives way for good: it is not
entered again, the graphs that start further on in the function are their
own again, and `torch.vm.regions` lists it as `"vm"` with the reason:

```python
def lets_go_every_time(x):
    held = HELD.pop()
    y = (x.sin() + 1).cos() * 2
    held = None
    return (y + 1).tanh(), held


g = torch.vm(lets_go_every_time, passes=(*torch.vm.DEFAULT_PASSES, P.Graphs(), keep))
for at in range(12):
    HELD.append(Ends() if at >= 6 else 0)
    out = g(x)
found = [f for f in torch.vm.regions(g) if "ops" in f][0]
found["state"], found["reason"]
# ('vm', 'where its path lands it would let go of an object whose end may run Python code')
```

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

- One function. A graph does not go on through a call of a Python function,
  as said above.
- A path per way. A function whose calls go many ways through it has a path
  and an export for each, and they share the `max_variants` of the graph.
  So does a loop that goes round another number of times on every call.
- No effects. An operation that writes to one of its arguments is in no
  graph, and a call whose operations turn out to write when they are taken
  down is not exported: a batch norm in training, which updates its running
  statistics, runs its operations on every call, and the same graph is
  exported for the calls in eval mode. A store into a subscript (`x[0] =
  y`) ends a graph, and so does an operation with several results
  (`x.split(2)`, `torch.max(x, 0)`).
- A graph does not take a call in which it would let go of the last
  reference that Python has to a tensor which requires a gradient ("Passes
  over graphs"), also where what the graph gives keeps the history of that
  tensor alive: who else holds the nodes of a history cannot be asked. That
  is a graph which starts in the middle of a function, takes a tensor that
  the operations in front of it gave, and lets go of the variable it was in.
- A path lets go of an object from outside it only where the end of the
  object cannot run Python code, or the object stays alive. Whether it
  stays alive is looked at in front of what stands for the graph: for a
  variable that the path binds anew, an object whose other reference is
  dropped while what a pass made of the graph runs (by that code itself, or
  by another thread it waits for) ends where the path lands.
- Sizes are fixed per export. Every size a graph is called with is a kind of
  call of its own, with a graph of its own, so a function that is called
  with many sizes uses up `max_variants`. Sizes of the tensors the graph
  computes are computed in it (`torch.ops.aten.sym_size.int`), but its
  inputs have the sizes of the call it was exported from. With `Inductor()`
  every export is a compilation.
- Under a `TorchDispatchMode`, a `TorchFunctionMode` or a transform of
  `torch.func` the operations of the function run, so that those see them,
  and the calls under one are not exported.
- A dict whose keys are not all texts, spread into a call (`f(**d)`), raises
  `TypeError` as it does without the engine, but the engine has compared such
  a key with the names of the parameters by then: an `__eq__` of the key's
  class that is written in Python runs where the interpreter runs none.

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
