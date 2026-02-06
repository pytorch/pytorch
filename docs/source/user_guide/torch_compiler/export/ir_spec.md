(export.ir_spec)=

# torch.export IR Specification

Export IR is an intermediate representation (IR) for compilers, which bears
similarities to [MLIR](https://mlir.llvm.org/) and TorchScript. It is specifically designed to express the
semantics of PyTorch programs. Export IR primarily represents computation in a
streamlined list of operations, with limited support for dynamism such as
control flows.

To create an Export IR graph, a frontend can be used that soundly captures a
PyTorch program via a trace-specializing mechanism. The resulting Export IR can
then be optimized and executed by a backend. This can be done today through
{func}`torch.export.export`.

The key concepts that will be covered in this document include:

- ExportedProgram: the data structure containing the Export IR program
- Graph: which consists of a list of nodes.
- Nodes: which represents operations, control flow, and metadata stored on this node.
- Values are produced and consumed by nodes.
- Types are associated with values and nodes.
- The size and memory layout of values are also defined.

## Assumptions

This doc assumes that the audience is sufficiently familiar with PyTorch,
specifically with {class}`torch.fx` and its related toolings. Thus it will stop
describing contents present in {class}`torch.fx` documentation and paper.

## What is Export IR

Export IR is a graph-based intermediate representation IR of PyTorch programs.
Export IR is realized on top of {class}`torch.fx.Graph`. In other words, **all
Export IR graphs are also valid FX graphs**, and if interpreted using standard
FX semantics, Export IR can be interpreted soundly. One implication is that an
exported graph can be converted to a valid Python program via standard FX
codegen.

This documentation will primarily focus on highlighting areas where Export IR
differs from FX in terms of its strictness, while skipping parts where it shares
similarities with FX.

## ExportedProgram

The top-level Export IR construct is an {class}`torch.export.ExportedProgram`
class. It bundles the computational graph of a PyTorch model (which is usually a
{class}`torch.nn.Module`) with the parameters or weights that this model
consumes.

Some notable attributes of the {class}`torch.export.ExportedProgram` class are:

- `graph_module` ({class}`torch.fx.GraphModule`): Data structure containing
  the flattened computational graph of the PyTorch model. The graph can be
  directly accessed through `ExportedProgram.graph`.
- `graph_signature` ({class}`torch.export.ExportGraphSignature`): The graph
  signature, which specifies the parameters and buffer names used and mutated
  within the graph. Instead of storing parameters and buffers as attributes of
  the graph, they are lifted as inputs to the graph. The graph_signature is
  utilized to keep track of additional information on these parameters and
  buffers.
- `state_dict` (`Dict[str, Union[torch.Tensor, torch.nn.Parameter]]`): Data
  structure containing the parameters and buffers.
- `range_constraints` (`Dict[sympy.Symbol, RangeConstraint]`): For programs
  that are exported with data dependent behavior, the metadata on each node will
  contain symbolic shapes (which look like `s0`, `i0`). This attribute maps
  the symbolic shapes to their lower/upper ranges.

## Graph

An Export IR Graph is a PyTorch program represented in the form of a DAG
(directed acyclic graph). Each node in this graph represents a particular
computation or operation, and edges of this graph consist of references between
nodes.

We can view Graph having this schema:

```python
class Graph:
  nodes: List[Node]
```

In practice, Export IR's graph is realized as {class}`torch.fx.Graph` Python class.

An Export IR graph contains the following nodes (Nodes will be described in more
details in the next section):

- 0 or more nodes of op type `placeholder`
- 0 or more nodes of op type `call_function`
- exactly 1 node of op type `output`

**Collorary:** The smallest valid Graph will be of one node. i.e. nodes is never empty.

**Definition:**
The set of `placeholder` nodes of a Graph represents the **inputs** of the
Graph of GraphModule. The `output` node of a Graph represents the **outputs**
of the Graph of GraphModule.

Example:

```python
import torch
from torch import nn

class MyModule(nn.Module):

    def forward(self, x, y):
      return x + y

example_args = (torch.randn(1), torch.randn(1))
mod = torch.export.export(MyModule(), example_args)
print(mod.graph)
```

```python
graph():
  %x : [num_users=1] = placeholder[target=x]
  %y : [num_users=1] = placeholder[target=y]
  %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %y), kwargs = {})
  return (add,)
```

The above is the textual representation of a Graph, with each line being a node.

## Node

A Node represents a particular computation or operation and is represented in
Python using the {class}`torch.fx.Node` class. Edges between nodes are
represented as direct references to other nodes via the `args` property of the
Node class. Using the same FX machinery, we can represent the following
operations that a computational graph typically needs, such as operator calls,
placeholders (aka inputs), conditionals, and loops.

The Node has the following schema:

```python
class Node:
  name: str # name of node
  op_name: str  # type of operation

  # interpretation of the fields below depends on op_name
  target: [str|Callable]
  args: List[object]
  kwargs: Dict[str, object]
  meta: Dict[str, object]
```

**FX Text Format**

As in the example above, notice that each line has this format:

```
%<name>:[...] = <op_name>[target=<target>](args = (%arg1, %arg2, arg3, arg4, …)), kwargs = {"keyword": arg5})
```

This format captures everything present in the Node class, with the exception of
`meta`, in a compact format.

Concretely:

- **&lt;name&gt;** is the name of the node as it would appear in `node.name`.
- **&lt;op_name&gt;** is the `node.op` field, which must be one of these:
  `<call_function>`, `<placeholder>`,
  `<get_attr>`, or `<output>`.
- **&lt;target&gt;** is the target of the node as `node.target`. The meaning of this
  field depends on `op_name`.
- **args1, … args 4…** are what is listed in the `node.args` tuple. If a
  value in the list is an {class}`torch.fx.Node`, then it will be especially
  indicated with a leading **%.**

For example, a call to the add operator would appear as:

```
%add1 = call_function[target = torch.op.aten.add.Tensor](args = (%x, %y), kwargs = {})
```

Where `%x`, `%y` are two other Nodes that have names x and y. Worth noting
that the string `torch.op.aten.add.Tensor` represents the callable object that
is actually stored in the target field, not merely its string name.

The final line of this text format is:

```
return [add]
```

which is a Node with `op_name = output`, indicating that we are returning this
one element.

### call_function

A `call_function` node represents a call to an operator.

**Definitions**

- **Functional:** We say a callable is “functional” if it satisfies all the
  following requirements:

  - Non-mutating: The operator does not mutate the value of its input (for
    tensors, this includes both metadata and data).
  - No side effects: The operator does not mutate states that are visible
    from outside, like changing values of module parameters.

- **Operator:** is a functional callable with a predefined schema. Examples of
  such operators include functional ATen operators.

**Representation in FX**

```
%name = call_function[target = operator](args = (%x, %y, …), kwargs = {})
```

**Differences from vanilla FX call_function**

1. In FX graph, a call_function can refer to any callable, in Export IR, we
   restrict it to only a select subset of ATen operators, custom operators, and
   control flow operators.
2. In Export IR, constant arguments will be embedded within the graph.
3. In FX graph, a get_attr node can represent reading any attribute stored in
   the graph module. However, in Export IR this is restricted to reading only
   submodules as all parameters/buffers will be passed in as inputs to the graph
   module.

#### Metadata

`Node.meta` is a dict attached to every FX node. However, the FX spec does not
specify what metadata can or will be there. Export IR provides a stronger
contract, specifically all `call_function` nodes will guarantee having and
only having the following metadata fields:

- `node.meta["stack_trace"]` is a string containing the Python stack trace
  referencing the original Python source code. An example stack trace looks
  like:

  ```
  File "my_module.py", line 19, in forward
  return x + dummy_helper(y)
  File "helper_utility.py", line 89, in dummy_helper
  return y + 1
  ```

- `node.meta["val"]` describes the output of running the operation. It can be
  of type `<symint>`, `<FakeTensor>`, a
  `List[Union[FakeTensor, SymInt]]`, or `None`.

- `node.meta["nn_module_stack"]` describes the "stacktrace" of the
  {class}`torch.nn.Module` from which the node came, if it was from a
  {class}`torch.nn.Module` call. For example, if a node containing the `addmm`
  op called from a {class}`torch.nn.Linear` module inside of a
  {class}`torch.nn.Sequential` module, the `nn_module_stack` would look
  something like:

  ```
  {'self_linear': ('self.linear', <class 'torch.nn.Linear'>), 'self_sequential': ('self.sequential', <class 'torch.nn.Sequential'>)}
  ```

- `node.meta["source_fn_stack"]` contains the torch function or the leaf
  {class}`torch.nn.Module` class this node was called from before decomposition.
  For example, a node containing the `addmm` op from a
  {class}`torch.nn.Linear` module call would contain {class}`torch.nn.Linear` in
  their `source_fn`, and a node containing the `addmm` op from a
  {class}`torch.nn.functional.Linear` module call would contain
  {class}`torch.nn.functional.Linear` in their `source_fn`.

### placeholder

Placeholder represents an input to a graph. Its semantics are exactly the same as in FX.
Placeholder nodes must be the first N nodes in the nodes list of a graph. N can be zero.

**Representation in FX**

```python
%name = placeholder[target = name](args = ())
```

The target field is a string which is the name of input.

`args`, if non-empty, should be of size 1 representing the default value of this input.

**Metadata**

Placeholder nodes also have `meta[‘val’]`, like `call_function` nodes. The
`val` field in this case represents the input shape/dtype that the graph is
expected to receive for this input parameter.

### output

An output call represents a return statement in a function; it thus terminates the
current graph. There is one and only one output node, and it will always be the
last node of the graph.

**Representation in FX**

```
output[](args = (%something, …))
```

This has the exact semantics as in {class}`torch.fx`. `args` represents the node
to be returned.

**Metadata**

Output node has the same metadata as `call_function` nodes.

### get_attr

`get_attr` nodes represent reading a submodule from the encapsulating
{class}`torch.fx.GraphModule`. Unlike a vanilla FX graph from
{func}`torch.fx.symbolic_trace` in which `get_attr` nodes are used to read
attributes such as parameters and buffers from the top-level
{class}`torch.fx.GraphModule`, parameters and buffers are passed in as
inputs to the graph module, and stored in the top-level
{class}`torch.export.ExportedProgram`.

**Representation in FX**

```python
%name = get_attr[target = name](args = ())
```

**Example**

Consider the following model:

```python
from functorch.experimental.control_flow import cond

def true_fn(x):
    return x.sin()

def false_fn(x):
    return x.cos()

def f(x, y):
    return cond(y, true_fn, false_fn, [x])
```

Graph:

```
graph():
    %x_1 : [num_users=1] = placeholder[target=x_1]
    %y_1 : [num_users=1] = placeholder[target=y_1]
    %true_graph_0 : [num_users=1] = get_attr[target=true_graph_0]
    %false_graph_0 : [num_users=1] = get_attr[target=false_graph_0]
    %conditional : [num_users=1] = call_function[target=torch.ops.higher_order.cond](args = (%y_1, %true_graph_0, %false_graph_0, [%x_1]), kwargs = {})
    return conditional
```

The line, `%true_graph_0 : [num_users=1] = get_attr[target=true_graph_0]`,
reads the submodule `true_graph_0` which contains the `sin` operator.

## References

### SymInt

A SymInt is an object that can either be a literal integer or a symbol that represents
an Integer (represented in Python by `sympy.Symbol` class). When SymInt is a
symbol, it describes a variable of type integer that is unknown to the graph at
compile time, that is, its value is only known at runtime.

### FakeTensor

A FakeTensor is an object that contains the metadata of a tensor. It can be
viewed as having the following metadata.

```python
class FakeTensor:
  size: List[SymInt]
  dtype: torch.dtype
  device: torch.device
  dim_order: List[int]  # This doesn't exist yet
```

The size field of FakeTensor is a list of integers or SymInts. If SymInts are
present, this means this tensor has a dynamic shape. If integers are present, it
is assumed that the tensor will have that exact static shape. The rank of the
TensorMeta is never dynamic. The dtype field represents the dtype of the
output of that node. There are no implicit type promotions in Edge IR. There
are no strides in FakeTensor.

In other words:

- If the operator in node.target returns a Tensor, then `node.meta['val']` is a
  FakeTensor describing that tensor.
- If the operator in node.target returns an n-tuple of Tensors, then
  `node.meta['val']` is an n-tuple of FakeTensors describing each tensor.
- If the operator in node.target returns an int/float/scalar that is known at
  compile time, then `node.meta['val']` is None.
- If the operator in node.target returns an int/float/scalar that is not known
  at compile time, then `node.meta['val']` is of type SymInt.

For example:

- `aten::add` returns a Tensor; so its spec will be a FakeTensor with dtype
  and size of the tensor returned by this operator.
- `aten::sym_size` returns an integer; so its val will be a SymInt because its
  value is only available at runtime.
- `max_pool2d_with_indexes` returns a tuple of (Tensor, Tensor); so the spec
  will also be a 2-tuple of FakeTensor objects, the first TensorMeta describes
  the first element of the return value etc.

Python code:

```python
def add_one(x):
  return torch.ops.aten(x, 1)
```

Graph:

```
graph():
  %ph_0 : [#users=1] = placeholder[target=ph_0]
  %add_tensor : [#users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%ph_0, 1), kwargs = {})
  return [add_tensor]
```

FakeTensor:

```python
FakeTensor(dtype=torch.int, size=[2,], device=CPU)
```

### Pytree-able Types

We define a type “Pytree-able”, if it is either a leaf type or a container type
that contains other Pytree-able types.

Note:

> The concept of pytree is the same as the one documented
> [here](https://jax.readthedocs.io/en/latest/pytrees.html) for JAX:

The following types are defined as **leaf type**:

```{eval-rst}
.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Type
     - Definition
   * - Tensor
     - :class:`torch.Tensor`
   * - Scalar
     - Any numerical types from Python, including integral types, floating point types, and zero dimensional tensors.
   * - int
     - Python int (bound as int64_t in C++)
   * - float
     - Python float (bound as double in C++)
   * - bool
     - Python bool
   * - str
     - Python string
   * - ScalarType
     - :class:`torch.dtype`
   * - Layout
     - :class:`torch.layout`
   * - MemoryFormat
     - :class:`torch.memory_format`
   * - Device
     - :class:`torch.device`
```

The following types are defined as **container type**:

```{eval-rst}
.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Type
     - Definition
   * - Tuple
     - Python tuple
   * - List
     - Python list
   * - Dict
     - Python dict with Scalar keys
   * - NamedTuple
     - Python namedtuple
   * - Dataclass
     - Must be registered through `register_dataclass <https://github.com/pytorch/pytorch/blob/901aa85b58e8f490631ce1db44e6555869a31893/torch/export/__init__.py#L693>`__
   * - Custom class
     - Any custom class defined with `_register_pytree_node <https://github.com/pytorch/pytorch/blob/901aa85b58e8f490631ce1db44e6555869a31893/torch/utils/_pytree.py#L72>`__
```
