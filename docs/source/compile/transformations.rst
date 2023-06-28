Writing Graph Transformations on ATen IR
========================================

Passes
------

Since the ATen IR sits at the FX Graph/GraphModule level, any
transformations written for FX Graphs can be easily applied onto the
ATen IR. If you’re familiar with writing FX graph transformations, then
this will be the same.

The most direct way of writing transformations is by looping through the
given graph and directly manipulating the nodes within the graph.

For example, let’s say we want to replace
``torch.ops.aten.add.Tensor()`` calls with
``torch.ops.aten.mul.Tensor()`` calls:

.. code:: python

   import torch

   def replace_add_with_mul(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
       for node in gm.graph.nodes:
           if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:
               node.target = torch.ops.aten.mul.Tensor

We can also delete and append new nodes through FX utility functions
that can be found in the
`Graph <https://pytorch.org/docs/stable/fx.html#torch.fx.Graph>`__
documentation. For example, if we want to insert a
``torch.ops.aten.relu.default()`` after the ``add`` call:

.. code:: python

   import torch

   def insert_relu_after_add(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
       for node in gm.graph.nodes:
           if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:

               # Specifies the insertion point. Any nodes added to the graph within
               # this scope will be inserted after `node`
               with gm.graph.inserting_after(node):
                   # Insert a new `call_function` node with op `torch.ops.aten.relu.default`
                   new_relu_node = gm.graph.call_function(torch.ops.aten.relu.default, args=(node,))
                   # Replace all the places that use `node` to now use the `new_relu_node`
                   node.replace_all_uses_with(new_relu_node)

In general, transformations can be roughly categorized into a couple of
axis:

Axis A: 1. Creating one-to-X mapping (eg. decomposition) 2. Creating
many-to-one mapping (eg. fusion)

Axis B: 1. Doing forwards iteration (eg. shape propagation) 2. Doing
backwards iteration (eg. dead code elimination)

Axis C: 1. Dependent on local node information (eg. out-variant
conversion) 2. Dependent on global graph information (eg. memory
planning)

Our projection on the frequency of these use cases are: 1. A.1, B.1, C.1
2. A.2 3. B.2, C.2

Although we can make all graph transformations through directly
manipulating the graph, we also provide some helper utilities for some
ease of use for the level 1 and 2 use-cases.

Transformer
~~~~~~~~~~~

For level 1 uses cases (creating one-to-X mappings, doing forwards
iterations, and looking at local node information), we can utilize the
`Transformer <https://pytorch.org/docs/stable/fx.html#torch.fx.Transformer>`__
class to execute each node and recreate a graph, except with the
transformations specified.

One-to-One Pass
^^^^^^^^^^^^^^^

An example for one-to-one mappings, if we wanted to replace an op A with
another op B, we can run the GraphModule, and very time we see op A,
return op B.

An example is:

.. code:: python

   class ReplaceAddWithMul(torch.fx.Transformer):
       def call_function(self, target, args, kwargs):
           if target != torch.ops.aten.add.Tensor:
               return super().call_function(target, args, kwargs)
           return super().call_function(torch.ops.aten.mul.Tensor, args, kwargs)

   transformed_graph_module = ReplaceAddWithMul(graph_module).transform()

The ``super().call_function(target, args, kwargs, meta)`` call creates a
``call_function`` FX node, and returns the result of running the
operator with the given arguments.

One-to-X Pass
^^^^^^^^^^^^^

If we wanted to do one-to-X mappings, like replacing op A with 2 other
ops B and C, we would then make 2 calls to ``super().call_function`` to
create 2 FX nodes, one with op B and another with op C, and return the
result of running op C.

For example:

.. code:: python

   class ReplaceAddWithMulSub(torch.fx.Transformer):
       """
       Original:
           def f(x, y):
               return x + y

       After pass:
           def f(x, y):
               z = x * y
               return z - y
       """
       def call_function(self, target, args, kwargs):
           if target != torch.ops.aten.add.Tensor:
               return super().call_function(target, args, kwargs)

           x, y = args

           mul_res = super().call_function(torch.ops.aten.mul.Tensor, args, {})
           return super().call_function(torch.ops.aten.sub.Tensor, (mul_res, y), {})

   transformed_graph_module = ReplaceAddWithMulSub(graph_module).transform()

One-to-None Pass
^^^^^^^^^^^^^^^^

If we wanted to remove an op, we can just return the value passed into
the function:

.. code:: python

   class RemoveDetachPass(torch.fx.Transformer):
       def call_function(self, target, args, kwargs):
           if target not in (
               torch.ops.aten.detach.default,
               torch.ops.aten.detach_copy.default,
           ):
               return super().call_function(target, args, kwargs, meta)

           assert len(args) == 1
           return args[0]

   transformed_graph_module = RemoveDetachPass(graph_module).transform()

Utilizing Local Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

An example of utilizing local node information is, if we wanted to
convert all the scalars within the graph to tensors, we can run the
given ``fx.GraphModule``, and for every argument that contains a scalar,
we convert it to a tensor. It might look something like:

.. code:: python

   def args_map(target, fn, args, kwargs):
       assert isinstance(args, tuple)
       assert isinstance(kwargs, dict)
       args = list(args)
       kwargs = kwargs.copy()

       # Update the argument based on the function passed
       def update(key, args, schema):
           args[key] = fn(args[key], schema)

       # Update each argument in the schema
       for i, schema in enumerate(target._schema.arguments):
           if schema.name in kwargs:
               update(schema.name, kwargs, schema)
           elif not schema.kwarg_only and i < len(args):
               update(i, args, schema)
       return tuple(args), kwargs

   class ScalarToTensorPass(torch.fx.Transformer):
       def call_function(self, target, args, kwargs):
           breakpoint()
           def try_coerce(value, arg):
               return (
                   torch.tensor(value)
                   if isinstance(value, (float, int, bool))
                   and type(arg.type) == torch.TensorType
                   else value
               )

           args, kwargs = args_map(target, try_coerce, args, kwargs)
           return super().call_function(target, args, kwargs)

   transformed_graph_module = ScalarToTensorPass(graph_module).transform()

Subgraph Rewriter
~~~~~~~~~~~~~~~~~

For creating many-to-one mappings, we can utilize FX’s `subgraph
rewriter <https://github.com/pytorch/pytorch/blob/main/torch/fx/subgraph_rewriter.py>`__.
Given a ``pattern``, it creates a subgraph of operators matching to the
pattern, and then replaces each matched subgraph with the
``replacement``.

Note:

::

   This is an inplace operation.

The ``pattern`` and ``replacement`` inputs must be callable functions or
GraphModules containing the same operators that are used within the
graph (ATen ops) so that the subgraph rewriter can find the correct
pattern in the graph. Inputs to the pattern/replacement callables will
be treated as wildcards when matching.

An example:

.. code:: python

   from torch.fx import subgraph_rewriter

   def replace_patterns(graph_module):
       def pattern(x, y):
           x = torch.ops.aten.add.Tensor(x, y)
           x = torch.ops.aten.mul.Tensor(x, y)
           return x

       def replacement(x, y):
           return torch.ops.aten.sub.Tensor(x, y)

   replaced_patterns = subgraph_rewriter.replace_pattern_with_filters(
       traced_module, pattern, replacement
   )

The subgraph rewriter returns a list of ``ReplacedPatterns``:

.. code:: python

   @dataclass
   class ReplacedPatterns:
       # Node from which the match was found
       anchor: Node
       # Maps nodes in the pattern subgraph to nodes in the larger graph
       nodes_map: Dict[Node, Node]
       # List of nodes that were added into the graph
       replacements: List[Node]

Note:

::

   The nodes created by the subgraph rewriter will not have the metadata that
   is populated in the matched nodes, but you can use
   `ReplacedPatterns.nodes_map` to find the nodes in the original graph that
   were matched, and `ReplacedPatterns.replacements` to find the nodes that
   were replaced in the transformed graph.

Pass Manager
------------

The
```PassManager`` <https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/infra/pass_manager.py>`__
is a class used to run multiple passes on a given graph module. When
initializing a ``PassManager`` instance, we pass in a list of passes
that we want to run and set a couple of flags. To run the collection of
passes on a graph module, we can pass the graph module directly to the
``PassManager`` instance.

An example:

.. code:: python

   from torch.fx.passes.infra.pass_manager import PassManager

   pm = PassManager(
       passes=[replace_add_with_div, replace_div_with_mul],
       run_checks_after_each_pass=True,
       suppress_check_failures=False,
   )
   graph_module_out = pm(graph_module)

To add a common set of checks that are run after each pass, we can call
the function ``set_checks(check: Callable)`` which takes in a callable
function as input. If the ``run_checks_after_each_pass`` flag is set,
the ``check`` will be called after each pass is run on the graph module.

An example:

.. code:: python

   pm = PassManager(passes=[replace_add_with_div, replace_div_with_mul])

   def check_div_target(graph_module):
       for node in graph_module.graph.nodes:
           if node.op == "call_function" and node.target != torch.div:
               raise ValueError("Target should be div!")

   pm.add_checks(check_div_target)

   pm(graph_module)    # raises ValueError after replace_div_with_mul pass

Partitioner
-----------

There are a couple of common FX graph based partitioners we can use to
partition the graph.

Subgraph Matcher
~~~~~~~~~~~~~~~~

For finding subgraphs within a graph that match a specific pattern, we
can utilize FX’s
```SubgraphMatcher`` <https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/utils/matcher_utils.py>`__.

Class Attributes:

-  ``pattern (Graph)``: The targeted matching pattern. Placeholder nodes
   in the graph will be treated as wildcards when matching.
-  ``match_output (bool)``: If True, output node in the pattern graph
   will be treated as a part of the targeted pattern. If False, output
   node is ignored during match.
-  ``match_placeholder (bool)``: If True, placeholder node in the
   pattern graph will be treated as a part of the targeted pattern. If
   False, placeholder nodes will be used a wildcard.
-  ``remove_overlapping_matches (bool)``: If True, in the case of
   overlapping matches, only the first match will be returned.
-  ``ignore_literals (bool)``: If True, will not check if literals are
   equal and will instead treat them as wildcards.

An example:

.. code:: python

   from torch.fx.passes.utils.matcher_utils import SubgraphMatcher

   class LargeModel(torch.nn.Module):
       def __init__(self):
           super().__init__()
           self._weight = torch.nn.Parameter(torch.ones(3, 3))
           self._bias = torch.nn.Parameter(torch.ones(3, 3))

       def forward(self, x):
           return torch.ops.aten.addmm.default(self._bias, x, self._weight)

   large_model_graph = torch.export(LargeModel(), inputs).graph

   class PatternModel(torch.nn.Module):
       def __init__(self):
           super().__init__()
           self._weight_1 = torch.nn.Parameter(torch.ones(5, 5))
           self._bias_1 = torch.nn.Parameter(torch.ones(5, 5))

       def forward(self, x):
           return torch.ops.aten.addmm.default(self._bias_1, x, self._weight_1)

   pattern_graph = torch.export(PatternModel(), inputs).graph

   subgraph_matcher = SubgraphMatcher(pattern_graph)
   match_result = subgraph_matcher.match(large_model_graph)

The ``match`` function returns a list of ``InternalMatch``:

.. code:: python

   @dataclass
   class InternalMatch():
       # Nodes from which the match was found
       anchors: List[Node]
       # Maps nodes in the pattern subgraph to nodes in the larger graph
       nodes_map: Dict[Node, Node] = field(default_factory=dict)
       # Nodes in target graph that are matched placeholder in pattern
       placeholder_nodes: List[Node] = field(default_factory=list)
       # Nodes in matched subgraph returned by output
       returning_nodes: List[Node] = field(default_factory=list)

Capability Based Partitioner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To find the largest subgraphs of nodes that support a specific
invariant, we can utilize FX’s
```CapabilityBasedPartitioner`` <https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/infra/partitioner.py#L34>`__.

Class Attributes

-  ``graph_module (torch.fx.GraphModule)``: The graph module we are
   partitioning on.
-  ``operator_support (OperatorSupportBase)``: The object used to
   determine if a node in the graph is supported in the partition.
-  ``allows_single_node_partition (bool)``: If True, allows single node
   partitions to be formed.
-  ``non_compute_ops (Optional[Sequence[str]])``: A set of ops that are
   considered to be “non-compute” (ex ``torch.ops.aten.view`` and
   ``_operator.getitem``, so that the partitioner will not create graphs
   that only contain these non-compute ops
-  ``allowed_single_node_partition_ops (Optional[Sequence[str]])``: A
   set of ops that are allowed to be in a single node partition.

The
```OperatorSupportBase`` <https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/operator_support.py#LL28C1-L28C1>`__
class is used by the partitioner to determine if a specific node in the
graph belongs in the partition. This is done by overriding the
``is_node_supported`` function. You can chain multiple
``OperatorSuppportBase`` by using
```chain`` <https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/operator_support.py#L150>`__\ (which
returns False if any of the OperatorSupportBase return False) and
```any_chain`` <https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/operator_support.py#L164>`__
(which returns True if any of the OperatorSupportBase returns True).

An example:

.. code:: python

   from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
   from torch.fx.passes.operator_support import any_chain, OperatorSupportBase

   class AddMulOperatorSupport(OperatorSupportBase):
       def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
           return node.op == "call_function" and node.target in [
               torch.ops.aten.add.Tensor, torch.ops.aten.mul.Tensor,
           ]

   capability_partitioner = CapabilityBasedPartitioner(
       graph_module,
       op_support,
   )

   # Returns a list of partitions (list of nodes that belong in each partition)
   partition_list = capability_partitioner.propose_partitions()
   # Fuses the partitions into graph modules and inserts `call_module` nodes in the graph
   fused_graph_module = capability_partitioner.fuse_partitions(partition_list)
