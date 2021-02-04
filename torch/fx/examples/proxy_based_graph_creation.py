import torch
from torch.fx import Proxy, Graph, GraphModule


'''
How to Create a Graph Using Proxy Objects Instead of Tracing

It's possible to directly create a Proxy object around a raw Node. This
can be used to create a Graph independently of symbolic tracing.

The following code demonstrates how to use Proxy with a raw Node to
append operations to a fresh Graph. We'll create two parameters (``x``
and ``y``), perform some operations on those parameters, then add
everything we created to the new Graph. We'll then wrap that Graph in
a GraphModule. Doing so creates a runnable instance of ``nn.Module``
where previously-created operations are represented in the Module's
``forward`` function.

By the end of the tutorial, we'll have added the following method to an
empty ``nn.Module`` class.

.. code-block:: python

    def forward(self, x, y):
        cat_1 = torch.cat([x, y]);  x = y = None
        tanh_1 = torch.tanh(cat_1);  cat_1 = None
        neg_1 = torch.neg(tanh_1);  tanh_1 = None
        return neg_1

'''


# Create a graph independently of symbolic tracing
graph = Graph()

# Create raw Nodes
raw1 = graph.placeholder('x')
raw2 = graph.placeholder('y')

# Initialize Proxies using the raw Nodes
y = Proxy(raw1)
z = Proxy(raw2)

# Create other operations using the Proxies `y` and `z`
a = torch.cat([y, z])
b = torch.tanh(a)
c = torch.neg(b)

# Create a new output Node and add it to the Graph. By doing this, the
# Graph will contain all the Nodes we just created (since they're all
# linked to the output Node)
graph.output(c.node)

# Wrap our created Graph in a GraphModule to get a final, runnable
# `nn.Module` instance
mod = GraphModule(torch.nn.Module(), graph)
