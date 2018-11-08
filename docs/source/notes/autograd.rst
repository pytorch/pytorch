Autograd mechanics
==================

This note will present an overview of how autograd works and records the
operations. It's not strictly necessary to understand all this, but we recommend
getting familiar with it, as it will help you write more efficient, cleaner
programs, and can aid you in debugging.

.. _excluding-subgraphs:

Excluding subgraphs from backward
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every Tensor has a flag: :attr:`requires_grad` that allows for fine grained
exclusion of subgraphs from gradient computation and can increase efficiency.

.. _excluding-requires_grad:

``requires_grad``
~~~~~~~~~~~~~~~~~

If there's a single input to an operation that requires gradient, its output
will also require gradient. Conversely, only if all inputs don't require
gradient, the output also won't require it. Backward computation is never
performed in the subgraphs, where all Tensors didn't require gradients.

.. code::

    >>> x = torch.randn(5, 5)  # requires_grad=False by default
    >>> y = torch.randn(5, 5)  # requires_grad=False by default
    >>> z = torch.randn((5, 5), requires_grad=True)
    >>> a = x + y
    >>> a.requires_grad
    False
    >>> b = a + z
    >>> b.requires_grad
    True

This is especially useful when you want to freeze part of your model, or you
know in advance that you're not going to use gradients w.r.t. some parameters.
For example if you want to finetune a pretrained CNN, it's enough to switch the
:attr:`requires_grad` flags in the frozen base, and no intermediate buffers will
be saved, until the computation gets to the last layer, where the affine
transform will use weights that require gradient, and the output of the network
will also require them.

.. code::

    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
    model.fc = nn.Linear(512, 100)

    # Optimize only the classifier
    optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)

How autograd encodes the history
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Autograd is reverse automatic differentiation system.  Conceptually,
autograd records a graph recording all of the operations that created
the data as you execute operations, giving you a directed acyclic graph
whose leaves are the input tensors and roots are the output tensors.
By tracing this graph from roots to leaves, you can automatically
compute the gradients using the chain rule.

Internally, autograd represents this graph as a graph of
:class:`Function` objects (really expressions), which can be
:meth:`~torch.autograd.Function.apply` ed to compute the result of
evaluating the graph.  When computing the forwards pass, autograd
simultaneously performs the requested computations and builds up a graph
representing the function that computes the gradient (the ``.grad_fn``
attribute of each :class:`torch.Tensor` is an entry point into this graph).
When the forwards pass is completed, we evaluate this graph in the
backwards pass to compute the gradients.

An important thing to note is that the graph is recreated from scratch at every
iteration, and this is exactly what allows for using arbitrary Python control
flow statements, that can change the overall shape and size of the graph at
every iteration. You don't have to encode all possible paths before you
launch the training - what you run is what you differentiate.

In-place operations with autograd
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Supporting in-place operations in autograd is a hard matter, and we discourage
their use in most cases. Autograd's aggressive buffer freeing and reuse makes
it very efficient and there are very few occasions when in-place operations
actually lower memory usage by any significant amount. Unless you're operating
under heavy memory pressure, you might never need to use them.

There are two main reasons that limit the applicability of in-place operations:

1. In-place operations can potentially overwrite values required to compute
   gradients.

2. Every in-place operation actually requires the implementation to rewrite the
   computational graph. Out-of-place versions simply allocate new objects and
   keep references to the old graph, while in-place operations, require
   changing the creator of all inputs to the :class:`Function` representing
   this operation. This can be tricky, especially if there are many Tensors
   that reference the same storage (e.g. created by indexing or transposing),
   and in-place functions will actually raise an error if the storage of
   modified inputs is referenced by any other :class:`Tensor`.

In-place correctness checks
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every tensor keeps a version counter, that is incremented every time it is
marked dirty in any operation. When a Function saves any tensors for backward,
a version counter of their containing Tensor is saved as well. Once you access
``self.saved_tensors`` it is checked, and if it is greater than the saved value
an error is raised. This ensures that if you're using in-place
functions and not seeing any errors, you can be sure that the computed
gradients are correct.
