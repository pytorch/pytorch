.. _autograd-mechanics:

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

.. _how-autograd-encodes-history:

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

How does Autograd work for Complex
^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyTorch supports gradients for both holomorphic and non-holomorphic functions.
We follow the `HIPS/autograd <https://github.com/HIPS/autograd/blob/master/docs/tutorial.md#complex-numbers>`_
convention to compute derivatives for complex numbers.
Consider a `C → C function, f, expressed in terms of R → R functions, u and v`:

.. code::

    def f(z):
            x, y = real(z), imag(z)
            return u(x, y) + v(x, y) * 1j

We define grad of f as:

.. code::

    def grad_f(z):
        x, y = real(z), imag(z)
        return grad(u, 0)(x, y) - i * grad(u, 1)(x, y)

This convention covers three important cases:

1. If f is holomorphic, then we get the usual complex derivative,
   since `grad(u, 0) equals grad(v, 1)` and `grad(u, 1) equals - grad(v, 0)`.

2. If f is a real-valued loss function of a complex parameter z,
   then we get a result that we can use in a gradient-based optimizer,
   by taking steps in the direction of the complex conjugate of grad(f)(z).

3. If f is a real-to-real function that happens to use complex primitives internally,
   some of which must necessarily be non-holomorphic (for example, FFTs to implement convolutions),
   then we get the same result that a purely real implementation would have given.

Our convention doesn't handle the case where f is a non-holomorphic function and
when all of du/dx, du/dy, dv/dx and dv/dy are of interest to you because the answer
would have to contain four real values and there would be no way to express it as
a single complex number.

We define primitive vector-Jacobian products of complex functions as:

.. code::

    def f_vjp(g, z):
        z_x, z_y = real(z), imag(z)
        g_x, g_y = real(g), imag(g)
        return (       g_x * grad(u, 0)(x, y)
                - i * g_x * grad(u, 1)(x, y)
                -     g_y * grad(v, 0)(x, y)
                + i * g_y * grad(v, 1)(x, y))

For holomorphic primitives, this is just the regular complex derivative multiplied by g,
so most simple math primitives don't need to be changed from their real implementations.
For non-holomorphic primitives, it preserves all four real partial derivatives as if we were
treating complex numbers as real 2-tuples. The negative signs are just to take care of
complex conjugation, and the fact that we’re working with covectors.

Multithreaded Autograd
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The autograd engine is responsible for running all the backward operations
necessary to compute the backward pass. This section will describe all the details
that can help you make the best use of it in a multithreaded environment.(this is
relevant only for PyTorch 1.6+ as the behavior in previous version was different).

User could train their model with multithreading code (e.g. Hogwild training), and
does not block on the concurrent backward computations, example code could be:

.. code::

    # Define a train function to be used in different threads
    def train_fn():
        x = torch.ones(5, 5, requires_grad=True)
        # forward
        y = (x + 3) * (x + 4) * 0.5
        # backward
        y.sum().backward()
        # potential optimizer update


    # User write their own threading code to drive the train_fn
    threads = []
    for _ in range(10):
        p = threading.Thread(target=train_fn, args=())
        p.start()
        threads.append(p)

    for p in threads:
        p.join()


Note that some behaviors that user should be aware of:

Concurrency on CPU
------------------

When you run ``backward()`` or ``grad()`` via python or C++ API in multiple
threads on CPU, you are expecting to see extra concurrency instead of
serializing all the backward calls in a specific order during execution
(behavior before PyTorch 1.6).

Non-determinism
------------------

If you are calling ``backward()`` on multiple thread concurrently but with
shared inputs (i.e. Hogwild CPU training). Since parameters are automatically
shared across threads, gradient accumulation might become non-deterministic on
backward calls across threads, because two backward calls might access and try
to accumulate the same ``.grad`` attribute. This is technically not safe, and
it might result in racing condition and the result might be invalid to use.

But this is expected pattern if you are using the multithreading approach to
drive the whole training process but using shared parameters, user who use
multithreading should have the threading model in mind and should expect this
to happen. User could use the functional API :func:`torch.autograd.grad` to
calculate the gradients instead of ``backward()`` to avoid non-determinism.

Graph retaining
------------------

If part of the autograd graph is shared between threads, i.e. run first
part of forward single thread, then run second part in multiple threads,
then the first part of graph is shared. In this case different threads
execute ``grad()`` or ``backward()`` on the same graph might have issue of
destroying the graph on the fly of one thread, and the other thread will
crash in this case. Autograd will error out to the user similar to what call
``backward()`` twice with out ``retain_graph=True``, and let the user know
they should use ``retain_graph=True``.

Thread Safety on Autograd Node
------------------------------

Since Autograd allows the caller thread to drive its backward execution for
potential parallelism, it's important that we ensure thread safety on CPU with
parallel backwards that share part/whole of the GraphTask.

Custom Python ``autograd.function`` is automatically thread safe because of GIL.
for built-in C++ Autograd Nodes(e.g. AccumulateGrad, CopySlices) and custom
``autograd::Function``, the Autograd Engine uses thread mutex locking to protect
thread safety on autograd Nodes that might have state write/read.

No thread safety on C++ hooks
------------------------------

Autograd relies on the user to write thread safe C++ hooks. If you want the hook
to be correctly applied in multithreading environment, you will need to write
proper thread locking code to ensure the hooks are thread safe.
