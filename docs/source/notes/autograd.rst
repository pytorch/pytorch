.. _autograd-mechanics:

Advanced autograd
=================

This note presents some more advanced topics about the autograd in PyTorch.
If you are looking for more basic information about autograd, check out
the `autograd introduction <https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html>`_.

.. contents::
    :local:
    :depth: 2

Mathematical formulation
------------------------

What does the autograd package do?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The autograd package allows us to perform automatic differentiation on top of tensors.
Automatic differentiation computes the dot product between a given vector and the Jacobian of a user-defined function.
Consider the function :math:`f: x \rightarrow y`, where :math:`x` is a vector of size :math:`I` and :math:`y` is a vector of size :math:`O`.
For simplicity, we'll consider inputs and ouputs to be 1-D; in practice, they could have an arbitrary number of dimensions, but automatic differentiation is indifferent to the dimensionality of tensors.
Then, the Jacobian matrix associated with this function is that matrix :math:`J_f` of size :math:`(O, I)` such that each entry is
given by :math:`(J_f)_{ij} = \dfrac{\partial{x_j}}{\partial{y_i}}`. We will write :math:`J_f = \dfrac{\partial{y}}{\partial{x}}`.

Currently, PyTorch only implements reverse mode automatic differentiation which, given
an arbitrary vector :math:`v` of size :math:`O`, will compute :math:`v^T J_f` (the so-called vector-Jacobian product, or vjp, as opposed to the Jacobian-vector product, which is computed by forward mode automatic differentiation).

When :math:`f`'s output is a scalar value, a vector-Jacobian product where :math:`v` is :math:`1` will compute exactly :math:`J_f`.
This fact is why reverse mode automatic differentiation used extensively in deep learning: loss functions are scalar and reverse mode automatic
differentiation will compute the gradients of the loss with respect to the weights of the model.
It is called the backpropagation algorithm in this case.



How does it do that?
^^^^^^^^^^^^^^^^^^^^

One simple interpretation of what it is doing is splitting the user function :math:`f` into composition of smaller
pre-defined operations.
We will call such operations elementary.
The gradient of the original function is then computed by using the chain rule on each of these elementary operations.

For example, if we split :math:`f` into two elementary operations, namely :math:`y = op_2(op_1(x))`,
the automatic differentiation will compute :math:`v^T J_f = (v^T J_{op_2}) J_{op_1} =  v^T J_{op_2} J_{op_1}`.

What is your code computing?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can request that PyTorch computes gradients for a tensor by passing the keyword argument `requires_grad=True`
when constructing the tensor (or, post facto, set this attribute with `requires_grad_(True)`).
The user can then use any PyTorch function on this tensor to compute the final value of its function.
Each of these functions is composed of one or more elementary operation.
Each operation will add an element to the "chain" if Jacobian products that the backward pass will compute.
In particular, consider the following function:

.. code::

    def f(w):
        x = op_1(w)
        y = op_2(x)
        return y

    # w is a Tensor of the appropriate size
    w.requires_grad_()

    # v is a Tensor of the appropriate size
    f(w).backward(v)

The backward pass will compute the following for a given vector :math:`v`: :math:`v^T J_f = v^T J_{op_2} J_{op_1} = v^T \dfrac{\partial{y}}{\partial{x}} \dfrac{\partial{x}}{\partial{w}} = v^T \dfrac{\partial{y}}{\partial{w}}`.

Adding a new operation as follows:

.. code::

    def f(w):
        x = op_1(w)
        y = op_2(x)
        z = op_3(y)
        return z

    w.requires_grad_()
    f(w).backward(v)

Will change the computation to: :math:`v^T J_f = v^T J_{op_3} J_{op_2} J_{op_1} = v^T \dfrac{\partial{z}}{\partial{y}} \dfrac{\partial{y}}{\partial{x}} \dfrac{\partial{x}}{\partial{w}} = v^T \dfrac{\partial{z}}{\partial{w}}`.

Handling in-place operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The only special case that needs to be considered here are in-place operations.
The main reason to use in-place operations is to preserve side effects and prevent additional memory requirements.

- The first point means that if a tensor is referenced in different places, the in-place operation should modify all these references.
- The second point means that no extra memory allocation should be done under the hood otherwise it would defeat the purpose of in-place operations.

To be able to fulfill these two needs, we make in-place operations actually be in-place during the forward pass giving exactly the same behavior
as any other non-autograd library that provides in-place operations.
To be able to compute the gradients, we consider that there are different versions of the tensor that existed and that the user tensor always points to the latest version.
This allows us to write the automatic differentiation rules as before.

See the two functions below for examples where we use subscripts to specify each version of the Tensor.:

.. code::

    def f(x):
        z = op_1(x)
        # z points to z_0
        z.op_2_()
        # z points to z_1
        return z

    def g(x):
        z = op_1(x)
        # z points to z_0
        y = z.select(0, 0)
        # y is a view into z
        # y points to y_0 = z_0.select(0, 0)
        z.op_2_()
        # z now points to z_1 which is the result of op_2_
        # y now points to z_1.select(0, 0)
        return y

The corresponding autograd computations can simply be written using the implicit variables: :math:`v^T J_f = v^T J_{op_2\_} J_{op_1} = v^T \dfrac{\partial{z_1}}{\partial{z_0}} \dfrac{\partial{z_0}}{\partial{x}} = v^T \dfrac{\partial{z_1}}{\partial{x}}`.
And for g, it is harder to write as only part of :code:`op_2_()` is used when computing :code:`y`, but it can be written as :math:`v^T J_g = v^T \dfrac{\partial{y_1}}{\partial{y_0}} \dfrac{\partial{y_0}}{\partial{z_0}} \dfrac{\partial{z_0}}{\partial{x}}`.

Computing gradient of a subset of your whole program
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In some cases, the user might need to compute something that is not the "true gradient" of its function.
We saw above how to start recording gradient only from a given Tensor using ``requires_grad`` but some cases require to disable them locally or stop tracking them.
We identify few use cases here:

- Ignore the gradients for some part of the computation as they could render the computed gradients unstable, e.g. gradients of square root near :math:`0`.
- Ignore the gradients in a set of computations where we know they will be :math:`0` and so don't need to be computed, e.g. a set of operations that compute values that are the used as integers to index a tensor.
- Update the value of a tensor that requires gradients without this update being recorded: initialization of the weights of a neural network.
- Work with tensors that may or may not require gradients to perform operations that do not need gradients computed: optimizers for neural networks or method that compute the 0-1 accuracy for classification.

To be able to perform all these operations, we provide three main constructs.

Detaching
"""""""""

The first construct is the method :code:`y = x.detach()` that can be used on any tensor.
This is a special function for which :math:`J_{detach} = \bm{0}`.
This means that however :code:`y` is used, the contribution via :code:`y` to the gradient of :code:`x` is always going to be :math:`0`.

.. code::

    def f(x):
        z = op_1(x)

        y = z.detach()
        return y

    # x is a Tensor of the appropriate size
    x.requires_grad_()

    # v is a Tensor of the appropriate size
    f(x).backward(v)

The backward pass will compute the following for a given vector :math:`v`: :math:`v^T J_f = v^T J_{detach} J_{op_1} = v^T \bm{0} \dfrac{\partial{z}}{\partial{x}} = \bm{0}`.

Note that the new Tensor shares the same data as the original one (it is a view).
This means that in-place operations on :code:`y` will change values in :code:`x` .

Disable gradient computation
""""""""""""""""""""""""""""

The second construct is the function decorator :code:`torch.no_grad()` (it can also be used as a context manager).
When it decorates a function :code:`outputs = f(inputs)`, it is defined as enforcing :math:`\forall in \in inputs, \forall out \in outputs, \dfrac{\partial{out}}{\partial{in}} = 0`.
Note that for the context manager version, outputs are all the variables that are assigned (on the left side of an :code:`=` in the block).

.. code::

    # Original function
    @torch.no_grad()
    def f(x, y):
        # Some ops using x and y, producing z and w
        z = foo(x, y)
        w = 3 * z
        return z, w

    # It can also be used as a context manager:
    # Some variables x, y
    with torch.no_grad():
        # Some ops using x, y, producing z, w
        z = foo(x, y)
        w = 3 * z

This is particularly useful to perform operations for which gradients are not needed or that the user does not want as part of the gradient computation.
This can range from model forward pass when doing inference to optimizer update of the parameters of a network.

Hooks
"""""

The third contruct is more general as it allows to change the gradients in arbitrary ways, not only setting them to :math:`0` as the first two.
A hook can be registered on any Tensor that requires gradients.
This hook function will be passes the computed gradient as input and can optionally return the new value for this gradient.

.. code::

  def h(g):
    return foo(g)

  def f(x):
      y = op_1(x)
      y.register_hook(h)

      z = op_2(y)
    return z

  # x is a Tensor of the appropriate size
  x.requires_grad_()

  # v is a Tensor of the appropriate size
  f(x).backward(v)

The backward pass will compute the following for a given vector :math:`v`: :math:`\text{foo}(v^T J_{op_2}) J_{op_1}`.



Equivalence
"""""""""""

detach and no_grad can be implemented using the others as follows:

.. code::

    # Original code
    y = x.detach()

    # torch.no_grad version
    with torch.no_grad():
        # Using `view_as` as an identity here
        y = x.view_as(x)

    # hook version
    # Using `view_as` as an identity here
    y = x.view_as(x)
    y.register_hook(lambda grad: torch.zeros_like(grad))


.. code::

    # Original code
    @torch.no_grad()
    def f(x, y):
        # Some ops using x and y, producing z and w
        return z, w

    # Equivalent formulation with detach
    def f_eq(x, y):
        x, y = x.detach(), y.detach()
        # Some ops using x and y, producing z and w
        return z, w

    # Equivalent formulation with detach
    def f_eq(x, y):
        x, y = x.view_as(x), y.view_as(y)
        x.register_hook(lambda grad: torch.zeros_like(grad))
        y.register_hook(lambda grad: torch.zeros_like(grad))
        # Some ops using x and y, producing z and w
        return z, w

Autograd quirks
---------------

Will in-place operations work?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We repeat the example from above here for clarity:

.. code::

    def f(x):
        z = op_1(x)
        # z points to z_0
        z.op_2_()
        # z points to z_1
        return z

The main limitation of our in-place operation strategy is that in this example, if :code:`op_1` needs the value of its output :code:`z_0` to compute :math:`J_{op_1}`, then the autograd computation cannot be performed correctly anymore.
Because we do not want to hide memory allocation from the user, the backward pass will raise an error.
In such a case, to be able to perform this backward pass we need to either replace :code:`op_2_` with an equivalent out-of-place operation or make sure :code:`op_2_` modifies in-place a copy of :code:`z` by adding a :code:`.clone()` for example.

Gradient of zero vs independent of the input
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As long as all gradients are finite values, these two can be seen as being the same.
In practice, this is the case for most pytorch programs.
The autograd engine does not make a difference between the two.

Unfortunately as soon as :code:`inf` or :code:`nan` appear, the two become different.
This is because a gradient of :math:`0` should propagate the non-finite values while an independent gradient should propagate an independent gradient (which is the same as :math:`0`).
This is a problem when trying to hide pathological points with indexing as can be seen for example in `Issue 9688 <https://github.com/pytorch/pytorch/issues/9688>`_.

How do we express gradients of zero
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A gradient of :math:`0` (or independent) can be expressed in any of the following ways:

- A Tensor full of zeros
- In python, a ``None`` Tensor
- In cpp, and ``undefined`` Tensor
- Using ``autograd.grad(allow_unused=False)``, an error stating that the output is independent of the input

Non-differentiable functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The chain rule can **only** be applied in the case where are considering each elementary operation
at a point where it is differentiable. If any of them is evaluated at a point where it is not,
the computed gradient can be arbitrarily incorrect. For example, consider the gradient at :math:`0` of the identity function
when it is decomposed as :math:`id(x) = relu(x) - relu(-x)`.

This means that the function that computes the :math:`2` norm and the one that square all the elements of a tensor, sum them and returns the square root
of the result will not have the same behavior with respect to gradients all the time.
In particular, if the sum is :math:`0`, the square root function is not differentiable at this point and so the returned gradient can be anything while the :math:`2`
norm function has a well defined gradient at :math:`0` of :math:`0`.

To try and reduce the impact of this limitation, we define the gradients of the elementary operations by applying the following rules in order:

#. If the function is not defined (:math:`sqrt(-1)` or :math:`log(-1)` for example) then this is undefined. Most function will return :math:`nan`, but for performance reasons, some functions will return not-nan values (:math:`log(-1)` for example).
#. If a gradient exist at the current point, return it.
#. If the function is convex, return one of it's subgradients at the current point, if more than one exists, return the subgradient with minimum norm (as it is always a descent direction, see "Convex Optimization Algorithms" by Dimitri Bertsekas, Proposition 4.3.1 for a proof).
#. Define the gradient at the current point by continuity (note that :math:`inf` is possible here). If multiple values are possible, pick one arbitrarily.

.. note::

    - Even though we try our best, unless all the used elementary operations used are in case 2, the returned gradient is not guaranteed to be correct.
    - Both the ``detach()`` method and the ``torch.no_grad`` decorator are expected not to follow these rules.
    - This is work in progress, if you find a :code:`Function` that does not follow these rules, please open a new issue with the tag "module:autograd".


Implementation details
----------------------

.. warn::

  All the following elements are implementation details.
  You should not rely on them being fixed but they can be useful to debug code that uses the current version.

Computational graph
^^^^^^^^^^^^^^^^^^^

To know in which order the chain rule should be applied, we need to record the order the operations are applied to be able to replay it in the reverse order later.
This is known as tape-based automatic differentiation.
In our case, we create a directed acyclic graph.
The nodes, called :code:`Node` represent the backward computations to be performed for a give elementary operation.
The edges link a given :code:`Node` to all the :code:`Node` s that created the inputs of the elementary operation it represents.

You can access the :code:`Node` that created a given Tensor (if the Tensor requires gradients and is not a leaf Tensor) using the ``.grad_fn`` field.
You can then access the edges that link the different :code:`Node` s by using the ``.next_functions`` field of a :code:`Node` .
A handy package to explore this graph is `torchviz <https://github.com/szagoruyko/pytorchviz>`_.

We use ``autograd.Function`` as a nice interface to create this graph.
Indeed, during the ``.apply``, the forward is invoked and all the inputs and outputs are connected properly to the exiting graph associated to the inputs.
The :code:`Node` that is created is a wrapper around the user-defined backward function (and the name is appended with ``Backward`` as you can see in the
graph examples in torchviz.


Not Gradients
^^^^^^^^^^^^^

The operations that do not compute gradients that are presented above are implemented as follows to obtain the behavior described above.

``detach`` is returning a new Tensor object that shares the same Storage.
The Tensor is the same except that it does not require gradient and its grad_fn is ``None``.

``torch.no_grad`` simply prevents any :code:`Node` from being added to the graph.


In-place handling
^^^^^^^^^^^^^^^^^

To get the required behavior for in-place operations, two things are needed.
First, to be able to ensure correctness, we need to track the versions of the different Tensors to make sure that a :code:`Node` has access to the version it needs, not a modified one.
Second, we need to build a graph that will perform the correct backward computation.

From the Python api, the current version of a Tensor can be queried with :code:`._version` .
The :code:`Function` wrapper takes care of managing this version by using the information provided by the :code:`.mark_dirty()` method available on the context.
It also takes care of making sure that the Tensors given to the :code:`.backward()` function have not been modified in-place.

Building the graph in the absence of any views is simple as all the in-place operations can be "reinterpreted" as out of place.
The in-place operations will only create a linear chain of :code:`Node` s as if :code:`x.op_()` was written :code:`x = x.op()` (assuming corresponding out of place operation exists).

Building the graph is more complex as we allow both views and in-place operations.
Here, the view operations are responsible to mark any views that they make as being linked to the input.
You can check if a Tensor is a view with :code:`._is_view` and query the Tensor it is a view of with :code:`._base` .
Two case can happen then:

- If this Tensor is not modified in-place, then its :code:`.grad_fn` is the :code:`Node` associated with the view operation. And the base's :code:`.grad_fn` is the :code:`Node` that corresponds to its forward function.
- If this Tensor is modified in-place, its graph is "rebased". This has a few effects:

  - The base :code:`.grad_fn` becomes a :code:`CopySlices` . This special :code:`Node` is a wrapper around the :code:`Node` corresponding to the in-place operations and apply it to the subset of the base on which the in-place operation was performed (through the view).
  - All the views' :code:`.grad_fn` become :code:`AsStridedBackard` s that point to the newly created :code:`CopySlices` from the base.

This is repeated every time an inplace operation is performed on a view and so the base's backward graph can contain a chain of :code:`CopySlices` before finally getting to the :code:`Node` that corresponds to its forward function.





// New things that were added since the original draft

Multithreaded Autograd
----------------------

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
^^^^^^^^^^^^^^^^^^

When you run ``backward()`` or ``grad()`` via python or C++ API in multiple
threads on CPU, you are expecting to see extra concurrency instead of
serializing all the backward calls in a specific order during execution
(behavior before PyTorch 1.6).

Non-determinism
^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^

If part of the autograd graph is shared between threads, i.e. run first
part of forward single thread, then run second part in multiple threads,
then the first part of graph is shared. In this case different threads
execute ``grad()`` or ``backward()`` on the same graph might have issue of
destroying the graph on the fly of one thread, and the other thread will
crash in this case. Autograd will error out to the user similar to what call
``backward()`` twice with out ``retain_graph=True``, and let the user know
they should use ``retain_graph=True``.

Thread Safety on Autograd Node
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since Autograd allows the caller thread to drive its backward execution for
potential parallelism, it's important that we ensure thread safety on CPU with
parallel backwards that share part/whole of the GraphTask.

Custom Python ``autograd.function`` is automatically thread safe because of GIL.
for built-in C++ Autograd Nodes(e.g. AccumulateGrad, CopySlices) and custom
``autograd::Function``, the Autograd Engine uses thread mutex locking to protect
thread safety on autograd Nodes that might have state write/read.

No thread safety on C++ hooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Autograd relies on the user to write thread safe C++ hooks. If you want the hook
to be correctly applied in multithreading environment, you will need to write
proper thread locking code to ensure the hooks are thread safe.

.. _complex_autograd-doc:

Autograd for Complex Numbers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**What notion of complex derivative does PyTorch use?**
*******************************************************

PyTorch follows `JAX's <https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#Complex-numbers-and-differentiation>`_
convention for autograd for Complex Numbers.

Suppose we have a function :math:`F: ℂ → ℂ` which we can decompose into functions u and v
which compute the real and imaginary parts of the function:

    .. code::

        def F(z):
            x, y = real(z), imag(z)
            return u(x, y) + v(x, y) * 1j

where :math:`1j` is a unit imaginary number.

We define the :math:`JVP` for function :math:`F` at :math:`(x, y)` applied to a tangent
vector :math:`c+dj \in C` as:

    .. math:: \begin{bmatrix} 1 & 1j \end{bmatrix} * J * \begin{bmatrix} c \\ d \end{bmatrix}

where

    .. math::
        J = \begin{bmatrix}
            \frac{\partial u(x, y)}{\partial x} & \frac{\partial u(x, y)}{\partial y}\\
            \frac{\partial v(x, y)}{\partial x} & \frac{\partial v(x, y)}{\partial y} \end{bmatrix} \\

This is similar to the definition of the JVP for a function defined from :math:`R^2 → R^2`, and the multiplication
with :math:`[1, 1j]^T` is used to identify the result as a complex number.

We define the :math:`VJP` of :math:`F` at :math:`(x, y)` for a cotangent vector :math:`c+dj \in C` as:

    .. math:: \begin{bmatrix} c & -d \end{bmatrix} * J * \begin{bmatrix} 1 \\ -1j \end{bmatrix}

In PyTorch, the `VJP` is mostly what we care about, as it is the computation performed when we do backward
mode automatic differentiation. Notice that d and :math:`1j` are negated in the formula above. Please look at
the `JAX docs <https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#Complex-numbers-and-differentiation>`_
to get explanation for the negative signs in the formula.

**What happens if I call backward() on a complex scalar?**
*******************************************************************************

The gradient for a complex function is computed assuming the input function is a holomorphic function.
This is because for general :math:`ℂ → ℂ` functions, the Jacobian has 4 real-valued degrees of freedom
(as in the `2x2` Jacobian matrix above), so we can’t hope to represent all of them with in a complex number.
However, for holomorphic functions, the gradient can be fully represented with complex numbers due to the
Cauchy-Riemann equations that ensure that `2x2` Jacobians have the special form of a scale-and-rotate
matrix in the complex plane, i.e. the action of a single complex number under multiplication. And so, we can
obtain that gradient using backward which is just a call to `vjp` with covector `1.0`.

The net effect of this assumption is that the partial derivatives of the imaginary part of the function
(:math:`v(x, y)` above) are discarded for :func:`torch.autograd.backward` on a complex scalar
(e.g., this is equivalent to dropping the imaginary part of the loss before performing a backwards).

For any other desired behavior, you can specify the covector `grad_output` in :func:`torch.autograd.backward` call accordingly.

**How are the JVP and VJP defined for cross-domain functions?**
***************************************************************

Based on formulas above and the behavior we expect to see (going from :math:`ℂ → ℝ^2 → ℂ` should be an identity),
we use the formula given below for cross-domain functions.

The :math:`JVP` and :math:`VJP` for a :math:`f1: ℂ → ℝ^2` are defined as:

    .. math:: JVP = J * \begin{bmatrix} c \\ d \end{bmatrix}

    .. math:: VJP = \begin{bmatrix} c & d \end{bmatrix} * J * \begin{bmatrix} 1 \\ -1j \end{bmatrix}

The :math:`JVP` and :math:`VJP` for a :math:`f1: ℝ^2 → ℂ` are defined as:

    .. math:: JVP = \begin{bmatrix} 1 & 1j \end{bmatrix} * J * \begin{bmatrix} c \\ d \end{bmatrix} \\ \\

    .. math:: VJP = \begin{bmatrix} c & -d \end{bmatrix} * J
