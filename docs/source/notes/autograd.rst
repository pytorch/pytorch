.. _autograd-mechanics:

Autograd mechanics
==================

This note will present an overview of how autograd works and records the
operations. It's not strictly necessary to understand all this, but we recommend
getting familiar with it, as it will help you write more efficient, cleaner
programs, and can aid you in debugging.

.. _how-autograd-encodes-history:

How autograd encodes the history
--------------------------------

Autograd is a reverse automatic differentiation system.  Conceptually,
autograd records a graph recording all of the operations that created
the data as you execute operations, giving you a directed acyclic graph
whose leaves are the input tensors and roots are the output tensors.
By tracing this graph from roots to leaves, you can automatically
compute the gradients using the chain rule.

Internally, autograd represents this graph as a graph of
:class:`Function` objects (really expressions), which can be
:meth:`~torch.autograd.Function.apply` ed to compute the result of
evaluating the graph.  When computing the forward pass, autograd
simultaneously performs the requested computations and builds up a graph
representing the function that computes the gradient (the ``.grad_fn``
attribute of each :class:`torch.Tensor` is an entry point into this graph).
When the forward pass is completed, we evaluate this graph in the
backwards pass to compute the gradients.

An important thing to note is that the graph is recreated from scratch at every
iteration, and this is exactly what allows for using arbitrary Python control
flow statements, that can change the overall shape and size of the graph at
every iteration. You don't have to encode all possible paths before you
launch the training - what you run is what you differentiate.

.. _saved-tensors-doc:

Saved tensors
^^^^^^^^^^^^^

Some operations need intermediary results to be saved during the forward pass
in order to execute the backward pass. For example, the function
:math:`x\mapsto x^2` saves the input :math:`x` to compute the gradient.

When defining a custom Python :class:`~torch.autograd.Function`, you can use
:func:`~torch.autograd.function._ContextMethodMixin.save_for_backward` to save
tensors during the forward pass and
:attr:`~torch.autograd.function.Function.saved_tensors` to retrieve them
during the backward pass. See :doc:`/notes/extending` for more information.

For operations that PyTorch defines (e.g. :func:`torch.pow`), tensors are
automatically saved as needed. You can explore (for educational or debugging
purposes) which tensors are saved by a certain ``grad_fn`` by looking for its
attributes starting with the prefix ``_saved``.

.. code::

    x = torch.randn(5, requires_grad=True)
    y = x.pow(2)
    print(x.equal(y.grad_fn._saved_self))  # True
    print(x is y.grad_fn._saved_self)  # True


In the previous code, ``y.grad_fn._saved_self`` refers to the same Tensor object as `x`.
But that may not always be the case. For instance:

.. code::

    x = torch.randn(5, requires_grad=True)
    y = x.exp()
    print(y.equal(y.grad_fn._saved_result))  # True
    print(y is y.grad_fn._saved_result)  # False


Under the hood, to prevent reference cycles, PyTorch has *packed* the tensor
upon saving and *unpacked* it into a different tensor for reading. Here, the
tensor you get from accessing ``y.grad_fn._saved_result`` is a different tensor
object than ``y`` (but they still share the same storage).

Whether a tensor will be packed into a different tensor object depends on
whether it is an output of its own `grad_fn`, which is an implementation detail
subject to change and that users should not rely on.

You can control how PyTorch does packing / unpacking with :ref:`saved-tensors-hooks-doc`.


.. _non-differentiable-func-grad:

Gradients for non-differentiable functions
------------------------------------------

The gradient computation using Automatic Differentiation is only valid when each elementary function being used is differentiable.
Unfortunately many of the functions we use in practice do not have this property (``relu`` or ``sqrt`` at ``0``, for example).
To try and reduce the impact of functions that are non-differentiable, we define the gradients of the elementary operations by applying the following rules in order:

#. If the function is differentiable and thus a gradient exists at the current point, use it.
#. If the function is convex (at least locally), use the sub-gradient of minimum norm (it is the steepest descent direction).
#. If the function is concave (at least locally), use the super-gradient of minimum norm (consider `-f(x)` and apply the previous point).
#. If the function is defined, define the gradient at the current point by continuity (note that ``inf`` is possible here, for example for ``sqrt(0)``). If multiple values are possible, pick one arbitrarily.
#. If the function is not defined (``sqrt(-1)``, ``log(-1)`` or most functions when the input is ``NaN``, for example) then the value used as the gradient is arbitrary (we might also raise an error but that is not guaranteed). Most functions will use ``NaN`` as the gradient, but for performance reasons, some functions will use other values (``log(-1)``, for example).
#. If the function is not a deterministic mapping (i.e. it is not a `mathematical function`_), it will be marked as non-differentiable. This will make it error out in the backward if used on tensors that require grad outside of a ``no_grad`` environment.

.. _mathematical function: https://en.wikipedia.org/wiki/Function_(mathematics)

.. _locally-disable-grad-doc:

Locally disabling gradient computation
--------------------------------------

There are several mechanisms available from Python to locally disable gradient
computation:

To disable gradients across entire blocks of code, there are context managers
like no-grad mode and inference mode.
For more fine-grained exclusion of subgraphs from gradient computation,
there is setting the ``requires_grad`` field of a tensor.

Below, in addition to discussing the mechanisms above, we also describe
evaluation mode (:meth:`nn.Module.eval()`), a method that is not used
to disable gradient computation but, because of its name, is often mixed up with the three.

Setting ``requires_grad``
^^^^^^^^^^^^^^^^^^^^^^^^^

:attr:`requires_grad` is a flag, defaulting to false *unless wrapped
in a* ``nn.Parameter``, that allows for fine-grained exclusion of
subgraphs from gradient computation. It takes effect in both the
forward and backward passes:

During the forward pass, an operation is only recorded in the backward graph if
at least one of its input tensors require grad.
During the backward pass (``.backward()``), only leaf tensors with
``requires_grad=True`` will have gradients accumulated into their ``.grad``
fields.

It is important to note that even though every tensor has this flag,
*setting* it only makes sense for leaf tensors (tensors that do not have a
``grad_fn``, e.g., a ``nn.Module``'s parameters).
Non-leaf tensors (tensors that do have ``grad_fn``) are tensors that have a
backward graph associated with them. Thus their gradients will be needed
as an intermediary result to compute the gradient for a leaf tensor that
requires grad. From this definition, it is clear that all non-leaf tensors
will automatically have ``require_grad=True``.

Setting ``requires_grad`` should be the main way you control which parts
of the model are part of the gradient computation, for example, if you need to
freeze parts of your pretrained model during model fine-tuning.

To freeze parts of your model, simply apply ``.requires_grad_(False)`` to
the parameters that you don't want updated. And as described above,
since computations that use these parameters as inputs would not be recorded in
the forward pass, they won't have their ``.grad`` fields updated in the backward
pass because they won't be part of the backward graph in the first place, as
desired.

Because this is such a common pattern, ``requires_grad`` can also be set at
the module level with :meth:`nn.Module.requires_grad_()`.
When applied to a module, ``.requires_grad_()`` takes effect on all
of the module's parameters (which have ``requires_grad=True`` by default).

Grad Modes
^^^^^^^^^^

Apart from setting ``requires_grad`` there are also three grad modes that can
be selected from Python that can affect how computations in PyTorch are
processed by autograd internally: default mode (grad mode), no-grad mode,
and inference mode, all of which can be togglable via context managers and
decorators.

.. list-table::
   :widths: 50 50 50 50 50
   :header-rows: 1

   * - Mode
     - Excludes operations from being recorded in backward graph
     - Skips additional autograd tracking overhead
     - Tensors created while the mode is enabled can be used in grad-mode later
     - Examples
   * - default
     -
     -
     - ✓
     - Forward pass
   * - no-grad
     - ✓
     -
     - ✓
     - Optimizer updates
   * - inference
     - ✓
     - ✓
     -
     - Data processing, model evaluation

Default Mode (Grad Mode)
^^^^^^^^^^^^^^^^^^^^^^^^

The "default mode" is the mode we are implicitly in when no other modes like
no-grad and inference mode are enabled. To be contrasted with
"no-grad mode" the default mode is also sometimes called "grad mode".

The most important thing to know about the default mode is that it is the only
mode in which ``requires_grad`` takes effect. ``requires_grad`` is always overridden
to be ``False`` in both the two other modes.

No-grad Mode
^^^^^^^^^^^^

Computations in no-grad mode behave as if none of the inputs require grad.
In other words, computations in no-grad mode are never recorded in the backward graph
even if there are inputs that have ``require_grad=True``.

Enable no-grad mode when you need to perform operations that should not be
recorded by autograd, but you’d still like to use the outputs of these
computations in grad mode later. This context manager makes it convenient to
disable gradients for a block of code or function without
having to temporarily set tensors to have ``requires_grad=False``, and then
back to ``True``.

For example, no-grad mode might be useful when writing an optimizer: when
performing the training update you’d like to update parameters
in-place without the update being recorded by autograd.
You also intend to use the updated parameters for computations in
grad mode in the next forward pass.

The implementations in :ref:`nn-init-doc` also
rely on no-grad mode when initializing the parameters as to avoid
autograd tracking when updating the initialized parameters in-place.

Inference Mode
^^^^^^^^^^^^^^

Inference mode is the extreme version of no-grad mode. Just like in no-grad
mode, computations in inference mode are not recorded in the backward graph, but
enabling inference mode will allow PyTorch to speed up your model even more.
This better runtime comes with a drawback: tensors created in inference mode
will not be able to be used in computations to be recorded by autograd after
exiting inference mode.

Enable inference mode when you are performing computations that don’t need
to be recorded in the backward graph, AND you don’t plan on using the tensors
created in inference mode in any computation that is to be recorded by autograd later.

It is recommended that you try out inference mode in the parts of your code
that do not require autograd tracking (e.g., data processing and model evaluation).
If it works out of the box
for your use case it’s a free performance win. If you run into errors after
enabling inference mode, check that you are not using tensors created in
inference mode in computations that are recorded by autograd after exiting inference
mode. If you cannot avoid such use in your case, you can always switch back
to no-grad mode.

For details on inference mode please see
`Inference Mode <https://pytorch.org/cppdocs/notes/inference_mode.html>`_.

For implementation details of inference mode see
`RFC-0011-InferenceMode <https://github.com/pytorch/rfcs/pull/17>`_.

Evaluation Mode (``nn.Module.eval()``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evaluation mode is not a mechanism to locally disable gradient computation.
It is included here anyway because it is sometimes confused to be such a mechanism.

Functionally, ``module.eval()`` (or equivalently ``module.train(False)``) are completely
orthogonal to no-grad mode and inference mode. How ``model.eval()`` affects
your model depends entirely on the specific modules used in your model and
whether they define any training-mode specific behavior.

You are responsible for calling ``model.eval()`` and ``model.train()`` if your
model relies on modules such as :class:`torch.nn.Dropout` and
:class:`torch.nn.BatchNorm2d` that may behave
differently depending on training mode, for example, to avoid updating your
BatchNorm running statistics on validation data.

It is recommended that you always use ``model.train()`` when
training and ``model.eval()`` when evaluating your model (validation/testing) even
if you aren’t sure your model has training-mode specific behavior, because a
module you are using might be updated to behave differently in training and
eval modes.

In-place operations with autograd
---------------------------------

Supporting in-place operations in autograd is a hard matter, and we discourage
their use in most cases. Autograd's aggressive buffer freeing and reuse makes
it very efficient and there are very few occasions when in-place operations
lower memory usage by any significant amount. Unless you're operating
under heavy memory pressure, you might never need to use them.

There are two main reasons that limit the applicability of in-place operations:

1. In-place operations can potentially overwrite values required to compute
   gradients.

2. Every in-place operation requires the implementation to rewrite the
   computational graph. Out-of-place versions simply allocate new objects and
   keep references to the old graph, while in-place operations, require
   changing the creator of all inputs to the :class:`Function` representing
   this operation. This can be tricky, especially if there are many Tensors
   that reference the same storage (e.g. created by indexing or transposing),
   and in-place functions will raise an error if the storage of
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

Multithreaded Autograd
----------------------

The autograd engine is responsible for running all the backward operations
necessary to compute the backward pass. This section will describe all the details
that can help you make the best use of it in a multithreaded environment. (This is
relevant only for PyTorch 1.6+ as the behavior in previous version was different.)

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

If you are calling ``backward()`` from multiple threads concurrently and have
shared inputs (i.e. Hogwild CPU training), then non-determinism should be expected.
This can occur because parameters are automatically shared across threads,
as such, multiple threads may access and try to accumulate the same ``.grad``
attribute during gradient accumulation. This is technically not safe, and
it might result in race condition and the result might be invalid to use.

Users developing multithreaded models featuring shared parameters should have the
threading model in mind and should understand the issues described above.

The functional API :func:`torch.autograd.grad` may be used to calculate the
gradients instead of ``backward()`` to avoid non-determinism.

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
parallel ``backward()`` calls that share part/whole of the GraphTask.

Custom Python ``autograd.Function``\s are automatically thread safe because of GIL.
For built-in C++ Autograd Nodes (e.g. AccumulateGrad, CopySlices) and custom
``autograd::Function``\s, the Autograd Engine uses thread mutex locking to ensure
thread safety on autograd Nodes that might have state write/read.

No thread safety on C++ hooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Autograd relies on the user to write thread safe C++ hooks. If you want the hook
to be correctly applied in multithreading environment, you will need to write
proper thread locking code to ensure the hooks are thread safe.

.. _complex_autograd-doc:

Autograd for Complex Numbers
----------------------------

The short version:

- When you use PyTorch to differentiate any function :math:`f(z)` with complex domain and/or codomain,
  the gradients are computed under the assumption that the function is a part of a larger real-valued
  loss function :math:`g(input)=L`. The gradient computed is :math:`\frac{\partial L}{\partial z^*}`
  (note the conjugation of z), the negative of which is precisely the direction of steepest descent
  used in Gradient Descent algorithm. Thus, all the existing optimizers work out of
  the box with complex parameters.
- This convention matches TensorFlow's convention for complex
  differentiation, but is different from JAX (which computes
  :math:`\frac{\partial L}{\partial z}`).
- If you have a real-to-real function which internally uses complex
  operations, the convention here doesn't matter: you will always get
  the same result that you would have gotten if it had been implemented
  with only real operations.

If you are curious about the mathematical details, or want to know how
to define complex derivatives in PyTorch, read on.

What are complex derivatives?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The mathematical definition of complex-differentiability takes the
limit definition of a derivative and generalizes it to operate on
complex numbers. Consider a function :math:`f: ℂ → ℂ`,

    .. math::
        f(z=x+yj) = u(x, y) + v(x, y)j

where :math:`u` and :math:`v` are two variable real valued functions
and :math:`j` is the imaginary unit.

Using the derivative definition, we can write:

    .. math::
        f'(z) = \lim_{h \to 0, h \in C} \frac{f(z+h) - f(z)}{h}

In order for this limit to exist, not only must :math:`u` and :math:`v` must be
real differentiable, but :math:`f` must also satisfy the Cauchy-Riemann `equations
<https://en.wikipedia.org/wiki/Cauchy%E2%80%93Riemann_equations>`_.  In
other words: the limit computed for real and imaginary steps (:math:`h`)
must be equal. This is a more restrictive condition.

The complex differentiable functions are commonly known as holomorphic
functions. They are well behaved, have all the nice properties that
you've seen from real differentiable functions, but are practically of no
use in the optimization world. For optimization problems, only real valued objective
functions are used in the research community since complex numbers are not part of any
ordered field and so having complex valued loss does not make much sense.

It also turns out that no interesting real-valued objective fulfill the
Cauchy-Riemann equations. So the theory with homomorphic function cannot be
used for optimization and most people therefore use the Wirtinger calculus.

Wirtinger Calculus comes into the picture ...
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

So, we have this great theory of complex differentiability and
holomorphic functions, and we can’t use any of it at all, because many
of the commonly used functions are not holomorphic. What’s a poor
mathematician to do? Well, Wirtinger observed that even if :math:`f(z)`
isn’t holomorphic, one could rewrite it as a two variable function
:math:`f(z, z*)` which is always holomorphic. This is because real and
imaginary of the components of :math:`z` can be expressed in terms of
:math:`z` and :math:`z^*` as:

    .. math::
        \begin{aligned}
            \mathrm{Re}(z) &= \frac {z + z^*}{2} \\
            \mathrm{Im}(z) &= \frac {z - z^*}{2j}
        \end{aligned}

Wirtinger calculus suggests to study :math:`f(z, z^*)` instead, which is
guaranteed to be holomorphic if :math:`f` was real differentiable (another
way to think of it is as a change of coordinate system, from :math:`f(x, y)`
to :math:`f(z, z^*)`.)  This function has partial derivatives
:math:`\frac{\partial }{\partial z}` and :math:`\frac{\partial}{\partial z^{*}}`.
We can use the chain rule to establish a
relationship between these partial derivatives and the partial
derivatives w.r.t., the real and imaginary components of :math:`z`.

    .. math::
        \begin{aligned}
            \frac{\partial }{\partial x} &= \frac{\partial z}{\partial x} * \frac{\partial }{\partial z} + \frac{\partial z^*}{\partial x} * \frac{\partial }{\partial z^*} \\
                                         &= \frac{\partial }{\partial z} + \frac{\partial }{\partial z^*}   \\
            \\
            \frac{\partial }{\partial y} &= \frac{\partial z}{\partial y} * \frac{\partial }{\partial z} + \frac{\partial z^*}{\partial y} * \frac{\partial }{\partial z^*} \\
                                         &= 1j * \left(\frac{\partial }{\partial z} - \frac{\partial }{\partial z^*}\right)
        \end{aligned}

From the above equations, we get:

    .. math::
        \begin{aligned}
            \frac{\partial }{\partial z} &= 1/2 * \left(\frac{\partial }{\partial x} - 1j * \frac{\partial }{\partial y}\right)   \\
            \frac{\partial }{\partial z^*} &= 1/2 * \left(\frac{\partial }{\partial x} + 1j * \frac{\partial }{\partial y}\right)
        \end{aligned}

which is the classic definition of Wirtinger calculus that you would find on `Wikipedia <https://en.wikipedia.org/wiki/Wirtinger_derivatives>`_.

There are a lot of beautiful consequences of this change.

- For one, the Cauchy-Riemann equations translate into simply saying that :math:`\frac{\partial f}{\partial z^*} = 0` (that is to say, the function :math:`f` can be written
  entirely in terms of :math:`z`, without making reference to :math:`z^*`).
- Another important (and somewhat counterintuitive) result, as we’ll see later, is that when we do optimization on a real-valued loss, the step we should
  take while making variable update is given by :math:`\frac{\partial Loss}{\partial z^*}` (not :math:`\frac{\partial Loss}{\partial z}`).

For more reading, check out: https://arxiv.org/pdf/0906.4835.pdf

How is Wirtinger Calculus useful in optimization?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Researchers in audio and other fields, more commonly, use gradient
descent to optimize real valued loss functions with complex variables.
Typically, these people treat the real and imaginary values as separate
channels that can be updated. For a step size :math:`\alpha/2` and loss
:math:`L`, we can write the following equations in :math:`ℝ^2`:

    .. math::
        \begin{aligned}
            x_{n+1} &= x_n - (\alpha/2) * \frac{\partial L}{\partial x}  \\
            y_{n+1} &= y_n - (\alpha/2) * \frac{\partial L}{\partial y}
        \end{aligned}

How do these equations translate into complex space :math:`ℂ`?

    .. math::
        \begin{aligned}
            z_{n+1} &= x_n - (\alpha/2) * \frac{\partial L}{\partial x} + 1j * (y_n - (\alpha/2) * \frac{\partial L}{\partial y}) \\
                    &= z_n - \alpha * 1/2 * \left(\frac{\partial L}{\partial x} + j \frac{\partial L}{\partial y}\right) \\
                    &= z_n - \alpha * \frac{\partial L}{\partial z^*}
        \end{aligned}

Something very interesting has happened: Wirtinger calculus tells us
that we can simplify the complex variable update formula above to only
refer to the conjugate Wirtinger derivative
:math:`\frac{\partial L}{\partial z^*}`, giving us exactly the step we take in optimization.

Because the conjugate Wirtinger derivative gives us exactly the correct step for a real valued loss function, PyTorch gives you this derivative
when you differentiate a function with a real valued loss.

How does PyTorch compute the conjugate Wirtinger derivative?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Typically, our derivative formulas take in `grad_output` as an input,
representing the incoming Vector-Jacobian product that we’ve already
computed, aka, :math:`\frac{\partial L}{\partial s^*}`, where :math:`L`
is the loss of the entire computation (producing a real loss) and
:math:`s` is the output of our function. The goal here is to compute
:math:`\frac{\partial L}{\partial z^*}`, where :math:`z` is the input of
the function.  It turns out that in the case of real loss, we can
get away with *only* calculating :math:`\frac{\partial L}{\partial s^*}`,
even though the chain rule implies that we also need to
have access to :math:`\frac{\partial L}{\partial s}`.  If you want
to skip this derivation, look at the last equation in this section
and then skip to the next section.

Let’s continue working with :math:`f: ℂ → ℂ` defined as
:math:`f(z) = f(x+yj) = u(x, y) + v(x, y)j`. As discussed above,
autograd’s gradient convention is centered around optimization for real
valued loss functions, so let’s assume :math:`f` is a part of larger
real valued loss function :math:`g`. Using chain rule, we can write:

    .. math::
        \frac{\partial L}{\partial z^*} = \frac{\partial L}{\partial u} * \frac{\partial u}{\partial z^*} + \frac{\partial L}{\partial v} * \frac{\partial v}{\partial z^*}
        :label: [1]

Now using Wirtinger derivative definition, we can write:

    .. math::
        \begin{aligned}
            \frac{\partial L}{\partial s} = 1/2 * \left(\frac{\partial L}{\partial u} - \frac{\partial L}{\partial v} j\right) \\
            \frac{\partial L}{\partial s^*} = 1/2 * \left(\frac{\partial L}{\partial u} + \frac{\partial L}{\partial v} j\right)
        \end{aligned}

It should be noted here that since :math:`u` and :math:`v` are real
functions, and :math:`L` is real by our assumption that :math:`f` is a
part of a real valued function, we have:

    .. math::
        \left( \frac{\partial L}{\partial s} \right)^* = \frac{\partial L}{\partial s^*}
        :label: [2]

i.e., :math:`\frac{\partial L}{\partial s}` equals to :math:`grad\_output^*`.

Solving the above equations for :math:`\frac{\partial L}{\partial u}` and :math:`\frac{\partial L}{\partial v}`, we get:

    .. math::
        \begin{aligned}
            \frac{\partial L}{\partial u} = \frac{\partial L}{\partial s} + \frac{\partial L}{\partial s^*} \\
            \frac{\partial L}{\partial v} = -1j * \left(\frac{\partial L}{\partial s} - \frac{\partial L}{\partial s^*}\right)
        \end{aligned}
        :label: [3]

Substituting :eq:`[3]` in :eq:`[1]`, we get:

    .. math::
        \begin{aligned}
            \frac{\partial L}{\partial z^*} &= \left(\frac{\partial L}{\partial s} + \frac{\partial L}{\partial s^*}\right) * \frac{\partial u}{\partial z^*} - 1j * \left(\frac{\partial L}{\partial s} - \frac{\partial L}{\partial s^*}\right) * \frac{\partial v}{\partial z^*}  \\
                                            &= \frac{\partial L}{\partial s} * \left(\frac{\partial u}{\partial z^*} + \frac{\partial v}{\partial z^*} j\right) + \frac{\partial L}{\partial s^*} * \left(\frac{\partial u}{\partial z^*} - \frac{\partial v}{\partial z^*} j\right)  \\
                                            &= \frac{\partial L}{\partial s^*} * \frac{\partial (u + vj)}{\partial z^*} + \frac{\partial L}{\partial s} * \frac{\partial (u + vj)^*}{\partial z^*}  \\
                                            &= \frac{\partial L}{\partial s} * \frac{\partial s}{\partial z^*} + \frac{\partial L}{\partial s^*} * \frac{\partial s^*}{\partial z^*}    \\
        \end{aligned}

Using :eq:`[2]`, we get:

    .. math::
        \begin{aligned}
            \frac{\partial L}{\partial z^*} &= \left(\frac{\partial L}{\partial s^*}\right)^* * \frac{\partial s}{\partial z^*} + \frac{\partial L}{\partial s^*} * \left(\frac{\partial s}{\partial z}\right)^*  \\
                                            &= \boxed{ (grad\_output)^* * \frac{\partial s}{\partial z^*} + grad\_output * \left(\frac{\partial s}{\partial z}\right)^* }       \\
        \end{aligned}
        :label: [4]

This last equation is the important one for writing your own gradients,
as it decomposes our derivative formula into a simpler one that is easy
to compute by hand.

How can I write my own derivative formula for a complex function?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The above boxed equation gives us the general formula for all
derivatives on complex functions.  However, we still need to
compute :math:`\frac{\partial s}{\partial z}` and :math:`\frac{\partial s}{\partial z^*}`.
There are two ways you could do this:

    - The first way is to just use the definition of Wirtinger derivatives directly and calculate :math:`\frac{\partial s}{\partial z}` and :math:`\frac{\partial s}{\partial z^*}` by
      using :math:`\frac{\partial s}{\partial x}` and :math:`\frac{\partial s}{\partial y}`
      (which you can compute in the normal way).
    - The second way is to use the change of variables trick and rewrite :math:`f(z)` as a two variable function :math:`f(z, z^*)`, and compute
      the conjugate Wirtinger derivatives by treating :math:`z` and :math:`z^*` as independent variables. This is often easier; for example, if the function in question is holomorphic, only :math:`z` will be used (and :math:`\frac{\partial s}{\partial z^*}` will be zero).

Let's consider the function :math:`f(z = x + yj) = c * z = c * (x+yj)` as an example, where :math:`c \in ℝ`.

Using the first way to compute the Wirtinger derivatives, we have.

.. math::
    \begin{aligned}
        \frac{\partial s}{\partial z} &= 1/2 * \left(\frac{\partial s}{\partial x} - \frac{\partial s}{\partial y} j\right) \\
                                      &= 1/2 * (c - (c * 1j) * 1j)  \\
                                      &= c                          \\
        \\
        \\
        \frac{\partial s}{\partial z^*} &= 1/2 * \left(\frac{\partial s}{\partial x} + \frac{\partial s}{\partial y} j\right) \\
                                        &= 1/2 * (c + (c * 1j) * 1j)  \\
                                        &= 0                          \\
    \end{aligned}

Using :eq:`[4]`, and `grad\_output = 1.0` (which is the default grad output value used when :func:`backward` is called on a scalar output in PyTorch), we get:

    .. math::
        \frac{\partial L}{\partial z^*} = 1 * 0 + 1 * c = c

Using the second way to compute Wirtinger derivatives, we directly get:

    .. math::
        \begin{aligned}
           \frac{\partial s}{\partial z} &= \frac{\partial (c*z)}{\partial z}       \\
                                         &= c                                       \\
            \frac{\partial s}{\partial z^*} &= \frac{\partial (c*z)}{\partial z^*}       \\
                                         &= 0
        \end{aligned}

And using :eq:`[4]` again, we get :math:`\frac{\partial L}{\partial z^*} = c`. As you can see, the second way involves lesser calculations, and comes
in more handy for faster calculations.

What about cross-domain functions?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some functions map from complex inputs to real outputs, or vice versa.
These functions form a special case of :eq:`[4]`, which we can derive using the
chain rule:

    - For :math:`f: ℂ → ℝ`, we get:

        .. math::
            \frac{\partial L}{\partial z^*} = 2 * grad\_output * \frac{\partial s}{\partial z^{*}}

    - For :math:`f: ℝ → ℂ`, we get:

        .. math::
            \frac{\partial L}{\partial z^*} = 2 * \mathrm{Re}(grad\_output^* * \frac{\partial s}{\partial z^{*}})

.. _saved-tensors-hooks-doc:

Hooks for saved tensors
-----------------------

You can control :ref:`how saved tensors are packed / unpacked
<saved-tensors-doc>` by defining a pair of ``pack_hook`` / ``unpack_hook``
hooks.  The ``pack_hook`` function should take a tensor as its single argument
but can return any python object (e.g. another tensor, a tuple, or even a
string containing a filename). The ``unpack_hook`` function takes as its single
argument the output of ``pack_hook`` and should return a tensor to be used in
the backward pass. The tensor returned by ``unpack_hook`` only needs to have
the same content as the tensor passed as input to ``pack_hook``. In particular,
any autograd-related metadata can be ignored as they will be overwritten during
unpacking.

An example of such pair is:

.. code::

    class SelfDeletingTempFile():
        def __init__(self):
            self.name = os.path.join(tmp_dir, str(uuid.uuid4()))

        def __del__(self):
            os.remove(self.name)

    def pack_hook(tensor):
        temp_file = SelfDeletingTempFile()
        torch.save(tensor, temp_file.name)
        return temp_file

    def unpack_hook(temp_file):
        return torch.load(temp_file.name)

Notice that the ``unpack_hook`` should not delete the temporary file because it
might be called multiple times: the temporary file should be alive for as long
as the returned `SelfDeletingTempFile` object is alive.  In the above example,
we prevent leaking the temporary file by closing it when it is no longer needed
(on deletion of the `SelfDeletingTempFile` object).

.. note::

    We guarantee that ``pack_hook`` will only be called once but ``unpack_hook`` can
    be called as many times as the backward pass requires it and we expect it to
    return the same data each time.

.. warning::

    Performing inplace operations on the input of any of the functions is forbidden
    as they may lead to unexpected side-effects. PyTorch will throw an error if the
    input to a pack hook is modified inplace but does not catch the case where the
    input to an unpack hook is modified inplace.


Registering hooks for a saved tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can register a pair of hooks on a saved tensor by calling the
:meth:`~torch.autograd.SavedTensor.register_hooks` method on a
:class:`SavedTensor` object. Those objects are exposed as attributes of a
``grad_fn`` and start with the ``_raw_saved_`` prefix.

.. code::

    x = torch.randn(5, requires_grad=True)
    y = x.pow(2)
    y.grad_fn._raw_saved_self.register_hooks(pack_hook, unpack_hook)

The ``pack_hook`` method is called as soon as the pair is registered.
The ``unpack_hook`` method is called each time the saved tensor needs to be
accessed, either by means of ``y.grad_fn._saved_self`` or during the backward
pass.

.. warning::

    If you maintain a reference to a :class:`SavedTensor` after the saved
    tensors have been released (i.e. after backward has been called), calling
    its :meth:`~torch.autograd.SavedTensor.register_hooks` is forbidden.
    PyTorch will throw an error most of the time but it may fail
    to do so in some cases and undefined behavior may arise.

Registering default hooks for saved tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, you can use the context-manager
:class:`~torch.autograd.graph.saved_tensors_hooks` to register a pair of
hooks which will be applied to *all* saved tensors that are created in
that context.

Example:

.. code::

    # Only save on disk tensors that have size >= 1000
    SAVE_ON_DISK_THRESHOLD = 1000

    def pack_hook(x):
        if x.numel() < SAVE_ON_DISK_THRESHOLD:
            return x
        temp_file = SelfDeletingTempFile()
        torch.save(tensor, temp_file.name)
        return temp_file

    def unpack_hook(tensor_or_sctf):
        if isinstance(tensor_or_sctf, torch.Tensor):
            return tensor_or_sctf
        return torch.load(tensor_or_sctf.name)

    class Model(nn.Module):
        def forward(self, x):
            with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
              # ... compute output
              output = x
            return output

    model = Model()
    net = nn.DataParallel(model)



The hooks defined with this context manager are thread-local.
Hence, the following code will not produce the desired effects because the hooks do not go
through `DataParallel`.

.. code::

      # Example what NOT to do

      net = nn.DataParallel(model)
      with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
          output = net(input)


Note that using those hooks disables all the optimization in place to reduce
Tensor object creation. For example:

.. code::

    with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
        x = torch.randn(5, requires_grad=True)
        y = x * x

Without the hooks, ``x``, ``y.grad_fn._saved_self`` and
``y.grad_fn._saved_other`` all refer to the same tensor object.
With the hooks, PyTorch will pack and unpack `x` into two new tensor objects
that share the same storage with the original `x` (no copy performed).

.. _backward-hooks-execution:

Backward Hooks execution
------------------------

This section will discuss when different hooks fire or don't fire.
Then it will discuss the order in which they are fired.
The hooks that will be covered are: hooks registered to Tensor via
:meth:`torch.tensor.register_hook`,
post-hooks registered to Node via :meth:`torch.autograd.graph.Node.register_hook`, and
pre-hooks registered to Node via :meth:`torch.autograd.graph.Node.register_prehook`.

Whether a particular hook will be fired
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Hooks registered to a Tensor via :meth:`torch.tensor.register_hook`
are executed when gradients are being computed for that Tensor. (Note that this does not require
the Tensor's grad_fn to be executed. For example, if the Tensor is passed
as part of the ``inputs`` argument to :func:`torch.autograd.grad`,
the Tensor's grad_fn may not be executed, but the hook register to that Tensor will always be executed.)

Hooks registered to :class:`torch.autograd.graph.Node` using
:meth:`torch.autograd.graph.Node.register_hook` or
:meth:`torch.autograd.graph.Node.register_prehook` are only fired if
the Node it was registered to is executed.

Whether a particular Node is executed may depend on whether the backward pass was called with
:func:`torch.autograd.grad` or :func:`torch.autograd.backward`.
Specifically, you should be aware of these differences when you register a hook on a
Node corresponding to a Tensor that you are passing to :func:`torch.autograd.grad` or
:func:`torch.autograd.backward` as part of the ``inputs`` argument.

If you are using :func:`torch.autograd.backward`, all of the above mentioned hooks will be executed,
whether or not you specified the ``inputs`` argument. This is because `.backward()` executes all
Nodes, even if they correspond to a Tensor specified as an input.
(Note that the execution of this additional Node corresponding to Tensors passed as  ``inputs``
is usually unnecessary, but done anyway. This behavior is subject to change;
you should not depend on it.)

On the other hand, if you are using :func:`torch.autograd.grad`, the backward hooks registered
to Nodes that correspond to the Tensors passed to ``input`` may not be executed, because
those Nodes will not be executed unless there is another input that depends on the gradient
result of this Node.

The order in which the different hooks are fired
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The order in which things happen are:
1. hooks registered to Tensor are executed
2. pre-hook registered to Node are executed (if Node is executed).
3. The ``.grad`` field is updated for Tensors that retain_grad
4. Node is executed (subject to rules above)
5. post-hook registered to Node are executed (if Node is executed)

If multiple hooks of the same type are registered on the same Tensor or Node
they are executed in the order in which they are registered.
Hooks that are executed later can observe the modifications to the gradient made by
earlier hooks.

Special hooks
^^^^^^^^^^^^^

:func:`torch.autograd.graph.register_multi_grad_hook` is implemented using hooks registered
to Tensors. Each individual Tensor hook is fired following the Tensor hook ordering
defined above and the registered multi-grad hook is called when the last Tensor gradient
is computed.

:meth:`torch.nn.modules.module.register_module_full_backward_hook` is implemented using hooks
registered to Node. As the forward is computed, hooks are registered to grad_fn corresponding
to the inputs and outputs of the module. Because a module may take multiple inputs and return
multiple outputs, a dummy custom autograd Function is first applied to the inputs of the module
before forward and the outputs of the module before the output of forward is returned to ensure
that those Tensors share a single grad_fn, which we can then attach our hooks to.

Behavior of Tensor hooks when Tensor is modified in-place
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Usually hooks registered to a Tensor receive the gradient of the outputs with respect to that
Tensor, where the value of the Tensor is taken to be its value at the time backward is computed.

However, if you register hooks to a Tensor, and then modify that Tensor in-place, hooks
registered before in-place modification similarly receive gradients of the outputs with
respect to the Tensor, but the value of the Tensor is taken to be its value before
in-place modification.

If you prefer the behavior in the former case,
you should register them to the Tensor after all in-place modifications to it have been made.
For example:

.. code::

    t = torch.tensor(1., requires_grad=True).sin()
    t.cos_()
    t.register_hook(fn)
    t.backward()

Furthermore, it can be helpful to know that under the hood,
when hooks are registered to a Tensor, they actually become permanently bound to the grad_fn
of that Tensor, so if that Tensor is then modified in-place,
even though the Tensor now has a new grad_fn, hooks registered before it was
modified in-place will continue to be associated with the old grad_fn, e.g. they will
fire when that Tensor's old grad_fn is reached in the graph by the autograd engine.
