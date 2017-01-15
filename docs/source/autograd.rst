.. role:: hidden
    :class: hidden-section

Automatic differentiation package - torch.autograd
==================================================

.. automodule:: torch.autograd
.. currentmodule:: torch.autograd

.. autofunction:: backward

:hidden:`Variable`
------------------

.. autoclass:: Variable
    :members:

API compatibility
^^^^^^^^^^^^^^^^^

Variable API is nearly the same as regular Tensor API (with the exception
of a couple in-place methods, that would overwrite inputs required for
gradient computation). In most cases Tensors can be safely replaced with
Variables and the code will remain to work just fine. Because of this,
we're not documenting all the operations on variables, and you should
refere to :class:`torch.Tensor` docs for this purpose.


Excluding subgraphs from backward
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every Variable has two flags: :attr:`requires_grad` and :attr:`volatile`.
They both allow for fine grained exclusion of subgraphs from gradient
computation and can increase efficiency.

If there's a single input to an operation that requires gradient, its output
will also require gradient. Conversely, only if all inputs don't require
gradient, the output also won't require it. Backward computation is never
performed in the subgraphs, where all Variables didn't require gradients.

Volatile is a similar setting, but differs in how the flag propagates.
If there's even a single volatile input to an operation, its output is also
going to be volatile. Volatility spreads accross the graph much easier than
non-requiring gradient - you only need a **single** volatile leaf to have a
volatile output, while you need **all** leaves to not require gradient to
have an output the doesn't require gradient. Using volatile flag you don't
need to change any settings of your model parameters to use it for
inference. It's enough to create a volatile input, and this will ensure that
no intermediate states are saved.

In-place operations on Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Supporting in-place operations in autograd is a hard matter, and we discourage
their use in most cases. Autograd's aggressive buffer freeing and reuse makes
it very efficient and there are very few occasions when in-place operations
actually lower memory usage by any significant amount. Unless you're operating
under heavy memory pressure, you might never need to use them.

There are two reasons that limit the applicability of in-place operations:

1. Overwriting values required to compute gradients. This is the main reason
   why variables don't support ``log_``. Its gradient formula requires the
   original input, and while it is possible to recreate it by doing the
   inverse operation, it is numerically unstable, and requires additional
   work that often defeats the purpose of using these functions.

2. Every in-place operation actually requires the implementation to rewrite the
   computational graph. Out-of-places version simply allocate new objects and
   keep references to the old graph, while in-place operations, require
   changing the creator of all inputs to the :class:`Function` representing
   this operation. This can be tricky, especially if there are many Variables
   that reference the same storage (e.g. created as a result of indexing and
   transposing), and in-place will actually raise an error if the storage of
   modified inputs is referenced by any other Variables.

In-place correctness checks
^^^^^^^^^^^^^^^^^^^^^^^^^^^

All :class:`Variable` s keep track of in-place operations applied to them, and
if the implementation detects that a variable was saved for backward in one of
the functions, but it was modified in-place afterwards, an error will be raised
once backward pass is started. This ensures that if you're using in-place
functions and not seing any errors, you can be sure that the computed gradients
are correct.

:hidden:`Function`
------------------

.. autoclass:: Function
    :members:

