PyTorch Design Philosophy
=========================

This document is designed to help contributors and module maintainers
understand the high-level design principles that have developed over
time in PyTorch. These are not meant to be hard-and-fast rules, but to
serve as a guide to help trade off different concerns and to resolve
disagreements that may come up while developing PyTorch. For more
information on contributing, module maintainership, and how to escalate a
disagreement to the Core Maintainers, please see `PyTorch
Governance <https://pytorch.org/docs/main/community/governance.html>`__.

Design Principles
-----------------

Principle 1: Usability over Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This principle may be surprising! As one Hacker News poster wrote:
*PyTorch is amazing! [...] Although I’m confused. How can a ML framework be
not obsessed with speed/performance?* See `Hacker News discussion on
PyTorch <https://news.ycombinator.com/item?id=28066093>`__.

Soumith’s blog post on `Growing the PyTorch
Community <https://soumith.ch/posts/2021/02/growing-opensource/?fbclid=IwAR1bvN_xZ8avGvu14ODJzS8Zp7jX1BOyfuGUf-zoRawpyL-s95Vjxf88W7s>`__
goes into this in some depth, but at a high-level:

-  PyTorch’s primary goal is usability
-  A secondary goal is to have *reasonable* performance

We believe the ability to maintain our flexibility to support
researchers who are building on top of our abstractions remains
critical. We can’t see what the future of what workloads will be, but we
know we want them to be built first on PyTorch and that requires
flexibility.

In more concrete terms, we operate in a *usability-first* manner and try
to avoid jumping to *restriction-first* regimes (for example, static shapes,
graph-mode only) without a clear-eyed view of the tradeoffs. Often there
is a temptation to impose strict user restrictions upfront because it
can simplify implementation, but this comes with risks:

-  The performance may not be worth the user friction, either because
   the performance benefit is not compelling enough or it only applies to
   a relatively narrow set of subproblems.
-  Even if the performance benefit is compelling, the restrictions can
   fragment the ecosystem into different sets of limitations that can
   quickly become incomprehensible to users.

We want users to be able to seamlessly move their PyTorch code to
different hardware and software platforms, to interoperate with
different libraries and frameworks, and to experience the full richness
of the PyTorch user experience, not a least common denominator subset.

Principle 2: Simple Over Easy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here, we borrow from `The Zen of
Python <https://peps.python.org/pep-0020/>`__:

-  *Explicit is better than implicit*
-  *Simple is better than complex*

A more concise way of describing these two goals is `Simple Over
Easy <https://www.infoq.com/presentations/Simple-Made-Easy/>`_. Let’s start with an example because *simple* and *easy* are
often used interchangeably in everyday English. Consider how one may
model `devices <https://pytorch.org/docs/main/tensor_attributes.html#torch.device>`__
in PyTorch:

-  **Simple / Explicit (to understand, debug):** every tensor is associated
   with a device. The user explicitly specifies tensor device movement.
   Operations that require cross-device movement result in an error.
-  **Easy / Implicit (to use):** the user does not have to worry about
   devices; the system figures out the globally optimal device
   placement.

In this specific case, and as a general design philosophy, PyTorch
favors exposing simple and explicit building blocks rather than APIs
that are easy-to-use by practitioners. The simple version is immediately
understandable and debuggable by a new PyTorch user: you get a clear
error if you call an operator requiring cross-device movement at the
point in the program where the operator is actually invoked. The easy
solution may let a new user move faster initially, but debugging such a
system can be complex: How did the system make its determination? What
is the API for plugging into such a system and how are objects
represented in its IR?

Some classic arguments in favor of this sort of design come from `A
Note on Distributed
Computation <https://dl.acm.org/doi/book/10.5555/974938>`__ (TLDR: Do not
model resources with very different performance characteristics
uniformly, the details will leak) and the `End-to-End
Principle <http://web.mit.edu/Saltzer/www/publications/endtoend/endtoend.pdf>`__
(TLDR: building smarts into the lower-layers of the stack can prevent
building performant features at higher layers in the stack, and often
doesn’t work anyway). For example, we could build operator-level or
global device movement rules, but the precise choices aren’t obvious and
building an extensible mechanism has unavoidable complexity and latency
costs.

A caveat here is that this does not mean that higher-level “easy” APIs
are not valuable; certainly there is a value in, for example,
higher-levels in the stack to support efficient tensor computations
across heterogeneous compute in a large cluster. Instead, what we mean
is that focusing on simple lower-level building blocks helps inform the
easy API while still maintaining a good experience when users need to
leave the beaten path. It also allows space for innovation and the
growth of more opinionated tools at a rate we cannot support in the
PyTorch core library, but ultimately benefit from, as evidenced by
our `rich ecosystem <https://pytorch.org/ecosystem/>`__. In other
words, not automating at the start allows us to potentially reach levels
of good automation faster.

Principle 3: Python First with Best In Class Language Interoperability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This principle began as **Python First**:

  PyTorch is not a Python binding into a monolithic C++ framework.
  It is built to be deeply integrated into Python. You can use it
  naturally like you would use `NumPy <https://www.numpy.org/>`__,
  `SciPy <https://www.scipy.org/>`__, `scikit-learn <(https://scikit-learn.org/>`__,
  or other Python libraries. You can write your new neural network
  layers in Python itself, using your favorite libraries and use
  packages such as `Cython <https://cython.org/>`__ and
  `Numba <http://numba.pydata.org/>`__. Our goal is to not reinvent
  the wheel where appropriate.

One thing PyTorch has needed to deal with over the years is Python
overhead: we first rewrote the `autograd` engine in C++, then the majority
of operator definitions, then developed TorchScript and the C++
frontend.

Still, working in Python provides easily the best experience for our
users: it is flexible, familiar, and perhaps most importantly, has a
huge ecosystem of scientific computing libraries and extensions
available for use. This fact motivates a few of our most recent
contributions, which attempt to hit a Pareto optimal point close to the
Python usability end of the curve:

-  `TorchDynamo <https://dev-discuss.pytorch.org/t/torchdynamo-an-experiment-in-dynamic-python-bytecode-transformation/361>`__,
   a Python frame evaluation tool capable of speeding up existing
   eager-mode PyTorch programs with minimal user intervention.
-  `torch_function <https://pytorch.org/docs/main/notes/extending.html#extending-torch>`__
   and `torch_dispatch <https://dev-discuss.pytorch.org/t/what-and-why-is-torch-dispatch/557>`__
   extension points, which have enabled Python-first functionality to be
   built on-top of C++ internals, such as the `torch.fx
   tracer <https://pytorch.org/docs/stable/fx.html>`__
   and `functorch <https://github.com/pytorch/functorch>`__
   respectively.

These design principles are not hard-and-fast rules, but hard won
choices and anchor how we built PyTorch to be the debuggable, hackable
and flexible framework it is today. As we have more contributors and
maintainers, we look forward to applying these core principles with you
across our libraries and ecosystem. We are also open to evolving them as
we learn new things and the AI space evolves, as we know it will.
