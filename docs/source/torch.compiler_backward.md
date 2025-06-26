``torch.compile`` has different autograd semantics
==================================================

When you apply ``torch.compile`` to a function in your model's forward pass,
it will automatically generate a backward pass for the compiled function.
During compilation, it will trace out a graph for the backward pass that
is used whenever autograd is invoked. We refer to the component inside
``torch.compile`` that is responsible for this as ``AOTDispatcher``
(sometimes known as ``AOTAutograd``).

As so, ``torch.compile`` bakes in details of the computation into the
traced-out backward graph during compilation of the function
in the forward pass.
However, in eager-mode PyTorch, the backward computation is dynamic:
outside of the forward pass, you can wrap the call to
``tensor.backward()`` or ``torch.autograd.grad(...)``
in a context manager that may change its behavior.

This page documents how ``torch.compile``'s autograd semantics differ from
eager-mode PyTorch and how to work around it.

``Autocast`` behavior
---------------------

``torch.compile`` bakes in an assumption on if the backward pass will be
run under an ambient autocast context manager. By default,
Use ``torch._functorch.config.backward_pass_autocast``
to control that assumption; an incorrect assumption may lead to silent
incorrectness.

The options are either:
- `"same_as_forward"` (default).
  We assume that the backward of the ``torch.compile``'ed region
  will be run under the same autocast context manager that the region was run
  under (if any). Use this if your code looks like the following:
  ```py
  with torch.amp.autocast(...):
      y = torch.compile(region)(x)
      ...
      # backward pass run under the same autocast context as the compiled region
      z.backward()
  ```
- `"off"`. We assume that the backward of the torch.compile'd region will
  not be run under any autocast context managers.
  Use this if your code looks like the following:
  ```py
  with torch.amp.autocast(...):
      y = torch.compile(region)(x)
      ...
  # Backward pass runs under no autocast.
  z.backward()
  ```
- There is a third option. If you set ``torch._functorch.config.backward_pass_autocast``
  to a list of kwargs, we will assume the backward pass runs under an autocast context
  constructed by those kwargs.

  For example, if your code looks like the following:
  ```py
  y = torch.compile(region)(x)
  ...
  # Backward pass runs under special context manager
  with torch.amp.autocast(**kwargs):
      z.backward()
  ```
  then set ``torch._functorch.config.backward_pass_autocast = kwargs``.

Use ``patch`` to apply the option to a specific ``torch.compile`` call:
```py
with torch.amp.autocast(...):
    with torch._functorch.config.patch(backward_pass_autocast="same_as_forward")
    y = torch.compile(region)(x)
    ...
    # backward pass run under the same autocast context as the compiled region
    z.backward()
```
