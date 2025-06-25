``torch._functorch.config.backward_pass_autocast``
--------------------------------------------------

``torch.compile`` traces out a backward graph at the time of the forward pass.
As so, it assumes something about if the backward pass will be run under an
ambient autocast context manager. Use ``torch._functorch.config.backward_pass_autocast``
to control that assumption; an incorrect assumption may lead to silent
incorrectness.

The options are either:

- `"same_as_forward"`. We assume that the backward of the ``torch.compile``'ed region
  will be run under the same autocast context manager that the region was run
  under. Use this if your code looks like the following:

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

Applying the option
===================

Use ``patch`` to apply the option to a specific ``torch.compile`` call:

```py
with torch.amp.autocast(...):
    with torch._functorch.config.patch(backward_pass_autocast="same_as_forward")
    y = torch.compile(region)(x)
    ...
    # backward pass run under the same autocast context as the compiled region
    z.backward()
```

Ignoring safety checks
======================

If, at runtime of the backward, ``torch.compile`` detects that there is an
ambient autocast context manager set that is *different* from what
``torch.compile`` assumed, and it materially would have changed the
compiled function, then we will either `"do_nothing"`, `"warn"` or `"error"`
depending on the value of ``torch._functorch.config.on_backward_autocast_mismatch``.

If you think that we're wrong or that the check is too conservative, please set
this to ``"do_nothing"`` and file an issue.
