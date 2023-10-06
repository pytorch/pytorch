PyTorch 2.0 nn.Module Support
=============================

**Author**: `Will Constable <https://github.com/wconstab>`_

`torch.compile` has special handling for torch.nn.Module objects, tracing them differently than it traces
arbitrary python classes, with the intent of producing faster code by making assumptions about the structure.

This doc describes some of the tradeoffs or edge cases that come up due to this specialization.

`nn.Module` Hooks Support
-------------------------
`torch.compile` now has partial support for forward and backward hooks on nn.Modules.

Hooks that are orchestrated via nn.Module.__call__ implementation include `_forward_pre_hooks`,
`forward_hooks`, `_backward_pre_hooks`, and `_backward_hooks`, and will be referred to as 'call hooks'.
These hooks are partially supported by `torch.compile` with limitations described below.

Another category of hooks includes `_state_dict_hooks` and its `pre` and `load_` variants, and are still
unsupported by `torch.compile`.

`nn.Module.__call__` Hooks Usage and limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
By default, `torch.compile` will trace the contents of `nn.Module.__call__` which means it will encounter
and run forward/pre-forward hooks. `torch.compile` installs guards that detect added and removed hooks,
and will trigger a recompilation if the forward/pre-forward hooks change.

Backward/Pre-backward hooks are generally also supported, with similar caveats: currently graph-breaks in dynamo
occur when accessing backward_hooks dicts, which is probably avoidable with some work.  Graph-breaks also impact the
timing of firing backward hooks, since graph-segments are run as autograd-functions which produce all their grads at
the same time.  Assuming it were possible for dynamo to not graph-break on the presence of backward-hooks, we would
still expect the backward hooks for a series of modules to all fire together after the whole compiled graph's backward
ran.

**hooks on 'allowed modules'**
`torch.compile` treats common modules such as torch.conv, as well as modules that are difficult to trace, specially
by allowing them to be called opaquely in the dynamo graph instead of traced into by dynamo.  For such modules, hooks
currently trigger a graph-break so that the affected modules run outside of dynamo.  Depending on the model, this could
introduce a significant performance regression, and additional work is required to improve this support.

state_dict Hooks
~~~~~~~~~~~~~~~~
State dict hooks have not yet been supported in `torch.compile`.


TODO: warn_once if graph-breaking on hooks.  warn_once to point to this doc if hooks are present.