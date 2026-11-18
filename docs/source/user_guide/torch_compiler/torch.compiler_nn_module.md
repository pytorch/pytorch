(compiler_nn_module)=

# PyTorch 2.0 NNModule Support

**Author**: [Will Constable](https://github.com/wconstab)

`torch.compile` has special handling for torch.nn.Module objects, tracing them differently than it traces
arbitrary python classes, with the intent of producing faster code by making assumptions about the structure.

This doc describes some of the tradeoffs or edge cases that come up due to this specialization.

## NNModule Hooks Support

Previously, `torch.compile` had no support for hooks on nn.Modules, and if hooks were registered
they would simply be ignored in the compiled program. Indeed many users do not
use nn.Module hooks at all, or only use them for debug workflows, but there are valid use cases
for composing nn.Module hooks with `torch.compile`.

Hooks that are orchestrated via nn.Module.__call__ implementation include `_forward_pre_hooks`,
`forward_hooks`, `_backward_pre_hooks`, and `_backward_hooks`, and will be referred to as 'call hooks'.
These hooks are partially supported by `torch.compile` with limitations described below.

Another category of hooks includes `_state_dict_hooks` and its `pre` and `load_` variants, and are still
unsupported by `torch.compile`.

## `nn.Module.__call__` Hooks Usage and limitations

By default, `torch.compile` will trace the contents of `nn.Module.__call__` which means it will encounter
and run forward/pre-forward hooks.  If you install hooks before calling `torch.compile` and then do not remove
or alter the hooks later, your use case should be supported by default.

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

**skip_nnmodule_hook_guards**
By default, `torch._dynamo.config.skip_nnmodule_hook_guards=True` skips checks that empty builtin
hook registries on a module stay empty. It also skips type checks on builtin global hook registries
and on hook IDs used to check flags such as `with_kwargs`. Dynamo still guards the callbacks and
flag membership used during dispatch. If user code observes the type of a registry, Dynamo guards it.

Set the flag to `False` to detect hooks added to an empty local registry after compilation. When a
module has no hooks at compile time, Dynamo can call `forward` directly. A global hook registered
afterwards can go undetected with either setting.

TODO: confirm if backward/pre_backward hooks are working or not and document accordingly

## state_dict Hooks

State dict hooks have not yet been supported in `torch.compile`.


TODO: warn_once if graph-breaking on hooks.  warn_once to point to this doc if hooks are present.
