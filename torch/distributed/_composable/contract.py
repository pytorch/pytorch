import torch.nn as nn

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional


# use state_slot as key for module.__dict__ to avoid coliding with other
# properties.
# TODO: since all composable distributed features can share the same slot.
class _StateKey:

    # implement operator < to avoid breaking dir()
    def __lt__(self, other: Any) -> bool:
        return True if isinstance(other, str) else id(self) < id(other)


class _State:
    pass


STATE_KEY = _StateKey()


def contract(func):
    r"""
    Decorate a function as a composable distributed API, where the first
    argument of the function must be an :class:`nn.Module` instance. The
    decorator verifies that the wrapped function does not modify parameter,
    buffer or sub-module fully-qualified names (FQN).

    When a function ``func`` is decorated by ``@contract``, a
    ``.state(module: nn.Module)`` method will be installed to the decorated
    function. Then you can retrieve and modify the state on a module by calling
    ``func.state(module)``.

    Example::
        >>> import torch.nn as nn
        >>>
        >>> class MyModel(nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.l1 = nn.Linear(10, 10)
        >>>         self.l2 = nn.Linear(10, 10)
        >>>
        >>>     def forward(self, x):
        >>>         return self.l2(self.l1(x))
        >>>
        >>> @contract
        >>> def my_feature(module: nn.Module) -> nn.Module:
        >>>     my_feature.state(module).some_state = "any value"
        >>>     return module
        >>>
        >>> model = MyModel()
        >>> my_feature(model.l1)
        >>> assert my_feature.state(model.l1).some_state == "any value"
        >>> my_feature(model.l2)
        >>> model(torch.randn(2, 10)).sum().backward()
    """

    def wrapper(module: nn.Module, *args, **kwargs) -> Optional[nn.Module]:
        # install states specific to the wrapped ``func``
        default_all_state: Dict[Callable, _State] = {}
        all_state: Dict[Callable, _State] = module.__dict__.setdefault(  # type: ignore[call-overload]
            STATE_KEY, default_all_state
        )
        assert isinstance(
            all_state, dict
        ), "Distributed composable API states corrupted"
        assert func not in all_state, (
            "Each distinct composable distributed API can only be applied to a "
            f"module once. {func} has already been applied to the following "
            f"module.\n{module}"
        )
        all_state.setdefault(func, _State())

        orig_named_params = OrderedDict(module.named_parameters())
        orig_named_buffers = OrderedDict(
            module.named_buffers(remove_duplicate=False)
        )
        orig_named_modules = OrderedDict(
            module.named_modules(remove_duplicate=False)
        )

        updated = func(module, *args, **kwargs)

        if updated is None:
            updated = module

        new_named_params = OrderedDict(updated.named_parameters())
        new_named_buffers = OrderedDict(
            updated.named_buffers(remove_duplicate=False)
        )
        new_named_modules = OrderedDict(
            updated.named_modules(remove_duplicate=False)
        )

        assert isinstance(updated, nn.Module), (
            "Output of composable distributed APIs must be either None or "
            f"nn.Module, but got {type(updated)}"
        )

        def check_fqn(orig_fqns: List[str], new_fqns: List[str]):
            if orig_fqns == new_fqns:
                return

            orig_fqn_set, new_fqn_set = set(orig_fqns), set(new_fqns)
            orig_only = orig_fqn_set - new_fqn_set
            new_only = new_fqn_set - orig_fqn_set
            if len(orig_only) or len(new_only):
                raise RuntimeError(
                    "Composable distributed API implementations cannot modify "
                    "FQNs.\n"
                    f"Only in original FQNs: {orig_only},\n"
                    f"Only in new FQNs: {new_only}"
                )
            else:
                raise RuntimeError(
                    "Composable distributed API implementations cannot modify "
                    "the order of FQNs.\n"
                    f"Original FQNs: {orig_only}\n"
                    f"New FQNs: {new_only}"
                )

        check_fqn(
            list(orig_named_params.keys()), list(new_named_params.keys())
        )
        check_fqn(
            list(orig_named_buffers.keys()), list(new_named_buffers.keys())
        )
        check_fqn(
            list(orig_named_modules.keys()), list(new_named_modules.keys())
        )

        # TODO: a stricter verification should also reject changing module
        # types and monkey-patching forward() method implementations.

        # TODO: verify that installed distributed paradigms are compatible with
        # each other.

        return updated

    def get_state(module: nn.Module) -> Optional[_State]:
        return module.__dict__.setdefault(  # type: ignore[call-overload]
            STATE_KEY,
            {},  # TODO(@yhcharles): this is a temporary fix, need a better way
        ).get(
            func
        )  # type: ignore[call-overload]

    wrapper.state = get_state  # type: ignore[attr-defined]

    return wrapper
