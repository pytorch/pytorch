# mypy: allow-untyped-defs
import functools
import inspect
import itertools
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.utils._pytree as pytree


log = logging.getLogger(__name__)
trace_shape_events_log = torch._logging.getArtifactLogger(
    __name__, "trace_shape_events"
)


__all__ = [
    "ShapeEnvEvent",
    "record_shapeenv_event",
    "replay_shape_env_events",
    "FakeTensorMeta",
    "shape_env_check_state_equal",
    "NotEqualError",
]

# [Note: Recording ShapeEnv Events]
# =================================
#
# What is a ShapeEnv event?
# -------------------------
# We consider a ShapeEnv event every function call (ShapeEnv method or
# independent function) that modifies the state of the ShapeEnv instance.
# Such calls are recorded alongside their positional and keyword arguments,
# so that it may be replayed over a different ShapeEnv instance.
#
# See [Note: ShapeEnv State Equality] for what is considered the state
# of a ShapeEnv instance.
#
# What is it for?
# ---------------
# ShapeEnv events recording is used for reconstructing the ShapeEnv in an
# arbitrary state in time.
#
# Being able to arbitrarily replay events like so is useful, mainly for
# translation validation bisection. i.e. if a ValidationException has been
# raised, find the earliest point in time where the translation validation
# fails.
#
# Besides that, it also allows us to inspect the given instance and,
# for example, check the guards that would actually be issued at that point.
#
# What kind of arguments can be stored in an event?
# -------------------------------------------------
# There's no specific rule for what cannot be used as an argument.
# That said, pay special attention to the following cases:
#
#   1. Tensor inputs: there are some tests that check whether the inputs
#      were garbage collected after execution. These will fail if there's
#      an event that is holding a reference to those inputs.
#
#   2. ShapeEnv arguments: if there is an argument of ShapeEnv type, that
#      will be automatically replaced by the new given ShapeEnv instance.
#
#   3. SymTypes arguments: they also hold references to ShapeEnv. So,
#      whenever we see them, we create a new instance, replacing the
#      ShapeEnv reference.
#
#   4. FX nodes: specifically, FX nodes from the FX graph for symbolic
#      shapes. That argument must be replaced when replaying the event at
#      ShapeEnvEvent.run, since it has to reference a node from the given
#      instance, and not from the recorded instance.


# Event class for reconstructing ShapeEnv at arbitrary time.
#
# Represents a method call that mutates ShapeEnv in a way that affects the
# issued guards, when ShapeEnv.produce_guards is called.
@dataclass
class ShapeEnvEvent:
    # ShapeEnv method.
    f: Callable

    # Arguments and keyword arguments called with.
    args: Optional[List[Any]] = None
    kwargs: Optional[Dict[str, Any]] = None

    # List of tracked_fakes at the time the method was called.
    tracked_fakes: Optional[List[Any]] = None

    # Name of the captured event.
    # Used for special handling of particular methods.
    name: Optional[str] = None

    # Replay itself, but using shape_env as self.
    def run(self, shape_env=None) -> Any:
        from torch.fx.experimental.symbolic_shapes import (
            is_symbolic,
            ShapeEnv,
            SymTypes,
        )

        # Special handling for the constructor event.
        if self.f is ShapeEnv:
            assert shape_env is None and self.args is None and self.kwargs is not None
            return ShapeEnv(**self.kwargs)

        assert shape_env is not None
        args = list(self.args or [])
        kwargs = dict(self.kwargs or {})

        # Replace any argument of type ShapeEnv by the given one.
        args, kwargs = pytree.tree_map_only(
            ShapeEnv, lambda _: shape_env, (args, kwargs)
        )

        # Replace any argument of type SymTypes by a new instance,
        # replacing its ShapeEnv reference.
        args, kwargs = pytree.tree_map_only(
            lambda x: isinstance(x, SymTypes) and is_symbolic(x),
            lambda a: type(a)(a.node.with_shape_env(shape_env)),
            (args, kwargs),
        )

        # Converts FX nodes using the mapping argument.
        def maybe_convert_node(x: Any) -> Any:
            if not isinstance(x, torch.fx.Node):
                # Don't do anything to x if it's not an FX node.
                return x

            # If, at some point, we created an FX node, it means that translation validation is on.
            # It also means we are building an FX graph for symbolic shapes at shape_env.graph, and
            # we are tracking node names at shape_env.name_to_node.
            assert hasattr(shape_env, "name_to_node")
            name_to_node = shape_env.name_to_node
            assert x.name in name_to_node
            return name_to_node[x.name]

        # Replaces the value of an specific argument by the result of fn.
        def replacearg(index: int, key: str, fn: Callable):
            if index < len(args):
                args[index] = fn(args[index])
            if key in kwargs:
                kwargs[key] = fn(kwargs[key])

        if self.is_create_fx_call_function():
            # ShapeEnv.create_fx_call_function:
            # "args" parameter is a tuple of FX nodes from the FX graph of the old ShapeEnv.
            # They must be replaced, since a "call_function" FX node with this tuple as argument
            # will be added to the FX graph of the new shape_env.
            replacearg(
                index=2,
                key="args",
                fn=lambda args: tuple(maybe_convert_node(a) for a in args),
            )
        if self.is_evaluate_expr() or self.is_defer_runtime_assert():
            # ShapeEnv.evaluate_expr and ShapeEnv.defer_runtime_assert:
            # "fx_node" parameter is an (optional) FX node that represents the evaluate expression.
            # They must be replaced, since it will be part of a "call_function" FX node for
            # torch._assert, which will be added to the FX graph of the new shape_env.
            replacearg(index=3, key="fx_node", fn=maybe_convert_node)

        # Actually call the method with the converted arguments.
        return self.f(*args, **kwargs)

    def __str__(self) -> str:
        name = self.name if self.name is not None else self.f.__name__
        return f"event: {name} ({self.args}, {self.kwargs})"

    def is_create_fx_call_function(self) -> bool:
        return self.name == "_create_fx_call_function"

    def is_evaluate_expr(self) -> bool:
        return self.name == "evaluate_expr"

    def is_defer_runtime_assert(self) -> bool:
        return self.name == "defer_runtime_assert"


NEST = 0


# Extracts a ShapeEnv instance inside args and kwargs.
# Specifically, it looks for:
#   1. ShapeEnv arguments
#   2. SymInt, SymFloat, or SymBool arguments
# If we find more than one object of any of the above types, we
# also check that the ShapeEnv instance is the same for all of them.
def _extract_shape_env_and_assert_equal(args, kwargs):
    from torch.fx.experimental.symbolic_shapes import is_symbolic, ShapeEnv, SymTypes

    def assert_equal(old: Optional[ShapeEnv], new: ShapeEnv) -> ShapeEnv:
        if old is not None:
            assert old is new, "call with different ShapeEnv"
        return new

    shape_env = None
    for val in itertools.chain(args, kwargs.values()):
        if isinstance(val, ShapeEnv):
            shape_env = assert_equal(shape_env, val)
        if isinstance(val, SymTypes) and is_symbolic(val):
            shape_env = assert_equal(shape_env, val.node.shape_env)

    return shape_env


# Decorator for recording the given function as a replayable event.
#
# This decorator should be used at every function that mutates the state of
# ShapeEnv in some way that affects the resulting issued guards (i.e. when
# ShapeEnv.produce_guards is called).
#
# save_tracked_fakes: saves a snapshot of the TrackedFake list.
# This is used when calling ShapeEnv.produce_guards at arbitrary points in time.
#
# When to save the list of TrackedFake?
# =====================================
# We should save the list of TrackedFake whenever the translation validation
# bisection may actually stop and call the produce_guards method at the moment
# right after the recorded function was played. In other words, since the
# bisection bisects through torch._assert calls, we should save in all methods
# that adds a torch._assert call to the symbolic shapes FX graph.
#
# At the moment, there are 2 methods that save the list:
#   - ShapeEnv.evaluate_expr
#   - ShapeEnv.defer_runtime_assert
def record_shapeenv_event(*, save_tracked_fakes: bool = False) -> Callable:
    def decorator(fn: Callable) -> Callable:
        assert callable(fn)
        args = inspect.getfullargspec(fn).args
        assert args and args[0] == "self", (
            "record_shapeenv_event should only wrap methods on ShapeEnv; refactor your "
            "code so that it calls into a method on ShapeEnv"
        )
        name = fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            from torch.fx.experimental.symbolic_shapes import ShapeEnv

            assert isinstance(args[0], ShapeEnv)

            global NEST

            trace_shape_events_log.debug(
                "%scall %s(*%r, **%r)", " " * NEST, name, args[1:], kwargs
            )
            NEST += 1

            def retlog(r):
                trace_shape_events_log.debug("%s-> %s", " " * (NEST - 1), r)
                return r

            try:
                shape_env = args[0]
                if not shape_env.should_record_events or shape_env.is_recording:
                    # If ShapeEnv is already recording an event, call the wrapped
                    # function directly.
                    #
                    # NB: here, we skip the check of whether all ShapeEnv instances
                    # are equal, in favor of a faster dispatch.
                    return retlog(fn(*args, **kwargs))

                # Retrieve an instance of ShapeEnv.
                # Assumption: the collection of args and kwargs may not reference
                # different ShapeEnv instances.
                self = _extract_shape_env_and_assert_equal(args, kwargs)

                # If we are calling this function without any ShapeEnv instance
                # alive in its arguments, we don't record and call the original.
                if self is None:
                    return retlog(fn(*args, **kwargs))

                # Otherwise, start recording and call the function.
                with self._recording():
                    # Take a snapshot of the current tracked_fakes.
                    tracked_fakes = (
                        self._snapshot_tracked_fakes() if save_tracked_fakes else None
                    )
                    # Record the event for 'fn'.
                    event = ShapeEnvEvent(
                        fn, list(args), kwargs, tracked_fakes, name=fn.__name__
                    )
                    # Play the event on this ShapeEnv.
                    # NB: It's important to put the event first, because running
                    # the event can trigger internal events that must be ordered
                    # after this event.  However, if an exception happens, we do
                    # NOT want to have the event in the list, so pop it off from
                    # the record if an error happened
                    self.events.append(event)
                    try:
                        return retlog(event.run(self))
                    except Exception:
                        self.events.pop()
                        raise

            except Exception:
                log.error(  # noqa: G201
                    "failed while running %s(*%s, **%s)",
                    name,
                    args[1:],
                    kwargs,
                    exc_info=log.isEnabledFor(logging.INFO),
                )
                raise

            finally:
                NEST -= 1

        return wrapper

    return decorator


# Replays the ShapeEnvEvents list.
# It assumes the first event is the constructor call.
#
# fn: transforms an old FX node into one corresponding to the newly created ShapeEnv.
def replay_shape_env_events(events):
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    constructor_event = events[0]
    assert constructor_event.f == ShapeEnv

    # Constructs the new ShapeEnv.
    shape_env = constructor_event.run()

    for event in events[1:]:
        try:
            # Actually replays each event.
            # We need to call create_mapping_fn every time, since the node list might
            # change after each event is replayed.
            event.run(shape_env)
        except Exception:
            log.error("failed when running event: %s", event)
            raise

    return shape_env


# FakeTensor metadata.
# This is to be used in place of FakeTensor placeholders when calling
# ShapeEnv.produce_guards.
@dataclass
class FakeTensorMeta:
    tensor_size: Tuple[Union[int, torch.SymInt], ...]
    tensor_stride: Tuple[Union[int, torch.SymInt], ...]
    tensor_storage_offset: Union[int, torch.SymInt]
    is_nested: bool

    def size(self) -> Tuple[Union[int, torch.SymInt], ...]:
        return self.tensor_size

    def stride(self) -> Tuple[Union[int, torch.SymInt], ...]:
        return self.tensor_stride

    def storage_offset(self) -> Union[int, torch.SymInt]:
        return self.tensor_storage_offset

    def dim(self) -> int:
        return len(self.tensor_size)

    @staticmethod
    def from_fake(fake) -> "FakeTensorMeta":
        return FakeTensorMeta(
            fake.size(), fake.stride(), fake.storage_offset(), fake.is_nested
        )


# [Note: ShapeEnv State Equality]
# ===============================
#
# What is considered ShapeEnv state?
# ----------------------------------
# We consider to be the state of a ShapeEnv instance everything that
# is not in the inline tuple inside remove_nonstate_variables function.
# That is: the fields within ShapeEnv that modify the flow of execution
# of the program.
#
# So, for example: the replacements field might influence on how an
# expression is simplified. That, in turn, may result in a guard being
# statically known (i.e. not added).
#
# On the other hand, var_to_stack serves only changes what is printed
# in the screen, i.e. used only for debugging purposes. Therefore, we
# should not consider it when comparing states.
#
# What to do on NotEqualError?
# ----------------------------
# Here are a few possible causes for getting a NotEqualError raised:
#
#   1. New field that does not belong in the ShapeEnv state.
#      For example: log field of type ShapeEnvLoggerAdapter. Different
#      ShapeEnv instances will always have different ShapeEnvLoggerAdapter
#      instances, i.e. equality comparison would fail.
#      Solution: add it to the inlined tuple inside remove_nonstate_variables
#      function inside check_equal method.
#
#   2. New field that is not directly comparable across instances.
#      For example: guards field of type List[ShapeGuard]. More specifically,
#      the ShapeGuard type holds an expression and a stack information
#      for debugging purposes. When replaying the even on a new ShapeEnv
#      instance, the stack would be different, which would trigger this error.
#      Solution: add a special case to the map_value function inside
#      check_equal function.
#
#   3. Mutation of ShapeEnv on some not recorded function.
#      If a mutation of the state of ShapeEnv happens inside a function
#      that is not recorded (or that no caller in the stack is recorded),
#      then, the replayed ShapeEnv won't catch that.
#      Solution: decorate the function with record_shape_env_event.


# Checks whether the state of two ShapeEnv are equal w.r.t. the guards
# returned by ShapeEnv.produce_guards.
def shape_env_check_state_equal(env1, env2, non_state_variable_names, map_value):
    # Collect and remove variables that don't necessarily represent the state
    # of a ShapeEnv. Note: we copy the dictionary so that we don't modify the
    # instance itself.
    env1_vars = vars(env1).copy()
    env2_vars = vars(env2).copy()

    for v in non_state_variable_names:
        if v in env1_vars:
            env1_vars.pop(v)
        if v in env2_vars:
            env2_vars.pop(v)

    # Function for transforming the mismatched values into string.
    # Needed, since dict and set entries order might not be the same every time.
    def value_to_str(value: Any) -> str:
        if isinstance(value, dict):
            return (
                "{"
                + ", ".join(f"{k}: {value[k]}" for k in sorted(value.keys(), key=str))
                + "}"
            )
        if isinstance(value, set):
            return "{" + ", ".join(f"{v}" for v in sorted(value)) + "}"
        return str(value)

    # Compares env1_vars with env2_vars.
    # Here, we allow the value of each field to be mapped, so that we appropriately
    # compare the two values.
    def compare_vars(
        map_value: Callable[[str, Any], Any]
    ) -> List[Tuple[str, str, str]]:
        env1_set, env2_set = set(env1_vars), set(env2_vars)

        # First, compare the set of keys in each vars dictionary.
        if env1_set != env2_set:
            raise NotEqualError(
                "field set mismatch:",
                [
                    (
                        "found unique fields:",
                        str(sorted(env1_set - env2_set)),
                        str(sorted(env2_set - env1_set)),
                    ),
                ],
            )

        # Then, sort the keys, and compare the mapped values of each key.
        sorted_keys = list(env1_set)
        sorted_keys.sort()

        mapped_dict = [
            (k, map_value(k, env1_vars[k]), map_value(k, env2_vars[k]))
            for k in sorted_keys
        ]

        # Return a list of tuples representing the fields that did not match
        # alongside their respective mapped values.
        return [
            (f"{k}: values don't match.", value_to_str(val1), value_to_str(val2))
            for k, val1, val2 in mapped_dict
            if val1 != val2
        ]

    # Accumulate the mismatching fields.
    errors = compare_vars(map_value)

    if len(errors) > 0:
        raise NotEqualError("field values don't match:", errors)


class NotEqualError(Exception):
    def __init__(
        self,
        msg: str,
        mismatched: List[Tuple[str, str, str]],
    ) -> None:
        details = "\n".join(
            [
                "\n".join(
                    [
                        f"==> {inner_msg}",
                        f"  >  Left: {str1}",
                        f"  > Right: {str2}",
                    ]
                )
                for inner_msg, str1, str2 in mismatched
            ]
        )

        super().__init__(
            f"""\
ShapeEnv not equal: {msg}

{details}
"""
        )
