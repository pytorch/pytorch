# mypy: allow-untyped-defs
import functools
import inspect
from enum import Enum

import torch


class _SnapshotState(Enum):
    r"""
    These are the snapshotting-related states that IterDataPipes can be in.

    `NotStarted` - allows you to restore a snapshot and create an iterator with reset
    `Restored` - cannot restore again, allows you to create an iterator without resetting the DataPipe
    `Iterating` - can restore, will reset if you create a new iterator
    """

    NotStarted = 0
    Restored = 1
    Iterating = 2


def _simplify_obj_name(obj) -> str:
    """Simplify the display strings of objects for the purpose of rendering within DataPipe error messages."""
    if inspect.isfunction(obj):
        return obj.__name__
    else:
        return repr(obj)


def _strip_datapipe_from_name(name: str) -> str:
    return name.replace("IterDataPipe", "").replace("MapDataPipe", "")


def _generate_input_args_string(obj):
    """Generate a string for the input arguments of an object."""
    signature = inspect.signature(obj.__class__)
    input_param_names = set(signature.parameters.keys())
    result = []
    for name, value in inspect.getmembers(obj):
        if name in input_param_names:
            result.append((name, _simplify_obj_name(value)))
    return ", ".join([f"{name}={value}" for name, value in result])


def _generate_iterdatapipe_msg(datapipe, simplify_dp_name: bool = False):
    output_string = (
        f"{datapipe.__class__.__name__}({_generate_input_args_string(datapipe)})"
    )
    if simplify_dp_name:
        output_string = _strip_datapipe_from_name(output_string)
    return output_string


def _gen_invalid_iterdatapipe_msg(datapipe):
    return (
        "This iterator has been invalidated because another iterator has been created "
        f"from the same IterDataPipe: {_generate_iterdatapipe_msg(datapipe)}\n"
        "This may be caused multiple references to the same IterDataPipe. We recommend "
        "using `.fork()` if that is necessary."
    )


_feedback_msg = (
    "\nFor feedback regarding this single iterator per IterDataPipe constraint, feel free "
    "to comment on this issue: https://github.com/pytorch/data/issues/45."
)


def _check_iterator_valid(datapipe, iterator_id, next_method_exists=False) -> None:
    r"""
    Given an instance of a DataPipe and an iterator ID, check if the IDs match, and if not, raises an exception.

    In the case of ChildDataPipe, the ID gets compared to the one stored in `main_datapipe` as well.
    """
    if next_method_exists:
        # This is the case where `IterDataPipe` has both `__iter__` and `__next__`.
        # The `_valid_iterator_id` should either be never set (`None`), or set by at most one
        # iterator (`0`). Otherwise, it means there are multiple iterators.
        if datapipe._valid_iterator_id is not None and datapipe._valid_iterator_id != 0:
            extra_msg = "\nNote that this exception is raised inside your IterDataPipe's a `__next__` method"
            raise RuntimeError(
                _gen_invalid_iterdatapipe_msg(datapipe) + extra_msg + _feedback_msg
            )
    elif (
        hasattr(datapipe, "_is_child_datapipe") and datapipe._is_child_datapipe is True
    ):
        if hasattr(datapipe, "_check_valid_iterator_id"):
            if not datapipe._check_valid_iterator_id(iterator_id):
                raise RuntimeError(
                    "This iterator has been invalidated, because a new iterator has been created "
                    f"from one of the ChildDataPipes of "
                    f"{_generate_iterdatapipe_msg(datapipe.main_datapipe)}."
                    + _feedback_msg
                )
        else:
            raise RuntimeError(
                "ChildDataPipe must have method `_check_valid_iterator_id`."
            )
    elif datapipe._valid_iterator_id != iterator_id:
        raise RuntimeError(_gen_invalid_iterdatapipe_msg(datapipe) + _feedback_msg)


def _set_datapipe_valid_iterator_id(datapipe):
    """Given a DataPipe, updates its valid iterator ID and reset the DataPipe."""
    if hasattr(datapipe, "_is_child_datapipe") and datapipe._is_child_datapipe is True:
        if hasattr(datapipe, "_set_main_datapipe_valid_iterator_id"):
            datapipe._set_main_datapipe_valid_iterator_id()  # reset() is called within this method when appropriate
        else:
            raise RuntimeError(
                "ChildDataPipe must have method `_set_main_datapipe_valid_iterator_id`."
            )
    else:
        if datapipe._valid_iterator_id is None:
            datapipe._valid_iterator_id = 0
        else:
            datapipe._valid_iterator_id += 1
        datapipe.reset()
    return datapipe._valid_iterator_id


def hook_iterator(namespace):
    r"""
    Define a hook that is applied to all `__iter__` of metaclass `_DataPipeMeta`.

    This is done for the purpose of profiling and checking if an iterator is still valid.
    """

    def profiler_record_fn_context(datapipe):
        if not hasattr(datapipe, "_profile_name"):
            datapipe._profile_name = _generate_iterdatapipe_msg(
                datapipe, simplify_dp_name=True
            )
        return torch.autograd.profiler.record_function(datapipe._profile_name)

    class IteratorDecorator:
        r"""
        Wrap the iterator and modifying its `__next__` method.

        This decorator is applied to DataPipes of which `__iter__` method is NOT a generator function.
        Those `__iter__` method commonly returns `self` but not necessarily.
        """

        def __init__(self, iterator, datapipe, iterator_id, has_next_method):
            self.iterator = iterator
            self.datapipe = datapipe
            self.iterator_id = iterator_id
            self._profiler_enabled = torch.autograd._profiler_enabled()
            # Check if `__iter__` returns `self` and `DataPipe` has `__next__`
            self.self_and_has_next_method = (
                self.iterator is self.datapipe and has_next_method
            )

        def __iter__(self):
            return self

        def _get_next(self):
            """Return next with logic related to iterator validity, profiler, and incrementation of samples yielded."""
            _check_iterator_valid(self.datapipe, self.iterator_id)
            result = next(self.iterator)
            if not self.self_and_has_next_method:
                self.datapipe._number_of_samples_yielded += 1
            return result

        def __next__(self):
            # TODO: Add try-except to in-place reduce traceback from the Exception
            # See: https://github.com/pytorch/data/issues/284
            if self._profiler_enabled:
                with profiler_record_fn_context(self.datapipe):
                    return self._get_next()
            else:  # Decided against using `contextlib.nullcontext` for performance reasons
                return self._get_next()

        def __getattr__(self, name):
            return getattr(self.iterator, name)

    func = namespace["__iter__"]

    # ``__iter__`` of IterDataPipe is a generator function
    if inspect.isgeneratorfunction(func):

        @functools.wraps(func)
        def wrap_generator(*args, **kwargs):
            gen = func(*args, **kwargs)
            datapipe = args[0]
            if datapipe._fast_forward_iterator:
                it = datapipe._fast_forward_iterator
                datapipe._fast_forward_iterator = None
                datapipe._snapshot_state = _SnapshotState.Iterating
                while True:
                    try:
                        yield next(it)
                    except StopIteration:
                        return
            iterator_id = _set_datapipe_valid_iterator_id(
                datapipe
            )  # This ID is tied to each created iterator
            _profiler_enabled = torch.autograd._profiler_enabled()
            try:
                if _profiler_enabled:
                    with profiler_record_fn_context(datapipe):
                        response = gen.send(None)
                else:
                    response = gen.send(None)

                while True:
                    datapipe._number_of_samples_yielded += 1
                    request = yield response
                    # Pass through here every time `__next__` is called
                    if _profiler_enabled:
                        with profiler_record_fn_context(datapipe):
                            _check_iterator_valid(datapipe, iterator_id)
                            response = gen.send(request)
                    else:  # Decided against using `contextlib.nullcontext` for performance reasons
                        _check_iterator_valid(datapipe, iterator_id)
                        response = gen.send(request)
            except StopIteration:
                return
            except Exception as e:
                # TODO: Simplify the traceback message to skip over `response = gen.send(None)`
                #       Part of https://github.com/pytorch/data/issues/284
                datapipe = args[0]
                msg = "thrown by __iter__ of"
                single_iterator_msg = "single iterator per IterDataPipe constraint"
                if hasattr(e.args, "__len__"):
                    full_msg = f"{msg} {datapipe.__class__.__name__}({_generate_input_args_string(datapipe)})"
                    if len(e.args) == 0 or not isinstance(
                        e.args[0], str
                    ):  # If an exception message doesn't exist
                        e.args = (f"\nThis exception is {full_msg}",)
                    elif msg not in e.args[0] and single_iterator_msg not in e.args[0]:
                        e.args = (
                            e.args[0] + f"\nThis exception is {full_msg}",
                        ) + e.args[1:]
                raise

        namespace["__iter__"] = wrap_generator
    else:  # ``__iter__`` of IterDataPipe is NOT a generator function
        # IterDataPipe is an iterator with both ``__iter__`` and ``__next__``
        # And ``__iter__`` may or may not return `self`
        if "__next__" in namespace:  # If `__next__` exists, put a wrapper around it
            next_func = namespace["__next__"]

            @functools.wraps(next_func)
            def wrap_next(*args, **kwargs):
                datapipe = args[0]
                if torch.autograd._profiler_enabled():
                    with profiler_record_fn_context(datapipe):
                        result = next_func(*args, **kwargs)
                else:
                    result = next_func(*args, **kwargs)
                datapipe._number_of_samples_yielded += 1
                return result

            namespace["__next__"] = wrap_next

            # Note that if the `__next__` and `__iter__` do something completely unrelated. It may cause issue but
            # the user will be violating the iterator protocol. Potential issue:
            # 1. Valid iterator ID may not update or checked properly
            # 2. The number of samples yielded will be miscounted

        # Regardless if `__next__` exists or not, `__iter__` needs a wrapper to track the number of valid iterators
        @functools.wraps(func)
        def wrap_iter(*args, **kwargs):
            iter_ret = func(*args, **kwargs)
            datapipe = args[0]
            datapipe._snapshot_state = _SnapshotState.Iterating
            if datapipe._fast_forward_iterator:
                iter_ret = datapipe._fast_forward_iterator
                datapipe._fast_forward_iterator = None
                return iter_ret
            iterator_id = _set_datapipe_valid_iterator_id(
                datapipe
            )  # This ID is tied to each created iterator
            return IteratorDecorator(
                iter_ret, datapipe, iterator_id, "__next__" in namespace
            )

        namespace["__iter__"] = wrap_iter
