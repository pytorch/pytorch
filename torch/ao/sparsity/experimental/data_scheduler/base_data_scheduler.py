from torch.ao.sparsity import BaseDataSparsifier
from functools import wraps
import weakref
import abc
import warnings

__all__ = ['BaseDataScheduler']


class BaseDataScheduler(object):
    r"""
    The BaseDataScheduler is the abstract scheduler class specifically for the
    BaseDataSparsifier class. This class controls a specific hyperparameter of
    the sparsifier class and varies it across the training process (or across time).

    Args:
        data_sparsifier (instance of BaseDataSparsifier)
            Implemented class data sparsifier class wherein the update_mask is implemented
        schedule_param (str)
            A specific hyperparameter of the passed sparsifier that needs to be scheduled/varied
        last_epoch (int, default=-1)
            This is specifically is passed when training needs to be resumed from a particular
            point.
        verbose (bool, default=False)
            Verbosity of the BaseDataScheduler

    The *get_hyperparam()* function needs to be implemented by the user.
    """
    def __init__(self, data_sparsifier, schedule_param: str, last_epoch=-1, verbose=False):
        # Attach sparsifier
        if not isinstance(data_sparsifier, BaseDataSparsifier):
            raise TypeError('{} is not an instance of torch.ao.sparsity.BaseDataSparsifier'.format(
                type(data_sparsifier).__name__))
        self.data_sparsifier = data_sparsifier
        self.schedule_param = schedule_param

        # Initialize epoch and base hyper-params
        self.base_param = {
            name: config.get(schedule_param, None)
            for name, config in self.data_sparsifier.data_groups.items()
        }

        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `scheduler.step()` is called after
        # `sparsifier.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `sparsifier.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the sparsifier instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1  # type: ignore[union-attr]
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True  # type: ignore[attr-defined]
            return wrapper

        self.data_sparsifier.step = with_counter(self.data_sparsifier.step)  # type: ignore[assignment]
        self.data_sparsifier._step_count = 0  # type: ignore[attr-defined]
        self._step_count: int = 0
        self.verbose = verbose

        # Housekeeping
        self._get_sl_called_within_step: bool = False

    @abc.abstractmethod
    def get_hyperparam(self):
        r"""Abstract method that needs to be implemented by the child class.
            Returns: Dictionary of mapping from name -> sparsifier_hyperparam value
        """
        raise NotImplementedError

    def step(self, epoch=None):
        # Raise warning if trying to call scheduler step before the sparsifier.
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.data_sparsifier.step, "_with_counter"):
                warnings.warn("Seems like `data_sparsifier.step()` has been overridden after sparsity scheduler "
                              "initialization. Please, make sure to call `data_sparsifier.step()` before "
                              "`scheduler.step()`.", UserWarning)

            # Just check if there were two first scheduler.step() calls before sparsifier.step()
            elif self.data_sparsifier._step_count < 1:  # type: ignore[attr-defined]
                warnings.warn("Detected call of `scheduler.step()` before `data_sparsifier.step()`. "
                              "You have to make sure you run the data_sparsifier.step() BEFORE any "
                              "calls to the scheduer.step().", UserWarning)
        self._step_count += 1

        class _enable_get_sl_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_sl_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_sl_called_within_step = False

        with _enable_get_sl_call(self):
            self.last_epoch += 1
            updated_scheduler_params = self.get_hyperparam()

        for name, param in updated_scheduler_params.items():
            self.data_sparsifier.data_groups[name][self.schedule_param] = param

        self._last_param = {
            name: config.get(self.schedule_param, None)
            for name, config in self.data_sparsifier.data_groups.items()
        }
        self.data_sparsifier.enable_mask_update = True
