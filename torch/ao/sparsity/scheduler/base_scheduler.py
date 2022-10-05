
from torch.ao.sparsity import BaseSparsifier

from functools import wraps
import warnings
import weakref

__all__ = ["BaseScheduler"]

class BaseScheduler(object):

    def __init__(self, sparsifier, last_epoch=-1, verbose=False):

        # Attach sparsifier
        if not isinstance(sparsifier, BaseSparsifier):
            raise TypeError('{} is not an instance of torch.ao.sparsity.BaseSparsifier'.format(
                type(sparsifier).__name__))
        self.sparsifier = sparsifier

        # Initialize epoch and base sparsity levels

        self.base_sl = [group['sparsity_level'] for group in sparsifier.groups]
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

        self.sparsifier.step = with_counter(self.sparsifier.step)  # type: ignore[assignment]
        self.sparsifier._step_count = 0  # type: ignore[attr-defined]
        self._step_count: int = 0
        self.verbose = verbose

        # Housekeeping
        self._get_sl_called_within_step: bool = False

        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the sparsifier.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'sparsifier'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_sl(self):
        """ Return last computed sparsity level by current scheduler.
        """
        return self._last_sl

    def get_sl(self):
        # Compute sparsity level using chainable form of the scheduler
        # Note: This method is not intended to be called directly, and is only
        #       used by the ".step" method. Use .get_last_sl() instead.
        if not self._get_sl_called_within_step:
            warnings.warn(
                "To get the last sparsity level computed by the scheduler, "
                "please use `get_last_sl()`.")
        raise NotImplementedError

    def print_sl(self, is_verbose, group, sl, epoch=None):
        """Display the current sparsity level.
        """
        if is_verbose:
            if epoch is None:
                print('Adjusting sparsity level'
                      ' of group {} to {:.4e}.'.format(group, sl))
            else:
                print('Epoch {:5d}: adjusting sparsity level'
                      ' of group {} to {:.4e}.'.format(epoch, group, sl))

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        format_string += '\n'
        format_string += 'Sparsifier {0}\n'.format(self.sparsifier)
        format_string += '    {0}: {1}\n'.format('base_sl', self.base_sl)
        format_string += ')'
        return format_string

    def step(self, epoch=None):
        # Raise warning if trying to call scheduler step before the sparsifier.
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.sparsifier.step, "_with_counter"):
                warnings.warn("Seems like `sparsifier.step()` has been overridden after sparsity scheduler "
                              "initialization. Please, make sure to call `sparsifier.step()` before "
                              "`scheduler.step()`.", UserWarning)

            # Just check if there were two first scheduler.step() calls before sparsifier.step()
            elif self.sparsifier._step_count < 1:  # type: ignore[attr-defined]
                warnings.warn("Detected call of `scheduler.step()` before `sparsifier.step()`. "
                              "You have to make sure you run the sparsifier.step() BEFORE any "
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
            values = self.get_sl()

        for i, data in enumerate(zip(self.sparsifier.groups, values)):
            param_group, sl = data
            param_group['sparsity_level'] = sl
            self.print_sl(self.verbose, i, sl, epoch)

        self._last_sl = [group['sparsity_level'] for group in self.sparsifier.groups]
        self.sparsifier.enable_mask_update = True

    def _make_sure_a_list(self, var):
        r"""Utility that extends it to the same length as the .groups, ensuring it is a list"""
        n = len(self.sparsifier.groups)
        if not isinstance(var, (list, tuple)):
            return [var] * n
        else:
            if len(var) != n:
                raise ValueError("Expected variable of length {n}, but got {got}".format(
                    n=n, got=len(var)
                ))
            return list(var)  # We want the result to be in a list, not tuple
