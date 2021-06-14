
from torch.ao.sparsity import BaseSparsifier

from functools import wraps
import weakref

class BaseScheduler(object):

    def __init__(self, sparsifier, last_epoch=-1, verbose=False):

        # Attach sparsifier
        if not isinstance(sparsifier, BaseSparsifier):
            raise TypeError('{} is not an instance of torch.ao.sparsity.BaseSparsifier'.format(
                type(sparsifier).__name__))
        self.sparsifier = sparsifier

        # Initialize epoch and base sparsity levels

        self.base_sl = [group['sparsity_level'] for group in sparsifier.module_groups]
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
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.sparsifier.step = with_counter(self.sparsifier.step)
        self.sparsifier._step_count = 0
        self._step_count = 0
        self.verbose = verbose

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
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
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

        for i, data in enumerate(zip(self.sparsifier.module_groups, values)):
            param_group, sl = data
            param_group['sparsity_level'] = sl
            self.print_sl(self.verbose, i, sl, epoch)

        self._last_sl = [group['sparsity_level'] for group in self.sparsifier.module_groups]
        self.sparsifier.enable_mask_update = True
