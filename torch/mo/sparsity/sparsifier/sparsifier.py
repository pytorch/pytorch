from collections import defaultdict

import torch

class Sparsifier(object):
    r"""Base class for all sparsifiers.

    Args:
        config: sparsification configuration. The exact configuration
                depends on the implementation of the sparsifier.
        kwargs: a dict containing default values of sparsification
                  options. These are only used if the `config` does not
                  specify them.

    Members:
        state: holds the current state of the sparsification. The implemented
               sparsifiers must store their state in this dict.
        config_groups: list of configurations that are applied to the model

    TODO:
        - hook for profiler
        - What to do if the config is empty?
        - __repr__ should conform with the design doc
        - `config_groups` checks
            - Need to add behavior for different types of configs (list, dict)
        - Add check for the required/optional arguments
    """
    def __init__(self, config, **kwargs):
        torch._C._log_api_usage_once("python.mo.sparsifier")
        self.defaults = kwargs

        if not isinstance(config, (dict, list, tuple)):
            raise TypeError("config argument given to sparsifier must be "
                            "a dict, a list, or a tuple, got " +
                            torch.typename(config))

        self.state = defaultdict(dict)
        self.config_groups = []

        config_groups = list(config)
        if len(config_groups) == 0:
            raise ValueError("Sparsifier for an ampty configuration")
        self._add_config_groups(config_groups)

    def _add_config_groups(self, config_groups):
        config_set = set()
        for config_group in config_groups:
            assert isinstance(config_group, dict), \
                   "config group must be a dict"
            assert 'params' in config_group, \
                   "config group must have a target to sparsify"
            if config_group['params'] in config_set:
                raise ValueError("some sparsification targets appear in more "
                                 "than one configuration group")
            else:
                config_set.add(config_group['params'])
            self._add_config_group(config_group)

    def _add_config_group(self, config_group):
        for name, default in self.defaults.items():
            config_group.setdefault(name, default)

    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'state': self.state,
            'config_groups': self.config_groups
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.config_groups):
            format_string += '\n'
            format_string += 'Sparsity Group {0}\n'.format(i)
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += '    {0}: {1}\n'.format(key, group[key])
        format_string += ')'
        return format_string

    def state_dict(self):
        r"""Returns the state of the sparsifier as a :class:`dict`.

        Keys:
            state: a dict holding the current state. Depends on the
                   implementation
            config_groups: a dict containing the sparsification configs
        """
        state = self.__getstate__()
        del state['defaults']
        return state

    def load_state_dict(self, state):
        self.__setstate__(state)

    def step(self):
        r"""Performs a single sparsification step."""
        raise NotImplementedError

    def prepare(self):
        r"""Prepares the model for sparsification."""
        raise NotImplementedError

    def convert(self):
        r"""Converts the model."""
        raise NotImplementedError
