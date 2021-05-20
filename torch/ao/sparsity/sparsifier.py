
import abc
from collections import defaultdict

from ._variables import SUPPORTED_MODULES

class BaseSparsifier(abc.ABC):
    r"""Base class for all sparsifiers.

    TODO: Docstring

    config keys:
        - 'module'
    """

    def __init__(self, config, defaults):
        self.config = config
        self.defaults = defaults

        self.state = defaultdict(dict)
        self.module_groups = []
        self.enable_mask_update = False

        modules_to_sparsify = []

        for module_config in self.config:
            local_args = copy.deepcopy(defaults)
            local_args.update(module_config)
            local_key = module_config['module']
            self.module_groups[local_key] = local_args

    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'state': self.state,
            'module_groups': self.module_groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.module_groups):
                format_string += '\n'
                format_string += f'Module Group {i}\n'
                for key in sorted(group.keys()):
                        if key != 'module':
                                format_string += f'    {key}: {group[key]}\n'
        format_string += ')'
        return format_string

    def state_dict(self):
        raise NotImplementedError("This is WIP")

    def load_state_dict(self, state_dict):
        raise NotImplementedError("This is WIP")


    @abc.abstractmethod
    def prepare(self, float_model, mapping=None):
        return

    @abc.abstractmethod
    def convert(self):
        return

    @abc.abstractmethod
    def step(self):
        return



