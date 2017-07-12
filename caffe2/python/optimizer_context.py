## @package optimizer_context
# Module caffe2.python.optimizer_context
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import context

DEFAULT_OPTIM = 'DEFAULT'


@context.define_context(allow_default=True)
class OptimizerContext(object):
    """
    provide context to allow param_info to have different optimizers
    """

    def __init__(self):
        self._optimizers = {}
        self._optimizers_list = []

    def _rebuild_optimizers(self):
        self._optimizers = {}
        for m in self._optimizers_list:
            self._optimizers.update(m)

    def has_optimizer(self, name):
        return name in self._optimizers

    def get_optimizer(self, name):
        assert self.has_optimizer(name), (
            "{} optimizer is not provided!".format(name))
        return self._optimizers.get(name)

    def push_optimizers(self, optimizers):
        # optimizer override is allowed
        self._optimizers_list.append(optimizers)
        self._optimizers.update(optimizers)

    def pop_optimizers(self):
        assert len(self._optimizers_list) > 0
        self._optimizers_list.pop()
        self._rebuild_optimizers()


class UseOptimizer(object):
    '''
    context class to allow setting the current context.
    Example usage with brew:
        - with UseOptimizer(optim):
            brew.func
        - with UseOptimizer({'WEIGHT': weight_optim}):
            brew.func
        - with UseOptimizer({'DEFAULT': optim, 'BIAS': bias_optim,
                                'WEIGHT': weight_optim}):
            brew.func
        - with UseOptimizer(optim1):
            brew.func
            with UseOptimizer(optim2):
                brew.func

    Example useage with layer:
        optimizers = {'optim1': optim1, 'optim2': optim2}
        with Optimizers(optimizers):
            optim = OptimizerContext.current().get_optimizer('optim1')
            layer(optim=optim)
    '''

    def __init__(self, optim_or_dict):
        if isinstance(optim_or_dict, dict):
            self._optimizers = optim_or_dict
        else:
            self._optimizers = {DEFAULT_OPTIM: optim_or_dict}

    def __enter__(self):
        OptimizerContext.current().push_optimizers(self._optimizers)
        return self

    def __exit__(self, type, value, traceback):
        OptimizerContext.current().pop_optimizers()
