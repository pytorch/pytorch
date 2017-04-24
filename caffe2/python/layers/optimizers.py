from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import context


@context.define_context(allow_default=True)
class OptimizerContext(object):
    """
    Scope driven way to provide optimizers to layers.
    Optimizer can be fetched through the 'get_optimizer' method.
    """

    def __init__(self):
        self._optimizers = {}
        self._optimizers_list = []

    def _rebuild_optimizers(self):
        self._optimizers = {}
        for m in self._optimizers_list:
            self._optimizers.update(m)

    def get_optimizer(self, name):
        assert name in self._optimizers, (
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


class Optimizers(object):
    """
    Optimizers context to provide optimizers to layers
    within the context.

    Example usage:
        optimizers = {'optim1': optim1, 'optim2': optim2}
        with Optimizers(optimizers):
            optim = OptimizerContext.current().get_optimizer('optim1')
            layer(optim=optim)
    """
    def __init__(self, optimizers):
        self._optimizers = optimizers

    def __enter__(self):
        OptimizerContext.current().push_optimizers(self._optimizers)
        return self

    def __exit__(self, type, value, traceback):
        OptimizerContext.current().pop_optimizers()
