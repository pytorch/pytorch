from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import six


class NetModifier(six.with_metaclass(abc.ABCMeta, object)):
    """
    An abstraction class for supporting modifying a generated net.
    Inherited classes should implement the modify_net method where
    related operators are added to the net.

    Example usage:
        modifier = SomeNetModifier(opts)
        modifier(net)
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def modify_net(self, net, init_net=None, grad_map=None, blob_to_device=None):
        pass

    def __call__(self, net, init_net=None, grad_map=None, blob_to_device=None):
        self.modify_net(
            net,
            init_net=init_net,
            grad_map=grad_map,
            blob_to_device=blob_to_device)
