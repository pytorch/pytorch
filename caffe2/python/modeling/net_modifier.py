




import abc


class NetModifier(metaclass=abc.ABCMeta):
    """
    An abstraction class for supporting modifying a generated net.
    Inherited classes should implement the modify_net method where
    related operators are added to the net.

    Example usage:
        modifier = SomeNetModifier(opts)
        modifier(net)
    """

    @abc.abstractmethod
    def modify_net(self, net, init_net=None, grad_map=None, blob_to_device=None):
        pass

    def __call__(self, net, init_net=None, grad_map=None, blob_to_device=None,
                 modify_output_record=False):
        self.modify_net(
            net,
            init_net=init_net,
            grad_map=grad_map,
            blob_to_device=blob_to_device,
            modify_output_record=modify_output_record)
