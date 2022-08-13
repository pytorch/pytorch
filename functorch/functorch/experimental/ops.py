from torch.dispatch.dispatcher import dispatcher_singleton, to_flat_tuple, has_torch_function

class PyOperator:
    def __init__(self, name):
        self.name = name
        self.table = {}
        self.entrance_rules = {}

        # TODO: torch_dispatch expects PyOperator to be an instance of a torch.ops.aten op.
        self.overloadpacket = self

        # Hack for FX tracing
        self.__name__ = f'functorch.experimental.ops.{name}'

    def impl(self, dispatch_key, fn, reentrant=False):
        assert dispatch_key not in self.table
        self.table[dispatch_key] = fn
        self.entrance_rules[dispatch_key] = reentrant

    def fallthrough(self, dispatch_key):
        assert dispatch_key not in self.table
        self.table[dispatch_key] = fallthrough_fn(self, dispatch_key)
        self.entrance_rules[dispatch_key] = False

    def __call__(self, *args, **kwargs):
        flat_args = to_flat_tuple(args, kwargs)
        if has_torch_function(flat_args):
            return handle_torch_function(self, flat_args, *args, **kwargs)

        return dispatcher_singleton.call(self, args, kwargs)

def fallthrough_fn(operator, dispatch_key):
    def inner(*args, **kwargs):
        return dispatcher_singleton.redispatch(operator, args, kwargs)
    return inner
