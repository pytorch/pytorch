# mypy: allow-untyped-defs
# Allows one to expose an API in a private submodule publicly as per the definition
# in PyTorch's public api policy.
#
# It is a temporary solution while we figure out if it should be the long-term solution
# or if we should amend PyTorch's public api policy. The concern is that this approach
# may not be very robust because it's not clear what __module__ is used for.
# However, both numpy and jax overwrite the __module__ attribute of their APIs
# without problem, so it seems fine.
def exposed_in(module):
    def wrapper(fn):
        fn.__module__ = module
        return fn

    return wrapper
