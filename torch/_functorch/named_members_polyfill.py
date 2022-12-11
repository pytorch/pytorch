# Polyfilled from pytorch core while we figure out the `remove_duplicate` issues.
def _named_members(mod, get_members_fn, prefix='', recurse=True, remove_duplicate=True):
    r"""Helper method for yielding various names + members of modules."""
    memo = set()
    modules = mod.named_modules(prefix=prefix, remove_duplicate=remove_duplicate) if recurse else [(prefix, mod)]
    for module_prefix, module in modules:
        members = get_members_fn(module)
        for k, v in members:
            if v is None or v in memo:
                continue
            if remove_duplicate:
                memo.add(v)
            name = module_prefix + ('.' if module_prefix else '') + k
            yield name, v


def _named_parameters(mod, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True):
    gen = _named_members(
        mod,
        lambda module: module._parameters.items(),
        prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
    for elem in gen:
        yield elem


def _named_buffers(mod, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True):
    gen = _named_members(
        mod,
        lambda module: module._buffers.items(),
        prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
    for elem in gen:
        yield elem
