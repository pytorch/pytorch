# mypy: allow-untyped-defs

import flydsl.expr as fx


def make_global_view(tensor, offset, shape, stride):
    iterator = fx.get_iter(fx.rocdl.make_buffer_tensor(tensor))
    if offset is not None:
        iterator = fx.add_offset(iterator, offset)
    return fx.make_view(iterator, fx.make_layout(shape, stride))


def make_shared_view(pointer, shape, stride):
    return fx.make_view(pointer, fx.make_layout(shape, stride))


def load_scalar(copy_atom, view, index, dtype):
    fragment = fx.make_rmem_tensor(1, dtype)
    source = fx.slice(
        fx.logical_divide(view, fx.make_layout(1, 1)),
        (None, index),
    )
    fx.copy(copy_atom, source, fragment)
    return fx.Vector(fragment.load())[0]


def store_scalar(copy_atom, view, index, value, dtype):
    fragment = fx.make_rmem_tensor(1, dtype)
    fragment.store(fx.Vector.from_elements([value], dtype).ir_value())
    destination = fx.slice(
        fx.logical_divide(view, fx.make_layout(1, 1)),
        (None, index),
    )
    fx.copy(copy_atom, fragment, destination)
