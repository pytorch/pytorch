import re
import sys
import ast
import inspect
import torch
from torch._C import DynamicType, TupleType
from textwrap import dedent


PY35 = sys.version_info >= (3, 5)


try:
    import typing
    from typing import Tuple

    def is_tuple(ann):
        # For some reason Python 3.7 violates the Type[A, B].__origin__ == Type rule
        return ann.__module__ == 'typing' and \
            (getattr(ann, '__origin__', None) is typing.Tuple or
             getattr(ann, '__origin__', None) is tuple)
except ImportError:
    # A minimal polyfill for versions of Python that don't have typing.
    # Note that this means that they also don't support the fancy annotation syntax, so
    # those instances will only be used in our tiny `type: ` comment interpreter.

    # The __getitem__ in typing is implemented using metaclasses, but I'm too lazy for that.
    class TupleCls(object):
        def __getitem__(self, types):
            return TupleInstance(types)

    class TupleInstance(object):
        def __init__(self, types):
            setattr(self, '__args__', types)

    Tuple = TupleCls()

    def is_tuple(ann):
        return isinstance(ann, TupleInstance)


class Module(object):
    def __init__(self, name, members):
        self.name = name
        self.members = members

    def __getattr__(self, name):
        try:
            return self.members[name]
        except KeyError:
            raise RuntimeError("Module {} has no member called {}".format(self.name, name))


_eval_env = {
    'torch': Module('torch', {'Tensor': torch.Tensor}),
    'Tensor': torch.Tensor,
    'typing': Module('typing', {'Tuple': Tuple}),
    'Tuple': Tuple,
}


def get_signature(fn, _n_arguments=None, _n_binders=None):
    # Python 3.5 adds support for the nice annotation syntax, so try that first.
    if PY35:
        sig = try_real_annotations(fn)
        if sig is not None:
            return sig

    type_line, source = None, None
    try:
        source = dedent(inspect.getsource(fn))
        type_line = get_type_line(source)
    except TypeError:
        pass
    # This might happen both because we failed to get the source of fn, or
    # because it didn't have any annotations.
    if type_line is None:
        return default_signature(fn, source, _n_arguments, _n_binders)

    return parse_type_line(type_line)


def parse_type_line(type_line):
    """Parses a type annotation specified as a comment.

    Example inputs:
        # type: (Tensor, torch.Tensor) -> Tuple[Tensor]
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tensor
    """
    arg_ann_str, ret_ann_str = split_type_line(type_line)

    try:
        arg_ann = eval(arg_ann_str, _eval_env)
    except SyntaxError:
        raise RuntimeError("Failed to parse the argument list of a type annotation")

    if not isinstance(arg_ann, tuple):
        arg_ann = (arg_ann,)

    try:
        ret_ann = eval(ret_ann_str, _eval_env)
    except SyntaxError:
        raise RuntimeError("Failed to parse the return type of a type annotation")

    return [ann_to_type(ann) for ann in arg_ann], ann_to_type(ret_ann)


def default_signature(fn, source, _n_arguments, _n_binders):
    """Returns the default signature for fn.

    The current formula is to use the source (if available) to determine the
    number of inputs and outputs, and set all their types as tensors.
    If the source is missing, we fall back to the numbers provided by the compiler,
    to make sure we don't cause an error there (although type mismatches can still happen).

    This method also accounts for the self argument if fn is a method.
    """
    if _n_binders is None:
        raise RuntimeError("default_signature needs to know the number of binders")
    if source is None and _n_arguments is None:
        raise RuntimeError("default_signature needs either the source or the number of arguments")

    ret_type = TupleType([DynamicType() for _ in range(_n_binders)])
    if source is not None:
        py_ast = ast.parse(source)
        if len(py_ast.body) != 1 or not isinstance(py_ast.body[0], ast.FunctionDef):
            raise RuntimeError("expected a single top-level function")
        py_def = py_ast.body[0]
        # TODO: ideally we'd ignore the type of varargs entirely, but we currently don't
        # allow passing in anything else than tensors anyway.
        if py_def.args.vararg is not None:
            arg_types = [DynamicType()] * _n_arguments
        else:
            arg_types = [DynamicType() for _ in py_def.args.args]
            if inspect.ismethod(fn):
                arg_types = arg_types[1:]
    else:
        arg_types = [DynamicType()] * _n_arguments

    return arg_types, ret_type


_def_end_regex = re.compile(r'.*\)\s*:.*')


def get_type_line(source):
    """Tries to find the line containing a comment with the type annotation."""
    lines = source.split('\n')

    def strip_comment(line):
        return line[:line.index('#') if '#' in line else None]

    i = 0
    while not _def_end_regex.match(strip_comment(lines[i])):
        i += 1
    i += 1

    type_line = lines[i].strip()
    if not type_line.startswith('# type:'):
        return None
    return type_line


def split_type_line(type_line):
    """Splits the comment with the type annotation into parts for argument and return types.

    For example, for an input of:
        # type: (Tensor, torch.Tensor) -> Tuple[Tensor, Tensor]

    This function will return:
        ("(Tensor, torch.Tensor)", "Tuple[Tensor, Tensor]")

    """
    start_offset = len('# type:')
    try:
        arrow_pos = type_line.index('->')
    except ValueError:
        raise RuntimeError("Syntax error in type annotation (cound't find `->`)")
    return type_line[start_offset:arrow_pos].strip(), type_line[arrow_pos + 2:].strip()


def try_real_annotations(fn):
    """Tries to use the Py3.5+ annotation syntax to get the type."""
    try:
        sig = inspect.signature(fn)
    except ValueError:
        return None

    all_annots = [sig.return_annotation] + [p.annotation for p in sig.parameters.values()]
    if all(ann is sig.empty for ann in all_annots):
        return None

    def as_ann(ann):
        # sig.empty is really annoying so convert it to None
        return ann if ann is not sig.empty else None

    param_types = [ann_to_type(as_ann(p.annotation))
                   for p in sig.parameters.values()]
    return_type = ann_to_type(as_ann(sig.return_annotation))
    return param_types, return_type


def ann_to_type(ann):
    if ann is None:
        return DynamicType()
    elif ann is torch.Tensor:
        return DynamicType()
    elif is_tuple(ann):
        return TupleType([ann_to_type(a) for a in ann.__args__])
    raise ValueError("The only supported annotations kinds are Tensor and Tuple[...]")
