import re
import sys
import ast
import inspect
import torch
from .._jit_internal import List, Tuple, BroadcastingList1, BroadcastingList2, BroadcastingList3, is_tuple
from torch._C import DynamicType, TupleType, FloatType, IntType
from textwrap import dedent


PY35 = sys.version_info >= (3, 5)


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


def get_signature(fn):
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
        return None

    return parse_type_line(type_line)


# This is essentially a weaker form of get_signature(), where we don't care if
# we have the types, we just care that we can figure out how many parameters
# a function takes.
def get_num_params(fn):
    try:
        source = dedent(inspect.getsource(fn))
    except (TypeError, IOError):
        return None
    if source is None:
        return None
    py_ast = ast.parse(source)
    if len(py_ast.body) != 1 or not isinstance(py_ast.body[0], ast.FunctionDef):
        raise RuntimeError("expected a single top-level function")
    py_def = py_ast.body[0]
    if py_def.args.vararg is not None:
        return None
    elif hasattr(py_def.args, 'kwonlyargs') and len(py_def.args.kwonlyargs) > 0:
        return None
    else:
        num_params = len(py_def.args.args)
        if inspect.ismethod(fn):
            num_params = num_params - 1
        return num_params


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

    arg_types = [ann_to_type(ann) for ann in arg_ann]
    return arg_types, ann_to_type(ret_ann)


def get_type_line(source):
    """Tries to find the line containing a comment with the type annotation."""
    lines = source.split('\n')

    type_line = None
    for line in lines:
        if '# type:' in line:
            type_line = line.strip()
            break

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

    arg_types = [ann_to_type(as_ann(p.annotation))
                 for p in sig.parameters.values()]
    return_type = ann_to_type(as_ann(sig.return_annotation))
    return arg_types, return_type


def ann_to_type(ann):
    if ann is None:
        return DynamicType.get()
    elif ann is torch.Tensor:
        return DynamicType.get()
    elif is_tuple(ann):
        return TupleType([ann_to_type(a) for a in ann.__args__])
    elif ann is float:
        return FloatType.get()
    elif ann is int:
        return IntType.get()
    raise ValueError("The only supported annotations kinds are Tensor and Tuple[...]")
