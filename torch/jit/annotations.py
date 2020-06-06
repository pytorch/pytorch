import sys
import ast
import inspect
import re
import torch
from .._jit_internal import List, BroadcastingList1, BroadcastingList2, \
    BroadcastingList3, Tuple, is_tuple, is_list, Dict, is_dict, Optional, \
    is_optional, _qualified_name, Any, RRef, is_rref, Future, is_future
from torch._C import TensorType, TupleType, FloatType, IntType, \
    ListType, StringType, DictType, BoolType, OptionalType, ClassType, InterfaceType, AnyType, NoneType, \
    DeviceObjType, RRefType, FutureType

from textwrap import dedent
from torch._six import builtins
from torch._utils_internal import get_source_lines_and_file

from typing import Callable


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


class EvalEnv(object):
    env = {
        'torch': Module('torch', {'Tensor': torch.Tensor}),
        'Tensor': torch.Tensor,
        'typing': Module('typing', {'Tuple': Tuple}),
        'Tuple': Tuple,
        'List': List,
        'Dict': Dict,
        'Optional': Optional,
        'RRef': RRef,
        'Future': Future,
    }

    def __init__(self, rcb):
        self.rcb = rcb

    def __getitem__(self, name):
        if name in self.env:
            return self.env[name]
        if self.rcb is not None:
            return self.rcb(name)
        return getattr(builtins, name, None)

def get_signature(fn, rcb, loc, is_method):
    # Python 3.5 adds support for the nice annotation syntax, so try that first.
    signature = None
    if PY35:
        signature = try_real_annotations(fn, loc)
        if signature is not None and is_method:
            # If this is a method, then the signaure will include a type for
            # `self`, but type comments do not contain a `self`. So strip it
            # away here so everything is consistent (`inspect.ismethod` does
            # not work here since `fn` is unbound at this point)
            param_types, return_type = signature
            param_types = param_types[1:]
            signature = (param_types, return_type)

    if signature is None:
        type_line, source = None, None
        try:
            source = dedent(''.join(get_source_lines_and_file(fn)[0]))
            type_line = get_type_line(source)
        except TypeError:
            pass
        # This might happen both because we failed to get the source of fn, or
        # because it didn't have any annotations.
        if type_line is not None:
            signature = parse_type_line(type_line, rcb, loc)

    return signature


def is_function_or_method(the_callable):
    # A stricter version of `inspect.isroutine` that does not pass for built-in
    # functions
    return inspect.isfunction(the_callable) or inspect.ismethod(the_callable)


def is_vararg(the_callable):
    if not is_function_or_method(the_callable) and hasattr(the_callable, '__call__'):  # noqa: B004
        # If `the_callable` is a class, de-sugar the call so we can still get
        # the signature
        the_callable = the_callable.__call__

    if is_function_or_method(the_callable):
        return inspect.getfullargspec(the_callable).varargs is not None
    else:
        return False


def get_param_names(fn, n_args):
    if not is_function_or_method(fn) and hasattr(fn, '__call__') and is_function_or_method(fn.__call__):  # noqa: B004
        # De-sugar calls to classes
        fn = fn.__call__

    if is_function_or_method(fn):
        return inspect.getfullargspec(fn).args
    else:
        # The `fn` was not a method or function (maybe a class with a __call__
        # method, so use a default param name list)
        return [str(i) for i in range(n_args)]


def check_fn(fn, loc):
    # Make sure the function definition is not a class instantiation
    try:
        source = dedent(''.join(get_source_lines_and_file(fn)[0]))
    except (TypeError, IOError):
        return
    if source is None:
        return

    py_ast = ast.parse(source)
    if len(py_ast.body) == 1 and isinstance(py_ast.body[0], ast.ClassDef):
        raise torch.jit.frontend.FrontendError(
            loc, "Cannot instantiate class '{}' in a script function".format(py_ast.body[0].name))
    if len(py_ast.body) != 1 or not isinstance(py_ast.body[0], ast.FunctionDef):
        raise torch.jit.frontend.FrontendError(loc, "Expected a single top-level function")


def parse_type_line(type_line, rcb, loc):
    """Parses a type annotation specified as a comment.

    Example inputs:
        # type: (Tensor, torch.Tensor) -> Tuple[Tensor]
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tensor
    """
    arg_ann_str, ret_ann_str = split_type_line(type_line)

    try:
        arg_ann = eval(arg_ann_str, {}, EvalEnv(rcb))  # noqa: P204
    except (NameError, SyntaxError) as e:
        raise RuntimeError("Failed to parse the argument list of a type annotation: {}".format(str(e)))

    if not isinstance(arg_ann, tuple):
        arg_ann = (arg_ann,)

    try:
        ret_ann = eval(ret_ann_str, {}, EvalEnv(rcb))  # noqa: P204
    except (NameError, SyntaxError) as e:
        raise RuntimeError("Failed to parse the return type of a type annotation: {}".format(str(e)))

    arg_types = [ann_to_type(ann, loc) for ann in arg_ann]
    return arg_types, ann_to_type(ret_ann, loc)


def get_type_line(source):
    """Tries to find the line containing a comment with the type annotation."""
    type_comment = '# type:'

    lines = source.split('\n')
    lines = [(line_num, line) for line_num, line in enumerate(lines)]
    type_lines = list(filter(lambda line: type_comment in line[1], lines))
    lines_with_type = list(filter(lambda line: 'type' in line[1], lines))

    if len(type_lines) == 0:
        type_pattern = re.compile('#[\t ]*type[\t ]*:')
        wrong_type_lines = list(filter(lambda line: type_pattern.search(line[1]), lines))
        if len(wrong_type_lines) > 0:
            raise RuntimeError("The annotation prefix in line " + str(wrong_type_lines[0][0])
                               + " is probably invalid.\nIt must be '# type:'"
                               + "\nSee PEP 484 (https://www.python.org/dev/peps/pep-0484/#suggested-syntax-for-python-2-7-and-straddling-code)" # noqa
                               + "\nfor examples")
        return None
    elif len(type_lines) == 1:
        # Only 1 type line, quit now
        return type_lines[0][1].strip()

    # Parse split up argument types according to PEP 484
    # https://www.python.org/dev/peps/pep-0484/#suggested-syntax-for-python-2-7-and-straddling-code
    return_line = None
    parameter_type_lines = []
    for line_num, line in type_lines:
        if '# type: (...) -> ' in line:
            return_line = (line_num, line)
            break
        elif type_comment in line:
            parameter_type_lines.append(line)
    if return_line is None:
        raise RuntimeError("Return type line '# type: (...) -> ...' not found on multiline "
                           "type annotation\n(See PEP 484 https://www.python.org/dev/peps/pep-0484/#suggested-syntax-for-python-2-7-and-straddling-code)")  # noqa

    def get_parameter_type(line):
        item_type = line[line.find(type_comment) + len(type_comment):]
        return item_type.strip()

    types = map(get_parameter_type, parameter_type_lines)
    parameter_types = ", ".join(types)

    return return_line[1].replace("...", parameter_types)


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


def try_real_annotations(fn, loc):
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

    arg_types = [ann_to_type(as_ann(p.annotation), loc)
                 for p in sig.parameters.values()]
    return_type = ann_to_type(as_ann(sig.return_annotation), loc)
    return arg_types, return_type


def try_ann_to_type(ann, loc):
    if ann is None:
        return TensorType.get()
    if inspect.isclass(ann) and issubclass(ann, torch.Tensor):
        return TensorType.get()
    if is_tuple(ann):
        return TupleType([try_ann_to_type(a, loc) for a in ann.__args__])
    if is_list(ann):
        elem_type = try_ann_to_type(ann.__args__[0], loc)
        if elem_type:
            return ListType(elem_type)
    if is_dict(ann):
        key = try_ann_to_type(ann.__args__[0], loc)
        value = try_ann_to_type(ann.__args__[1], loc)
        return DictType(key, value)
    if is_optional(ann):
        if issubclass(ann.__args__[1], type(None)):
            return OptionalType(try_ann_to_type(ann.__args__[0], loc))
        else:
            return OptionalType(try_ann_to_type(ann.__args__[1], loc))
    if is_rref(ann):
        return RRefType(try_ann_to_type(ann.__args__[0], loc))
    if is_future(ann):
        return FutureType(try_ann_to_type(ann.__args__[0], loc))
    if ann is float:
        return FloatType.get()
    if ann is int:
        return IntType.get()
    if ann is str:
        return StringType.get()
    if ann is bool:
        return BoolType.get()
    if ann is Any:
        return AnyType.get()
    if ann is type(None):
        return NoneType.get()
    if inspect.isclass(ann) and hasattr(ann, "__torch_script_interface__"):
        return InterfaceType(_qualified_name(ann))
    if ann is torch.device:
        return DeviceObjType.get()
    if inspect.isclass(ann):
        if hasattr(ann, "__torch_script_class__"):
            return ClassType(_qualified_name(ann))
        ignored_builtin_classes = (torch.nn.Module, tuple, list, Callable)
        if torch._jit_internal.can_compile_class(ann) and not issubclass(ann, ignored_builtin_classes):
            torch.jit._recursive_compile_class(ann, loc)
            return ClassType(_qualified_name(ann))

    # Maybe resolve a NamedTuple to a Tuple Type
    def fake_rcb(key):
        return None
    return torch._C._resolve_type_from_object(ann, loc, fake_rcb)


def ann_to_type(ann, loc):
    the_type = try_ann_to_type(ann, loc)
    if the_type is not None:
        return the_type
    raise ValueError("Unknown type annotation: '{}'".format(ann))


__all__ = [
    'Any',
    'List',
    'BroadcastingList1',
    'BroadcastingList2',
    'BroadcastingList3',
    'Tuple',
    'is_tuple',
    'is_list',
    'Dict',
    'is_dict',
    'TensorType',
    'TupleType',
    'FloatType',
    'IntType',
    'ListType',
    'StringType',
    'DictType',
    'AnyType',
    'Module',
    # TODO: Consider not exporting these during wildcard import (reserve
    # that for the types; for idiomatic typing code.)
    'get_signature',
    'check_fn',
    'get_param_names',
    'parse_type_line',
    'get_type_line',
    'split_type_line',
    'try_real_annotations',
    'try_ann_to_type',
    'ann_to_type',
]
