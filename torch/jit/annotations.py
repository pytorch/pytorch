import sys
import ast
import inspect
import re
import torch
from .._jit_internal import List, BroadcastingList1, BroadcastingList2, \
    BroadcastingList3, Tuple, is_tuple, is_list, Dict, is_dict, Optional, \
    is_optional, _qualified_name
from torch._C import TensorType, TupleType, FloatType, IntType, \
    ListType, StringType, DictType, BoolType, OptionalType, ClassType
from textwrap import dedent
from torch._six import builtins
from torch._utils_internal import get_source_lines_and_file


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
    'List': List,
    'Dict': Dict,
    'Optional': Optional,
}
class EvalEnv(object):
    env = {
        'torch': Module('torch', {'Tensor': torch.Tensor}),
        'Tensor': torch.Tensor,
        'typing': Module('typing', {'Tuple': Tuple}),
        'Tuple': Tuple,
        'List': List,
        'Dict': Dict,
        'Optional': Optional,
    }

    def __init__(self, rcb):
        self.rcb = rcb

    def __getitem__(self, name):
        if name in self.env:
            return self.env[name]
        if self.rcb is not None:
            return self.rcb(name)
        return getattr(builtins, name, None)

def get_signature(fn, rcb, loc):
    # Python 3.5 adds support for the nice annotation syntax, so try that first.
    if PY35:
        sig = try_real_annotations(fn)
        if sig is not None:
            return sig

    type_line, source = None, None
    try:
        source = dedent(''.join(get_source_lines_and_file(fn)[0]))
        type_line = get_type_line(source)
    except TypeError:
        pass
    # This might happen both because we failed to get the source of fn, or
    # because it didn't have any annotations.
    if type_line is None:
        return None

    return parse_type_line(type_line, rcb, loc)


# This is essentially a weaker form of get_signature(), where we don't care if
# we have the types, we just care that we can figure out how many parameters
# a function takes.
def get_num_params(fn, loc):
    try:
        source = dedent(''.join(get_source_lines_and_file(fn)[0]))
    except (TypeError, IOError):
        return None
    if source is None:
        return None
    py_ast = ast.parse(source)
    if len(py_ast.body) == 1 and isinstance(py_ast.body[0], ast.ClassDef):
        raise torch.jit.frontend.FrontendError(
            loc, "Cannot instantiate class '{}' in a script function".format(py_ast.body[0].name))
    if len(py_ast.body) != 1 or not isinstance(py_ast.body[0], ast.FunctionDef):
        raise torch.jit.frontend.FrontendError(loc, "Expected a single top-level function")
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

    resolver = (rcb, loc)
    arg_types = [ann_to_type(ann, resolver) for ann in arg_ann]
    return arg_types, ann_to_type(ret_ann, resolver)


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
    for line_num, line in reversed(type_lines):
        if '# type: (...) -> ' in line:
            return_line = (line_num, line)
        elif type_comment in line:
            if return_line is None:
                raise RuntimeError("Return type line '# type: (...) -> ...' not found on multiline "
                                   "type annotation\n(See PEP 484 https://www.python.org/dev/peps/pep-0484/#suggested-syntax-for-python-2-7-and-straddling-code)")  # noqa
            if line_num < return_line[0]:
                parameter_type_lines.insert(0, line)

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


def ann_to_type(ann, resolver=None):
    # resolver should be a Tuple[Callable, SourceRange] where the Callable
    # is a resolutionCallback
    if ann is None:
        return TensorType.get()
    elif ann is torch.Tensor:
        return TensorType.get()
    elif is_tuple(ann):
        return TupleType([ann_to_type(a) for a in ann.__args__])
    elif is_list(ann):
        return ListType(ann_to_type(ann.__args__[0]))
    elif is_dict(ann):
        key = ann_to_type(ann.__args__[0])
        value = ann_to_type(ann.__args__[1])
        return DictType(key, value)
    elif is_optional(ann):
        if issubclass(ann.__args__[1], type(None)):
            return OptionalType(ann_to_type(ann.__args__[0]))
        else:
            return OptionalType(ann_to_type(ann.__args__[1]))
    elif ann is float:
        return FloatType.get()
    elif ann is int:
        return IntType.get()
    elif ann is str:
        return StringType.get()
    elif ann is bool:
        return BoolType.get()
    elif hasattr(ann, "__torch_script_class__"):
        return ClassType(_qualified_name(ann))
    elif resolver is not None:
        # Maybe resolve a NamedTuple to a Tuple Type
        rcb, loc = resolver
        the_type = torch._C._resolve_type(ann.__name__, loc, rcb)
        if the_type is not None:
            return the_type
    raise ValueError("Unknown type annotation: '{}'".format(ann))


__all__ = [
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
    'Module',
    # TODO: Consider not exporting these during wildcard import (reserve
    # that for the types; for idiomatic typing code.)
    'get_signature',
    'get_num_params',
    'parse_type_line',
    'get_type_line',
    'split_type_line',
    'try_real_annotations',
    'ann_to_type',
]
