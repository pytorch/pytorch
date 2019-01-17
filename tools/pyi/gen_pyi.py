from __future__ import print_function

import os
import sys
import inspect

needed_modules = set()


def type_to_python(typename):
    typename = {'int64_t': 'int',
                'Scalar': 'float',
                'ScalarType': 'dtype',
                'Device': 'Union[device, str, None]',
                'IntList': 'Tuple[int, ...]',
                'Tensor?': 'Optional[Tensor]',
                'Tensor': 'Tensor',
                'bool': 'bool',
                'double': 'float',
                'Generator *': 'Generator',
                'Generator*': 'Generator',
                'std::vector<Tensor>': 'Tuple[Tensor, ...]',
                'TensorList': 'Tuple[Tensor, ...]',
                'Storage': 'Storage',  # inaccurate (FloatStorage...)
                'SparseTensorRef': 'Tensor',
                'void': 'None',
                'Layout': 'layout',
                'void*': 'int',  # dataptr
                'std::string': 'str',
                'real': 'float',
                'accreal': 'float',
                'Union[int, Tuple[int, ...]]': 'Union[int, Tuple[int, ...]]',
                'IntegerTensor': 'Tensor',
                'BoolTensor': 'Tensor',
                'IndexTensor': 'Tensor',
                }[typename]
    return typename


def arg_to_type_hint(arg):
    name = arg['name']
    if name == 'from':  # keyword...
        name += '_'
    typename = type_to_python(arg['dynamic_type'])
    if arg.get('is_nullable'):
        typename = 'Optional[' + typename + ']'
    if 'default' in arg:
        default = '=' + str(arg['default'])
    else:
        default = ''
    return name + ': ' + typename + default


def generate_type_hints(fname, decls, is_tensor=False):
    type_hints = []
    dnames = ([d['name'] for d in decls])
    has_out = fname + '_out' in dnames
    if has_out:
        decls = [d for d in decls if d['name'] != fname + '_out']
    for decl in decls:
        skip = 'Type' in [a['dynamic_type'] for a in decl['arguments']]
        if not skip:
            has_tensor_options = 'TensorOptions' in [a['dynamic_type'] for a in decl['arguments']]
            python_args = [arg_to_type_hint(a) for a in decl['arguments'] if a['dynamic_type'] != 'TensorOptions']
            if is_tensor:
                if 'self: Tensor' in python_args:
                    python_args.remove('self: Tensor')
                    python_args = ['self'] + python_args
                elif 'self: float' in python_args:
                    python_args.remove('self: float')
                    python_args = ['self'] + python_args
                else:
                    raise Exception("method without self is unexpected")
            if has_out:
                python_args += ['*', 'out: Optional[Tensor]=None']
            if has_tensor_options:
                if '*' not in python_args:
                    python_args.append('*')
                python_args += ["dtype: dtype=None",
                                "layout: layout=torch.strided",
                                "device: device=None",
                                "requires_grad:bool=False"]
            python_args_s = ', '.join(python_args)
            python_returns = [type_to_python(r['dynamic_type']) for r in decl['returns']]
            if len(python_returns) > 1:
                python_returns_s = 'Tuple[' + ', '.join(python_returns) + ']'
            else:
                python_returns_s = python_returns[0]
            type_hint = "def {}({}) -> {}: ...".format(fname, python_args_s, python_returns_s)
            numargs = len(decl['arguments'])
            have_vararg_version = (numargs > 0 and decl['arguments'][0]['dynamic_type'] in {'IntList', 'TensorList'} and
                                   (numargs == 1 or python_args[1] == '*'))
            if len(decls) > 1 or have_vararg_version:
                type_hint = "@overload\n" + type_hint
            type_hints.append(type_hint)
            if have_vararg_version:
                typelist = decl['arguments'][0]['dynamic_type']
                if typelist == 'IntList':
                    vararg_type = 'int'
                else:
                    vararg_type = 'Tensor'
                # replace first argument and eliminate '*' if present
                python_args = ['*' + decl['arguments'][0]['name'] + ': ' + vararg_type] + python_args[2:]
                python_args_s = ', '.join(python_args)
                type_hint = "@overload\ndef {}({}) -> {}: ...".format(fname, python_args_s, python_returns_s)
                type_hints.append(type_hint)
    return type_hints


def parameters_from_signature(sig):
    # adapted from standard library inspect
    result = []
    render_pos_only_separator = False
    render_kw_only_separator = True
    for param in sig.parameters.values():
        kind = param.kind
        formatted = param._name

        if param._annotation is not inspect._empty:
            formatted = '{}:{}'.format(formatted, inspect.formatannotation(param._annotation))

        if param._default is not inspect._empty:
            if type(param._default).__name__ == 'module':  # better way?
                default_repr = param._default.__name__
                needed_modules.add(default_repr)
            else:
                default_repr = repr(param._default)
            formatted = '{}={}'.format(formatted, default_repr)

        if kind == inspect._VAR_POSITIONAL:
            formatted = '*' + formatted
        elif kind == inspect._VAR_KEYWORD:
            formatted = '**' + formatted

        if kind == inspect._POSITIONAL_ONLY:
            render_pos_only_separator = True
        elif render_pos_only_separator:
            result.append('/')
            render_pos_only_separator = False

        if kind == inspect._VAR_POSITIONAL:
            render_kw_only_separator = False
        elif kind == inspect._KEYWORD_ONLY and render_kw_only_separator:
            result.append('*')
            render_kw_only_separator = False

        formatted = formatted.replace('torch.Tensor', 'Tensor')
        result.append(formatted)

    if render_pos_only_separator:
        result.append('/')
    return result


def type_hint_from_python_fn(fname, fn):
    sig = inspect.signature(fn)
    python_parameters = parameters_from_signature(sig)

    return_annotation = None
    if sig.return_annotation is not inspect._empty:
        return_annotation = inspect.formatannotation(sig.return_annotation)
        return_annotation = return_annotation.replace('torch.Tensor', 'Tensor')

    python_parameters = ', '.join(python_parameters)
    if return_annotation:
        return ["def {}({}) -> {}: ...".format(fname, python_parameters, return_annotation)]
    else:
        return ["def {}({}): ...".format(fname, python_parameters, return_annotation)]


def do_gen_py(build_lib_path):
    import torch
    import yaml
    import inspect
    import types
    import re
    import collections

    fns = collections.defaultdict(list)
    yaml_loader = getattr(yaml, 'CLoader', yaml.loader)

    for d in yaml.load(open('torch/share/ATen/Declarations.yaml'),
                       Loader=yaml_loader):

        name = d['name']
        if name.endswith('_out'):
            name = name[:-4]
        if not name.startswith('_'):
            fns[name].append(d)

    type_hints = []
    for fname in dir(torch):
        fn = getattr(torch, fname)
        docstr = inspect.getdoc(fn)
        if docstr:
            if isinstance(fn, types.BuiltinFunctionType):
                if fname in fns:
                    type_hints += generate_type_hints(fname, fns[fname])
                else:
                    pass  # todo
            elif isinstance(fn, types.FunctionType):
                type_hints += type_hint_from_python_fn(fname, fn)
    type_hints_s = '\n\n'.join(type_hints) + '\n'

    tensor_type_hints = []
    for fname in dir(torch.Tensor):
        fn = getattr(torch.Tensor, fname)
        docstr = inspect.getdoc(fn)
        if docstr and not fname.startswith('_'):
            if getattr(fn, '__qualname__', '').startswith('_TensorBase.'):  # better check?
                if fname in fns:
                    tensor_type_hints += generate_type_hints(fname, fns[fname], is_tensor=True)
                else:
                    pass  # these require magic... print (fname)
            elif isinstance(fn, types.FunctionType):  # python defined
                tensor_type_hints += type_hint_from_python_fn(fname, fn)
    tensor_type_hints_s = """class Tensor:\n""" + '\n\n'.join(
        ['    ' + re.sub(r"\bTensor\b", "'Tensor'", s.replace('\n', '\n' + '    '))
         for s in tensor_type_hints]) + '\n\n'

    header = """
from typing import Tuple, Optional, Union
"""
    header += '\n'.join(["import " + m for m in needed_modules])
    with open(os.path.join(build_lib_path, 'torch', '__init__.pyi'), 'w') as f:
        print(header, file=f)
        print(tensor_type_hints_s, file=f)
        print(type_hints_s, file=f)


def gen_pyi(build_lib_path):
    # we import torch, better do that in a subprocess
    if os.fork() == 0:
        do_gen_py(build_lib_path)
    else:
        os.wait()
