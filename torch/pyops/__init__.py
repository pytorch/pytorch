import torch  # TODO: move to separate ops file

from typing import List, Tuple
import inspect
import ast

from tools.codegen.model import (
    BaseOperatorName, OperatorName, FunctionSchema,
    BaseTy, BaseType, ListType, Argument, Arguments,
    Return, Annotation,
    DeviceCheckType,
    Variant,
    Location,
    NativeFunction,
    DispatchKey, BackendMetadata,)

# Writes a file if its contents differ from what is being written
def _write_if_changed(filename: str, contents: str) -> None:
    old_contents: Optional[str]
    try:
        with open(filename, 'r') as f:
            old_contents = f.read()
    except IOError:
        old_contents = None
    if contents != old_contents:
        with open(filename, 'w') as f:
            f.write(contents)

def TORCH_CHECK(cond, str):
    assert cond, str

class TensorList(object):
    pass

class Tensor(object):
    pass

class MutatedTensorA(object):
    pass

class PyOp(object):
    pass

class dstack_op(PyOp):
    pass

# Function variants definitions
# TODO: link back to op class with decorator or attribute

#   TORCH_CHECK(tensors.size() > 0, "dstack expects a non-empty TensorList");
#   auto rep = at::atleast_3d(tensors);
#   return at::cat(rep, 2);
def dstack(tensors: TensorList) -> Tensor:
    TORCH_CHECK(len(tensors) > 0, "dstack expects a non-empty list of tensors")
    rep = torch.atleast_3d(tensors)
    return torch.cat(rep, 2)

# TORCH_CHECK(tensors.size() > 0, "dstack expects a non-empty TensorList");
#   auto rep = at::atleast_3d(tensors);
#   return at::cat_out(result, rep, 2);
def dstack__out(tensors: TensorList, *, out: MutatedTensorA) -> MutatedTensorA:
    TORCH_CHECK(len(tensors) > 0, "dstack expects a non-empty list of tensors")
    rep = torch.atleast_3d(tensors)
    return torch.cat(rep, 2, out=out)

# TODO: auto-harvest the ops so this sequence isn't needed
pyops_sequence = (
    dstack,
    dstack__out
)

# Types
supported_types = {"TensorList", "Tensor", "MutatedTensorA"}

# Parsed representations
class _Arg(object):
    def __init__(self, name, *, type):
        super().__init__()
        self.name = name
        self.type = type

    def __repr__(self):
        return "[Arg. name:{0}, type:{1}]".format(self.name, self.type)

class _Signature(object):
    def __init__(self, name, *, args, kwargs, type):
        super().__init__()
        self.name = name
        self.args = args
        self.kwargs = kwargs
        self.type = type

        if '__' in name:
            self.base_name, self.overload_name = name.split('__')
        else:
            self.base_name = name
            self.overload_name = ""

    def __repr__(self):
        return "[Signature. name:{0}, args:{1}, type:{2}]".format(self.name, self.args, self.type)

# Parsing functions
# Returns the type of an annotation
def parseType(annotation):
    assert isinstance(annotation, ast.Name), "annotation {0} is not an ast.Name".format(annotation)
    id = annotation.id
    assert id in supported_types, "Unsupported type {0}".format(id)
    return id

# Returns a class describing an arg
# TODO: name, type, default, is_kwarg, is_modified
def parseArg(arg):
    arg_name = arg.arg
    arg_type = parseType(arg.annotation)

    parsed_arg = _Arg(arg_name, type=arg_type)
    return parsed_arg

# Returns a list of the args
def parseArgs(arguments):
    assert len(arguments.args) > 0
    parsed_args = []
    parsed_kwargs = []
    for arg in arguments.args:
        parsed_args.append(parseArg(arg))
    for arg in arguments.kwonlyargs:
        parsed_kwargs.append(parseArg(arg))

    return parsed_args, parsed_kwargs

# Returns a _Signature
def parseSignature(node):
    assert isinstance(node, ast.FunctionDef)

    fn_name = node.name
    args = node.args
    returns_node = node.returns

    parsed_args, parsed_kwargs = parseArgs(args)
    return_type = parseType(returns_node)

    signature = _Signature(
        fn_name,
        args=parsed_args,
        kwargs=parsed_kwargs,
        type=return_type)
    return signature

# Maps from parsed objects to gen.py objects
def map_type(type):
    if type == "TensorList":
        return ListType(elem=BaseType(BaseTy.Tensor), size=None)
    elif type == "Tensor":
        return BaseType(name=BaseTy.Tensor)
    elif type == "MutatedTensorA":
        return BaseType(name=BaseTy.Tensor)

    assert False, "Unknown type to map!"



# Generates native functions
def gen_pyops_functions(rs, bs):
    results = []
    for op in pyops_sequence:
        src = inspect.getsource(op)
        tree = ast.parse(src)

        # asserts the tree is a single FunctionDef node
        assert len(tree.body) == 1
        fn_node = tree.body[0]

        sig = parseSignature(fn_node)

        # TODO: handle inplace variants (and dunders?)
        bon = BaseOperatorName(
            base=sig.base_name,
            inplace=False,
            dunder_method=False)
        on = OperatorName(
            name=bon,
            overload_name=sig.overload_name)

        # TODO: handle args with default values
        # TODO: handle args with annotations
        mapped_args = []
        for arg in sig.args:
            mapped_args.append(Argument(
                name=arg.name,
                type=map_type(arg.type),
                default=None,
                annotation=None
            ))

        # TODO: handle kwargs other than out=
        # TODO: handle out= with multiple tensors
        out = ()
        out_annotation = None
        for kwarg in sig.kwargs:
            assert kwarg.name == "out"
            out_annotation = Annotation(
                alias_set=('a',),
                is_write=True
            )
            out = (Argument(
                name='out',
                type=map_type(kwarg.type),
                default=None,
                annotation=out_annotation
            ),)

        # TODO: handle method variants
        # TODO: handle tensor options
        args = Arguments(
            pre_self_positional=(),
            self_arg=None,
            post_self_positional=tuple(mapped_args),
            pre_tensor_options_kwarg_only=(),
            tensor_options=None,
            post_tensor_options_kwarg_only=(),
            out=out
        )

        # TODO: handle named tuple returns
        # TODO: handle ops with multiple returns
        ret = (Return(
            name=None,
            type=BaseType(BaseTy.Tensor),
            annotation=out_annotation),)

        fs = FunctionSchema(
            name=on,
            arguments=args,
            returns=ret)

        # TODO: consider improving location
        loc = Location(file="generated", line=0)

        # TODO: review structured options
        # TODO: review making device guard an option
        # TODO: support python modules
        nf = NativeFunction(
            func=fs,
            use_const_ref_for_mutable_tensors=False,
            device_guard=True,
            device_check=DeviceCheckType.ExactSame,
            python_module=None,
            category_override=None,
            variants=set((Variant.function,)),
            manual_kernel_registration=False,
            manual_cpp_binding=False,
            loc=loc,
            structured=False,
            structured_delegate=None,
            structured_inherits=None,
            cpp_no_default_args=set(),
            is_abstract=False,
            has_composite_implicit_autograd_kernel=True,
            has_composite_explicit_autograd_kernel=False
        )

        bm = BackendMetadata(
            kernel=sig.name.replace("__", "_"),
            structured=False)
        bs = {DispatchKey.CompositeImplicitAutograd: {on: bm}}

        results.append((nf, bs))

    return results

cpp_template = r"""
#include <ATen/ATen.h>
#include <c10/util/Exception.h>

// NOTICE! This file is autogenerated!

namespace at {{ namespace native {{

{function_cpp}

}}}} // at::native
"""

dstack_cpp = r"""
Tensor dstack(TensorList tensors) {
  TORCH_CHECK(tensors.size() > 0,
           "dstack expects a non-empty TensorList");
  auto rep = at::atleast_3d(tensors);
  return at::cat(rep, 2);
}

Tensor& dstack_out(TensorList tensors, Tensor& result) {
  TORCH_CHECK(tensors.size() > 0,
           "dstack expects a non-empty TensorList");
  auto rep = at::atleast_3d(tensors);
  return at::cat_out(result, rep, 2);
}
"""

# cpp print helpers
def map_type_to_cpp(type):
    if type == "TensorList":
        return "TensorList"
    elif type == "Tensor":
        return "Tensor"
    elif type == "MutatedTensorA":
        return "Tensor&"

    assert False, "Unknown type to map to cpp!"

def print_arg_cpp(arg):
    return map_type_to_cpp(arg.type) + " " + arg.name

# Base class for representing all Python nodes
class _Statement(object):
    def __init__(self, stmt):
        super().__init__()

        self.stmt = stmt

type_to_class_map = {}

def parse_stmt(stmt):
    return type_to_class_map[type(stmt)](stmt)

# An assignment
# https://greentreesnakes.readthedocs.io/en/latest/nodes.html?highlight=assign#Assign
class _Assign(_Statement):
    def __init__(self, stmt):
        super().__init__(stmt)

        assert isinstance(stmt, ast.Assign)
        assert len(stmt.targets) == 1
        self.targets = parse_stmt(stmt.targets[0])
        self.value = parse_stmt(stmt.value)

        self.fields = ('targets', 'value')

    # TODO: don't auto all new variables
    def __str__(self):
        return "auto " + str(self.targets) + " = " + str(self.value)


# Attribute access, like foo.bar
# https://docs.python.org/3/library/ast.html?highlight=attribute#ast.Attribute
class _Attribute(_Statement):
    def __init__(self, stmt):
        super().__init__(stmt)

        assert isinstance(stmt, ast.Attribute)
        self.value = parse_stmt(stmt.value)
        self.attr = stmt.attr  # attr is a string
        self.ctx = stmt.ctx  # ctx is load/store/del

        self.fields = ('value', 'attr', 'ctx')

    def __str__(self):
        # TODO: allow other namespaces
        # translates torch calls to at::
        s = None
        if str(self.value) == "torch":
            s = "at::" + self.attr
        else:
            s = str(self.value) + "." + self.attr

        if type(self.ctx) == ast.Load:
            s += "()"

        return s

# A function call
# https://greentreesnakes.readthedocs.io/en/latest/nodes.html?highlight=call#Call
class _Call(_Statement):
    def __init__(self, stmt):
        super().__init__(stmt)
        assert isinstance(stmt, ast.Call)

        self.func = parse_stmt(stmt.func)
        # args and keywords can be empty lists
        # TODO: map out-of-order Python calls to CPP
        self.args = tuple(map(parse_stmt, stmt.args),)
        self.keywords = tuple(map(parse_stmt, stmt.keywords),)
        # TODO: handle starargs

        self.fields = ('func', 'args', 'keywords')

    def __str__(self):
        # TORCH_CHECK is a special-cased prim
        if str(self.func) == "TORCH_CHECK":
            return "TORCH_CHECK(" + ", ".join(map(str, self.args)) + ")"

        # Note: The func_name may have a () because of how attributes
        #   are printed in the load ctx
        func_name = str(self.func).replace('()', '')
        s = func_name

        # special-cases out by assuming it's the first argument
        # TODO: FIXME
        if len(self.keywords) > 0:
            # TODO: handle more kwargs than out=
            # TODO: don't assume the out arg is always named "result"
            # TODO: don't assume that the out variant is always named _out
            assert len(self.keywords) == 1
            kwarg = self.keywords[0]
            assert kwarg.arg == "out"
            s += "_out(result, "
        else:
            s += "("


        s += ", ".join(map(str, self.args)) + ")"

        return s

# A comparison of two or more values
# https://docs.python.org/3/library/ast.html?highlight=compare#ast.Compare
class _Compare(_Statement):
    def __init__(self, stmt):
        super().__init__(stmt)

        assert isinstance(stmt, ast.Compare)
        self.left = parse_stmt(stmt.left)  # first value in the comparison
        assert len(stmt.ops) == 1
        self.op = parse_stmt(stmt.ops[0])
        assert len(stmt.comparators) == 1
        self.right = parse_stmt(stmt.comparators[0])

        self.fields = ('left', 'op', 'right')

    def __str__(self):
        return str(self.left) + " " + str(self.op) + " " + str(self.right)

# A constant value
# https://docs.python.org/3/library/ast.html?highlight=constant#ast.Constant
class _Constant(_Statement):
    def __init__(self, stmt):
        super().__init__(stmt)
        assert isinstance(stmt, ast.Constant)

        self.value = parse_stmt(stmt.value)
        self.fields = ('value',)

    def __str__(self):
        return str(self.value)

# A function call by itself
# https://greentreesnakes.readthedocs.io/en/latest/nodes.html#expressions
class _Expr(_Statement):
    def __init__(self, stmt):
        super().__init__(stmt)

        assert isinstance(stmt, ast.Expr)
        self.value = parse_stmt(stmt.value)

        self.fields = ('value',)

    def __str__(self):
        return str(self.value)

# Greater than operator
# https://docs.python.org/3/library/ast.html?highlight=gt#ast.Gt
class _Gt(_Statement):
    def __init__(self, stmt):
        super().__init__(stmt)
        assert isinstance(stmt, ast.Gt)

    def __str__(self):
        return ">"


# An integer
# https://docs.python.org/3/library/functions.html?highlight=int#int
class _Int(_Statement):
    def __init__(self, stmt):
        super().__init__(stmt)
        assert isinstance(stmt, int)

        self.value = stmt

    def __str__(self):
        return str(self.value)

# A keyword argument to a function call or class definition
# https://docs.python.org/3/library/ast.html?highlight=keyword#ast.keyword
class _Keyword(_Statement):
    def __init__(self, stmt):
        super().__init__(stmt)
        assert isinstance(stmt, ast.keyword)

        self.arg = stmt.arg  # a str
        self.value = parse_stmt(stmt.value)

        self.fields = ('arg', 'value',)

    def __str__(self):
        return self.arg + "=" + str(self.value)

# TODO: support typed names
# A variable name
# https://docs.python.org/3/library/ast.html?highlight=name#ast.Name
class _Name(_Statement):
    def __init__(self, stmt):
        super().__init__(stmt)

        assert isinstance(stmt, ast.Name)
        self.id = stmt.id  # string name
        # ctx is one of Load, Store, Del
        self.ctx = stmt.ctx

        self.fields = ('id', 'ctx')

    def __str__(self):
        return self.id

# A return statement
# https://greentreesnakes.readthedocs.io/en/latest/nodes.html?highlight=return#Return
class _Return(_Statement):
    def __init__(self, stmt):
        super().__init__(stmt)

        assert isinstance(stmt, ast.Return)
        self.value = parse_stmt(stmt.value)

        self.fields = ('value',)

    def __str__(self):
        return "return " + str(self.value)


# A Python string
class _String(_Statement):
    def __init__(self, stmt):
        super().__init__(stmt)
        assert isinstance(stmt, str)

        self.value = stmt

    def __str__(self):
        return '"' + self.value + '"'

type_to_class_map_impl = {
    ast.Assign : _Assign,
    ast.Attribute : _Attribute,
    ast.Call : _Call,
    ast.Compare : _Compare,
    ast.Constant : _Constant,
    ast.Expr : _Expr,
    ast.Gt : _Gt,
    int : _Int,
    ast.keyword : _Keyword,
    ast.Name : _Name,
    ast.Return : _Return,
    str : _String,
}

type_to_class_map.update(type_to_class_map_impl)

def parseBody(body):
    stmts = []
    for stmt in body:
        stmts.append(type_to_class_map[type(stmt)](stmt))

    return stmts

def cppify_helper(parent, child):
    if not isinstance(child, _Statement):
        return

    # maps len() calls to .size() attribute accesses
    if isinstance(child, _Call):
        if str(child.func) == "len":
            for field in parent.fields:
                if getattr(parent, field) == child:
                    size_access = _Attribute(
                        ast.Attribute(
                            value=child.args[0].stmt,
                            attr="size",
                            ctx=ast.Load()))
                    setattr(parent, field, size_access)
                    return

    if not hasattr(child, 'fields'):
        return

    for field in child.fields:
        attr = getattr(child, field)

        if isinstance(attr, Tuple) or isinstance(attr, List):
            for a in attr:
                cppify_helper(child, a)
        else:
            cppify_helper(child, attr)

# Converts Python statements into C++ equivalents where necessary
def cppify(stmts):
    for stmt in stmts:
        cppify_helper(None, stmt)


def write_pyops_cpp(path):
    cpp = ""
    for op in pyops_sequence:
        src = inspect.getsource(op)
        tree = ast.parse(src)

        # asserts the tree is a single FunctionDef node
        assert len(tree.body) == 1
        fn_node = tree.body[0]

        # constructs the signature
        sig = parseSignature(fn_node)

        name = sig.name.replace("__", "_")
        cpp_type = map_type_to_cpp(sig.type)
        cpp = cpp + "{0} {1}(".format(cpp_type, name)

        l = []
        for arg in sig.args:
            l.append(print_arg_cpp(arg))

        # TODO: handle kwargs other than out=
        for kwarg in sig.kwargs:
            assert kwarg.name == "out"
            l.append("Tensor& result")

        cpp = cpp + ", ".join(l) + ") {\n"

        # constructs the body
        body_list = fn_node.body
        parsed_body = parseBody(body_list)
        cppify(parsed_body)

        # TODO: add adjustable indentation level
        for stmt in parsed_body:
            cpp = cpp + "\t" + str(stmt) + ";\n"

        # prints closing curly brace
        cpp += "}\n\n"

        # print(cpp)

    cpp = cpp_template.format(function_cpp=cpp)

    if path is not None:
        _write_if_changed(path, cpp)
    else:
        return cpp


def main():
    cpp = write_pyops_cpp(None)
    print(cpp)


if __name__ == '__main__':
    main()

# Backend meta
# CompositeImplicitAutograd
# BackendMetadata(kernel='dstack', structured=False)


# NativeFunction(
# func=FunctionSchema(
    # name=OperatorName(
        # name=BaseOperatorName(
        #   base='dstack',
        #   inplace=False,
        #   dunder_method=False
        # ),
    # overload_name=''
    # ),
#   arguments=Arguments(
#       pre_self_positional=(),
#       self_arg=None,
#       post_self_positional=(
    #   Argument(
            #   name='tensors',
            #   type=ListType(elem=BaseType(name=<BaseTy.Tensor: 3>), size=None),
            #   default=None,
            #   annotation=None),
    #   ),
        # pre_tensor_options_kwarg_only=(),
        # tensor_options=None,
        # post_tensor_options_kwarg_only=(), out=()
    # ),
    # returns=(Return(name=None, type=BaseType(name=<BaseTy.Tensor: 3>), annotation=None),)
    # ),
# use_const_ref_for_mutable_tensors=False,
# device_guard=True,
# device_check=<DeviceCheckType.ExactSame: 1>,
# python_module=None,
# category_override=None,
# variants={<Variant.function: 1>},
# manual_kernel_registration=False,
# manual_cpp_binding=False,
# loc=Location(file='/private/home/mruberry/git/pytorch/cmake/../aten/src/ATen/native/native_functions.yaml', line=4006),
# structured=False,
# structured_delegate=None,
# structured_inherits=None,
# cpp_no_default_args=set(),
# is_abstract=False,
# has_composite_implicit_autograd_kernel=True,
# has_composite_explicit_autograd_kernel=False)

# out= variant
#NativeFunction(
# func=FunctionSchema(
# name=OperatorName(
# name=BaseOperatorName(
# base='dstack',
# inplace=False,
# dunder_method=False),
# overload_name='out'),
# arguments=Arguments(
# pre_self_positional=(),
# self_arg=None,
# post_self_positional=(
# Argument(
# name='tensors',
# type=ListType(
# elem=BaseType(name=<BaseTy.Tensor: 3>),
# size=None),
# default=None,
# annotation=None),),
# pre_tensor_options_kwarg_only=(),
# tensor_options=None,
# post_tensor_options_kwarg_only=(),
# out=(Argument(name='out',
# type=BaseType(name=<BaseTy.Tensor: 3>),
# default=None,
# annotation=Annotation(alias_set=('a',), is_write=True)),)),
# returns=(Return(name=None,
# type=BaseType(name=<BaseTy.Tensor: 3>),
# annotation=Annotation(alias_set=('a',), is_write=True)),)),
# use_const_ref_for_mutable_tensors=False,
# device_guard=True,
# device_check=<DeviceCheckType.ExactSame: 1>,
# python_module=None,
# category_override=None,
# variants={<Variant.function: 1>},
# manual_kernel_registration=False,
# manual_cpp_binding=False,
# loc=Location(file='/private/home/mruberry/git/pytorch/cmake/../aten/src/ATen/native/native_functions.yaml', line=4008),
# structured=False,
# structured_delegate=None,
# structured_inherits=None,
# cpp_no_default_args=set(),
# is_abstract=False,
# has_composite_implicit_autograd_kernel=True,
# has_composite_explicit_autograd_kernel=False)

# Backend meta
# CompositeImplicitAutograd
# BackendMetadata(kernel='dstack', structured=False)
# CompositeImplicitAutograd
# BackendMetadata(kernel='dstack_out', structured=False)