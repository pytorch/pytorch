# TODO: this import doesn't work if torch is not installed
# from tools.codegen.model import (
#     BaseOperatorName, OperatorName, FunctionSchema,
#     BaseTy, BaseType, ListType, Argument, Arguments,
#     Return, Annotation,
#     DeviceCheckType,
#     Variant,
#     Location,
#     NativeFunction,
#     DispatchKey, BackendMetadata,)

import inspect
import ast

# This file returns extends the intermediate native function objects based
#   on the signatures in ops.py.

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