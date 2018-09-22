import torch.jit
# this file is for generating documentation using sphinx autodoc
# > help(torch.jit.supported_ops) will also give a nice listed of the
# supported ops programmatically


def _list_supported_ops():
    def emit_type(type):
        return str(type)

    def emit_arg(indent, i, arg):
        v = "{} : {}".format(arg.name, emit_type(arg.type))
        default = arg.default_value
        if default is not None:
            v = "{}={}".format(v, str(default))
        if i > 0:
            v = "\n{}{}".format(" " * indent, v)
        return v

    def emit_args(indent, arguments):
        return ",".join(emit_arg(indent, i, arg) for i, arg in enumerate(arguments))

    def emit_ret(ret):
        return emit_type(ret.type)

    def emit_rets(returns):
        if len(returns) == 1:
            return emit_ret(returns[0])
        return "Tuple[{}]".format(", ".join(emit_ret(r) for r in returns))

    def emit_schema(mod, name, schema, arg_start=0):
        qualified_name = "{}.{}".format(mod, name)
        schema = "{}({}) -> {}".format(qualified_name,
                                       emit_args(len(qualified_name) + 1 + 4, schema.arguments[arg_start:]),
                                       emit_rets(schema.returns))
        return schema

    def hidden(name):
        return name.startswith('_') and not name.startswith('__')

    functions = []

    for mod in torch.jit._modules_containing_builtins:
        name = mod.__name__
        for elem in dir(mod):
            builtin = torch.jit._find_builtin(getattr(mod, elem))
            if builtin is not None:
                schemas = torch._C._jit_get_schemas_for_operator(builtin)
                for schema in schemas:
                    # remove _tan but not __and__
                    if not hidden(elem):
                        functions.append(emit_schema(name, elem, schema))

    def is_tensor_method(schema):
        if len(schema.arguments) == 0:
            return False
        self = schema.arguments[0]
        if self.name != 'self':
            return False
        if not self.type.isSubtypeOf(torch._C.DynamicType.get()):
            return False
        return True

    methods = []
    # discover methods
    for elem in dir(torch.Tensor):
        if not hidden(elem):
            schemas = torch._C._jit_get_schemas_for_operator("aten::" + elem)
            for schema in schemas:
                if is_tensor_method(schema):
                    methods.append(emit_schema('Tensor', elem, schema, arg_start=1))

    def emit_block(decls):
        return '\n::\n\n{}\n'.format(''.join('    {}\n\n'.format(d) for d in decls))
    body = """
Supported Functions
~~~~~~~~~~~~~~~~~~~
{}

Supported Methods
~~~~~~~~~~~~~~~~~
{}
"""
    return body.format(emit_block(functions), emit_block(methods))

__doc__ = _list_supported_ops()
