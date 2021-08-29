


# TODO: package and calling convention/documentation here
# python -m tools.codegen.gen --pyops_cpp_dry_run

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