from typing import List, Optional, Union
from typing_extensions import Literal
from dataclasses import dataclass
import re

from tools.codegen.context import *
from tools.codegen.utils import *
from tools.codegen.model import *
from tools.codegen.api.types import *

# TODO: if any of these are XLA specific, we should move them into yaml.
_FN_DENYLIST_REGEX = [
    # ATEN functions
    r'[^(]*cudnn',
    r'slow_conv_transpose2d_backward.grad_output',
    r'slow_conv_transpose3d_backward.grad_output',
    r'slow_conv3d_backward.grad_input',
    r'thnn_conv2d_backward.grad_input',
    r'thnn_conv_depthwise2d_backward.grad_input',
    # XLA/TPU functions
]

# TODO: remove this list.
# Instead, the codegen will figure out which ops to generate _out wrappers for
# entirely from the yaml. Maintaining the same behavior as current XLA codegen for now.
_FN_OUT = [
    'abs',
    'add',
    'acos',
    'acosh',
    'asin',
    'asinh',
    'atan',
    'atan2',
    'atanh',
    'baddbmm',
    'bernoulli',
    'binary_cross_entropy',
    'binary_cross_entropy_backward',
    'clamp',
    'div',
    'gather',
    'ger',
    'hardsigmoid',
    'kthvalue',
    'index_select',
    'inverse',
    'log',
    'masked_select',
    'maximum',
    'minimum',
    'pow',
    'prod',
    'nonzero',
    'round',
    'normal',
    'std',
    'take',
    'topk',
    'var',
]

# TODO: I can split this out into two functions later: one for the cpu fallback codegen,
# and one for the xla kernel wrapper codegen.
def requires_backend_wrapper(f: ExternalBackendFunction) -> bool:
    # TODO: Need to keep this in sync with `default` in RegistrationDeclarations.yaml, until we kill it
    requires_lowering = not any(is_generic_dispatch_key(k) for k in f.native_function.dispatch)
    has_xla_lowering = f.metadata is not None
    in_denylist = any([re.match(frx, str(f.native_function.func.name)) for frx in _FN_DENYLIST_REGEX])
    return not in_denylist and (requires_lowering or has_xla_lowering)

def xla_tensor_creation_api(ret_name: str, ret: Return, device_param_name: str, *, tuple_idx: int = None) -> str:
    if ret.type == BaseType(BaseTy.Tensor) and not ret.is_write:
        # Only raw Tensor (non-reference) returns need to go through the XLA tensor creation API.
        # Tensor references can be returned directly, since they've already been converted to XLA tensors.
        # See Note [Tensor Copy Returns]
        bridge_api = 'CreateXlaTensor'
    elif isinstance(ret.type, ListType) and ret.type.elem == BaseType(BaseTy.Tensor):
        bridge_api = 'CreateXlaTensors'
    else:
        # for non tensor-types, there's no need to wrap the output in an xla bridge api.
        return ret_name

    out_name = 'x_result'
    if tuple_idx is not None:
        out_name = f"std::get<{tuple_idx}>(x_result)"

    return f"bridge::{bridge_api}({out_name}, bridge::GetXlaDevice({device_param_name}))"



# Generates aten_xla_type_default.h and aten_xla_type_default.cpp.
#
#   - This function registers external kernels that fallback to cpu.
#     This is useful because pretty much all external backends (e.g. XLA)
#     do not have full aten coverage.
#     For operators not implemented by the external backend, our codegen
#     will register these fallbacks instead.
#   - Why do we generate fallback for ALL aten ops, including ops that
#     external backends have already implemented?
#     Many external backend kernels only work with specific input shapes,
#     and are written to call into a cpu fallback when given inputs
#     that they cannot handle.
@dataclass(frozen=True)
class GenExternalAtenFallback:
    target: Union[
        Literal[Target.NAMESPACED_DEFINITION],
        Literal[Target.NAMESPACED_DECLARATION],
        # TODO: split registration out from aten fallbacks so they're in different files.
        Literal[Target.REGISTRATION],
    ]

    @method_with_native_function
    def __call__(self, g: Union[ExternalBackendFunctionsGroup, ExternalBackendFunction]) -> List[str]:

        def gen_out_wrapper(f: ExternalBackendFunction) -> Optional[str]:
            dispatcher_sig = DispatcherSignature.from_schema(f.native_function.func)
            name = dispatcher_sig.name()

            # TODO: keep this in sync with gen_unstructured_external
            dispatcher_order_args = list(dispatcher.jit_arguments(f.native_function.func))
            tensors = [a for a in dispatcher_order_args if a.type == BaseType(BaseTy.Tensor)]
            print_args_str = ''.join([f' << " {a.name}=" << {a.name}.toString()' for a in tensors])

            func_name = f'AtenXlaTypeDefault::{name}'
            return_names = cpp.return_names(f.native_function, override_name="x_result")
            if len(return_names) > 1:
                updates = '\n  '.join(
                    f'bridge::XlaUpdateTensors({{{ret_name}}}, {{std::get<{i}>({name}_tmp)}}, {{0}});'
                    for i, ret_name in enumerate(return_names))
                returns = f'{dispatcher_sig.returns_type().cpp_type()}({", ".join(return_names)})'
            else:
                ret_name = return_names[0]
                updates = f'bridge::XlaUpdateTensors({{{ret_name}}}, {{{name}_tmp}}, {{0}});'
                returns = ret_name

            # TODO: instead of hardcoding out wrappers that call into functional kernels,
            # make that toggleable by the backend.
            functional_sig = DispatcherSignature.from_schema(g.functional.native_function.func)

            return f"""\
{dispatcher_sig.defn(name=func_name)} {{
  XLA_FN_TRACK(3);
  TF_VLOG(3) << "XLA {name} :"{print_args_str};
  auto {name}_tmp = AtenXlaType::{functional_sig.name()}({", ".join(a.name for a in functional_sig.arguments())});
  {updates}
  return {returns};
}}

"""

        def gen_unstructured_external(f: ExternalBackendFunction) -> Optional[str]:
            if not requires_backend_wrapper(f):
                return None

            def get_device_param(args: List[Argument]) -> str:
                # TODO: xla codegen gives const / self Tensor args higher precendence other tensor args
                # When deciding which argument to determine the device from.
                # That's probably not necessary, and we can merge the two conditions.
                const_tensor_or_self = [
                    a for a in args if (a.type == BaseType(BaseTy.Tensor) or a.type == OptionalType(BaseType(BaseTy.Tensor)))
                    and not a.is_write]
                if any(const_tensor_or_self):
                    return const_tensor_or_self[0].name
                tensor_like = [a for a in args if a.type.is_tensor_like()]
                if any(tensor_like):
                    return tensor_like[0].name
                device_like = [a for a in args if a.type == BaseType(BaseTy.Device)
                               or a.type == OptionalType(BaseType(BaseTy.Device))]
                if any(device_like):
                    return device_like[0].name
                assert_never("Need a tensor-like or device argument in order to determine the output device")

            # XLA appears to have used the dispatcher convention to write their kernel signatures,
            # probably because they based their signatures off of our RegistrationDeclarations.h
            dispatcher_sig = DispatcherSignature.from_schema(f.native_function.func)
            name = dispatcher_sig.name()
            args = dispatcher_sig.arguments()

            if self.target is Target.NAMESPACED_DECLARATION:
                # TODO: does it make sense for fallback kernels to be named directly based off dispatcher schema,
                # but explicit xla kernels to have custom names?
                return f"  static {dispatcher_sig.decl()};"

            elif self.target is Target.REGISTRATION:
                if f.metadata is not None:
                    # xla has their own kernel: register it
                    namespace = 'AtenXlaType'
                else:
                    # xla doesn't have a kerne: register a cpu fallback
                    namespace = 'AtenXlaTypeDefault'
                payload = f"static_cast<{dispatcher_sig.decl(func_ptr_cast=True)}>(&{namespace}::{name})"
                return f'  m.impl("{f.native_function.func.name}", {payload});\n'

            # TODO: we should generate out wrappers for ALL valid out kernels; not just ones in xla's hardcoded list
            # TODO: byte-for-byte compatibility. We should move out wrappers into a different file than cpu fallbacks
            if f.native_function.func.kind() is SchemaKind.out and str(f.native_function.func.name.name) in _FN_OUT:
                return gen_out_wrapper(f)

            dispatcher_order_args = list(dispatcher.jit_arguments(f.native_function.func))

            # maps each argument to it's intermediate variable name in the fallback
            name_ctx: Dict[Argument, str] = {}

            tensorlist_args: Dict[Argument, str] = {
                a: f'l_{a.name}' for a in dispatcher_order_args
                if isinstance(a.type, ListType) and a.type.elem == BaseType(BaseTy.Tensor)}

            opt_tensors = [
                a for a in dispatcher_order_args
                if isinstance(a.type, OptionalType) and a.type.elem == BaseType(BaseTy.Tensor)]
            opt_tensor_args: Dict[Argument, str] = {a: f'xlatens_opt[{i}]' for i, a in enumerate(opt_tensors)}

            tensors = [a for a in dispatcher_order_args if a.type == BaseType(BaseTy.Tensor)]
            tensor_args: Dict[Argument, str] = {a: f'xlatens[{i}]' for i, a in enumerate(tensors)}
            annotated_tensor_indices: List[int] = [
                i for i, a in enumerate(tensors) if a.annotation is not None and a.annotation.is_write]

            print_args_str = ''.join([f' << " {a.name}=" << {a.name}.toString()' for a in tensor_args.keys()])


            tensorlist_intermediates_str = ''
            if len(tensorlist_args) > 0:
                tensorlist_intermediates_str = '\n'.join([f'  auto {updated_name} = bridge::XlaCreateTensorList({arg.name});'
                                                          for arg, updated_name in tensorlist_args.items()])

            opt_tensor_intermediates_str = ''
            if len(opt_tensor_args) > 0:
                arg_str = ", ".join([a.name for a in opt_tensor_args.keys()])
                opt_tensor_intermediates_str = f'\n  std::vector<c10::optional<at::Tensor>> xlatens_opt_tensors = {{{arg_str}}};'
                opt_tensor_intermediates_str += '\n  auto xlatens_opt = bridge::XlaCreateOptTensorList(xlatens_opt_tensors);'

            intermediates = ''
            if tensorlist_intermediates_str != '':
                intermediates += tensorlist_intermediates_str + '\n'
            intermediates += f"  std::vector<at::Tensor> xlatens_tensors = {{{', '.join([a.name for a in tensor_args.keys()])}}};"
            intermediates += "\n  auto xlatens = bridge::XlaCreateTensorList(xlatens_tensors);"
            if opt_tensor_intermediates_str != '':
                intermediates += opt_tensor_intermediates_str


            is_method = Variant.function not in f.native_function.variants
            func_name = f'AtenXlaTypeDefault::{name}'


            # Just use the original binding names if we didn't create explicit intermediate variables
            updated_bindings: List[str] = [
                tensorlist_args.get(a, opt_tensor_args.get(a, tensor_args.get(a, a.name))) for a in dispatcher_order_args]
            at_call_name = CppSignatureGroup.from_native_function(f.native_function, method=is_method) \
                .most_faithful_signature().name()

            if is_method:
                at_call = f'{updated_bindings[0]}.{at_call_name}({", ".join(name for name in updated_bindings[1:])});'
            else:
                at_call = f'at::{at_call_name}({", ".join(name for name in updated_bindings)});'
            avoid_warning = ''
            if f.native_function.func.returns:
                at_call = 'auto&& x_result = ' + at_call
                avoid_warning = '\n  static_cast<void>(x_result); // Avoid warnings in case not used'

            collect_mutated_tensors = ''
            update_tensors = ''
            if len(annotated_tensor_indices) > 0:
                indices_str = ", ".join([str(i) for i in annotated_tensor_indices])
                collect_mutated_tensors = f'\n  std::vector<size_t> xlatens_update_indices = {{{indices_str}}};'
                update_tensors = '\n  bridge::XlaUpdateTensors(xlatens_tensors, xlatens, xlatens_update_indices);'

            returns = ''
            if f.native_function.func.returns:
                ret_names = cpp.return_names(f.native_function, override_name="x_result")
                if len(ret_names) == 1:
                    return_args = xla_tensor_creation_api(ret_names[0], f.native_function.func.returns[0], get_device_param(dispatcher_order_args))
                    returns = return_args
                else:
                    return_args = [xla_tensor_creation_api(ret_names[i], f.native_function.func.returns[i], get_device_param(dispatcher_order_args), tuple_idx=i) for i in range(len(f.native_function.func.returns))]
                    returns = f'{dispatcher_sig.returns_type().cpp_type()}({", ".join(return_args)})'
            return_str = ''
            if returns != '':
                return_str = f'\n  return {returns};'

            return f"""\
{dispatcher_sig.defn(name=func_name)} {{
  XLA_FN_TRACK(3);
  XLA_COUNTER("aten::{name}", 1);
  TF_VLOG(3) << "XLA {name} :"{print_args_str};
{intermediates}
  {at_call}{collect_mutated_tensors}{update_tensors}{avoid_warning}{return_str}
}}

"""
        if isinstance(g, ExternalBackendFunctionsGroup):
            if g.structured:
                # We can probably only bother generating fallbacks for one of the variants, for structured
                assert_never("Not Implemented")
            else:
                return list(mapMaybe(gen_unstructured_external, g.functions()))
        elif isinstance(g, ExternalBackendFunction):
            f = g
            x = gen_unstructured_external(f)
            return [x] if x else []
        else:
            assert_never(f)
