from typing import List, Optional, Union
import itertools
from typing_extensions import Literal
from dataclasses import dataclass
import re
from functools import reduce

# TODO: clean up imports
from tools.codegen.context import *
from tools.codegen.utils import *
from tools.codegen.model import *
from tools.codegen.api.types import *
import tools.codegen.api.meta as meta
import tools.codegen.api.structured as structured
from tools.codegen.api.translate import translate
import tools.codegen.local as local
from tools.codegen.selective_build.selector import SelectiveBuilder

# List of non-leaf ops we want to override both forward + backward.
# TODO(https://github.com/pytorch/pytorch/issues/39959)
_FN_AUTOGRAD_XLA = set([
    'max_pool2d'
    'max_pool3d'
])

# TODO: Consider moving these to yaml
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

def requires_backend_wrapper(f: ExternalBackendFunction) -> bool:
    # TODO: Need to keep this in sync with `default` in RegistrationDeclarations.yaml, until we kill it
    requires_lowering = not any(is_generic_dispatch_key(k) for k in f.native_function.dispatch)
    has_xla_lowering = f.metadata is not None
    # TODO: Can't tell if this is a temporary hack, or intended longterm behavior.
    # If it's longterm behavior, move the hardcoded list into yaml
    has_autogradxla = f.native_function.func.name.name in _FN_AUTOGRAD_XLA
    in_denylist = any([re.match(frx, str(f.native_function.func.name)) for frx in _FN_DENYLIST_REGEX])
    return not in_denylist and (requires_lowering or has_xla_lowering or has_autogradxla)

def xla_tensor_creation_api(out_name: str, out_type: Type, device_param_name: str, *, tuple_idx: int = None) -> str:
    if tuple_idx != None:
        out_name = f"std::get<{tuple_idx}>({out_name})"
    if out_type == BaseType(BaseTy.Tensor):
        bridge_api = 'CreateXlaTensor'
    elif isinstance(out_type, ListType) and out_type.elem == BaseType(BaseTy.Tensor):
        bridge_api = 'CreateXlaTensors'
    else:
        # for non tensor-types, there's no need to wrap the output in an xla bridge api.
        return out_name
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
        Literal[Target.NAMESPACED_DECLARATION]
    ]

    @method_with_native_function
    def __call__(self, g: Union[ExternalBackendFunctionsGroup, ExternalBackendFunction]) -> List[str]:

        def gen_out_wrapper(f: ExternalBackendFunction) -> Optional[str]:
            dispatcher_sig = DispatcherSignature.from_schema(f.native_function.func)
            name = dispatcher_sig.name()

            # TODO: keep this in sync with gen_unstructured_external
            tensor_args = [a for a in dispatcher_sig.arguments() if (
                           (isinstance(a.ctype, ConstRefCType) or isinstance(a.ctype, MutRefCType)) and
                           isinstance(a.ctype.elem, BaseCType) and a.ctype.elem.type == 'Tensor')]
            print_args_str = ''.join([f' << " {a.name}=" << {a.name}.toString()' for a in tensor_args])

            func_name = f'AtenXlaTypeDefault::{name}'
            return_names = cpp.return_names(f.native_function)
            if len(return_names) > 1:
                updates = '\n  '.join(f'bridge::XlaUpdateTensors({{{ret_name}}}, {{std::get<{i}>({name}_tmp)}}, {{0}});' for i, ret_name in enumerate(return_names))
                returns = f'std::tuple<{",".join(["at::Tensor &"] * len(return_names))}>({", ".join(return_names)})'
            else:
                ret_name = return_names[0]
                updates = f'bridge::XlaUpdateTensors({{{ret_name}}}, {{{name}_tmp}}, {{0}});'
                returns = ret_name

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

            def get_device_param(binds: List[Binding]) -> str:
                # TODO: xla codegen gives Tensor args higher precendence than Tensorlist args
                # When deciding which argument to determine the device from.
                # That's probably not necessary, and we can merge the two conditions.
                tensor_like = [b for b in binds if b.argument.type.is_tensor_like() and not isinstance(b.argument.type, ListType)]
                if any(tensor_like):
                    return tensor_like[0].name
                tensor_list_like = [b for b in binds if b.argument.type.is_tensor_like() and isinstance(b.argument.type, ListType)]
                if any(tensor_list_like):
                    return tensor_list_like[0].name
                device_like = [b for b in binds if b.argument.type == BaseType(BaseTy.Device)
                               or b.argument.type == OptionalType(BaseType(BaseTy.Device))]
                if any(device_like):
                    return device_like[0].name
                assert_never("Need a tensor-like or device argument in order to determine the output device")

            # This is an accumulator function used by itertools.reduce - Python's version of fold_left.
            # It's used to compute several pieces of information used by the aten fallback:
            #   `bindings` - The names of intermediate variables that each input arg maps to.
            #   `tensorlist_args` - Intermediate TensorArg variables
            #   `tensor_args` - Intermediate Tensor variables
            #   `tensor_opt_args` - Intermediate optional Tensor variables
            #   `annotated_tensor_indices` - The indices of the input Tensor arguments that are annotated
            # Fold is useful here because we need to keep track of intermediate state as we loop through the arguments:
            # Specifically, the number of optional/immutable/mutable tensors we've seen so far.
            def map_xla_binding(acc: Tuple[List[Binding], List[str], List[str], List[str], List[int]], binding: Binding) -> Tuple[List[Binding], List[str], List[str], List[str], List[int]]:
                bindings, tensorlist_args, tensor_args, tensor_opt_args, annotated_tensor_indices = acc
                t = binding.type
                # case: TensorList
                if t == 'at::TensorList':
                # if isinstance(binding.ctype, BaseCType) and binding.ctype.type == 'TensorList':
                    new_bind = binding.with_name(name=f'l_{binding.name}')
                    return (bindings + [new_bind], tensorlist_args + [binding.name], tensor_args, tensor_opt_args, annotated_tensor_indices)
                # case: optional<Tensor>
                elif t == 'c10::optional<at::Tensor>' or t == 'c10::optional<at::Tensor> &' or t == 'const c10::optional<at::Tensor> &':
                # elif (isinstance(binding.ctype, OptionalCType) and isinstance(binding.ctype.elem, BaseCType) and
                        # binding.ctype.elem.type == 'Tensor'):
                    new_bind = binding.with_name(name=f'xlatens_opt[{len(tensor_opt_args)}]')
                    return (bindings + [new_bind], tensorlist_args, tensor_args, tensor_opt_args + [binding.name], annotated_tensor_indices)
                elif t == 'const at::Tensor &' or t == 'at::Tensor &' or t == 'at::Tensor':
                # elif ((isinstance(binding.ctype, ConstRefCType) or isinstance(binding.ctype, MutRefCType)) and
                        # isinstance(binding.ctype.elem, BaseCType) and binding.ctype.elem.type == 'Tensor'):
                    new_bind = binding.with_name(name=f'xlatens[{len(tensor_args)}]')
                    # case: Tensor (annotations)
                    if binding.argument.annotation is not None and binding.argument.annotation.is_write:
                        mutable_idx = len(tensor_args)
                        return (bindings + [new_bind], tensorlist_args, tensor_args + [binding.name], tensor_opt_args, annotated_tensor_indices + [mutable_idx])
                    # case: Tensor (no annotations)
                    else:
                        return (bindings + [new_bind], tensorlist_args, tensor_args + [binding.name], tensor_opt_args, annotated_tensor_indices)
                else:
                    return (bindings + [binding], tensorlist_args, tensor_args, tensor_opt_args, annotated_tensor_indices)
            # tensor_count = 0
            # opt_tensor_count = 0
            # tensorlist_count = 0
            # annotated_tensor_indices = []
            # arg_renames = {}
            # for binding in args:

                # TODO: Confirm. xla codegen uses "which Tensor& vars are mutable" to determine which ones to update.
                # But I think what they really want is "which Tensor& vars have type aliases"

            # XLA appears to have used the dispatcher convention to write their kernel signatures,
            # probably because they based their signatures off of our RegistrationDeclarations.h
            dispatcher_sig = DispatcherSignature.from_schema(f.native_function.func)
            name = dispatcher_sig.name()
            args = dispatcher_sig.arguments()

            if self.target is Target.NAMESPACED_DECLARATION:
                # TODO: does it make sense for fallback kernels to be named directly based off dispatcher schema,
                # but explicit xla kernels to have custom names?
                return f"  static {dispatcher_sig.decl()};"

            # TODO: we should generate out wrappers for ALL valid out kernels; not just ones in xla's hardcoded list
            # TODO: byte-for-byte compatibility. We should move out wrappers into a different file than cpu fallbacks
            if f.native_function.func.kind() is SchemaKind.out and str(f.native_function.func.name.name) in _FN_OUT:
                return gen_out_wrapper(f)

            updated_bindings, tensorlist_args, tensor_args, tensor_opt_args, annotated_tensor_indices = reduce(map_xla_binding, args, ([], [], [], [], []))

            print_args_str = ''.join([f' << " {a}=" << {a}.toString()' for a in tensor_args])


            tensorlist_intermediates_str = ''
            if any(tensorlist_args):
                tensorlist_intermediates_str = '\n'.join([f'  auto l_{argname} = bridge::XlaCreateTensorList({argname});' for argname in tensorlist_args])

            opt_tensor_intermediates_str = ''
            if any(tensor_opt_args):
                opt_tensor_intermediates_str = f'\n  std::vector<c10::optional<at::Tensor>> xlatens_opt_tensors = {{{", ".join(tensor_opt_args)}}};'
                opt_tensor_intermediates_str += '\n  auto xlatens_opt = bridge::XlaCreateOptTensorList(xlatens_opt_tensors);'

            intermediates = ''
            if tensorlist_intermediates_str != '':
                intermediates += tensorlist_intermediates_str + '\n'
            intermediates += f"  std::vector<at::Tensor> xlatens_tensors = {{{', '.join(tensor_args)}}};"
            intermediates += "\n  auto xlatens = bridge::XlaCreateTensorList(xlatens_tensors);"
            if opt_tensor_intermediates_str != '':
                intermediates += opt_tensor_intermediates_str


            # TODO: I'm using the CppSignature purely to get the faithful API.
            # I'm not doing any translating from Dispatcher to cpp faithful. Do I need to?
            is_method = Variant.function not in f.native_function.variants
            func_name = f'AtenXlaTypeDefault::{name}'


            at_call_name = CppSignatureGroup.from_native_function(f.native_function, method=is_method).most_faithful_signature().name()
            if is_method:
                at_call = f'{updated_bindings[0].name}.{at_call_name}({", ".join(b.name for b in updated_bindings[1:])});'
            else:
                at_call = f'at::{at_call_name}({", ".join(b.name for b in updated_bindings)});'
            avoid_warning = ''
            if f.native_function.func.returns:
                at_call = 'auto&& x_result = ' + at_call
                avoid_warning = '\n  static_cast<void>(x_result); // Avoid warnings in case not used'

            collect_mutated_tensors = ''
            update_tensors = ''
            if len(annotated_tensor_indices) > 0:
                collect_mutated_tensors = f'\n  std::vector<size_t> xlatens_update_indices = {{{", ".join([str(i) for i in annotated_tensor_indices])}}};'
                update_tensors = '\n  bridge::XlaUpdateTensors(xlatens_tensors, xlatens, xlatens_update_indices);'

            returns = ''
            if f.native_function.func.returns:
                return_names = cpp.return_names(f.native_function)
                if len(return_names) == 1:
                    if len(annotated_tensor_indices) > 0:
                        return_args = return_names[0]
                    else:
                        return_args = xla_tensor_creation_api("x_result", f.native_function.func.returns[0].type, get_device_param(args))
                    returns = f'\n  return {return_args};'
                else:
                    if len(annotated_tensor_indices) > 0:
                        return_args = ", ".join([a for (i, a) in enumerate(tensor_args) if i in annotated_tensor_indices])
                        if len(annotated_tensor_indices) != len(return_names):
                            import pdb; pdb.set_trace()
                    else:
                        return_args = ", ".join([xla_tensor_creation_api("x_result", f.native_function.func.returns[i].type, get_device_param(args), tuple_idx=i) for i in range(len(return_names))])
                    returns = f'\n  return {dispatcher_sig.returns_type()}({return_args});'
            if '_amp_update_scale' in str(f.native_function.func.name):
                import pdb; pdb.set_trace()

            # TODO: if I don't do a translate from dispatcher -> cpp, will I be shooting myself in the foot somewhere?
            # sig_group = CppSignatureGroup.from_native_function(f.native_function, method=Variant.method in f.native_function.variants)
            # cpp_sig = sig_group.faithful_signature if sig_group.faithful_signature is not None else sig_group.signature
    # {store_result}at::{cpp_sig.name()}({', '.join(e.expr for e in translate(updated_bindings, [a.ctype for a in cpp_sig.arguments()]))});

            return f"""\
{dispatcher_sig.defn(name=func_name)} {{
  XLA_FN_TRACK(3);
  XLA_COUNTER("aten::{name}", 1);
  TF_VLOG(3) << "XLA {name} :"{print_args_str};
{intermediates}
  {at_call}{collect_mutated_tensors}{update_tensors}{avoid_warning}{returns}
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
