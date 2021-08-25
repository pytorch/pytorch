import copy
import operator

import torch
import torch.fx

class AllModuleTracer(torch.fx.Tracer):
    """
    This is a tracer that knows how to convert quantizeable ops with
    dynamic dispatch into their corresponding quantized subgraphs.
    """

    def is_leaf_module(self, m, module_qualified_name) -> bool:
        return True

    def _maybe_update_args_with_quants(self, args, arg_quant_infos):
        # insert quants for inputs, if needed
        if len(arg_quant_infos):
            new_args = []
            for idx, input_arg_quant_info in enumerate(arg_quant_infos):
                if input_arg_quant_info is None:
                    new_args.append(args[idx])
                else:
                    # create a quant node
                    scale, zp = input_arg_quant_info
                    quant = super().create_node(
                        'call_function', torch.quantize_per_tensor,
                        (args[idx], scale.item(), zp.item(), torch.quint8), {}, None, None)
                    new_args.append(quant)
            args = tuple(new_args)
        return args

    def create_node(self, kind, target, args, kwargs, name=None, type_expr=None):
        if target == operator.add:
            target = torch.add
        if target == operator.mul:
            target = torch.mul

        if kind == 'call_function':
            if self.root._auto_quant_state.cur_op_needs_hooks(target):
                self.root._auto_quant_state.validate_cur_op(target)

                old_target = target
                target, arg_quant_infos, additional_kwargs = \
                    self.root._auto_quant_state.get_op_convert_info(
                        target, unwrap_scale_zp=True)

                args = self._maybe_update_args_with_quants(args, arg_quant_infos)
                kwargs.update(**additional_kwargs)

                self.root._auto_quant_state.mark_cur_op_complete(old_target)

        elif kind == 'call_module':
            # TODO: handle fqn
            module_instance = getattr(self.root, target)
            if self.root._auto_quant_state.cur_op_needs_hooks(module_instance):
                self.root._auto_quant_state.validate_cur_op(module_instance)

                _, arg_quant_infos, additional_kwargs = \
                    self.root._auto_quant_state.get_op_convert_info(
                        module_instance, unwrap_scale_zp=True)

                args = self._maybe_update_args_with_quants(args, arg_quant_infos)
                kwargs.update(**additional_kwargs)

                self.root._auto_quant_state.mark_cur_op_complete(module_instance)

        out = super().create_node(kind, target, args, kwargs, name, type_expr)
        return out

# TODO(future PR): handle cases where the module is not symbolically
# traceable
def rewrite_for_scripting(mod: torch.nn.Module) -> torch.nn.Module:
    """
    Makes the dynamically dispatched ops in `mod` be explicit, so they
    can be visibile to `torch.jit.script`. In detail:

    1. symbolically traces the forward with FX, without any leaves
    2. for each quantizeable op with dynamic dispatch, rewrites the graph to
       contain the quantized subgraph (quant if necessary, quantized op,
       dequant if necessary).
    3. recursively repeat (1 - 2) for each child
    """

    def rewrite_helper(mod : torch.nn.Module):
        copied = copy.copy(mod)
        for name, child in mod.named_children():
            setattr(copied, name, rewrite_helper(child))

        if hasattr(mod, '_auto_quant_state') and \
                mod._auto_quant_state.has_at_least_one_seen_op():  # type: ignore[union-attr, arg-type]
            copied._auto_quant_state.reset_to_new_call()  # type: ignore[union-attr]

            graph = AllModuleTracer().trace(copied)
            return torch.fx.GraphModule(copied, graph, copied.__class__.__name__)
        else:
            return copied

    return rewrite_helper(mod)
