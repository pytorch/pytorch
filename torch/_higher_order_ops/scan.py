from dataclasses import dataclass
from functools import partial
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
import torch.utils._pytree as pytree
from torch._C import DispatchKey, DispatchKeySet, _ExcludeDispatchKeyGuard
from torch._functorch.eager_transforms import _unwrap_all_tensors_from_functional, _wrap_all_tensors_to_functional, \
    functionalize
from torch._functorch.aot_autograd import create_joint, AOTConfig
from torch._ops import HigherOrderOperator
from torch.multiprocessing.reductions import StorageWeakRef
from torch.fx.experimental.proxy_tensor import (
    ProxyTorchDispatchMode,
    disable_proxy_modes_tracing,
    make_fx,
    track_tensor_tree,
    unwrap_proxy,
)
import torch.fx.traceback as fx_traceback
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode,
    _pop_mode_temporarily,
)
from torch._dispatch.python import suspend_functionalization


# TODO: We add this to prevent dymamo from tracing into map_wrapper,
# remove the wrapper call when it's ready.
class ScanWrapper(HigherOrderOperator):
    def __call__(self, f, init, xs):
        return scan_wrapper(f, init, xs)


scan = ScanWrapper("scan")
scan_impl = HigherOrderOperator("scan_impl")


def _maybe_run_with_interpreter(fn):
    #import pdb
    #pdb.set_trace()
    maybe_interpreted_fn = fn
    if isinstance(fn, torch.fx.GraphModule) and fx_traceback.has_preserved_node_meta():
        # Running graph with interpreter is needed for propagating the stack_trace
        def graph_with_interpreter(*args):
            with fx_traceback.preserve_node_meta():
                return torch.fx.Interpreter(fn).run(*args)

        maybe_interpreted_fn = graph_with_interpreter
    return maybe_interpreted_fn


dummy_aot_config = AOTConfig(fw_compiler=None,
                             bw_compiler=None,
                             partition_fn=None,
                             decompositions={},
                             num_params_buffers=0,
                             aot_id=0,
                             keep_inference_input_mutations=False)


def create_fw_bw_graph(f, flat_init, flat_xs):
    # Note: We create "clean" environments for make_fx by suspending all dispatch keys
    # between Autograd and Python key. Currently, we only suspend functionalization but more can be
    # added when required. Will encounter two problems if we don't suspend functionalization:
    #
    # 1. make_fx fails to capture operations on input: the inputs are wrapped as _to_functional_tensor_wrapper,
    # but they will be unwrapped before entering ProxyTorchDispatchMode as part of the dispatching.
    # However, it's the outside wrapper that tracer creates proxies for. This casuses tracer fail to
    # fetch the proxy for the inputs and fail to capture any operations on them.
    #
    # 2. make_fx fails to capture output: the outputs after ProxyTorchDispatchMode are further
    # wrapped as FunctionalTensorWrapper in Functionalize key after return. However, the tracer
    # only associates the inner tensor with proxy in ProxyTorchDispatchMode. Therefore,
    # when creating the output node, it fails to associate the wrapped tensor with its proxy.
    # Instead, it will create _tensor_constant as output.

    with suspend_functionalization():
        with disable_proxy_modes_tracing():
            def from_fun(t):
                if isinstance(t, torch.Tensor):
                    if t.dtype != torch.bool:
                        return torch.empty_strided(
                            t.size(),
                            t.stride(),
                            dtype=t.dtype,
                            requires_grad=t.requires_grad,
                        )
                    else:
                        # clone of a functional tensor produces a functional tensor
                        # but we want to avoid it so we clone a non-functional version
                        maybe_unfunc_t = t
                        if isinstance(t, FunctionalTensor):
                            torch._sync(t)
                            maybe_unfunc_t = from_fun(t)
                        elif torch._is_functional_tensor(t):
                            # need to handle both types of functionalization here:
                            # these are the tensors that came from the user,
                            # which could be either FunctionalTensorWrapper or FunctionalTensor
                            torch._sync(t)
                            maybe_unfunc_t = torch._from_functional_tensor(t)
                        return maybe_unfunc_t.clone()
                return t

            # import pdb
            # pdb.set_trace()
            example_init = [from_fun(init_element) for init_element in flat_init]
            # example_xs = [from_fun(xs) for xs in _unstack_pytree(flat_xs)[0]]
            # example_xs = [from_fun(xs[0, :]) for xs in flat_xs]
            # example_xs = [from_fun(flat_xs[0])]

            # example_all_xs = [from_fun(xs[0:1, :]) for xs in flat_xs]
            # example_all_xs = [from_fun(flat_xs[0][0:1, :])]
            example_all_xs = [from_fun(xs[0:1, :]) for xs in flat_xs]
            example_xs = [xs[0, :] for xs in example_all_xs]

            carry_length = len(example_init)
            leading_dim_size = example_xs[0].shape[0]
            # import pdb
            # pdb.set_trace()
            # Expect BxF, BxF
            flattened_out = f(*example_init, *example_xs)
            # example_flat_carry_out, example_flat_ys = _stack_pytree([flattened_out[:carry_length]]*(leading_dim_size+1)), _stack_pytree([flattened_out[carry_length:]]*(leading_dim_size))
            # example_flat_carry_out, example_flat_ys = _stack_pytree([flattened_out[:carry_length]]), _stack_pytree([flattened_out[carry_length:]])
            example_flat_carry_out, example_flat_ys = flattened_out[:carry_length], flattened_out[carry_length:]
            # import pdb
            # pdb.set_trace()
            # example_flat_carry_out, example_flat_ys = pytree.tree_map(from_fun, example_flat_carry_out), pytree.tree_map(from_fun, example_flat_ys)
            if any(not isinstance(out, torch.Tensor) for out in example_flat_ys if out is not None):
                raise RuntimeError("Expect outputs of scan only contains tensors or None. "
                                   f"Got types {[type(out) for out in example_flat_ys]}.")
            if any(not isinstance(out, torch.Tensor) for out in example_flat_carry_out if out is not None):
                raise RuntimeError("Expect output carry of scan only contains tensors or None. "
                                   f"Got types {[type(out) for out in example_flat_carry_out]}.")
            # import pdb
            # pdb.set_trace()
            # example_grad_carry_out = [from_fun(out[leading_dim_size-1]) for out in example_flat_carry_out]
            # example_grad_carry_out = [from_fun(out[0]) for out in example_flat_carry_out]
            example_grad_carry_out = [from_fun(out) for out in example_flat_carry_out]
            example_grad_ys = [from_fun(out) for out in example_flat_ys]

            num_grad_carry_out_args = len(example_grad_carry_out)
            num_grad_ys_args = len(example_grad_ys)
            num_init_args = len(example_init)

            # import pdb
            # pdb.set_trace()
            # Expects BxF, BxF
            print('Expected: BxF, BxF')
            print((example_init[0].shape, example_xs[0].shape))
            fw_graph = make_fx(f)(*example_init, *example_xs)

        # TODO: why is the bw graph not the same as the fw graph, but just receiving the grad_carry_out and grad_ys?
        def bw_f(*flat_args):
            # import pdb
            # pdb.set_trace()
            # the order of arguments are: grad_carry_out, grad_ys, init, xs,  all flattened
            grad_carry_out = flat_args[:num_grad_carry_out_args]
            # import pdb
            # pdb.set_trace()
            grad_ys = flat_args[num_grad_carry_out_args:num_grad_carry_out_args + num_grad_ys_args]
            # grad_ys = [x[0, :] for x in flat_args[num_grad_carry_out_args:num_grad_carry_out_args + num_grad_ys_args]]
            init = flat_args[
                   num_grad_carry_out_args + num_grad_ys_args:num_grad_carry_out_args + num_grad_ys_args + num_init_args]
            xs = flat_args[num_grad_carry_out_args + num_grad_ys_args + num_init_args:]
            # xs = [x[0, :] for x in xs]
            # grad_ys = flat_args[1][:num_grad_ys_args]
            # init = flat_args[1][num_grad_ys_args:num_grad_ys_args + num_init_args]
            # xs = flat_args[1][num_grad_ys_args + num_init_args:]
            # Expected BxF, 1xBxF
            grad_args = list(grad_carry_out) + list(grad_ys)

            # import pdb
            # pdb.set_trace()

            def fw_with_masks(*args):
                # import pdb
                # pdb.set_trace()
                # Expects: BxF, BxF
                flattened_out = f(*args)
                return flattened_out, [True if isinstance(ret, torch.Tensor) and ret.requires_grad else False for ret in
                                       flattened_out]

            joint = create_joint(fw_with_masks, aot_config=dummy_aot_config)
            # import pdb
            # pdb.set_trace()
            _, grads = joint(list(init) + list(xs),
                             [grad for grad in grad_args if grad is not None and grad.requires_grad])
            # import pdb
            # pdb.set_trace()

            # In order to keep map functional for backward graph,
            # we clone outputs that are aliasing inputs
            input_storage = {StorageWeakRef(arg._typed_storage()) for arg in flat_args if isinstance(arg, torch.Tensor)}

            def maybe_clone(t):
                if isinstance(t, torch.Tensor) and StorageWeakRef(t._typed_storage()) in input_storage:
                    return t.clone()
                return t

            # flat list of grad_carry + grad_xs
            return pytree.tree_map(maybe_clone, grads)

        # import pdb
        # pdb.set_trace()
        # Expects BxF, BxF, BxF, BxF
        print('Expected: BxF, BxF, BxF, BxF')
        print((example_grad_carry_out[0].shape, example_grad_ys[0].shape, example_init[0].shape, example_xs[0].shape))
        # bw_graph = make_fx(bw_f)(*example_grad_carry_out, *example_grad_ys, *example_init, *example_all_xs)
        bw_graph = make_fx(bw_f)(*example_grad_carry_out, *example_grad_ys, *example_init, *example_xs)
        return fw_graph, bw_graph

        # bw_graph = make_fx(f)(*example_grad_carry_out, *example_grad_ys)
        # return fw_graph, bw_graph


def scan_wrapper(f, init, xs):
    r"""
    Scan a function `f` over the leading array axes while carrying along a state.

    .. warning::
        `torch.cond` is currently a prototype feature in PyTorch. It uses the same interface as defined in https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html,
        but the flags `length`, `reverse` and `unroll` are not implemented yet. Please look forward to a more version in a future version of PyTorch.

    Args:
        f (Callable): A callable to be scanned of type c -> a -> (c, b), meaning that f accepts two arguments where the first is a value of the loop carry and the second is a slice of xs along its leading axis, and that f returns a pair where the first element represents a new value for the loop carry and the second represents a slice of the output.

        init (torch.Tensor): The initial loop carry value of type c, which can be a scalar, array, or any pytree (nested Python tuple/list/dict) thereof, representing the initial loop carry value. This value must have the same structure as the first element of the pair returned by f.

        xs (torch.Tensor): The value of type [a] over which to scan along the leading axis

    Example::

        def f(carry: torch.Tensor, x: torch.Tensor):
            return carry+1, x+carry

        init = torch.rand(1, 2)
        xs = torch.rand(10, 2)
        carry_out, ys = control_flow.scan(f, init, xs)

    Outputs:
        carry_out (torch.Tensor): The final loop carry when the function f has been scanned over the leading axis of xs

        ys (torch.Tensor): The final output when the function f has been scanned over the leading axis of xs

    """

    flat_init, init_spec = pytree.tree_flatten(init)
    if not all(isinstance(t, torch.Tensor) for t in flat_init):
        raise RuntimeError(f"Scanned init can only consist of tensors. Got init {flat_init}.")
    flat_xs, xs_spec = pytree.tree_flatten(xs)
    if not all(isinstance(t, torch.Tensor) for t in flat_xs):
        raise RuntimeError(f"Scanned xs can only consist of tensors. Got xs {flat_xs}.")

    # TODO: Introduce more shape checks
    # TODO: Is it necessary to restrict the shape of scanned or mapped elements?
    # for scan, the only requirement should be that f spits out a carry that
    # always has the same shape
    shapes = [xs.shape for xs in flat_xs]
    leading_dim_size = shapes[0][0]
    if leading_dim_size == 0:
        raise RuntimeError(
            "Leading dimensions of scanned xs cannot be 0.")

    if any(cur_shape[0] != leading_dim_size for cur_shape in shapes):
        raise RuntimeError(
            f"Leading dimensions of scanned xs must be consistent. Got shapes {shapes}.")

    carry_out_spec = None
    out_spec = None
    num_init_args = len(flat_init)

    def flat_fn(*flat_args):
        # import pdb
        # pdb.set_trace()
        # carry and init should have the same spec
        flat_init, flat_xs = flat_args[:num_init_args], flat_args[num_init_args:]
        # import pdb
        # pdb.set_trace()
        carry = pytree.tree_unflatten(flat_init, init_spec)
        xs = pytree.tree_unflatten(flat_xs, xs_spec)
        # import pdb
        # pdb.set_trace()
        unflattened_carry_out, unflattened_out = f(carry, xs)
        flat_carry_out, tmp_carry_out_spec = pytree.tree_flatten(unflattened_carry_out)
        flat_out, tmp_out_spec = pytree.tree_flatten(unflattened_out)

        nonlocal carry_out_spec
        nonlocal out_spec
        carry_out_spec = tmp_carry_out_spec
        out_spec = tmp_out_spec
        return *flat_carry_out, *flat_out

    # import pdb
    # pdb.set_trace()
    print((flat_init[0].shape, flat_xs[0].shape))
    flat_carry_out, flat_out = scan_impl(flat_fn, flat_init, flat_xs, reverse=False)  # (carry, ys)
    # import pdb
    # pdb.set_trace()
    return pytree.tree_unflatten(flat_carry_out, carry_out_spec), pytree.tree_unflatten(flat_out, out_spec)


class ScanAutogradOp(torch.autograd.Function):
    # Note: The scan operation when executed forward produces
    # scan_impl(f, init, xs, reverse=False) -> carry, ys
    #
    # When executed backward, the scan_impl is reused, but the inputs have a different meaning.
    # In particular, the init should be initialized as the carry gradient
    # the xs should contain the list of gradients of ys, i.e., the output gradients.
    # scan_impl(f, grad_carry, grad_ys, reverse=True) -> grad_init, grad_xs
    # Essentially, the scan_impl then works its way back through time
    # and computes the gradients for the init and the xs
    #
    # Example for a single iteration:
    # In the forward pass:
    # init \in \RR^{1, 2}
    # xs \in \RR^{1, 1, 2}
    # fw_graph(f, init, xs) -> carry, ys
    # carry \in \RR^{1, 2}
    # ys \in \RR^{1, 1, 2}
    #
    # In the backwards pass:
    # Compute gradients of carry and/or ys based on a loss function
    # E = MSE(target, ys) + carry -> grad_carry, grad_ys
    # grad_carry \in \RR^{1, 2}
    # grad_ys \in \RR^{1, 1, 2}
    # joint_graph(f, grad_carry, grad_ys, init, xs) -> grad_init, grad_xs
    # grad_init \in \RR^{1, 2}
    # grad_xs \in \RR^{1, 1, 2}
    #
    # Example for n iterations:
    # In the forward pass:
    # init \in \RR^{1, 2}
    # xs \in \RR^{n, 1, 2}
    # fw_graph(f, init, xs) -> carry, ys
    # carry \in \RR^{1, 2}
    # ys \in \RR^{n, 1, 2}
    #
    # In the backwards pass:
    # Compute gradients of carry and/or ys based on a loss function
    # E = MSE(target, ys) + carry -> grad_carry, grad_ys
    # grad_carry \in \RR^{1, 2}
    # grad_ys \in \RR^{n, 1, 2}
    # joint_graph(f, grad_carry, grad_ys, init, xs) -> grad_init, grad_xs
    # grad_init \in \RR^{1, 2}
    # grad_xs \in \RR^{n, 1, 2}

    @staticmethod
    def forward(ctx, fw_graph, bw_graph, num_init_args, *flat_args):
        # import pdb
        # pdb.set_trace()
        ctx._bw_graph = bw_graph
        ctx._num_input_args = len(flat_args)
        ctx._num_init_args = num_init_args
        ctx._leading_dim_size = flat_args[:num_init_args][0].shape[0]
        with torch._C._AutoDispatchBelowAutograd():
            # import pdb
            # pdb.set_trace()
            # print((flat_args[:num_init_args][0].shape, flat_args[num_init_args:][0].shape))
            flat_carries, flat_out = scan_impl(fw_graph, flat_args[:num_init_args], flat_args[num_init_args:],
                                               reverse=False)
            #import pdb
            #pdb.set_trace()
            try:
                leading_dim = flat_carries[0].shape[0]
                '''if type(flat_carries[0]) == tuple or type(flat_carries[0]) == list:
                    import pdb
                    pdb.set_trace()
                    leading_dim = flat_carries[0][0].shape[0]
                else:
                    import pdb
                    pdb.set_trace()
                    leading_dim = flat_carries[0].shape[0]'''
            except:
                print('Failed!')
                import pdb
                pdb.set_trace()

            # flat_carries = _unstack_pytree(flat_carries)

            # # needs to flat the carries nested list
            # # second call to save_for_backward will override whatever that is saved earlier
            # # therefore, need to save everything that is needed at once
            # flat_carries = [carry_entry for carry in flat_carries for carry_entry in carry]

            # flat_carries_chunk = flat_carries[:-len(flat_carries) // leading_dim]
            # flat_carries_out_chunk = flat_carries[-len(flat_carries) // leading_dim:]

            # flat_carries_chunk = [carry_entry for carry in flat_carries_chunk for carry_entry in carry]
            
            flat_carries_chunk = [carry[:-1, :] for carry in flat_carries]
            flat_carries_out_chunk = [carry[-1, :] for carry in flat_carries]
            
            ctx.save_for_backward(*flat_args, *flat_carries_chunk)
            ctx._num_out_args = len(flat_out)
            #import pdb
            #pdb.set_trace()
            return (*flat_carries_out_chunk, *flat_out)
            #return (*flat_out, *flat_out)

    @staticmethod
    def backward(ctx, *flat_grads):
        # import pdb
        # pdb.set_trace()
        flat_carries = ctx.saved_tensors[ctx._num_input_args:]
        fwd_args = ctx.saved_tensors[:ctx._num_input_args]
        # bring back the nested carries list by chunking up the list to equal lengthed nested lists
        # all carries should have length num_init_args
        # From 30x2 into 10x3x2; 20x1x2
        carries = [_stack_pytree(flat_carries[i * ctx._leading_dim_size:(i + 1) * ctx._leading_dim_size]) for i in
                   range(len(flat_carries) // ctx._leading_dim_size)]
        # From 20x1x2 into 2 10x1x2
        carries = [carries[i * ctx._num_init_args:(i + 1) * ctx._num_init_args] for i in
                   range(len(carries) // ctx._num_init_args)]

        flat_xs = fwd_args[ctx._num_init_args:]
        # carry and init should have the same spec
        final_carry_grad = flat_grads[:ctx._num_init_args]
        # ys should only be up to ctx._num_init_args + ctx._num_out_args, since the remaining should be grads for flat_carries
        # and should be empty
        ys_grad = flat_grads[ctx._num_init_args:ctx._num_init_args + ctx._num_out_args]
        with torch._C._AutoDispatchBelowAutograd():
            # import pdb
            # pdb.set_trace()
            flat_carry_out_grads, flat_out_grads = scan_impl(ctx._bw_graph, final_carry_grad,
                                                             (*ys_grad, carries, *flat_xs), reverse=True)
            # import pdb
            # pdb.set_trace()
            # flat_carry_out_grads, flat_out_grads = scan_impl(ctx._bw_graph, final_carry_grad, *ys_grad, reverse=True)
            return None, None, None, *[grad[0] for grad in flat_carry_out_grads], *flat_out_grads


def trace_scan(proxy_mode, func_overload, f, flat_init, flat_xs, reverse=False):
    pre_dispatch = getattr(proxy_mode, "pre_dispatch", False)

    # import pdb
    # pdb.set_trace()
    xs = flat_xs
    leading_dim = xs[0].shape[0]
    num_init = len(flat_init)
    example_input = _unstack_pytree(xs)
    body_graph = f
    out_carries = []
    out_pytrees = []
    direction = -1 if reverse else 1

    #import pdb
    #pdb.set_trace()
    with disable_proxy_modes_tracing():
        if not isinstance(body_graph, torch.fx.GraphModule):
            # FW Expects BxF, BxF
            # BW Expects BxF, BxF, BxF, BxF
            body_graph = make_fx(_maybe_run_with_interpreter(body_graph))(*flat_init, *flat_xs)

        # FW Expects BxF, BxF
        # BW Expects BxF, BxF, BxF, BxF
        # import pdb
        # pdb.set_trace()
        # carry = flat_init
        # for inp in _unstack_and_flatten_tensors_or_lists(flat_xs)[::direction]:
        #     #loop_values = [xs_el[::direction] for xs_el in flat_xs]
        #     #for inp_t in range(leading_dim):
        #     #inp = [lv[inp_t, :] for lv in loop_values]
        #     out_carries.append(carry)
        #     example_outs = body_graph(*carry, *inp)
        #     out_pytrees.append(example_outs[num_init:])
        #     carry = example_outs[:num_init]
        # out_carries.append(carry)
        # #import pdb
        # #pdb.set_trace()
        # ys = _stack_pytree(out_pytrees[::direction])
        # cs = _stack_pytree(out_carries[::direction])
        # cs = out_carries[::direction][-1]
        
        example_outs = body_graph(*flat_init, *flat_xs)
        expanded_carries_out, expanded_outs = (example_outs[:num_init], example_outs[num_init:])

    expanded_outs_comb = (expanded_carries_out, expanded_outs)
    #expanded_outs_comb = (cs, ys)

    #import pdb
    #pdb.set_trace()
    next_name = None
    i = 0
    while not next_name:
        candidate = f"body_graph_{i}"
        if hasattr(proxy_mode.tracer.root, candidate):
            i += 1
        else:
            next_name = candidate

    # import pdb
    # pdb.set_trace()
    proxy_mode.tracer.root.register_module(next_name, body_graph)
    #proxy_mode.tracer.root.register_module(next_name, f)
    #import pdb
    #pdb.set_trace()
    #carry = torch.empty(expanded_outs_comb[0][0].shape, device=expanded_outs_comb[0][0].device)
    #out = torch.empty(expanded_outs_comb[1][0].shape, device=expanded_outs_comb[1][0].device)
    #node_args = (body_graph, flat_init, flat_xs, carry, out)
    #node_args = (body_graph, flat_init, flat_xs)
    # import pdb
    # pdb.set_trace()
    
    # from torch._inductor.utils import run_and_get_code, run_and_get_triton_code
    # #ret = run_and_get_triton_code(args[0], args[1][0], args[2][0])
    # #ret = run_and_get_triton_code(f, *flat_init, *flat_xs)
    # def fn(c, x):
    #     return c+1, x+1
    # ret = run_and_get_triton_code(fn, torch.rand(1, 2, device='cuda'), torch.rand(10, 1, 2, device='cuda'))
    
    # import inspect
    # import dis
    # #fn_src = inspect.getsource(('def '.join(body_graph.print_readable().split('def')[1:])).strip())
    # code = dis.dis(f)
    # dis.compile(source, filename, mode, flags=0, dont_inherit=False, optimize=- 1)
    
    #node_args = (fn_src, flat_init, flat_xs)
    node_args = (body_graph, flat_init, flat_xs)
    proxy_args = pytree.tree_map(partial(unwrap_proxy, proxy_mode), node_args)
    out_proxy = proxy_mode.tracer.create_proxy('call_function', func_overload, proxy_args, {"reverse": reverse},
                                               name="scan_impl")
    
    #import pdb
    #pdb.set_trace()
    
    '''
    (Pdb) qualified_op_name
    'aten::_assert_async'
    (Pdb) op
    <built-in method _assert_async of PyCapsule object at 0x1473c0c75da0>
    (Pdb) namespace_name
    'aten'
    (Pdb) op_name
    '_assert_async'
    (Pdb) overload_names
    ['', 'msg']
    overload = OpOverload(self, op_, op_dk_, schema, []])
    torch.ops.scan_impl
    qualified_op_name, op_name, op, overload_names
    '''
    
    #import pdb
    #pdb.set_trace()
    #opoverloadpacket = torch._ops.OpOverloadPacket("torch.ops::scan_impl", "scan_impl", scan, [""])
    #schema = torch._C._get_schema("torch.ops::scan_impl", "")
    #opoverload = torch._ops.OpOverload(opoverloadpacket, scan, scan, schema, [])
    op_, op_dk_, tags = torch._C._get_operation_overload("aten::_assert_async", "msg")
    schema = torch._C._get_schema("aten::_assert_async", "msg")
    opoverloadpacket = torch._ops.OpOverloadPacket("aten::_assert_async", "_assert_async", torch.ops.aten._assert_async, ['', 'msg'])
    opoverload = torch._ops.OpOverload(opoverloadpacket, op_, op_dk_, schema, tags)
    
    out_proxy.node.meta['original_aten'] = opoverload
    # import pdb
    # pdb.set_trace()
    return track_tensor_tree(expanded_outs_comb, out_proxy, constant=None, tracer=proxy_mode.tracer)


def _unstack_and_flatten_tensors_or_lists(flat_xs):
    '''
        This function performs unstacking and flattening of a list of mixed items, where items can be of type
        Tensor, List or Tuple
        For example, with the following input:
        flat_xs = (
                    tensor([[0.8338, 0.8524],
                           [0.3481, 0.6204],
                           [0.9394, 0.1453]]),
                    tensor([[0.7662, 0.1674],
                            [0.9786, 0.1188],
                            [0.8857, 0.0750]]),
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                  )
        The output would be:
        output = [
                    [tensor([0.8338, 0.8524]), tensor([0.7662, 0.1674]), 1, 2, 3],
                    [tensor([0.3481, 0.6204]), tensor([0.9786, 0.1188]), 4, 5, 6],
                    [tensor([0.9394, 0.1453]), tensor([0.8857, 0.0750]), 7, 8, 9],
                 ]
    '''
    if not all(isinstance(xs, torch.Tensor) or isinstance(xs, tuple) or isinstance(xs, list) for xs in flat_xs):
        # import pdb
        # pdb.set_trace()
        raise RuntimeError(f"Leaves of xs must be Tensor or Tuple or List {flat_xs}")
    if not all(len(xs) == len(flat_xs[0]) for xs in flat_xs):
        # if not all(len(xs)==len(flat_xs[0]) for xs in flat_xs):
        # import pdb
        # pdb.set_trace()
        raise RuntimeError(f"Leaves of xs must have same leading dimension size {[len(xs) for xs in flat_xs]}")

    a = zip(*flat_xs)
    unstacked_results = []
    for entry in a:
        # flatten lists
        flat_entry = []
        for item in entry:
            if isinstance(item, tuple) or isinstance(item, list):
                flat_entry.extend(item)
            else:
                flat_entry.append(item)
        unstacked_results.append(flat_entry)
    return unstacked_results


# TODO: this can function can be shared with the torch._higher_order_ops.map._unstack_pytree from main
def _unstack_pytree(xs):
    flat_xs, inspec = pytree.tree_flatten(xs)
    if not all(isinstance(xs, torch.Tensor) for xs in flat_xs):
        raise RuntimeError(f"Leaves of xs must be Tensor {flat_xs}")

    if not all(xs.shape[0] == flat_xs[0].shape[0] for xs in flat_xs):
        raise RuntimeError(f"Leaves of xs must have same leading dimension size {[xs.shape for xs in flat_xs]}")

    a = zip(*flat_xs)
    pytrees = []
    for tuple in a:
        pytrees.append(pytree.tree_unflatten(tuple, inspec))
    return pytrees


# TODO: this can function can be shared with the torch._higher_order_ops.map._stack_pytree from main
def _stack_pytree(pytrees):
    flat_out = []
    out_spec = None
    for pt in pytrees:
        flat_pt, out_spec = pytree.tree_flatten(pt)
        flat_out.append(flat_pt)
    b = zip(*flat_out)
    stacked_out = []
    for leaves in b:
        if all(isinstance(leaf, torch.Tensor) for leaf in leaves):
            stacked_out.append(torch.stack(leaves))
        elif all(leaf is None for leaf in leaves):
            # Backward graph can return None output when forward inputs doesn't require grad.
            # When we eagerly execute backward graph, we need to call _stack_pytree on its output,
            # therefore we need to deal with None output.
            stacked_out.append(None)
        else:
            raise RuntimeError(f"Cannot stack {leaves}.")
    return pytree.tree_unflatten(stacked_out, out_spec)


@scan_impl.py_impl(DispatchKey.CompositeExplicitAutograd)
def scan_dense(f, flat_init, flat_xs, reverse=False):
    '''
        The scan_dense implementation is reused for executing both forward and backward graph.
        When executing forward graph, f is the fwd_graph, and flat_init, flat_xs are the typical flattened
        init and xs lists, reverse should be False (not exposing reverse argument at the top level yet).

        When executing backward graph, f is the bwd_graph, and flat_init should be the initial output carry gradient,
        flat_xs should be a concatenated list of ys_grad (the output gradient), carries(the list of all the intermediate carries generated during
        forward pass less the final output carry) and input xs.
        This list of carries are needed for computing the backward gradient at each iteration.
        When running bwd_graph with scan_dense, we should set reverse = True because the gradients are computed bottom up.
    '''
    # import pdb
    # pdb.set_trace()
    carry = flat_init
    carry_length = len(carry)
    # each carry output is needed for calculating gradients backwards
    out_carries = []
    out_pytrees = []
    direction = -1 if reverse else 1
    # print(flat_xs)
    # print(flat_xs[0].shape)
    # import pdb
    # pdb.set_trace()
    for inp in _unstack_and_flatten_tensors_or_lists(flat_xs)[::direction]:
        # saves the initial carry input for each iteration
        out_carries.append(carry)
        flattened_out = f(*carry, *inp)
        # try:
        #     print('--------------------------------------')
        #     print(carry)
        #     print(inp)
        #     flattened_out = f(*carry, *inp)
        #     print(flattened_out)
        #     print('======================================')
        #     #import pdb
        #     #pdb.set_trace()
        # except:
        #     print('Failed dense')
        #     import pdb
        #     pdb.set_trace()
        #     #flattened_out = f(*carry, *inp)
        # flattened_out_includes carry and output
        out_pytrees.append(flattened_out[carry_length:])
        carry = flattened_out[:carry_length]
    out_carries.append(carry)
    ys = _stack_pytree(out_pytrees[::direction])
    return _stack_pytree(out_carries[::direction]), ys


@scan_impl.py_impl(DispatchKey.Autograd)
def scan_autograd(f, flat_init, flat_xs, reverse=False):
    fw_graph, bw_graph = create_fw_bw_graph(f, flat_init, flat_xs)
    num_carry_out = len(flat_init)
    #import pdb
    #pdb.set_trace()
    flat_all_out = ScanAutogradOp.apply(fw_graph, bw_graph, num_carry_out, *flat_init, *flat_xs)
    #import pdb
    #pdb.set_trace()
    return flat_all_out[:num_carry_out], flat_all_out[num_carry_out:]


@scan_impl.py_impl(ProxyTorchDispatchMode)
def map_proxy_torch_dispatch_mode(f, flat_init, flat_xs, reverse=False):
    mode = _get_current_dispatch_mode()
    assert (mode is not None), "Mode should always be enabled for python fallback key"
    with _pop_mode_temporarily() as mode:
        if mode.enable_tracing:
            ret = trace_scan(mode, scan_impl, f, flat_init, flat_xs, reverse=reverse)
            #import pdb
            #pdb.set_trace()
            return ret
        else:
            ret = scan_impl(f, flat_init, flat_xs, reverse=reverse)
            import pdb
            pdb.set_trace()
            return ret


@scan_impl.py_impl(FakeTensorMode)
def map_fake_tensor_mode(f, flat_init, flat_xs, reverse=False):
    return scan_dense(f, flat_init, flat_xs, reverse=reverse)


# TODO: In the main branch this function is not needed anymore.
# Here, this is a duplicate of the function form functorch.experimental.control_flow._map
def _has_potential_branch_input_mutation(branch, inputs):
    """
    Dispatch-trace the branch with inputs and check if
    producing graph has mutable op on the input. This is
    bit restrictive as the branch must be traceable.
    """
    try:
        gm = make_fx(branch)(*inputs)
    except UnsupportedAliasMutationException:
        # this can happen when nested cond is
        # functionalized
        return True
    except Exception as e:
        raise e

    def _detect_input_mutation(gm):
        input_nodes = set()
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                input_nodes.add(node)
            if node.op == "call_function":
                target = node.target
                if isinstance(target, torch._ops.OpOverload) and target._schema.is_mutable:
                    for arg in node.args:
                        if arg in input_nodes:
                            return True

        for _, module in gm.named_children():
            if isinstance(module, torch.fx.GraphModule):
                if _detect_input_mutation(module):
                    return True

        return False

    return _detect_input_mutation(gm)


# TODO: In the main branch this function is not needed anymore.
# Here, this is a duplicate of the function form functorch.experimental.control_flow._map
def _has_potential_branch_input_alias(branch, inputs):
    """
    Dispatch-trace the branch with inputs and check if
    producing graph has output aliasing the branch input. This is
    bit restrictive as the branch must be traceable.
    """
    try:
        gm = make_fx(branch)(*inputs)

    except UnsupportedAliasMutationException:
        # this can happen when nested cond is
        # functionalized
        return True
    except Exception as e:
        raise e

    def _detect_input_alias(gm):
        input_storages = set()
        for node in gm.graph.nodes:
            # We need to check existence of "val" because we reuse the logic here
            # for map operator, where num_mapped_args is a scalar
            # and doesn't have a "val" meta.
            if node.op == "placeholder" and "val" in node.meta:
                input_storages.add(StorageWeakRef(node.meta['val']._typed_storage()))
            if node.op == "output":
                def check_alias(out):
                    if out is not None and "val" in out.meta:
                        out_storage = StorageWeakRef(out.meta['val']._typed_storage())
                        return out_storage in input_storages
                    return False

                if any(pytree.tree_flatten(pytree.tree_map(check_alias, node.args))[0]):
                    return True

        for _, module in gm.named_children():
            if isinstance(module, torch.fx.GraphModule) and _detect_input_alias(module):
                return True

        return False

    return _detect_input_alias(gm)


# TODO: In the main branch this function is not needed anymore.
# Here, this is a duplicate of the function form functorch.experimental.control_flow._map
@dataclass
class UnsupportedAliasMutationException(RuntimeError):
    reason: str


# Functions for the lowering
# Cond uses functionalization as well in the main branch of Pytorch. Consolidate this while merging
@scan_impl.py_impl(DispatchKey.Functionalize)
def scan_func(f, init, xs, reverse=False):
    reapply_views = torch._C._functionalization_reapply_views_tls()
    unwrapped_init = _unwrap_all_tensors_from_functional(init, reapply_views=reapply_views)
    unwrapped_xs = _unwrap_all_tensors_from_functional(xs, reapply_views=reapply_views)
    mode = 'mutations_and_views' if reapply_views else 'mutations'

    leading_dim = unwrapped_xs[0].shape[0]
    num_init = len(unwrapped_init)

    with _ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.Functionalize)):
        functional_map_fn = functionalize(f, remove=mode)
        # with disable_proxy_modes_tracing():
        #     example_inputs = (unwrapped_init[0], unwrapped_xs[0])

        # if _has_potential_branch_input_mutation(f, example_inputs):
        #     raise UnsupportedAliasMutationException(
        #         "torch._higher_order_ops.scan is mutating the input!"
        #     )

        # if _has_potential_branch_input_alias(f, example_inputs):
        #     raise UnsupportedAliasMutationException(
        #         "torch._higher_order_ops.scan is aliasing the input!"
        #     )

        # import pdb
        # pdb.set_trace()
        example_outs = scan_impl(functional_map_fn, unwrapped_init, unwrapped_xs, reverse=reverse)
        #example_outs = scan_impl(f, unwrapped_init, unwrapped_xs, reverse=reverse)
        # import pdb
        # pdb.set_trace()
        # TODO: This may need adjustment for different scenarios of the carry, e.g. nested lists
        expanded_carries_out, expanded_outs = (example_outs[:num_init], example_outs[num_init:])
        # flat_carries = _unstack_pytree(expanded_carries_out)
        flat_carries = [carry_entry for carry in expanded_carries_out for carry_entry in carry]
        # flat_carries_chunk = flat_carries[:-len(flat_carries)//leading_dim]
        # #flat_carries_out_chunk = flat_carries[-len(flat_carries)//leading_dim:]
        # flat_carries_chunk = [carry_entry for carry in flat_carries_chunk for carry_entry in carry]
        flat_carries_chunk = [carry_entry[-1, :] for carry_entry in flat_carries]

        # expanded_carries_out, expanded_outs = (example_outs[:num_init], example_outs[num_init:])
        # expanded_outs_comb = (expanded_carries_out[-1, :], expanded_outs)
        example_outs_new = (tuple(flat_carries_chunk), expanded_outs[0])
        # import pdb
        # pdb.set_trace()
        return _wrap_all_tensors_to_functional(example_outs, level=0)


# TODO: This is just a replication from the map functionalization and needs to be adapted
@scan_impl.py_impl(torch._C._functorch.TransformType.Functionalize)
def scan_functionalize(f, init, xs, reverse=False):
    """
    Functionalization implementation for torch.map. Currently:
      1. We don't allow any input mutation inside the map function
      2. Our check for above condition is not exhaustive
    """
    import pdb
    pdb.set_trace()
    xs = args[:num_mapped]
    pos_args = args[num_mapped:]
    reapply_views = interpreter.functionalize_add_back_views()
    mode = 'mutations_and_views' if reapply_views else 'mutations'
    # At this point, we will see functionalized tensors, so need to unwrap them first
    unwrapped_xs = _unwrap_all_tensors_from_functional(xs, reapply_views=reapply_views)
    unwrapped_args = _unwrap_all_tensors_from_functional(pos_args, reapply_views=reapply_views)

    functional_map_fn = functionalize(f, remove=mode)

    with interpreter.lower():
        with disable_proxy_modes_tracing():
            example_inputs = (*_unstack_pytree(unwrapped_xs)[0], *unwrapped_args)
        if _has_potential_branch_input_mutation(f, example_inputs):
            raise UnsupportedAliasMutationException(
                "torch.map is mutating the input!"
            )

        if _has_potential_branch_input_alias(f, example_inputs):
            raise UnsupportedAliasMutationException(
                "torch.map is aliasing the input!"
            )

        map_return = map_impl(functional_map_fn, num_mapped, *unwrapped_xs, *unwrapped_args)
        return _wrap_all_tensors_to_functional(map_return, level=interpreter.level())


# TODO(voz) Make this automatic for keys, this is very ugly atm
scan_impl.fallthrough(DispatchKey.PythonDispatcher)
scan_impl.fallthrough(DispatchKey.PythonTLSSnapshot)
scan_impl.fallthrough(DispatchKey.ADInplaceOrView)
scan_impl.fallthrough(DispatchKey.BackendSelect)
scan_impl.fallthrough(DispatchKey.AutocastCPU)