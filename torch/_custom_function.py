import torch
from torch._ops import PyOperator
from torch._C._functorch import TransformType
from functorch._src.vmap import stupid_vmap, unwrap_batched, wrap_batched
from functorch import vmap, grad
import functools
import torch.utils._pytree as pytree
from torch._C import (
    DispatchKey,
)



custom_vjp_call = PyOperator('custom_vjp_call')
custom_vjp_call.fallthrough(DispatchKey.PythonTLSSnapshot)

def unwrap_grad(level, t):
    if isinstance(t, torch.Tensor):
        return torch._C._functorch._unwrap_for_grad(t, level)
    return t


def wrap_grad(level, t):
    if isinstance(t, torch.Tensor):
        return torch._C._functorch._wrap_for_grad(t, level)
    return t


def index_of(lst, tensor):
    for idx, t in enumerate(lst):
        if tensor is t:
            return idx
    return None


def wrap_outs_and_saved(
        unwrapped_outs,
        unwrapped_saved,
        inputs,
        unwrapped_inputs,
        level):
    outs = pytree.tree_map(functools.partial(wrap_grad, level), unwrapped_outs)

    saved = []
    for s in unwrapped_saved:
        idx = index_of(unwrapped_inputs, s)
        if idx is not None:
            saved.append(inputs[idx])
            continue
        idx = index_of(unwrapped_outs, s)
        if idx is not None:
            saved.append(outs[idx])
            continue
        saved.append(wrap_grad(level, s))
    return outs, saved


def custom_vjp_call_grad_generic(maybe_interpreter, f_fwd, f_bwd, *operands):
    class Generated(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *operands):
            if maybe_interpreter:
                level = maybe_interpreter.level()
                unwrapped_operands = pytree.tree_map(functools.partial(unwrap_grad, level), operands)

                with torch.enable_grad(), maybe_interpreter.lower():
                    output = custom_vjp_call(f_fwd, f_bwd, *unwrapped_operands)
                    results, saved = output

                results, out_spec = pytree.tree_flatten(results)
                outs, saved = wrap_outs_and_saved(results, saved, operands, unwrapped_operands, level)
                outs = pytree.tree_unflatten(outs, out_spec)
                ctx.save_for_backward(*saved)
            else:
                outs, saved = f_fwd(*operands)
                # TODO: the user may save things that aren't Tensors
                ctx.save_for_backward(*saved)

            flat_outs, outs_spec = pytree.tree_flatten(outs)
            outs_and_saved_and_spec = flat_outs + [saved, outs_spec]
            return tuple(outs_and_saved_and_spec)

        @staticmethod
        def backward(ctx, *grads):
            # Accounts for saved and spec
            assert grads[-1] is None
            assert grads[-2] is None
            grads = grads[:-2]

            saved = ctx.saved_tensors
            result = f_bwd(saved, grads)
            return result

    outs_and_saved_and_spec = Generated.apply(*operands)
    out_spec = outs_and_saved_and_spec[-1]
    saved = outs_and_saved_and_spec[-2]
    flat_outs = outs_and_saved_and_spec[:-2]
    return pytree.tree_unflatten(flat_outs, out_spec), saved


@custom_vjp_call.py_impl(TransformType.Grad)
def custom_vjp_call_grad(interpreter, f_fwd, f_bwd, *operands):
    return custom_vjp_call_grad_generic(interpreter, f_fwd, f_bwd, *operands)


# TODO: registering to 'Autograd' doesn't work (alias keys don't work with py_impl)
@custom_vjp_call.py_impl(DispatchKey.AutogradCPU)
def custom_vjp_call_autograd(f_fwd, f_bwd, *operands):
    return custom_vjp_call_grad_generic(None, f_fwd, f_bwd, *operands)


def reductify_leaf(tensor, tensor_bdim, desired_bdim):
    if tensor_bdim is None and desired_bdim is None:
        return tensor
    if tensor_bdim is None and desired_bdim is not None:
        raise RuntimeError('NYI: A')
    if tensor_bdim is not None and desired_bdim is None:
        return tensor.sum(tensor_bdim)
    return tensor.movedim(tensor_bdim, desired_bdim)


def reductify(tensors, tensor_bdims, desired_bdims):
    tensors, spec = pytree.tree_flatten(tensors)
    tensor_bdims, _ = pytree.tree_flatten(tensor_bdims)
    desired_bdims, _ = pytree.tree_flatten(desired_bdims)

    result = [reductify_leaf(tensor, bdim, desired_bdim)
              for tensor, bdim, desired_bdim
              in zip(tensors, tensor_bdims, desired_bdims)]
    return pytree.tree_unflatten(result, spec)


def batchify(f_fwd, f_bwd, in_dims, batch_size):
    out_dims = None

    def new_f_fwd(*args):
        nonlocal out_dims
        outs, out_dims2 = stupid_vmap(f_fwd, in_dims, batch_size)(*args)
        out_dims = out_dims2
        return outs

    def new_f_bwd(grad_outs, saved):
        assert out_dims is not None
        grad_ins, grad_ins_dims = stupid_vmap(f_bwd, out_dims, batch_size)(grad_outs, saved)
        return reductify(grad_ins, grad_ins_dims, in_dims)

    def get_out_dims():
        assert out_dims is not None
        return out_dims

    return new_f_fwd, new_f_bwd, get_out_dims


@custom_vjp_call.py_impl(TransformType.Vmap)
def custom_vjp_call_vmap(interpreter, f_fwd, f_bwd, *operands):
    current_level = interpreter.level()
    unwrapped_operands, in_dims = unwrap_batched(operands)
    new_f_fwd, new_f_bwd, get_out_dims = batchify(f_fwd, f_bwd, in_dims, interpreter.batch_size())

    with interpreter.lower():
        result = custom_vjp_call(new_f_fwd, new_f_bwd, *unwrapped_operands)

    out_dims = get_out_dims()
    return wrap_batched(current_level, result, out_dims)


class CustomVjp:
    # TODO: support kwargs (or not)
    @classmethod
    def apply(cls, *args):
        outs, saved = custom_vjp_call(cls.forward, cls.backward, *args)
        return outs

# TODO: somehow we need to raise an error if
# (1) intermediate tensors are being saved
# (2) user is computing higher order gradients. e.g. grad(grad(...
class MockCtx:
    def __init__(self):
        self.saved_tensors = []
        self.saved_values = {}

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

    def __setattr__(self, name, value):
        if name in ('saved_tensors', 'saved_values'):
            return super().__setattr__(name, value)
        self.saved_values[name] = value

def to_custom_vjp(af):
    class Generated(CustomVjp):
        @staticmethod
        def forward(*args):
            ctx = MockCtx()
            output = af.forward(ctx, *args)
            return output, ctx.saved_tensors

        @staticmethod
        def backward(saved, grads):
            ctx = MockCtx()
            ctx.saved_tensors = saved
            return af.backward(ctx, *grads)

    return Generated


# This is autograd.Function that has a vmap_rule
custom_function_call = PyOperator('custom_function_call')
custom_function_call.fallthrough(DispatchKey.PythonTLSSnapshot)


# autograd.Function translation:
# custom_function(f_fwd, f_bwd, f_vmap, *operands) -> Output
#
# Output object:
# - outputs
# - saved_tensors
# - saved_values (non-tensors)
# flatten()
# unflatten()
#
# asdf

def pyreturn_flatten(output):
    if isinstance(output, tuple):
        return list(output), len(output)
    return [output], None


def pyreturn_unflatten(spec, output):
    if spec is None:
        assert len(output) == 1
        return output[0]
    assert spec == len(output)
    return tuple(output)


class OutputAndCtx:
    def __init__(self, output, saved_tensors, saved_values):
        self.output = output
        self.saved_tensors = saved_tensors
        self.saved_values = saved_values

    def flatten(self):
        flat_output, output_spec = pyreturn_flatten(self.output)
        result = [output_spec, self.saved_tensors, self.saved_values] + flat_output
        return tuple(result)

    @staticmethod
    def unflatten(flattened_list):
        output_spec, saved_tensors, saved_values, *remainder = flattened_list
        output = pyreturn_unflatten(output_spec, remainder)
        return OutputAndCtx(output, saved_tensors, saved_values)

    @staticmethod
    def get_relevant_grads(output_spec, grads):
        assert isinstance(grad, tuple)
        return grads[3:]


# TODO: registering to 'Autograd' doesn't work (alias keys don't work with py_impl)
@custom_function_call.py_impl(DispatchKey.AutogradCPU)
def custom_function_call_autograd(f_fwd, f_bwd, f_vmap, *operands):
    class Generated(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *operands):
            output_and_ctx = f_fwd(*operands)
            ctx.save_for_backward(*output_and_ctx.saved_tensors)
            ctx.saved_values = output_and_ctx.saved_values
            return output_and_ctx.flatten()

        @staticmethod
        def backward(ctx, _0, _1, _2, *grads):
            result = f_bwd(ctx.saved_tensors, ctx.saved_values, grads)
            return result

    flat_out = Generated.apply(*operands)
    return OutputAndCtx.unflatten(flat_out)


@custom_function_call.py_impl(TransformType.Grad)
def custom_function_call_grad(interpreter, f_fwd, f_bwd, f_vmap, *operands):
    print(f'grad {interpreter.level()}')
    maybe_interpreter = interpreter
    level = maybe_interpreter.level()

    class Generated(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *operands):
            print("generated forward")
            unwrapped_operands = pytree.tree_map(functools.partial(unwrap_grad, level), operands)

            with torch.enable_grad(), maybe_interpreter.lower():
                output_and_ctx = custom_function_call(f_fwd, f_bwd, f_vmap, *unwrapped_operands)

            flat_output, output_spec = pyreturn_flatten(output_and_ctx.output)
            flat_output, saved_tensors = wrap_outs_and_saved(
                flat_output, output_and_ctx.saved_tensors,
                operands, unwrapped_operands, level)

            ctx.save_for_backward(*saved_tensors)
            ctx.saved_values = output_and_ctx.saved_values

            output_and_ctx.output = pyreturn_unflatten(output_spec, flat_output)
            output_and_ctx.saved_tensors = saved_tensors
            result = output_and_ctx.flatten()
            return result

        @staticmethod
        def backward(ctx, _0, _1, _2, *grads):
            print("generated backward")
            grads = pytree.tree_map(functools.partial(unwrap_grad, level), grads)
            result = f_bwd(ctx.saved_tensors, ctx.saved_values, grads)
            return result

    flat_out = Generated.apply(*operands)
    return OutputAndCtx.unflatten(flat_out)


@custom_function_call.py_impl(TransformType.Vmap)
def custom_function_call_vmap(interpreter, f_fwd, f_bwd, f_vmap, *operands):
    print(f'vmap {interpreter.level()}')
    current_level = interpreter.level()
    unwrapped_operands, in_dims = unwrap_batched(operands)
    o, spec = pytree.tree_flatten(unwrapped_operands)
    i, _ = pytree.tree_flatten(in_dims)
    k = [(oo, ii) for oo, ii in zip(o, i)]
    operands = pytree.tree_unflatten(k, spec)

    with interpreter.lower():
        output_and_ctx = f_vmap(*operands)

    # TODO: people need to specify bdims for tensors that are saved :/
    # I don't think this is going to work out
    out = output_and_ctx.output
    saved_values = output_and_ctx.saved_values
    assert len(output_and_ctx.saved_tensors) == 0

    if len(out) == 2 and not isinstance(out[1], tuple):
        # single output
        return OutputAndCtx(wrap_batched(current_level, out[0], out[1]), [], saved_values)

    outs, outs_bdims = zip(*out)
    return OutputAndCtx(wrap_batched(current_level, outs, outs_bdims), [], saved_values)


class MockCtx2:
    pass


def to_custom_function(af):
    def f_fwd(*args):
        ctx = MockCtx()
        output = af.forward(ctx, *args)
        return OutputAndCtx(output, ctx.saved_tensors, ctx.saved_values)

    def f_bwd(saved_tensors, saved_values, grads):
        ctx = MockCtx2()
        for k, v in saved_values.items():
            setattr(ctx, k, v)
        ctx.saved_tensors = saved_tensors
        return af.backward(ctx, *grads)

    def f_vmap(*args):
        ctx = MockCtx()
        output = af.vmap_rule(ctx, *args)
        return OutputAndCtx(output, ctx.saved_tensors, ctx.saved_values)

    def blah(*args):
        output_and_ctx = custom_function_call(f_fwd, f_bwd, f_vmap, *args)
        return output_and_ctx

    return blah
