# mypy: ignore-errors

import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing

class ErrorTensor(torch.Tensor):
    t: torch.Tensor
    error: float
    __slots__ = ["t", "error"]

    @staticmethod
    def __new__(cls, t, error):
        # shape = outer_size
        # kwargs = {}
        # kwargs["strides"] = outer_stride
        # kwargs["storage_offset"] = t.storage_offset()
        # kwargs["device"] = t.device
        # kwargs["layout"] = t.layout
        # kwargs["requires_grad"] = t.requires_grad
        # kwargs["dtype"] = t.dtype
        # return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)
        if not t.is_floating_point():
            breakpoint()
        assert t.is_floating_point()
        return torch.Tensor._make_subclass(cls, t)

    def __init__(self, t, error):
        self.t = t
        self.error = error

    def __repr__(self):
        t_repr = repr(self.t)
        error_repr = repr(self.error)
        return f"ErrorTensor({t_repr}, error={error_repr})"

    def __tensor_flatten__(self):
        return ["t"], (self.error,)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        t = inner_tensors["t"]
        error, = meta
        return ErrorTensor(t, error, outer_size, outer_stride)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        args_t = pytree.tree_map_only(ErrorTensor, lambda x: x.t, args)
        kwargs_t = pytree.tree_map_only(ErrorTensor, lambda x: x.t, kwargs)

        out_t = func(*args_t, **kwargs_t)
        out_t_flat, spec = pytree.tree_flatten(out_t)

        in_errs = []
        import itertools
        for l in itertools.chain(pytree.tree_leaves(args), pytree.tree_leaves(kwargs)):
            if isinstance(l, ErrorTensor):
                in_errs.append(l.error)

        # out_t - true output
        print(f"XXX ErrorTensor {func} errors:{in_errs}")
        if func == torch.ops.aten.cudnn_batch_norm.default:
            out_t_max_error = [in_errs[0]]
            print(f"XXX ErrorTensor -> batch_norm errors:{out_t_max_error}")
            out_flat = []
            for o_t, err in zip(out_t_flat, out_t_max_error):
                if err is not None:
                    out_flat.append(cls(o_t, err))
                else:
                    out_flat.append(o_t)
            breakpoint()
            out = pytree.tree_unflatten(out_flat, spec)
            from torch._higher_order_ops.cond import cond_op

            if func is cond_op:
                return out
            else:
                return return_and_correct_aliasing(func, args, kwargs, out)


        N = 5
        out_t_e_flats = []
        for i in range(N):
            def _clone_add_error(t, error):
                assert t.is_floating_point()
                _t = t.detach().clone()
                import random
                r = random.uniform(0, 1)
                _t = _t + error * (2 * r - 1)
                return _t

            args_t_e = pytree.tree_map(
                lambda t: _clone_add_error(t.t, t.error) if isinstance(t, ErrorTensor) else t, 
                args
            ) 
            kwargs_t_e = pytree.tree_map(
                lambda t: _clone_add_error(t.t, t.error) if isinstance(t, ErrorTensor) else t, 
                kwargs
            )

            out_t_e = func(*args_t_e, **kwargs_t_e)
            out_t_e_flat, _ = pytree.tree_flatten(out_t_e)

            out_t_e_flats.append(out_t_e_flat)

        out_t_max_error = []
        for out_idx, o_t in enumerate(out_t_flat):
            if not isinstance(o_t, torch.Tensor) or not o_t.is_floating_point() or o_t.numel() == 0:
                out_t_max_error.append(None)

            o_t_es = [otef[out_idx] for otef in out_t_e_flats]
            # o_t - true value
            # o_t_es - distribution of +error
            
            _max_error = 0.0
            for e in o_t_es:
                delta = torch.abs(e - o_t)
                val = 0.0
                if delta.numel() > 0:
                    val = delta.max().item()   
                _max_error = max(_max_error, val)
            out_t_max_error.append(_max_error)
            
        print(f"XXX ErrorTensor -> errors:{out_t_max_error}")
        out_flat = []
        for o_t, err in zip(out_t_flat, out_t_max_error):
            if err is not None:
                out_flat.append(cls(o_t, err))
            else:
                out_flat.append(o_t)
        out = pytree.tree_unflatten(out_flat, spec)
        from torch._higher_order_ops.cond import cond_op

        if func is cond_op:
            return out
        else:
            return return_and_correct_aliasing(func, args, kwargs, out)

    def get_elem_a(self):
        return self.a
