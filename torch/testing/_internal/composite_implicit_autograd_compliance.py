import torch
import logging
import contextlib
from typing import List, Iterator
from torch.utils._pytree import tree_map
from functools import partial

# TODO: move this into library proper
@contextlib.contextmanager
def no_dispatch() -> Iterator[None]:
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard

class CompositeImplicitAutogradCompliantError(Exception):
    pass

def check_metadata_consistency(wrapper_tensor, op_name):
    if not isinstance(wrapper_tensor, LoggingTensor):
        return
    elem = wrapper_tensor.elem
    if wrapper_tensor.shape != elem.shape:
        raise CompositeImplicitAutogradCompliantError(
            f"{op_name} is not CompositeImplicitAutograd compliant: the "
            f"shape of the tensor was modified directly without "
            f"going through the PyTorch dispatcher.")
    if wrapper_tensor.dtype != elem.dtype:
        raise CompositeImplicitAutogradCompliantError(
            f"{op_name} is not CompositeImplicitAutograd compliant: the "
            f"dtype of the tensor was modified directly without "
            f"going through the PyTorch dispatcher.")

def is_view_fn(func):
    return func.__name__ in {
        'as_strided',
        'detach',
        'diagonal',
        'expand',
        'expand_as',
        'movedim',
        'narrow',
        'permute',
        'select',
        'squeeze',
        'transpose',
        't',
        'real',
        'imag',
        'view_as_real',
        'view_as_complex',
        'unflatten',
        'unfold',
        'unsqueeze',
        'view',
        'view_as',
        'unbind',
        'split',
        'split_with_sizes',
        'chunk',
        'swapaxes',
    }

# How the chain of calls works for LoggingTensor:
# 1. Call torch.sin
# 2. Attempt __torch_function__. In LoggingTensor torch function is disabled so we bypass it entirely
# 3. Enter dispatcher, wind your way through Autograd
# 4. Hit Python dispatch key, call __torch_dispatch__

# TODO: TensorBase should work
class LoggingTensor(torch.Tensor):
    elem: torch.Tensor

    __slots__ = ['elem']

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        # The wrapping tensor (LoggingTensor) is just a meta tensor, so it
        # doesn't hold any memory (meta tensor is generally the preferred type
        # of tensor you want to make a subclass from)...
        r = torch.Tensor._make_subclass(cls, elem.to('meta'), elem.requires_grad)
        # ...the real tensor is held as an element on the tensor.
        r.elem = elem
        return r

    def __repr__(self):
        return f"LoggingTensor()"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(e):
            return e.elem if isinstance(e, LoggingTensor) else e

        def wrap(e):
            return LoggingTensor(e) if isinstance(e, torch.Tensor) else e

        with no_dispatch():
            unwrapped_args = tree_map(unwrap, args)
            unwrapped_kwargs = tree_map(unwrap, kwargs)
            rs = tree_map(wrap, func(*unwrapped_args, **unwrapped_kwargs))

        if is_view_fn(func):
            # Create meta tensor using storage
            aliased_tensor = args[0]
            with no_dispatch():
                x = torch.empty(aliased_tensor.shape, dtype=aliased_tensor.dtype,
                                device=aliased_tensor.device)
                x.set_(aliased_tensor)
                args_with_x = list(args)
                args_with_x[0] = x
                result = func(*args_with_x, **kwargs)
                if isinstance(result, tuple) or isinstance(result, list):
                    for a, b in zip(rs, result):
                        a.set_(b)
                else:
                    rs.set_(result)
                    assert torch._C._is_alias_of(rs, args[0])
        with no_dispatch():
            logging.getLogger("LoggingTensor").info(f"{func.__module__}.{func.__name__}", args, rs)

        # Keep the metadata in sync for these operations (are there more?)
        with no_dispatch():
            if func.__name__ == 'squeeze_':
                args[0].set_(rs)
            if func.__name__ == 'unsqueeze_':
                args[0].set_(rs)
            if func.__name__ == 'transpose_':
                args[0].set_(rs)
            if func.__name__ == 't_':
                args[0].set_(rs)
            if func.__name__ in ('set_', 'resize_'):
                raise CompositeImplicitAutogradCompliantError(
                    f"{func.__name__} is not allowed to be called inside of "
                    f"CompositeImplicitAutograd operators.")

        check = partial(check_metadata_consistency, op_name=func.__name__)

        tree_map(check, args)
        tree_map(check, kwargs)
        tree_map(check, rs)
        return rs

# https://stackoverflow.com/questions/36408496/python-logging-handler-to-append-to-list
class LoggingTensorHandler(logging.Handler):
    log_list: List[str]
    next_shortid: int

    def __init__(self, log_list: List[str]) -> None:
        logging.Handler.__init__(self)
        self.log_list = log_list
        self.next_shortid = 0

    # WARNING: not deterministic over multiple threads, this matters for
    # autograd
    def _shortid(self, o: object) -> int:
        if not hasattr(o, '_shortid'):
            o._shortid = self.next_shortid
            self.next_shortid += 1
        return o._shortid

    def _fmt(self, a: object) -> str:
        return f'${self._shortid(a)}' if isinstance(a, LoggingTensor) else repr(a)

    def emit(self, record):
        fmt_args = "(" + ", ".join(self._fmt(a) for a in record.args[0]) + ")"
        fmt_rets = ", ".join(self._fmt(a) for a in record.args[1]) \
            if isinstance(record.args[1], (list, tuple)) else self._fmt(record.args[1])
        self.log_list.append(f'{fmt_rets} = {record.msg}{fmt_args}')

def log_input(name: str, var: object):
    logging.getLogger("LoggingTensor").info("input", (name,), (var,))

@contextlib.contextmanager
def capture_logs() -> Iterator[List[str]]:
    logger = logging.getLogger("LoggingTensor")
    log_list = []
    handler = LoggingTensorHandler(log_list)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield log_list
    finally:
        logger.removeHandler(handler)

def _check_CompositeImplicitAutograd_compliance(op, args, kwargs):
    def unwrap(e):
        return e.elem if isinstance(e, LoggingTensor) else e

    def wrap(e):
        return LoggingTensor(e) if isinstance(e, torch.Tensor) else e

    def mark_input(e):
        if not isinstance(e, LoggingTensor):
            return e
        log_input('a', e)
        return e

    try:
        x = torch.empty([])
        rep = LoggingTensor(x)
        torch._C._python_mode_set_torch_dispatch(rep)
        try:
            with capture_logs() as logs:
                args = tree_map(wrap, args)
                kwargs = tree_map(wrap, kwargs)
                op(*args, **kwargs)
        finally:
            torch._C._python_mode_reset_torch_dispatch()

        code = '\n'.join(logs)
        assert 'torch._ops.resize_(' not in code
    except RuntimeError as err:
        if 'The tensor has a non-zero number of elements' in err.args[0]:
            raise CompositeImplicitAutogradCompliantError(
                f'{op} or a CompositeImplicitAutograd operation it calls is '
                f'directly accessing the data_ptr in its implementation. '
                f'This is not allowed in operators marked as '
                f'CompositeImplicitAutograd.')
        raise
        

if __name__ == '__main__':
    rep =  LoggingTensor(torch.tensor([[[1, 0], [0, 1]], [[1., 0], [0, 1]]], requires_grad=True))
    x0 = torch.tensor([[0.1459, 2.0105], [1.3002, 1.8342]], requires_grad=True)
    y0 = torch.tensor([[0.1583, 0.8006], [0.1518, 0.5681]], requires_grad=True)
    # torch._C._python_mode_set_torch_dispatch(rep)
    with capture_logs() as logs:
        x = LoggingTensor(torch.tensor([[[1, 0], [0, 1]], [[1., 0], [0, 1]]], requires_grad=True))
        y = LoggingTensor(torch.tensor([[[1, 0], [0, 1]], [[1., 0], [0, 1]]], requires_grad=True))
        # log_input("x", x)
        # print(y.shape)
        # y.squeeze(0)
        # print(y.shape)
        # print(y.elem.shape)
        y = torch.linalg.cholesky_ex(x)
        # y = torch.linalg.matrix_norm(x)
        z = y + y

    print('\n'.join(logs))

    _check_CompositeImplicitAutograd_compliance(torch.sin, (torch.randn(5),), {})
    _check_CompositeImplicitAutograd_compliance(torch.float_power, (x0, y0), {})
    _check_CompositeImplicitAutograd_compliance(torch.signbit, (x0,), {})

