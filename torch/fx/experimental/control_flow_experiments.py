import torch
import torch.fx
from torch.fx.node import Argument, Target
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Union

class IfStmtModule(torch.nn.Module):
    def __new__(cls: 'Type[IfStmtModule]', *args, **kwargs):
        class IfStmtImpl(cls):  # type: ignore
            pass
        return super().__new__(IfStmtImpl)

    def __init__(self, true_block : Union[Callable, torch.nn.Module],
                 false_block : Union[Callable, torch.nn.Module]):
        super().__init__()
        self.true_block_gm = torch.fx.symbolic_trace(true_block)
        self.false_block_gm = torch.fx.symbolic_trace(false_block)

        true_placeholders = [n for n in self.true_block_gm.graph.nodes if n.op == 'placeholder']
        false_placeholders = [n for n in self.false_block_gm.graph.nodes if n.op == 'placeholder']

        if len(true_placeholders) != len(false_placeholders):
            raise RuntimeError('True and False blocks must have same number of inputs!')

        arg_names : List[str] = []
        for t_ph, f_ph in zip(true_placeholders, false_placeholders):
            arg_names.append(t_ph.target)
        args = ', '.join(arg_names)

        forward_code = f"""
def forward(self, cond, {args}):
    if cond:
        return self.true_block_gm({args})
    else:
        return self.false_block_gm({args})
"""

        gbls = {}
        exec(forward_code, gbls)
        self.__class__.forward = gbls['forward']

def if_stmt(self, cond, true_block, false_block, *args) -> Any:
    i = 0
    while hasattr(self, f'_if_stmt_{i}'):
        i += 1

    # FIXME: will make a new module every time this is called in forward()
    setattr(self, f'_if_stmt_{i}', IfStmtModule(true_block, false_block))

    return getattr(self, f'_if_stmt_{i}')(cond, *args)

class WhileStmtModule(torch.nn.Module):
    def __new__(cls: 'Type[WhileStmtModule]', *args, **kwargs):
        class WhileStmtImpl(cls):  # type: ignore
            pass
        return super().__new__(WhileStmtImpl)

    def __init__(self, body : Union[Callable, torch.nn.Module],):
        super().__init__()
        self.body_gm = torch.fx.symbolic_trace(body)

        body_placeholders = [n for n in self.body_gm.graph.nodes if n.op == 'placeholder']

        arg_names : List[str] = []
        for b_ph in body_placeholders:
            arg_names.append(b_ph.target)
        args = ', '.join(arg_names)

        forward_code = f"""
def forward(self, cond, {args}):
    while cond:
        cond, {args} = self.body_gm({args})
    return {args}
"""

        print(forward_code)

        gbls = {}
        exec(forward_code, gbls)
        self.__class__.forward = gbls['forward']

def while_loop(self, cond, body_block, *args) -> Tuple[Any, ...]:
    i = 0
    while hasattr(self, f'_while_stmt_{i}'):
        i += 1

    # FIXME: will make a new module every time this is called in forward()
    setattr(self, f'_while_stmt_{i}', WhileStmtModule(body_block))

    return getattr(self, f'_while_stmt_{i}')(cond, *args)

class ShapePropRecord:
    def __init__(self, size, dtype):
        self.size = size
        self.dtype = dtype

    def __str__(self):
        return f'ShapePropRecord({self.size}, {self.dtype})'

    def __repr__(self):
        return str(self)

    size : torch.Size
    dtype : torch.dtype

class ShapePropagation(torch.fx.Interpreter):
    """
    TODO:
    1. Switch ShapePropRecord to contain sets of values
    2. Implement joins in control flow operators
    """
    def placeholder(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        val = super().placeholder(target, args, kwargs)
        return torch.fx.node.map_aggregate(val, lambda a: ShapePropRecord(a.shape, a.dtype) if isinstance(a, torch.Tensor) else a)

    def get_attr(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        attr_val = super().get_attr(target, args, kwargs)
        attr_val = torch.fx.node.map_aggregate(attr_val, lambda a: ShapePropRecord(a.shape, a.dtype) if isinstance(a, torch.Tensor) else a)
        return attr_val

    def call_function(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        def fake_val(a):
            if isinstance(a, ShapePropRecord):
                return torch.randn(a.size).to(a.dtype)
            return a
        processed_args = torch.fx.node.map_aggregate(args, fake_val)
        processed_kwargs = torch.fx.node.map_aggregate(kwargs, fake_val)
        real_call = super().call_function(target, processed_args, processed_kwargs)
        return torch.fx.node.map_aggregate(real_call, lambda a: ShapePropRecord(a.shape, a.dtype) if isinstance(a, torch.Tensor) else a)

    def call_method(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        def fake_val(a):
            if isinstance(a, ShapePropRecord):
                return torch.randn(a.size).to(a.dtype)
            return a
        processed_args = torch.fx.node.map_aggregate(args, fake_val)
        processed_kwargs = torch.fx.node.map_aggregate(kwargs, fake_val)
        real_call = super().call_method(target, processed_args, processed_kwargs)
        return torch.fx.node.map_aggregate(real_call, lambda a: ShapePropRecord(a.shape, a.dtype) if isinstance(a, torch.Tensor) else a)

    def call_module(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any]) -> Any:
        def fake_val(a):
            if isinstance(a, ShapePropRecord):
                return torch.randn(a.size).to(a.dtype)
            return a
        processed_args = torch.fx.node.map_aggregate(args, fake_val)
        processed_kwargs = torch.fx.node.map_aggregate(kwargs, fake_val)
        real_call = super().call_module(target, processed_args, processed_kwargs)
        return torch.fx.node.map_aggregate(real_call, lambda a: ShapePropRecord(a.shape, a.dtype) if isinstance(a, torch.Tensor) else a)

if __name__ == '__main__':
    def true(x, y):
        return x + y

    def false(x, y, z):
        return x - y

    # if_mod = IfStmtModule(true, false)

    def false(x, y):
        return x - y

    if_mod = IfStmtModule(true, false)

    x, y = torch.randn(3, 4), torch.randn(3, 4)
    torch.testing.assert_allclose(if_mod(True, x, y), x + y)
    torch.testing.assert_allclose(if_mod(False, x, y), x - y)


    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # FIXME: fix handling of submodule calls. Reentrant tracing?
            self.lin = torch.nn.Linear(4, 4)

        def forward(self, x, y):
            cond = x.sum() > 0
            # Would make these lambdas but we can't symtrace those
            def true(x, y):
                return x + y
            def false(x, y):
                return x - y
            return if_stmt(self, cond, true, false, x, y)

    mm = MyModule()
    x = torch.ones(3, 4)
    y = torch.randn(3, 4)

    torch.testing.assert_allclose(mm(x, y), x + y)

    x = torch.ones(3, 4) * -1
    torch.testing.assert_allclose(mm(x, y), x - y)

    class ControlFlowTracer(torch.fx.Tracer):
        def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str) -> bool:
            return m.__class__.__name__ == 'IfStmtImpl' or m.__class__.__name__ == 'WhileStmtImpl' \
                    or super().is_leaf_module(m, module_qualified_name)

    traced = ControlFlowTracer().trace(mm)
    print(traced)

    def body(i, x, y):
        x = x + 1
        y = y - 1
        i = i + 1
        return i <= 10, i, x, y

    while_mod = WhileStmtModule(body)

    x, y = torch.randn(1), torch.randn(1)

    x_ref = x.clone()
    y_ref = y.clone()
    i = 0
    while i <= 10:
        x_ref += 1
        y_ref -= 1
        i += 1

    i = 0
    _, x_test, y_test = while_mod(i <= 10, i, x, y)
    torch.testing.assert_allclose(x_test, x_ref)
    torch.testing.assert_allclose(y_test, y_ref)

    class MyModule(torch.nn.Module):
        def forward(self, x, y):
            i = 0
            cond = i <= 10
            # Would make these lambdas but we can't symtrace those
            def body(i, x, y):
                x = x + 1
                y = y - 1
                i = i + 1
                return i <= 10, i, x, y
            return while_loop(self, cond, body, i, x, y)

    mm = MyModule()
    x = torch.randn(3, 4)
    y = torch.randn(3, 4)

    x_ref = x.clone()
    y_ref = y.clone()
    i = 0
    while i <= 10:
        x_ref += 1
        y_ref -= 1
        i += 1

    _, x, y = mm(x, y)
    torch.testing.assert_allclose(x, x_ref)
    torch.testing.assert_allclose(y, y_ref)

    traced = ControlFlowTracer().trace(mm)
    print(traced)

    class MyModule(torch.nn.Module):
        def forward(self, x, y):
            cond = x.sum() > 0
            # Would make these lambdas but we can't symtrace those
            def true(x, y):
                return x + y
            def false(x, y):
                return x - y
            x = if_stmt(self, cond, true, false, x, y)

            i = 0
            cond = i <= 10
            # Would make these lambdas but we can't symtrace those
            def body(i, x, y):
                x = x + 1
                y = y - 1
                i = i + 1
                return i <= 10, i, x, y
            return while_loop(self, cond, body, i, x, y)

    mm = MyModule()
    x = torch.ones(3, 4)
    y = torch.randn(3, 4)

    x_ref = x.clone()
    y_ref = y.clone()

    x_ref = x_ref + y_ref if x_ref.sum() > 0 else x_ref - y_ref

    i = 0
    while i <= 10:
        x_ref += 1
        y_ref -= 1
        i += 1

    _, x, y = mm(x, y)
    torch.testing.assert_allclose(x, x_ref)
    torch.testing.assert_allclose(y, y_ref)

    traced = ControlFlowTracer().trace(mm)
    print(traced)

    traced_gm = torch.fx.GraphModule(mm, traced)
    shape_prop = ShapePropagation(traced_gm)
    print(shape_prop.run(x, y))
