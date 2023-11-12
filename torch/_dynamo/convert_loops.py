"""
Set of general utilities to convert a loop into a function call.

For example, the following loop:

```
    for i in range(3):
        x[i] = y
        y = y + y

```

is converted to

```
def __loop_functionalized__9(y, x):

    def loop_body(y, x, i):
        x[i] = y
        y = y + y
        return (y,)
    i = 0
    y, = loop_body(y=y, x=x, i=i)
    i = 1
    y, = loop_body(y=y, x=x, i=i)
    i = 2
    y, = loop_body(y=y, x=x, i=i)
    return (y, x)
```

This opens up opportunities to speed up loop compilation in dynamo.
Especially since loop_body compilation can be cached.
"""
import ast
import inspect
import types
import weakref
from typing import Any, Dict, List, MutableMapping, Optional, Set, Tuple

from .exc import TorchDynamoException


class CannotConvertLoop(TorchDynamoException):
    pass


class CollectLoadsAndStore(ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.loads: Set[str] = set()
        self.stores: Set[str] = set()
        self.id = 0

    def visit_Many(self, nodes: list[Any]) -> Any:
        for node in nodes:
            self.visit(node)

    def visit_Name(self, node) -> Any:
        if isinstance(node.ctx, ast.Load):
            self.loads.add(node.id)
            self.id += 1
        else:
            assert isinstance(node.ctx, ast.Store)
            self.stores.add(node.id)
            self.id += 1


class FunctionalizeLoops(ast.NodeVisitor):
    def __init__(
        self,
        range_start: int,
        range_stop: int,
        range_step: int,
        loop_lineno: int,
    ) -> None:
        super().__init__()
        self.range_start = range_start
        self.range_stop = range_stop
        self.range_step = range_step
        self.loop_lineno = loop_lineno
        self.detected_loop = False
        self.for_loop: Optional[List[Any]] = None
        self.outer_parameters: Optional[Set[str]] = None

    def visit_For(self, node):
        if node.lineno != self.loop_lineno:
            # print(node.lineno, self.loop_lineno)
            return
        assert not self.detected_loop
        self.detected_loop = True
        target = node.target
        body = node.body

        # First, convert loop body to a pure function.

        # All used locals shall become argument values.
        # Including the for loop target, which shall be assigned a special name.
        collect_loads_and_store = CollectLoadsAndStore()
        collect_loads_and_store.visit_Many(body)
        parameters = collect_loads_and_store.loads
        self.outer_parameters = parameters.copy()
        # The name binding of the for loop should not be registered
        # as a parameter.
        if isinstance(target, ast.Name):
            self.outer_parameters.remove(target.id)
        # print(parameters)
        # print(collect_loads_and_store.loads)

        # All assignment shall become return values.
        retvals = collect_loads_and_store.stores

        # Construct a pure function
        # Note: we should construct the AST directly here instead of using
        # ast.unparse, for speed reasons and compatibility with Python 3.8.
        # However, this means that all the line numbers and column offsets will be wrong
        # unless we fix those manually.
        # The following is the equivalent of
        #         fun =\
        # f"""
        # def loop_body({', '.join(parameters)}):
        #     {ast.unparse(body)}
        #     return ({', '.join(retvals)},)
        # """
        fun = ast.FunctionDef(
            name="loop_body",
            args=ast.arguments(
                posonlyargs=[],
                defaults=[],
                kwonlyargs=[],
                kw_defaults=[],
                args=[
                    ast.arg(
                        arg=name,
                    )
                    for name in parameters
                ],
            ),
            body=body
            + [
                ast.Return(
                    value=ast.Tuple(
                        ctx=ast.Load(),
                        elts=[
                            ast.Name(
                                id=name,
                                ctx=ast.Load(),
                            )
                            for name in retvals
                        ],
                    )
                )
            ],
            decorator_list=[],
        )
        # print(ast.unparse(fun))
        # Then, convert all the loop iterations to function calls
        # With the target being assigned before each iteration
        # This creates stuff in the form
        # target = val
        # target = func_body(arg1=arg1, arg2=arg2)
        # and so on...
        new_body = []

        for i in range(self.range_start, self.range_stop, self.range_step):
            assgn = ast.Assign(targets=[target], value=ast.Constant(value=i))
            call = ast.Assign(
                targets=[
                    ast.Tuple(
                        elts=[ast.Name(id=name, ctx=ast.Store()) for name in retvals],
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Call(
                    func=ast.Name(id="loop_body", ctx=ast.Load()),
                    args=[],
                    keywords=[
                        ast.keyword(arg=name, value=ast.Name(id=name, ctx=ast.Load()))
                        for name in parameters
                    ],
                ),
            )
            new_body.append(assgn)
            new_body.append(call)

        self.for_loop = [fun] + new_body


parse_cache: MutableMapping[types.CodeType, ast.Module] = weakref.WeakKeyDictionary()


def parse(fun):
    try:
        if (res := parse_cache.get(fun)) is not None:
            return res
        val = ast.parse(inspect.getsource(fun))
        parse_cache[fun] = val
        return val
    except OSError:
        raise CannotConvertLoop("Loop translation cannot find source")
    except SyntaxError:
        raise CannotConvertLoop("Can't parse the loop source code")


def functionalize_loop_body(
    fun: types.CodeType,
    new_wrapper_name: str,
    range: range,
    loop_lineno: int,
) -> Tuple[types.CodeType, Set[str]]:
    if not loop_lineno:
        raise CannotConvertLoop("No associated lineno")

    tree = parse(fun)
    # print(inspect.getsourcelines(fun))
    # Because inspect.getsource returns wrong line numbers,
    # we need to adjust the line number to that and hope it finds it
    inspected_startlineno = tree.body[0].lineno
    real_startlineno = fun.co_firstlineno
    loop_lineno += (inspected_startlineno - real_startlineno) - 1
    try:
        transformed = FunctionalizeLoops(
            range.start, range.stop, range.step, loop_lineno
        )
        transformed.visit(tree)
        stmts = transformed.for_loop
        if stmts is None:
            raise CannotConvertLoop("Erronouesly returned none when converting loop")
        if transformed.outer_parameters is None:
            raise CannotConvertLoop("Outer parameteres cannot be None.")
        stmts += [
            ast.Return(
                value=ast.Tuple(
                    ctx=ast.Load(),
                    elts=[
                        ast.Name(
                            id=name,
                            ctx=ast.Load(),
                        )
                        for name in transformed.outer_parameters
                    ],
                )
            )
        ]
        # Wrap the statements in a dummy function
        final = ast.Module(
            body=[
                ast.FunctionDef(
                    name=new_wrapper_name,
                    args=ast.arguments(
                        posonlyargs=[],
                        defaults=[],
                        kwonlyargs=[],
                        kw_defaults=[],
                        args=[
                            ast.arg(
                                arg=name,
                            )
                            for name in transformed.outer_parameters
                        ],
                    ),
                    body=stmts,
                    decorator_list=[],
                ),
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(final)
        # print(ast.unparse(final))
        res = compile(final, filename="<string>", mode="exec")
        env: Dict[str, Any] = {}
        exec(res, env)
        # print(dir(res))
        # print(res.co_varnames)
        fun = env[new_wrapper_name]
        assert isinstance(fun, types.FunctionType)
        return (fun.__code__, transformed.outer_parameters)
    except SyntaxError:
        raise CannotConvertLoop(
            "SyntaxError in transformed loop. This is a bug. Please submit a report."
        )
