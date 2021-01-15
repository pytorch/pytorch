"""Default set of benchmarks."""
import functools

from core.api import Setup, TimerArgs, GroupedTimerArgs
from core.types import FlatIntermediateDefinition
from core.utils import flatten, iter_parsed_lines, parse_stmts
from worker.main import CostEstimate


# Convenience methods for small, simple benchmarks to reduce boilerplate.
_small_trivial = functools.partial(
    GroupedTimerArgs,
    setup=Setup.TRIVIAL,
    cost=CostEstimate.LESS_THAN_10_US)

_small_generic = functools.partial(
    GroupedTimerArgs,
    setup=Setup.GENERIC,
    cost=CostEstimate.LESS_THAN_10_US)


BENCHMARKS: FlatIntermediateDefinition = flatten({
    "empty": {
        "no allocation": GroupedTimerArgs(
            r"torch.empty(())",
            r"torch::empty({0});",
            Setup.NONE,
            cost=CostEstimate.LESS_THAN_10_US,
        ),

        "with allocation": GroupedTimerArgs(
            r"torch.empty((1,))",
            r"torch::empty({1});",
            Setup.NONE,
            cost=CostEstimate.LESS_THAN_10_US,
        ),
    },

    ("Pointwise", "Math"): {
        "add": {
            "Tensor-Scalar": _small_generic(
                r"x += 1.0",
                r"x += 1.0;",
            ),

            "Tensor-Tensor": _small_generic(
                r"x += y_float",
                r"x += y_float;",
                signature=r"f(x, y_float) -> None",
            ),

            "Tensor-Tensor (type promotion)": _small_generic(
                r"x += y_int",
                r"x += y_int;",
            ),

            "Tensor-Tensor (out of place)": _small_generic(
                r"x + y_float",
                r"x + y_float;",
            ),
        },

        "multiply": _small_generic(
            r"x * y_float",
            r"x * y_float;",
        ),

        "equality": {
            "Tensor-Tensor": _small_generic(
                r"x == y_float",
                r"x == y_float;",
            ),

            "Tensor-Scalar": _small_generic(
                r"x == 1.0",
                r"x == 1.0;",
            ),
        },
    },

    ("Pointwise", "Data movement"): {
        "contiguous (trivial)": _small_trivial(
            r"x.contiguous()",
            r"x.contiguous();",
        ),

        "contiguous (non-trivial)": _small_trivial(
            r"x.t().contiguous()",
            r"x.t().contiguous();",
        ),

        "clone": _small_trivial(
            r"x.clone()",
            r"x.clone();",
        ),

        "copy_": _small_generic(
            r"x.copy_(y_float)",
            r"x.copy_(y_float);",
        ),

        "zero_": _small_trivial(
            r"x.zero_()",
            r"x.zero_();",
        ),

        "RNG": _small_trivial(
            r"x.uniform_()",
            r"x.uniform_();",
        ),
    },

    "Reduction": {
        "Max": _small_generic(
            r"x.max()",
            r"x.max();"
        ),

        "Sum": _small_generic(
            r"x.sum()",
            r"x.sum();",
        ),

        "Variance": _small_generic(
            r"x.var(0)",
            r"x.var(0);",
        ),
    },

    "Indexing": {
        py_index: GroupedTimerArgs(
            f"""
                x{py_index} = 1
                x{py_index} = y{py_index}
            """,
            f"""
                x.index_put_({cpp_index}, 1);
                x.index_put_({cpp_index}, y.index({cpp_index}));
            """,
            Setup.INDEXING,
            cost=CostEstimate.LESS_THAN_50_US,
        )

        for py_index, cpp_index in iter_parsed_lines(r"""
            Python                                   | C++
            ---------------------------------------- | ----------------------------------------
            [0]                                      | {0}
            [0, 0]                                   | {0, 0}
            [0, 0, 0]                                | {0, 0, 0}
            [...]                                    | {"..."}
            [:]                                      | {Slice(None, None, None)}
            [None]                                   | {None}
            [False]                                  | {false}
            [True]                                   | {true}
        """)
    },

    ("Indexing", "Tensor index"): GroupedTimerArgs(
        r"a[b, None, ...] = 1",
        r"a.index_put_({b, None, Ellipsis}, 1);",
        Setup.INDEXING,
        signature=r"f(a, b) -> None",
        cost=CostEstimate.LESS_THAN_50_US,
    ),

    "Metadata and views": {
        "size": _small_trivial(
            r"x.size()[0]",
            r"x.sizes()[0];",
        ),

        "stride": _small_trivial(
            r"x.stride(0)",
            r"x.stride(0);",
        ),

        "as_strided": _small_generic(
            r"torch.as_strided(x, (2, 3), (4, 1), 2)",
            r"torch::as_strided(x, {2, 3}, {4, 1}, 2);",
        ),

        "select": _small_trivial(
            r"x.select(1, 1)",
            r"x.select(1, 1);",
        ),

        "unsqueeze": _small_trivial(
            r"x.unsqueeze(0)",
            r"x.unsqueeze(0);",
        ),

        "view": _small_trivial(
            r"x.view(-1, 1)",
            r"x.view({-1, 1});",
        ),
    },

    "Misc": {
        "resize_": _small_trivial(
            r"""
                x.resize_(0)
                x.resize_((4, 4))
            """,
            r"""
                x.resize_(0);
                x.resize_({4, 4});
            """,
        ),

        "Sort": GroupedTimerArgs(
            r"torch.sort(x)",
            r"torch::sort(x);",
            Setup.GENERIC,
            cost=CostEstimate.LESS_THAN_50_US,
        ),
    },

    "MatMul": {
        "Broadcasting (matmul)": GroupedTimerArgs(
            r"z = torch.matmul(x, y_float)",
            r"auto z = torch::matmul(x, y_float);",
            Setup.GENERIC,
            cost=CostEstimate.LESS_THAN_50_US,
        ),

        "Non-broadcasting (mm)": _small_generic(
            r"z = torch.mm(x, y_float)",
            r"auto z = torch::mm(x, y_float);",
        ),
    },

    "Mesoscale": {
        "MatMul-Bias-ReLU": GroupedTimerArgs(
            *parse_stmts(r"""
                Python                                   | C++
                ---------------------------------------- | ----------------------------------------
                z0 = torch.mm(x, y) + bias               | auto z0 = torch::mm(x, y) + bias;
                z1 = torch.nn.functional.relu(z0)        | auto z1 = torch::nn::functional::relu(z0);
            """),
            Setup.MESOSCALE,
            signature=r"f(x, y, bias) -> z1",
            cost=CostEstimate.LESS_THAN_50_US,
        ),

        "Off diagonal indices": GroupedTimerArgs(
            *parse_stmts(r"""
                Python                                   | C++
                ---------------------------------------- | ----------------------------------------
                indices = torch.arange(eye_4.numel())    | auto indices = torch::arange(eye_4.numel());
                z = indices[eye_4.flatten() == 0]        | auto z = indices.index({eye_4.flatten() == 0});
            """),
            Setup.MESOSCALE,
            signature=r"f(eye_4) -> z",
            cost=CostEstimate.LESS_THAN_100_US,
        ),
    },

    "AutoGrad": {
        "simple": GroupedTimerArgs(
            *parse_stmts(r"""
                Python                                   | C++
                ---------------------------------------- | ----------------------------------------
                a0 = torch.nn.functional.relu(x * w0)    | auto a0 = torch::nn::functional::relu(x * w0);
                y = a0 * w1                              | auto y = a0 * w1;
                y.backward()                             | y.backward();
            """),
            Setup.AUTOGRAD,
            num_threads=(1, 2),
            signature=r"f(x, w0, w1) -> None",
            cost=CostEstimate.LESS_THAN_250_US,
        ),

        "intermediate": GroupedTimerArgs(
            *parse_stmts(r"""
                Python                                   | C++
                ---------------------------------------- | ----------------------------------------
                a0 = torch.nn.functional.gelu(x * w0)    | auto a0 = torch::nn::functional::gelu(x * w0);
                a1 = torch.nn.functional.prelu(y, w1)    | auto a1 = torch::nn::functional::prelu(y, w1);
                z = torch.nn.functional.normalize(       | auto z = torch::nn::functional::normalize(
                    torch.cat([a0, a1]),                 |     torch::cat({a0, a1}),
                    p=2.0, dim=0,                        |     torch::nn::functional::NormalizeFuncOptions().p(2).dim(0)
                ).dot(w2)                                | ).dot(w2);
                z.backward()                             | z.backward();
            """),
            Setup.AUTOGRAD,
            num_threads=(1, 2),
            signature=r"f(x, y, w0, w1, w2) -> None",
            cost=CostEstimate.LESS_THAN_1000_US,
        ),
    },
})
