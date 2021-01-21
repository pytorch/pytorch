"""Default set of benchmarks."""
import functools

from core.api_impl import GroupedModules, GroupedStmts
from core.types import FlatIntermediateDefinition
from core.utils import flatten, iter_parsed_lines, parse_stmts
from definitions.setup import Setup


# Convenience methods for small, simple benchmarks to reduce boilerplate.
TrivialSetup_GroupedStmts = functools.partial(
    GroupedStmts,
    setup=Setup.TRIVIAL_2D.value,
)

GenericSetup_GroupedStmts = functools.partial(
    GroupedStmts,
    setup=Setup.GENERIC.value,
)


BENCHMARKS: FlatIntermediateDefinition = flatten({
    "empty": {
        "no allocation": GroupedStmts(
            r"torch.empty(())",
            r"torch::empty({0});",
        ),

        "with allocation": GroupedStmts(
            r"torch.empty((1,))",
            r"torch::empty({1});",
        ),
    },

    ("Pointwise", "Math"): {
        "add": {
            "Tensor-Scalar": GenericSetup_GroupedStmts(
                r"x += 1.0",
                r"x += 1.0;",
            ),

            "Tensor-Tensor": GenericSetup_GroupedStmts(
                r"x += y_float",
                r"x += y_float;",
                signature=r"f(x, y_float) -> None",
                torchscript=True,
            ),

            "Tensor-Tensor (type promotion)": GenericSetup_GroupedStmts(
                r"x += y_int",
                r"x += y_int;",
            ),

            "Tensor-Tensor (out of place)": GenericSetup_GroupedStmts(
                r"x + y_float",
                r"x + y_float;",
            ),
        },

        "multiply": GenericSetup_GroupedStmts(
            r"x * y_float",
            r"x * y_float;",
        ),

        "equality": {
            "Tensor-Tensor": GenericSetup_GroupedStmts(
                r"x == y_float",
                r"x == y_float;",
            ),

            "Tensor-Scalar": GenericSetup_GroupedStmts(
                r"x == 1.0",
                r"x == 1.0;",
            ),
        },
    },

    ("Pointwise", "Data movement"): {
        "contiguous (trivial)": TrivialSetup_GroupedStmts(
            r"x.contiguous()",
            r"x.contiguous();",
        ),

        "contiguous (non-trivial)": TrivialSetup_GroupedStmts(
            r"x.t().contiguous()",
            r"x.t().contiguous();",
        ),

        "clone": TrivialSetup_GroupedStmts(
            r"x.clone()",
            r"x.clone();",
        ),

        "copy_": GenericSetup_GroupedStmts(
            r"x.copy_(y_float)",
            r"x.copy_(y_float);",
        ),

        "zero_": TrivialSetup_GroupedStmts(
            r"x.zero_()",
            r"x.zero_();",
        ),

        "RNG": TrivialSetup_GroupedStmts(
            r"x.uniform_()",
            r"x.uniform_();",
        ),
    },

    "Reduction": {
        "Max": GenericSetup_GroupedStmts(
            r"x.max()",
            r"x.max();"
        ),

        "Sum": GenericSetup_GroupedStmts(
            r"x.sum()",
            r"x.sum();",
        ),

        "Variance": GenericSetup_GroupedStmts(
            r"x.var(0)",
            r"x.var(0);",
        ),
    },

    "Indexing": {
        py_index: GroupedStmts(
            f"""
                x{py_index} = 1
                x{py_index} = y{py_index}
            """,
            f"""
                x.index_put_({cpp_index}, 1);
                x.index_put_({cpp_index}, y.index({cpp_index}));
            """,
            Setup.INDEXING.value,
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

    ("Indexing", "Tensor index"): GroupedStmts(
        r"a[b, None, ...] = 1",
        r"a.index_put_({b, None, Ellipsis}, 1);",
        Setup.INDEXING.value,
        signature=r"f(a, b) -> None",
        torchscript=True,
    ),

    "Metadata and views": {
        "size": TrivialSetup_GroupedStmts(
            r"x.size()[0]",
            r"x.sizes()[0];",
        ),

        "stride": TrivialSetup_GroupedStmts(
            r"x.stride(0)",
            r"x.stride(0);",
        ),

        "as_strided": GenericSetup_GroupedStmts(
            r"torch.as_strided(x, (2, 3), (4, 1), 2)",
            r"torch::as_strided(x, {2, 3}, {4, 1}, 2);",
        ),

        "select": TrivialSetup_GroupedStmts(
            r"x.select(1, 1)",
            r"x.select(1, 1);",
        ),

        "unsqueeze": TrivialSetup_GroupedStmts(
            r"x.unsqueeze(0)",
            r"x.unsqueeze(0);",
        ),

        "view": TrivialSetup_GroupedStmts(
            r"x.view(-1, 1)",
            r"x.view({-1, 1});",
        ),
    },

    "Misc": {
        "resize_": TrivialSetup_GroupedStmts(
            r"""
                x.resize_(0)
                x.resize_((4, 4))
            """,
            r"""
                x.resize_(0);
                x.resize_({4, 4});
            """,
        ),

        "Sort": GenericSetup_GroupedStmts(
            r"torch.sort(x)",
            r"torch::sort(x);",
        ),
    },

    "MatMul": {
        "Broadcasting (matmul)": GenericSetup_GroupedStmts(
            r"z = torch.matmul(x, y_float)",
            r"auto z = torch::matmul(x, y_float);",
        ),

        "Non-broadcasting (mm)": GenericSetup_GroupedStmts(
            r"z = torch.mm(x, y_float)",
            r"auto z = torch::mm(x, y_float);",
        ),
    },

    "nn Modules": {
        py_constructor.split("(")[0]: GroupedModules(
            f"model = torch.nn.{py_constructor}",
            f"auto model = torch::nn::{cpp_constructor};",
            setup=setup.value,
            signature="f(x) -> y",
            torchscript=torchscript,
        )

        for setup, torchscript, (py_constructor, cpp_constructor) in (
            (Setup.TRIVIAL_4D, True, ("BatchNorm2d(4)",) * 2),
            (Setup.TRIVIAL_4D, True, ("GroupNorm(2, 4)",) * 2),
            (Setup.TRIVIAL_4D, True, (
                "LayerNorm(4)",
                "LayerNorm(torch::nn::LayerNormOptions({4}))"
            )),
            (Setup.TRIVIAL_3D, True, ("Conv1d(4, 4, 1)",) * 2),
            (Setup.TRIVIAL_4D, True, ("Conv2d(4, 4, 1)",) * 2),
            (Setup.TRIVIAL_4D, True, ("MaxPool2d(2)",) * 2),
            (Setup.TRIVIAL_2D, True, ("ReLU()",) * 2),
            (Setup.TRIVIAL_2D, True, ("Sigmoid()",) * 2),
            (Setup.TRIVIAL_4D, True, ("Linear(4, 2)",) * 2),

            # TODO: LSTM can't be TorchScript'd
            (Setup.TRIVIAL_3D, False, ("LSTM(4, 2)",) * 2),
        )
    },

    "Mesoscale": {
        "MatMul-Bias-ReLU": GroupedStmts(
            *parse_stmts(r"""
                Python                                   | C++
                ---------------------------------------- | ----------------------------------------
                z0 = torch.mm(x, y) + bias               | auto z0 = torch::mm(x, y) + bias;
                z1 = torch.nn.functional.relu(z0)        | auto z1 = torch::nn::functional::relu(z0);
            """),
            Setup.MESOSCALE.value,
            signature=r"f(x, y, bias) -> z1",
            torchscript=True,
        ),

        "Off diagonal indices": GroupedStmts(
            *parse_stmts(r"""
                Python                                   | C++
                ---------------------------------------- | ----------------------------------------
                indices = torch.arange(eye_4.numel())    | auto indices = torch::arange(eye_4.numel());
                z = indices[eye_4.flatten() == 0]        | auto z = indices.index({eye_4.flatten() == 0});
            """),
            Setup.MESOSCALE.value,
            signature=r"f(eye_4) -> z",
            torchscript=True,
        ),
    },

    "training": {
        "simple": GroupedStmts(
            *parse_stmts(r"""
                Python                                   | C++
                ---------------------------------------- | ----------------------------------------
                a0 = torch.nn.functional.relu(x * w0)    | auto a0 = torch::nn::functional::relu(x * w0);
                y = a0 * w1                              | auto y = a0 * w1;
            """),
            Setup.TRAINING.value,
            num_threads=(1, 2),
            signature=r"f(x, w0, w1) -> y",
            torchscript=True,
            autograd=True,
        ),

        "ensemble": GroupedStmts(
            *parse_stmts(r"""
                Python                                   | C++
                ---------------------------------------- | ----------------------------------------
                a0 = torch.nn.functional.gelu(x * w0)    | auto a0 = torch::nn::functional::gelu(x * w0);
                a1 = torch.nn.functional.prelu(y, w1)    | auto a1 = torch::nn::functional::prelu(y, w1);
                z = torch.nn.functional.normalize(       | auto z = torch::nn::functional::normalize(
                    torch.cat([a0, a1]),                 |     torch::cat({a0, a1}),
                    p=2.0, dim=0,                        |     torch::nn::functional::NormalizeFuncOptions().p(2).dim(0)
                ).dot(w2)                                | ).dot(w2);
            """),
            Setup.TRAINING.value,
            num_threads=(1, 2),
            signature=r"f(x, y, w0, w1, w2) -> z",
            torchscript=True,
            autograd=True,
        ),
    },
})
