"""Default set of benchmarks.

Parser notes:
    `parse_stmts`:
        - Width for the left (Python) column MUST be 40 characters.
        - The column separator is " | ", not "|". Whitespace matters.

    `GroupedVariants`:
        - `Setup` and `Global_Setup` (case insensitive) are reserved keywords
          to populate `setup` and `global_setup` for every generated benchmark.
        - To set a label for the succeeding block, add `# @YOUR_LABEL` (Python)
          or `// @YOUR_LABEL` (C++).
"""

from core.api import GroupedModules, GroupedStmts, GroupedVariants
from core.types import FlatIntermediateDefinition
from core.utils import flatten, parse_stmts
from definitions.setup import Setup


BENCHMARKS: FlatIntermediateDefinition = flatten({
    "Empty": {
        "no allocation": GroupedStmts(
            r"torch.empty(())",
            r"torch::empty({0});",
        ),

        "with allocation": GroupedStmts(
            r"torch.empty((1,))",
            r"torch::empty({1});",
        ),

        "overloads": GroupedVariants(
            cpp_block=r"""
                // @Setup
                auto options_empty = c10::TensorOptions();
                auto options_full = c10::TensorOptions().dtype(at::kFloat).device(at::kCPU);
                auto optional_float = c10::make_optional(at::kFloat);

                // @TensorOptions overload
                at::empty({0}, options_empty);
                at::empty({0}, options_full);
                at::empty({0}, at::kFloat); // implicit conversion

                // @Faithful overload
                at::empty({0}, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);
                at::empty({0}, at::kFloat, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);
                at::empty({0}, optional_float, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);
            """
        ),
    },

    "Pointwise": {
        "Math": {
            "add": {
                "Tensor-Scalar": GroupedStmts(
                    r"x += 1.0",
                    r"x += 1.0;",
                    setup=Setup.GENERIC.value,
                ),
            },
        },
    },

    "Indexing": GroupedVariants(*parse_stmts(r"""
        Python                                   | C++
        ---------------------------------------- | ----------------------------------------
        # @setup                                 | // @setup
                                                 | using namespace torch::indexing;
        torch.manual_seed(6626_10_34)            | torch::manual_seed(66261034);
                                                 |
        x = torch.randn(1, 1, 1)                 | auto x = torch::randn({1, 1, 1});
        y = torch.randn(1, 1, 1)                 | auto y = torch::randn({1, 1, 1});
                                                 |
        # @Tensor-Scalar                         | // @Tensor-Scalar
        x[0] = 1                                 | x.index_put_({0}, 1);
        x[0, 0] = 1                              | x.index_put_({0, 0}, 1);
        x[0, 0, 0] = 1                           | x.index_put_({0, 0, 0}, 1);
                                                 |
        # @Tensor-Scalar (Advanced)              | // @Tensor-Scalar (Advanced)
        x[...] = 1                               | x.index_put_({"..."}, 1);
        x[:] = 1                                 | x.index_put_({Slice(None, None, None)}, 1);
        x[None] = 1                              | x.index_put_({None}, 1);
        x[False] = 1                             | x.index_put_({false}, 1);
        x[True] = 1                              | x.index_put_({true}, 1);
                                                 |
        # @Tensor-Tensor                         | // @Tensor-Tensor
        x[0] = y[0]                              | x.index_put_({0}, y.index({0}));
        x[0, 0] = y[0, 0]                        | x.index_put_({0, 0}, y.index({0, 0}));
        x[0, 0, 0] = y[0, 0, 0]                  | x.index_put_({0, 0, 0}, y.index({0, 0, 0}));
                                                 |
        # @Tensor-Tensor (Advanced)              | // @Tensor-Tensor (Advanced)
        x[...] = y[...]                          | x.index_put_({"..."}, y.index({"..."}));
        x[:] = y[:]                              | x.index_put_({Slice(None, None, None)}, y.index({Slice(None, None, None)}));
        x[None] = y[None]                        | x.index_put_({None}, y.index({None}));
        x[False] = y[False]                      | x.index_put_({false}, y.index({false}));
        x[True] = y[True]                        | x.index_put_({true}, y.index({true}));
    """)),

    "nn Modules": {
        "Linear": GroupedModules(
            "model = torch.nn.Linear(4, 2)",
            "auto model = torch::nn::Linear(4, 2);",
            setup=Setup.TRIVIAL_4D.value,
            signature="f(x) -> y",
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
