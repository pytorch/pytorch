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
# mypy: ignore-errors

from core.api import GroupedModules, GroupedStmts, GroupedVariants
from core.types import FlatIntermediateDefinition
from core.utils import flatten, parse_stmts

from definitions.setup import Setup


BENCHMARKS: FlatIntermediateDefinition = flatten(
    {
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
                auto optional_float = std::make_optional(at::kFloat);

                // @TensorOptions overload
                at::empty({0}, options_empty);
                at::empty({0}, options_full);
                at::empty({0}, at::kFloat); // implicit conversion

                // @Faithful overload
                at::empty({0}, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
                at::empty({0}, at::kFloat, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
                at::empty({0}, optional_float, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
            """
            ),
        },
        "Pointwise": {
            "Math": GroupedVariants(
                *parse_stmts(
                    r"""
            Python                                   | C++
            ---------------------------------------- | ----------------------------------------
            # @setup                                 | // @setup
            torch.manual_seed(138_10_23)             | torch::manual_seed(1381023);
            x = torch.rand((4, 4))                   | auto x = torch::rand({4, 4});
            y_float = torch.ones((4, 4))             | auto y_float = torch::ones({4, 4});
            y_vector = torch.ones((4, 1))            | auto y_vector = torch::ones({4, 1});
            y_int = torch.ones(                      | auto y_int = torch::ones({4, 4}, at::kInt);
                (4, 4), dtype=torch.int32)           |
                                                     |
            # @add                                   | // @add
            x += 1.0                                 | x += 1;
            x += y_float                             | x += y_float;
            x += y_vector                            | x += y_vector;
            x += y_int                               | x += y_int;
            x + y_float                              | x + y_float;
            torch.add(x, y_float)                    | torch::add(x, y_float);
            torch.add(x, y_float, out=x)             | torch::add_out(/*out=*/x, x, y_float);
                                                     |
            # @multiply                              | // @multiply
            x *= 1.0                                 | x *= 1;
            x *= y_float                             | x *= y_float;
            x *= y_vector                            | x *= y_vector;
            x *= y_int                               | x *= y_int;
            x * y_float                              | x * y_float;
            torch.mul(x, y_float)                    | torch::mul(x, y_float);
            torch.mul(x, y_float, out=x)             | torch::mul_out(/*out=*/x, x, y_float);
                                                     |
            # @equality                              | // @equality
            x == y_float                             | x == y_float;
            x == 1.0                                 | x == 1.0;
        """
                )
            ),
            "Data movement": GroupedVariants(
                *parse_stmts(
                    r"""
            Python                                   | C++
            ---------------------------------------- | ----------------------------------------
            # @setup                                 | // @setup
            x = torch.ones((4, 4))                   | auto x = torch::ones({4, 4});
            y = torch.ones((4, 4))                   | auto y = torch::ones({4, 4});
            x_t = x.t()                              | auto x_t = x.t();
                                                     |
            # @contiguous (trivial)                  | // @contiguous (trivial)
            x.contiguous()                           | x.contiguous();
                                                     |
            # @contiguous (non-trivial)              | // @contiguous (non-trivial)
            x_t.contiguous()                         | x_t.contiguous();
                                                     |
            # @clone                                 | // @clone
            x.clone()                                | x.clone();
                                                     |
            # @copy_                                 | // @copy_
            x.copy_(y)                               | x.copy_(y);
                                                     |
            # @zero_                                 | // @zero_
            x.zero_()                                | x.zero_();
                                                     |
            # @RNG                                   | // @RNG
            x.uniform_()                             | x.uniform_();
        """
                )
            ),
        },
        "Reduction": GroupedVariants(
            *parse_stmts(
                r"""
        Python                                   | C++
        ---------------------------------------- | ----------------------------------------
        # @setup                                 | // @setup
        x = torch.ones((4, 4))                   | auto x = torch::ones({4, 4});
                                                 |
        # @max                                   | // @max
        x.max()                                  | x.max();
                                                 |
        # @sum                                   | // @sum
        x.sum()                                  | x.sum();
                                                 |
        # @variance                              | // @variance
        x.var(0)                                 | x.var(0);
    """
            )
        ),
        "Indexing": GroupedVariants(
            *parse_stmts(
                r"""
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
    """
            )
        ),
        "Metadata and views": GroupedVariants(
            *parse_stmts(
                r"""
        Python                                   | C++
        ---------------------------------------- | ----------------------------------------
        # @setup                                 | // @setup
        x = torch.ones((4, 4))                   | auto x = torch::ones({4, 4});
                                                 |
        # @size                                  | // @size
        x.size()[0]                              | x.sizes()[0];
                                                 |
        # @stride                                | // @stride
        x.stride(0)                              | x.stride(0);
                                                 |
        # @as_strided                            | // @as_strided
        torch.as_strided(x, (2, 3), (4, 1), 2)   | torch::as_strided(x, {2, 3}, {4, 1}, 2);
                                                 |
        # @select                                | // @select
        x.select(1, 1)                           | x.select(1, 1);
                                                 |
        # @unsqueeze                             | // @unsqueeze
        x.unsqueeze(0)                           | x.unsqueeze(0);
                                                 |
        # @view                                  | // @view
        x.view(-1, 1)                            | x.view({-1, 1});
                                                 |
        # @transpose                             | // @transpose
        x.t()                                    | x.t();
                                                 |
        # @reshape                               | // @reshape
        x.reshape((16, 1))                       | x.reshape({16, 1});
    """
            )
        ),
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
                (
                    Setup.TRIVIAL_4D,
                    True,
                    ("LayerNorm(4)", "LayerNorm(torch::nn::LayerNormOptions({4}))"),
                ),
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
        "training": {
            "simple": GroupedStmts(
                *parse_stmts(
                    r"""
                Python                                   | C++
                ---------------------------------------- | ----------------------------------------
                a0 = torch.nn.functional.relu(x * w0)    | auto a0 = torch::nn::functional::relu(x * w0);
                y = a0 * w1                              | auto y = a0 * w1;
            """
                ),
                Setup.TRAINING.value,
                num_threads=(1, 2),
                signature=r"f(x, w0, w1) -> y",
                torchscript=True,
                autograd=True,
            ),
            "ensemble": GroupedStmts(
                *parse_stmts(
                    r"""
                Python                                   | C++
                ---------------------------------------- | ----------------------------------------
                a0 = torch.nn.functional.gelu(x * w0)    | auto a0 = torch::nn::functional::gelu(x * w0);
                a1 = torch.nn.functional.prelu(y, w1)    | auto a1 = torch::nn::functional::prelu(y, w1);
                z = torch.nn.functional.normalize(       | auto z = torch::nn::functional::normalize(
                    torch.cat([a0, a1]),                 |     torch::cat({a0, a1}),
                    p=2.0, dim=0,                        |     torch::nn::functional::NormalizeFuncOptions().p(2).dim(0)
                ).dot(w2)                                | ).dot(w2);
            """
                ),
                Setup.TRAINING.value,
                num_threads=(1, 2),
                signature=r"f(x, y, w0, w1, w2) -> z",
                torchscript=True,
                autograd=True,
            ),
        },
        "InferenceMode": GroupedVariants(
            # In general, the mixed input scenario is less common so its
            # perf can be less important than pure inference tensor inputs.
            cpp_block=r"""
            // @Setup
            auto s = torch::ones({3, 3});  // Normal Tensor
            c10::InferenceMode guard;
            auto x = torch::ones({3, 3});  // Inference Tensor

            // @View
            torch::Tensor y = x.view({9});

            // @Inplace
            torch::Tensor y = x.mul_(x);

            // @Mixed
            torch::Tensor y = x + s;
        """
        ),
    }
)
