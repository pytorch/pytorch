from typing import Dict, Union

from torchgen.model import NativeFunctionsGroup, NativeFunctionsViewGroup


def func_name_base_str(g: Union[NativeFunctionsGroup, NativeFunctionsViewGroup]) -> str:
    if isinstance(g, NativeFunctionsGroup):
        return str(g.functional.func.name.name.base)
    else:
        return str(g.view.root_name)


is_hand_written_ops_ = frozenset(
    (
        "abs",
        "add",
        "addmm",
        "all",
        "any",
        "argmin",
        "bmm",
        "clamp",
        "clamp_min",
        "cumsum",
        "div",
        "fmod",
        "index_select",
        "leaky_relu",
        "linear",
        "log",
        "matmul",
        "mul",
        "narrow_copy",
        "nonzero",
        "pow",
        "remainder",
        "sigmoid",
        "sign",
        "sub",
        "tanh",
        "detach",
        "expand_as",
        "flatten",
        "narrow",
        "reshape_as",
        "select",
        "slice",
        "softmax",
        "split",
        "squeeze",
        "transpose",
        "view",
        "where",
    )
)


def is_hand_written(g: Union[NativeFunctionsGroup, NativeFunctionsViewGroup]) -> bool:
    name_base = func_name_base_str(g)
    return name_base in is_hand_written_ops_


def override_test_values(arg_map: Dict[str, str], op_name: str, index: int) -> None:
    assert index == 0 or index == 1
    if op_name == "addr":
        if index == 0:
            arg_map["self"] = "at::rand({6, 6})"
            arg_map["vec1"] = "at::rand({6})"
            arg_map["vec2"] = "at::rand({6})"
        else:
            arg_map["self"] = "at::rand({22, 22})"
            arg_map["vec1"] = "at::rand({22})"
            arg_map["vec2"] = "at::rand({22})"
        return
    if op_name == "mv":
        if index == 0:
            arg_map["self"] = "at::rand({6, 6})"
            arg_map["vec"] = "at::rand({6})"
        else:
            arg_map["self"] = "at::rand({22, 22})"
            arg_map["vec"] = "at::rand({22})"
        return
    if op_name == "addbmm":
        if index == 0:
            arg_map["self"] = "at::rand({6, 6})"
        else:
            arg_map["self"] = "at::rand({22, 22})"
        return
    if op_name == "cross":
        if index == 0:
            arg_map["self"] = "at::rand({3, 3, 3})"
            arg_map["other"] = "at::rand({3, 3, 3})"
        else:
            arg_map["self"] = "at::rand({22, 3, 22})"
            arg_map["other"] = "at::rand({22, 3, 22})"
        return
    if op_name == "take":
        if index == 0:
            arg_map["index"] = "at::randint(0, 216, {20}, torch::kInt64)"
        else:
            arg_map["index"] = "at::randint(0, 1000, {100}, torch::kInt64)"
        return
    if op_name == "take_along_dim":
        if index == 0:
            arg_map["indices"] = "at::argsort(self0, 1, true)"
        else:
            arg_map["indices"] = "at::argsort(self1, 1, true)"
        return
    if op_name == "masked_select":
        if index == 0:
            arg_map["mask"] = "at::randn({6, 6, 6}) > 0.5"
        else:
            arg_map["mask"] = "at::rand({22, 22, 22}) > 0.5"
        return
    if op_name == "orgqr":
        if index == 0:
            arg_map["input2"] = "at::rand({6, 6})"
        else:
            arg_map["input2"] = "at::rand({22, 22})"
        return
    if op_name == "ormqr":
        if index == 0:
            arg_map["input2"] = "at::rand({6, 6})"
        else:
            arg_map["input2"] = "at::rand({22, 22})"
        return
    if op_name == "quantile":
        if index == 0:
            arg_map["q"] = "at::rand({6})"
            arg_map["interpolation"] = '"linear"'
        else:
            arg_map["q"] = "at::rand({22})"
            arg_map["interpolation"] = '"linear"'
        return
    if op_name == "nanquantile":
        if index == 0:
            arg_map["q"] = "at::rand({6})"
            arg_map["interpolation"] = '"linear"'
        else:
            arg_map["q"] = "at::rand({22})"
            arg_map["interpolation"] = '"linear"'
        return
    if op_name == "multi_margin_loss":
        if index == 0:
            arg_map["self"] = "at::rand({6, 6})"
            arg_map["target"] = "at::randint(6, {6}, torch::kInt64)"
            arg_map["weight"] = "at::rand({6})"
        else:
            arg_map["self"] = "at::rand({22, 22})"
            arg_map["target"] = "at::randint(22, {22}, torch::kInt64)"
            arg_map["weight"] = "at::rand({22})"
        return
    if op_name == "multilabel_margin_loss":
        if index == 0:
            arg_map["self"] = "at::rand({6, 6})"
            arg_map["target"] = "at::randint(6, {6, 6}, torch::kInt64)"
        else:
            arg_map["self"] = "at::rand({22, 22})"
            arg_map["target"] = "at::randint(22, {22, 22}, torch::kInt64)"
        return
    if op_name == "nll_loss":
        if index == 0:
            arg_map["self"] = "at::rand({6, 6})"
            arg_map["target"] = "at::randint(6, {6}, torch::kInt64)"
            arg_map["weight"] = "at::rand({6})"
        else:
            arg_map["self"] = "at::rand({22, 22})"
            arg_map["target"] = "at::randint(22, {22}, torch::kInt64)"
            arg_map["weight"] = "at::rand({22})"
        return
    if op_name == "nll_loss2d":
        if index == 0:
            arg_map["self"] = "at::rand({6, 6, 6, 6})"
            arg_map["target"] = "at::randint(6, {6, 6, 6}, torch::kInt64)"
            arg_map["weight"] = "at::rand({6})"
        else:
            arg_map["self"] = "at::rand({22, 22, 22, 22})"
            arg_map["target"] = "at::randint(22, {22, 22, 22}, torch::kInt64)"
            arg_map["weight"] = "at::rand({22})"
        return
    if op_name in (
        "fft_fft",
        "fft_ifft",
        "fft_rfft",
        "fft_irfft",
        "fft_hfft",
        "fft_ihfft",
    ):
        arg_map["norm"] = '"forward"'
        return
    if op_name == "linalg_tensorinv":
        if index == 0:
            arg_map["self"] = "at::rand({6, 6, 6, 6})"
            arg_map["ind"] = "2"
        else:
            arg_map["self"] = "at::rand({22, 22, 22, 22})"
            arg_map["ind"] = "2"
        return
    if op_name == "addmv":
        if index == 0:
            arg_map["self"] = "at::rand({2})"
            arg_map["mat"] = "at::rand({2, 2})"
            arg_map["vec"] = "at::rand({2})"
        else:
            arg_map["self"] = "at::rand({35})"
            arg_map["mat"] = "at::rand({35, 35})"
            arg_map["vec"] = "at::rand({35})"
        return
    if op_name == "acosh":
        if index == 0:
            arg_map["self"] = "at::rand({2, 2, 2}) + at::ones({2, 2, 2})"
        else:
            arg_map["self"] = "at::rand({5, 5, 5}) + at::ones({5, 5, 5})"
        return
    if op_name == "adaptive_max_pool2d_backward":
        if index == 0:
            arg_map["grad_output"] = "at::rand({2, 2, 2}, at::kFloat)"
            arg_map["self"] = "at::rand({2, 2, 2}, at::kFloat)"
            arg_map["indices"] = "at::randint(0, 1, {2, 2, 2}, at::kLong)"
        else:
            arg_map["grad_output"] = "at::rand({3, 3, 3}, at::kFloat)"
            arg_map["self"] = "at::rand({3, 3, 3}, at::kFloat)"
            arg_map["indices"] = "at::randint(0, 1, {3, 3, 3}, at::kLong)"
        return
    if op_name == "adaptive_max_pool3d_backward":
        if index == 0:
            arg_map["grad_output"] = "at::rand({2, 2, 2, 2}, at::kFloat)"
            arg_map["self"] = "at::rand({2, 2, 2, 2}, at::kFloat)"
            arg_map["indices"] = "at::randint(0, 1, {2, 2, 2, 2}, at::kLong)"
        else:
            arg_map["grad_output"] = "at::rand({3, 3, 3, 3}, at::kFloat)"
            arg_map["self"] = "at::rand({3, 3, 3, 3}, at::kFloat)"
            arg_map["indices"] = "at::randint(0, 1, {3, 3, 3, 3}, at::kLong)"
        return
    if op_name == "bitwise_left_shift":
        if index == 0:
            arg_map["self"] = "at::randint(1, 1 << 4, {6, 6, 6}, at::kInt)"
            arg_map["other"] = "at::randint(1, 26, {6, 6, 6}, at::kInt)"
        else:
            arg_map["self"] = "at::randint(1, 1 << 4, {22, 22, 22}, at::kInt)"
            arg_map["other"] = "at::randint(1, 26, {22, 22, 22}, at::kInt)"
        return
    if op_name == "bitwise_right_shift":
        if index == 0:
            arg_map["self"] = "at::randint(1 << 21, 1 << 30, {6, 6, 6}, at::kInt)"
            arg_map["other"] = "at::randint(1, 22, {6, 6, 6}, at::kInt)"
        else:
            arg_map["self"] = "at::randint(1 << 21, 1 << 30, {22, 22, 22}, at::kInt)"
            arg_map["other"] = "at::randint(1, 22, {22, 22, 22}, at::kInt)"
        return
    if op_name == "gather":
        if index == 0:
            arg_map["self"] = "at::randint(1, 100, {2,2,2}, at::kInt)"
            arg_map["dim"] = "1"
            arg_map["index"] = "at::randint(0, 1, {2,2,2}, torch::kInt64)"
            arg_map["sparse_grad"] = "false"
        else:
            arg_map["self"] = "at::randint(1, 100, {5,5,5}, at::kInt)"
            arg_map["dim"] = "1"
            arg_map["index"] = "at::randint(0, 4, {5,5,5}, torch::kInt64)"
            arg_map["sparse_grad"] = "false"
        return
    if op_name == "gelu":
        if index == 0:
            arg_map["self"] = "at::rand({6, 6, 6})"
            arg_map["approximate"] = '"tanh"'
        else:
            arg_map["self"] = "at::rand({22, 22, 22})"
            arg_map["approximate"] = '"tanh"'
        return
    if op_name == "gelu_backward":
        if index == 0:
            arg_map["grad_output"] = "at::rand({6, 6, 6})"
            arg_map["self"] = "at::rand({6, 6, 6})"
            arg_map["approximate"] = '"tanh"'
        else:
            arg_map["grad_output"] = "at::rand({22, 22, 22})"
            arg_map["self"] = "at::rand({22, 22, 22})"
            arg_map["approximate"] = '"tanh"'
        return
    if op_name == "index_add":
        if index == 0:
            arg_map["self"] = "at::rand({2})"
            arg_map["dim"] = "0"
            arg_map["index"] = "at::randint(0, 1, {2}, at::kInt)"
            arg_map["source"] = "at::rand({2})"
            arg_map["alpha"] = "2"
        else:
            arg_map["self"] = "at::rand({16})"
            arg_map["dim"] = "0"
            arg_map["index"] = "at::randint(0, 10, {16}, at::kInt)"
            arg_map["source"] = "at::rand({16})"
            arg_map["alpha"] = "2"
        return
    if op_name == "index_copy":
        if index == 0:
            arg_map["self"] = "at::rand({2})"
            arg_map["dim"] = "0"
            arg_map["index"] = "at::randint(0, 1, {2}, at::kLong)"
            arg_map["source"] = "at::rand({2})"
        else:
            arg_map["self"] = "at::rand({32})"
            arg_map["dim"] = "0"
            arg_map["index"] = "at::randint(0, 10, {32}, at::kLong)"
            arg_map["source"] = "at::rand({32})"
        return
    if op_name == "linalg_cross":
        if index == 0:
            arg_map["self"] = "at::rand({6, 3, 6})"
            arg_map["other"] = "at::rand({6, 3, 6})"
            arg_map["dim"] = "1"
        else:
            arg_map["self"] = "at::rand({22, 3, 22})"
            arg_map["other"] = "at::rand({22, 3, 22})"
            arg_map["dim"] = "1"
        return
    if op_name == "nll_loss_backward":
        if index == 0:
            arg_map["grad_output"] = "at::rand({})"
            arg_map["self"] = "at::rand({6})"
            arg_map["target"] = "at::randint(0, 5, {6}, torch::kInt64)"
            arg_map["weight"] = "at::rand({6})"
            arg_map["reduction"] = "1"
            arg_map["ignore_index"] = "1"
            arg_map["total_weight"] = "at::rand({})"
        else:
            arg_map["grad_output"] = "at::rand({})"
            arg_map["self"] = "at::rand({36})"
            arg_map["target"] = "at::randint(0, 11, {36}, torch::kInt64)"
            arg_map["weight"] = "at::rand({36})"
            arg_map["reduction"] = "1"
            arg_map["ignore_index"] = "1"
            arg_map["total_weight"] = "at::rand({})"
        return
    if op_name in ["scatter", "scatter_add", "_scatter_reduce"]:
        if index == 0:
            arg_map["self"] = "at::randint(1, 100, {2,2,2}, torch::kInt64)"
            arg_map["index"] = "at::randint(0, 1, {2,2,2}, torch::kInt64)"
            arg_map["src"] = "at::randint(1, 100, {2,2,2}, torch::kInt64)"
        else:
            arg_map["self"] = "at::randint(1, 100, {5,5,5}, torch::kInt64)"
            arg_map["index"] = "at::randint(0, 1, {5,5,5}, torch::kInt64)"
            arg_map["src"] = "at::randint(1, 100, {5,5,5}, torch::kInt64)"
        if "reduce" in arg_map:
            arg_map["reduce"] = '"sum"' if op_name == "_scatter_reduce" else '"add"'
        return
    if op_name == "scatter_reduce":
        arg_map["reduce"] = '"mean"'
        if index == 0:
            arg_map["index"] = "at::randint(6, {6, 6, 6}, torch::kInt64)"
        else:
            arg_map["index"] = "at::randint(22, {22, 22, 22}, torch::kInt64)"
        return
    if op_name == "special_zeta":
        if index == 0:
            arg_map["self"] = "at::rand({2,2,2}, at::kDouble) + at::ones({2,2,2})"
            arg_map["other"] = "at::rand({2,2,2}, at::kDouble) + at::ones({2,2,2})"
        else:
            arg_map["self"] = "at::rand({5,5,5}, at::kDouble) + at::ones({5,5,5})"
            arg_map["other"] = "at::rand({5,5,5}, at::kDouble) + at::ones({5,5,5})"
        return
    if op_name == "_convert_indices_from_csr_to_coo":
        if index == 0:
            arg_map["crow_indices"] = "torch::tensor({1}, torch::kInt32)"
            arg_map["col_indices"] = "torch::tensor({0, 1, 0}, torch::kInt32)"
            arg_map["out_int32"] = "false"
        else:
            arg_map["crow_indices"] = "torch::tensor({0}, torch::kInt32)"
            arg_map[
                "col_indices"
            ] = "torch::tensor({0, 1, 0, 2, 1, 2, 0, 1, 0, 2, 1, 2}, torch::kInt32)"
            arg_map["out_int32"] = "false"
        return
    if op_name == "_convert_indices_from_coo_to_csr":
        if index == 0:
            arg_map["self"] = "at::randint(0, 3, {2}, at::kInt)"
            arg_map["size"] = "10"
            arg_map["out_int32"] = "false"
        else:
            arg_map["self"] = "at::randint(0, 3, {12}, at::kInt)"
            arg_map["size"] = "24"
            arg_map["out_int32"] = "false"
        return
    if op_name in ("diagonal", "linalg_diagonal"):
        arg_map["offset"] = "0"
        arg_map["dim0"] = "1"
        arg_map["dim1"] = "2"
        return
