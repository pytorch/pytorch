from torchgen.model import NativeFunctionsGroup

from typing import Dict


def func_name_base_str(g: NativeFunctionsGroup) -> str:
    return str(g.functional.func.name.name.base)


is_hand_written_ops_ = frozenset(
    (
        "add",
        "addmm",
        "all",
        "any",
        "argmin",
        "bmm",
        "clamp",
        "cumsum",
        "div",
        "fmod",
        "leaky_relu",
        "log",
        "mul",
        "pow",
        "remainder",
        "sigmoid",
        "sign",
        "sub",
        "tanh",
    )
)


def is_hand_written(g: NativeFunctionsGroup) -> bool:
    name_base = func_name_base_str(g)
    return name_base in is_hand_written_ops_


def override_test_values(arg_map: Dict[str, str], op_name: str, index: int) -> None:
    assert index == 0 or index == 1
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
            arg_map["grad_output"] = "at::randint(-3, 2, {2,2,2})"
            arg_map["self"] = "at::randint(-3, 2, {2,2,2})"
            arg_map["indices"] = "at::randint(0, 1, {2,2,2}, at::kLong)"
        else:
            arg_map["grad_output"] = "at::randint(-3, 3, {3,3,3})"
            arg_map["self"] = "at::randint(-3, 2, {3,3,3})"
            arg_map["indices"] = "at::randint(0, 1, {3,3,3}, at::kLong)"
        return
    if op_name == "adaptive_max_pool3d_backward":
        if index == 0:
            arg_map["grad_output"] = "at::randint(-3, 2, {2,2,2,2})"
            arg_map["self"] = "at::randint(-3, 2, {2,2,2,2})"
            arg_map["indices"] = "at::randint(0, 1, {2,2,2,2}, at::kLong)"
        else:
            arg_map["grad_output"] = "at::randint(-3, 3, {3,3,3,3})"
            arg_map["self"] = "at::randint(-3, 2, {3,3,3,3})"
            arg_map["indices"] = "at::randint(0, 1, {3,3,3,3}, at::kLong)"
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
