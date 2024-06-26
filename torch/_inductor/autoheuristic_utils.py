import functools

from typing import Any, Callable, Dict, List

Feedback = float
Choice = str
Value = Any
ContextDictT = Dict[str, Value]

CHOICE_COL = "choice"
FEEDBACK_COL = "feedback"


class AHOperation:
    def __init__(
        self, name: str, func: Callable[[Any], Value], is_categorical: bool = False
    ):
        self.name = name
        self.func = func
        self.is_categorical = is_categorical

    def apply_operation(self, data: Any) -> None:
        data[self.name] = self.func(data)


def pad_mm_operations() -> List[AHOperation]:
    m_times_k_op = AHOperation("m*k", lambda data: data["m"] * data["k"])
    m_times_n_op = AHOperation("m*n", lambda data: data["m"] * data["n"])
    k_times_n_op = AHOperation("k*n", lambda data: data["k"] * data["n"])
    k_div_m_times_n_op = AHOperation(
        "k/(m*n)", lambda data: data["k"] / (data["m"] * data["n"])
    )

    def bfloat_perf_hit(data: Any) -> bool:
        m = data["m"]
        k = data["k"]
        n = data["n"]
        is_bfloat = str(data["mat1_dtype"]) == "torch.bfloat16"
        return k > (m * 1024) and k > (n * 1024) and is_bfloat

    bfloat_perf_hit_op = AHOperation(
        "bfloat_perf_hit", bfloat_perf_hit, is_categorical=True
    )

    def get_arith_intensity(data: Any) -> float:
        m = data["m"]
        k = data["k"]
        n = data["n"]
        return m * k * n / (m * k + k * n + m * n)

    arith_intensity_op = AHOperation("arith_intensity", get_arith_intensity)
    dims_need_padding_ops = get_dims_need_padding_ops()
    dims_multiple_ops = get_dims_multiple_ops()

    ah_operations = [
        m_times_k_op,
        m_times_n_op,
        k_times_n_op,
        k_div_m_times_n_op,
        bfloat_perf_hit_op,
        arith_intensity_op,
    ]
    ah_operations.extend(dims_need_padding_ops)
    ah_operations.extend(dims_multiple_ops)
    return ah_operations


def is_multiple(data: Any, dim: str, mult: int) -> bool:
    return data[dim] % mult == 0


def get_dims_multiple_ops() -> List[AHOperation]:
    multiples = [2, 4, 8, 16, 32]
    dims = ["m", "k", "n"]
    dims_multiple_ops = []
    for dim in dims:
        for mult in multiples:
            is_multiple_fn = functools.partial(is_multiple, dim=dim, mult=mult)
            dims_multiple_op = AHOperation(
                f"{dim}_multiple_{mult}", is_multiple_fn, is_categorical=True
            )
            dims_multiple_ops.append(dims_multiple_op)
    return dims_multiple_ops


def get_dims_need_padding_ops() -> List[AHOperation]:
    def mat1_innermost_needs_padding_fn(data: Any) -> bool:
        mat1_stride_0 = data["mat1_stride_0"]
        mat1_stride_1 = data["mat1_stride_1"]
        m_padded_length = data["m_padded_length"]
        k_padded_length = data["k_padded_length"]
        mat1_innermost_needs_padding = False
        if mat1_stride_0 == 1 and m_padded_length != 0:
            mat1_innermost_needs_padding = True
        if mat1_stride_1 == 1 and k_padded_length != 0:
            mat1_innermost_needs_padding = True
        return mat1_innermost_needs_padding

    mat1_innermost_op = AHOperation(
        "mat1_innermost_needs_padding",
        mat1_innermost_needs_padding_fn,
        is_categorical=True,
    )

    def mat2_innermost_needs_padding_fn(data: Any) -> bool:
        mat2_stride_0 = data["mat2_stride_0"]
        mat2_stride_1 = data["mat2_stride_1"]
        k_padded_length = data["k_padded_length"]
        n_padded_length = data["n_padded_length"]
        mat2_innermost_needs_padding = False
        if mat2_stride_0 == 1 and k_padded_length != 0:
            mat2_innermost_needs_padding = True
        if mat2_stride_1 == 1 and n_padded_length != 0:
            mat2_innermost_needs_padding = True
        return mat2_innermost_needs_padding

    mat2_innermost_op = AHOperation(
        "mat2_innermost_needs_padding",
        mat2_innermost_needs_padding_fn,
        is_categorical=True,
    )

    def num_dims_needs_padding_fn(data: Any) -> int:
        m_padded_length = data["m_padded_length"]
        k_padded_length = data["k_padded_length"]
        n_padded_length = data["n_padded_length"]
        num_dims_needs_padding = 0
        if m_padded_length != 0:
            num_dims_needs_padding += 1
        if k_padded_length != 0:
            num_dims_needs_padding += 1
        if n_padded_length != 0:
            num_dims_needs_padding += 1
        return num_dims_needs_padding

    num_dims_op = AHOperation("num_dims_needs_padding", num_dims_needs_padding_fn)
    return [mat1_innermost_op, mat2_innermost_op, num_dims_op]
