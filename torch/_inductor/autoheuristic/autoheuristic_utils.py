import functools
from collections.abc import Callable
from typing import Any

import torch


Feedback = float
Choice = str
Value = Any

CHOICE_COL = "choice"
FEEDBACK_COL = "feedback"


class AHFeature:
    """
    The context, that AutoHeuristic stores, is a list of features. AutoHeuristic needs to know whether a feature is
    categorical (i.e., not a continuous variable) to learn a machine learning model.
    """

    def __init__(self, name: str, value: Value, is_categorical: bool = False) -> None:
        self.name = name
        self.value = value
        self.is_categorical = is_categorical


class AHOperation:
    """
    AHOperation can be used to augment the data collected by AutoHeuristic.
    One might for example store features like m, k, n, but also want to use
    features like m*n, or k*n, to learn a heuristic. Instead of storing features
    that can be created from the collected data, one can use AHOperation to
    create new features from the collected data.
    """

    def __init__(
        self, name: str, func: Callable[[Any], Value], is_categorical: bool = False
    ) -> None:
        self.name = name
        self.func = func
        self.is_categorical = is_categorical

    def apply_operation(self, data: Any) -> None:
        data[self.name] = self.func(data)


class AHContext:
    """
    This class is used to specify which information AutoHeuristic should store. For each choice, AutoHeursitic will
    store the context and the collected feedback. The context could be something like the shape of a tensor, i.e.,
    information that will help to learn a heuristic.
    """

    features: list[AHFeature]
    context_dict: dict[str, Value]

    def __init__(self) -> None:
        self.features = []
        self.context_dict = {}

    def add_feature(
        self, name: str, value: Value, is_categorical: bool = False
    ) -> None:
        self.features.append(AHFeature(name, value, is_categorical=is_categorical))
        self.context_dict[name] = value

    def get_numerical_and_categorical_features(self) -> tuple[list[str], list[str]]:
        numerical_features = []
        categorical_features = []
        for feature in self.features:
            if feature.is_categorical:
                categorical_features.append(feature.name)
            else:
                numerical_features.append(feature.name)

        return numerical_features, categorical_features

    def get_feature_names_csv(self) -> str:
        return ",".join(feature.name for feature in self.features)

    def get_feature_values_csv(self) -> str:
        return ",".join(str(feature.value) for feature in self.features)

    def get_value(self, name: str) -> Value:
        return self.context_dict[name]

    def apply_operations(self, operations: list[AHOperation]) -> None:
        for op in operations:
            op.apply_operation(self.context_dict)


class AHMetadata:
    def __init__(
        self,
        shared_memory: Any,
        device_capa: tuple[int, int],
        choices: list[Choice],
        name: str,
    ) -> None:
        # use amount of shared_memory and device_capability to identify GPU
        # TODO(AlnisM): there might be a better way to do this
        self.shared_memory = shared_memory
        self.device_capa = device_capa
        self.choices = choices
        self.name = name

    def to_dict(self) -> dict[str, Value]:
        return {
            "shared_memory": self.shared_memory,
            "device_capa": self.device_capa,
            "name": self.name,
        }


def get_metadata_str_from_log(log_path: str) -> str:
    with open(log_path, newline="") as file:
        json_string = file.readline().strip()
        return json_string


def check_minsize(context: AHContext, minsize: int) -> bool:
    return (
        context.get_value("m") >= minsize
        and context.get_value("k") >= minsize
        and context.get_value("n") >= minsize
    )


def pad_mm_precondition(metadata: AHMetadata, context: AHContext) -> bool:
    if metadata.shared_memory == 166912 and metadata.device_capa == (8, 0):
        # A100 precondition
        return check_minsize(context, 512)
    elif metadata.shared_memory == 232448 and metadata.device_capa == (9, 0):
        # H100 precondition
        return check_minsize(context, 768)
    return True


def get_mixedmm_precondition(metadata: AHMetadata, context: AHContext) -> bool:
    m = context.get_value("m")
    k = context.get_value("k")
    n = context.get_value("n")
    if m > 128 or k < 1024 or n < 1024:
        return False
    mat1_iscontig = context.get_value("mat1_iscontig")
    mat2_iscontig = context.get_value("mat2_iscontig")
    return mat1_iscontig and not mat2_iscontig


def get_mult_dims_ops() -> list[AHOperation]:
    m_times_k_op = AHOperation("m*k", lambda data: data["m"] * data["k"])
    m_times_n_op = AHOperation("m*n", lambda data: data["m"] * data["n"])
    k_times_n_op = AHOperation("k*n", lambda data: data["k"] * data["n"])
    return [m_times_k_op, m_times_n_op, k_times_n_op]


def get_arith_intensity(data: Any) -> float:
    m = data["m"]
    k = data["k"]
    n = data["n"]
    if m == 0 or k == 0 or n == 0:
        return 0.0
    return m * k * n / (m * k + k * n + m * n)


def pad_mm_operations() -> list[AHOperation]:
    mult_dims_ops = get_mult_dims_ops()
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

    arith_intensity_op = AHOperation("arith_intensity", get_arith_intensity)
    dims_need_padding_ops = get_dims_need_padding_ops()
    dims_multiple_ops = get_dims_multiple_ops()
    is_contig_ops = get_is_contig_ops()

    ah_operations = mult_dims_ops + [
        k_div_m_times_n_op,
        bfloat_perf_hit_op,
        arith_intensity_op,
    ]
    ah_operations.extend(dims_need_padding_ops)
    ah_operations.extend(dims_multiple_ops)
    ah_operations.extend(is_contig_ops)
    return ah_operations


def between_op(data: Any, dim: str, lower: int, upper: int) -> bool:
    return data[dim] >= lower and data[dim] <= upper


def between_ops() -> list[AHOperation]:
    dims = ["m", "k", "n"]
    limits = [(1, 16), (17, 32), (33, 64), (65, 128), (129, 256)]
    ah_operations = []
    for dim in dims:
        for lower, upper in limits:
            between_op_fn = functools.partial(
                between_op, dim=dim, lower=lower, upper=upper
            )
            # using 'LEQ' instead of '<=' because '<=' cannot be exported to dot
            between_op_name = f"{lower}LEQ{dim}LEQ{upper}"
            ah_operations.append(
                AHOperation(between_op_name, between_op_fn, is_categorical=True)
            )
    return ah_operations


def pow2_op(data: Any, dim: str, exponent: int) -> bool:
    return data[dim] == 2**exponent


def mm_operations() -> list[AHOperation]:
    mult_dims_ops = get_mult_dims_ops()
    arith_intensity_op = AHOperation("arith_intensity", get_arith_intensity)
    return mult_dims_ops + [arith_intensity_op]


def mixed_mm_operations() -> list[AHOperation]:
    return mm_operations() + between_ops()


def is_multiple(data: Any, dim: str, mult: int) -> bool:
    return data[dim] % mult == 0


def get_dims_multiple_ops() -> list[AHOperation]:
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


def get_dims_need_padding_ops() -> list[AHOperation]:
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


def get_is_contig_ops() -> list[AHOperation]:
    def mat1_is_contig_fn(data: Any) -> bool:
        stride_0 = data["mat1_stride_0"]
        stride_1 = data["mat1_stride_1"]
        k = data["k"]
        return stride_0 == k and stride_1 == 1

    mat1_is_contig_op = AHOperation(
        "mat1_iscontig", mat1_is_contig_fn, is_categorical=True
    )

    def mat2_is_contig_fn(data: Any) -> bool:
        stride_0 = data["mat2_stride_0"]
        stride_1 = data["mat2_stride_1"]
        n = data["n"]
        return stride_0 == n and stride_1 == 1

    mat2_is_contig_op = AHOperation(
        "mat2_iscontig", mat2_is_contig_fn, is_categorical=True
    )

    return [mat1_is_contig_op, mat2_is_contig_op]


def context_add_strides(context: AHContext, name: str, stride: tuple[int, ...]) -> None:
    for i, s in enumerate(stride):
        context.add_feature(f"{name}_stride_{i}", s)


def context_add_using_tf32(context: AHContext, dtype: torch.dtype) -> None:
    using_tf32 = "not_float_32"
    if dtype == torch.float32:
        using_tf32 = torch.backends.cuda.matmul.allow_tf32
    context.add_feature("using_tf32", using_tf32, is_categorical=True)
