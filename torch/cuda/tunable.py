from typing import Tuple

import torch


__all__ = [
    "enable",
    "is_enabled",
    "tuning_enable",
    "tuning_is_enabled",
    "set_max_tuning_duration",
    "get_max_tuning_duration",
    "set_max_tuning_iterations",
    "get_max_tuning_iterations",
    "set_max_warmup_duration",
    "get_max_warmup_duration",
    "set_max_warmup_iterations",
    "get_max_warmup_iterations",
    "set_filename",
    "get_filename",
    "get_results",
    "get_validators",
]


def enable(val: bool = True) -> None:
    torch._C._cuda_tunableop_enable(val)


def is_enabled():
    return torch._C._cuda_tunableop_is_enabled()


def tuning_enable(val: bool = True) -> None:
    torch._C._cuda_tunableop_tuning_enable(val)


def tuning_is_enabled():
    return torch._C._cuda_tunableop_tuning_is_enabled()


def set_max_tuning_duration(duration: int) -> None:
    torch._C._cuda_tunableop_set_max_tuning_duration(duration)


def get_max_tuning_duration() -> int:
    return torch._C._cuda_tunableop_get_max_tuning_duration()


def set_max_tuning_iterations(iterations: int) -> None:
    torch._C._cuda_tunableop_set_max_tuning_iterations(iterations)


def get_max_tuning_iterations() -> int:
    return torch._C._cuda_tunableop_get_max_tuning_iterations()


def set_max_warmup_duration(duration: int) -> None:
    torch._C._cuda_tunableop_set_max_warmup_duration(duration)


def get_max_warmup_duration() -> int:
    return torch._C._cuda_tunableop_get_max_warmup_duration()


def set_max_warmup_iterations(iterations: int) -> None:
    torch._C._cuda_tunableop_set_max_warmup_iterations(iterations)


def get_max_warmup_iterations() -> int:
    return torch._C._cuda_tunableop_get_max_warmup_iterations()


def set_filename(filename: str) -> None:
    torch._C._cuda_tunableop_set_filename(filename)


def get_filename() -> str:
    return torch._C._cuda_tunableop_get_filename()


def get_results() -> Tuple[str, str, str, float]:
    return torch._C._cuda_tunableop_get_results()


def get_validators() -> Tuple[str, str]:
    return torch._C._cuda_tunableop_get_validators()
