from typing import Optional, Tuple

import torch


__all__ = [
    "enable",
    "is_enabled",
    "tuning_enable",
    "tuning_is_enabled",
    "numerics_check_enable",
    "numerics_check_is_enabled",
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
    "write_file_on_exit",
    "write_file",
    "read_file",
    "icache_flush_enable",
    "icache_flush_is_enabled",
    "set_rotating_buffer_size",
    "get_rotating_buffer_size",
]


def enable(val: bool = True) -> None:
    torch._C._cuda_tunableop_enable(val)


def is_enabled() -> bool:
    return torch._C._cuda_tunableop_is_enabled()


def tuning_enable(val: bool = True) -> None:
    torch._C._cuda_tunableop_tuning_enable(val)


def tuning_is_enabled() -> bool:
    return torch._C._cuda_tunableop_tuning_is_enabled()


def numerics_check_enable(val: bool = True) -> None:
    torch._C._cuda_tunableop_numerics_check_enable(val)


def numerics_check_is_enabled() -> bool:
    return torch._C._cuda_tunableop_numerics_check_is_enabled()


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


def set_filename(filename: str, insert_device_ordinal: bool = False) -> None:
    torch._C._cuda_tunableop_set_filename(filename, insert_device_ordinal)


def get_filename() -> str:
    return torch._C._cuda_tunableop_get_filename()


def get_results() -> Tuple[str, str, str, float]:
    return torch._C._cuda_tunableop_get_results()


def get_validators() -> Tuple[str, str]:
    return torch._C._cuda_tunableop_get_validators()


def write_file_on_exit(val: bool) -> None:
    torch._C._cuda_tunableop_write_file_on_exit(val)


def write_file(filename: Optional[str] = None) -> bool:
    if filename is None:
        filename = get_filename()
    return torch._C._cuda_tunableop_write_file(filename)


def read_file(filename: Optional[str] = None) -> bool:
    if filename is None:
        filename = get_filename()
    return torch._C._cuda_tunableop_read_file(filename)


def icache_flush_enable(val: bool = True) -> None:
    torch._C._cuda_tunableop_icache_flush_enable(val)


def icache_flush_is_enabled() -> bool:
    return torch._C._cuda_tunableop_icache_flush_is_enabled()


def set_rotating_buffer_size(size: int) -> None:
    torch._C._cuda_tunableop_set_rotating_buffer_size(size)


def get_rotating_buffer_size() -> int:
    return torch._C._cuda_tunableop_get_rotating_buffer_size()
