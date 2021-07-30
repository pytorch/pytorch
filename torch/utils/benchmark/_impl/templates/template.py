""" Python template for Timer methods.

This template will replace:
    `SETUP_TEMPLATE_LOCATION`
      and
    `STMT_TEMPLATE_LOCATION`
sections with user provided statements.
"""
import timeit
import typing

import torch


# Note: The name of this class (PythonTemplate) is a magic word in compile.py
class PythonTemplate:

    @staticmethod
    def call(n_iter: int) -> None:
        # SETUP_TEMPLATE_LOCATION

        for _ in range(n_iter):
            # STMT_TEMPLATE_LOCATION
            pass

    @staticmethod
    def measure_wall_time(
        n_iter: int,
        n_warmup_iter: int,
        cuda_sync: bool,
        timer: typing.Callable[[], float] = timeit.default_timer,
    ) -> float:
        # SETUP_TEMPLATE_LOCATION

        for _ in range(n_warmup_iter):
            # STMT_TEMPLATE_LOCATION
            pass

        if cuda_sync:
            torch.cuda.synchronize()
        start_time = timer()

        for _ in range(n_iter):
            # STMT_TEMPLATE_LOCATION
            pass

        if cuda_sync:
            torch.cuda.synchronize()

        return timer() - start_time
