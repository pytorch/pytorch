# mypy: allow-untyped-defs
import itertools
import logging

from torch.hub import _Faketqdm, tqdm


# Disable progress bar by default, not in dynamo config because otherwise get a circular import
disable_progress = True


# Return all loggers that torchdynamo/torchinductor is responsible for
def get_loggers():
    return [
        logging.getLogger("torch.fx.experimental.symbolic_shapes"),
        logging.getLogger("torch._dynamo"),
        logging.getLogger("torch._inductor"),
    ]


# Creates a logging function that logs a message with a step # prepended.
# get_step_logger should be lazily called (i.e. at runtime, not at module-load time)
# so that step numbers are initialized properly. e.g.:

# @functools.lru_cache(None)
# def _step_logger():
#     return get_step_logger(logging.getLogger(...))

# def fn():
#     _step_logger()(logging.INFO, "msg")

_step_counter = itertools.count(1)

# Update num_steps if more phases are added: Dynamo, AOT, Backend
# This is very inductor centric
# _inductor.utils.has_triton() gives a circular import error here

if not disable_progress:
    try:
        import triton  # noqa: F401

        num_steps = 3
    except ImportError:
        num_steps = 2
    pbar = tqdm(total=num_steps, desc="torch.compile()", delay=0)


def get_step_logger(logger):
    if not disable_progress:
        pbar.update(1)
        if not isinstance(pbar, _Faketqdm):
            pbar.set_postfix_str(f"{logger.name}")

    step = next(_step_counter)

    def log(level, msg, **kwargs):
        logger.log(level, "Step %s: %s", step, msg, **kwargs)

    return log
