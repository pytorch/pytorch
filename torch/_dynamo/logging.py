import itertools
import logging
import os

# logging level for dynamo generated graphs/bytecode/guards
logging.CODE = 15
logging.addLevelName(logging.CODE, "CODE")


# Return all loggers that torchdynamo/torchinductor is responsible for
def get_loggers():
    return [
        logging.getLogger("torch._dynamo"),
        logging.getLogger("torch._inductor"),
    ]


# Set the level of all loggers that torchdynamo is responsible for
def set_loggers_level(level):
    for logger in get_loggers():
        logger.setLevel(level)


LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "torchdynamo_format": {
            "format": "[%(asctime)s] %(name)s: [%(levelname)s] %(message)s"
        },
    },
    "handlers": {
        "torchdynamo_console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "torchdynamo_format",
            "stream": "ext://sys.stderr",
        },
    },
    "loggers": {
        "torch._dynamo": {
            "level": "DEBUG",
            "handlers": ["torchdynamo_console"],
            "propagate": False,
        },
        "torch._inductor": {
            "level": "DEBUG",
            "handlers": ["torchdynamo_console"],
            "propagate": False,
        },
    },
    "disable_existing_loggers": False,
}


# initialize torchdynamo loggers
def init_logging(log_level, log_file_name=None):
    if "PYTEST_CURRENT_TEST" not in os.environ:
        logging.config.dictConfig(LOGGING_CONFIG)
        if log_file_name is not None:
            log_file = logging.FileHandler(log_file_name)
            log_file.setLevel(log_level)
            for logger in get_loggers():
                logger.addHandler(log_file)

        if bool(os.environ.get("TORCH_COMPILE_DEBUG", False)):
            from .utils import get_debug_dir

            log_level = logging.DEBUG
            log_path = os.path.join(get_debug_dir(), "torchdynamo")
            if not os.path.exists(log_path):
                os.makedirs(log_path)

            log_file = logging.FileHandler(os.path.join(log_path, "debug.log"))
            log_file.setLevel(logging.DEBUG)
            logger = logging.getLogger("torch._dynamo")
            logger.addHandler(log_file)

        set_loggers_level(log_level)


# Creates a logging function that logs a message with a step # prepended.
# get_step_logger should be lazily called (i.e. at runtime, not at module-load time)
# so that step numbers are initialized properly. e.g.:

# @functools.lru_cache(None)
# def _step_logger():
#     return get_step_logger(logging.getLogger(...))

# def fn():
#     _step_logger()(logging.INFO, "msg")

_step_counter = itertools.count(1)


def get_step_logger(logger):
    step = next(_step_counter)

    def log(level, msg):
        logger.log(level, f"Step {step}: {msg}")

    return log
