r"""Functional interface."""

from .adadelta import adadelta  # type: ignore[attr-defined]  # noqa: F401
from .adagrad import _make_sparse, adagrad  # type: ignore[attr-defined]  # noqa: F401
from .adam import adam  # type: ignore[attr-defined]  # noqa: F401
from .adamax import adamax  # type: ignore[attr-defined]  # noqa: F401
from .adamw import adamw  # type: ignore[attr-defined]  # noqa: F401
from .asgd import asgd  # type: ignore[attr-defined]  # noqa: F401
from .nadam import nadam  # type: ignore[attr-defined]  # noqa: F401
from .radam import radam  # type: ignore[attr-defined]  # noqa: F401
from .rmsprop import rmsprop  # type: ignore[attr-defined]  # noqa: F401
from .rprop import rprop  # type: ignore[attr-defined]  # noqa: F401
from .sgd import sgd  # type: ignore[attr-defined]  # noqa: F401
from .sparse_adam import sparse_adam  # type: ignore[attr-defined]  # noqa: F401
