# mypy: allow-untyped-defs
import logging

from torch._ops import HigherOrderOperator


log = logging.getLogger(__name__)


class PreserveMeta(HigherOrderOperator):
    def __init__(self):
        super().__init__("preserve_meta")

    def __call__(self, op, args, kwargs):
        return super().__call__(op, args, kwargs)


def trace_preserve_meta(mode, op, args, kwargs):
    return mode.tracer.create_proxy(
        "call_function", op, (op, args, kwargs), {}, name="preserve_meta"
    )
