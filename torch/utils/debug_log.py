import logging

import torch
import torch._higher_order_ops
import torch.distributed as dist
from torch._dynamo.decorators import leaf_function


log = logging.getLogger(__name__)


@leaf_function
def debug_log(t, tag):
    torch._higher_order_ops.print("[{}][fwd] norm={}", tag, t.norm())
    return None


@debug_log.register_fake  # pyrefly: ignore[missing-attribute]
def debug_log_fake(t, tag):
    return None


@debug_log.register_hook  # pyrefly: ignore[missing-attribute]
def debug_log_hook(t_grad, tag):
    torch._higher_order_ops.print("[{}][bwd] norm={}", tag, t_grad.norm())


def _should_log(ranks):
    if ranks is None:
        return True
    if not dist.is_initialized():
        return True
    current_rank = dist.get_rank()
    if isinstance(ranks, int):
        return current_rank == ranks
    return current_rank in ranks


def _get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


@leaf_function
def debug_log_rank(t, tag, ranks=None):
    if _should_log(ranks):
        log.info("[rank %d][%s][fwd] norm=%s", _get_rank(), tag, t.norm().item())
    return None


@debug_log_rank.register_fake  # pyrefly: ignore[missing-attribute]
def debug_log_rank_fake(t, tag, ranks=None):
    return None


@debug_log_rank.register_hook  # pyrefly: ignore[missing-attribute]
def debug_log_rank_hook(t_grad, tag, ranks=None):
    if _should_log(ranks):
        log.info("[rank %d][%s][bwd] norm=%s", _get_rank(), tag, t_grad.norm().item())
