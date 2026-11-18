# Copyright (c) Meta Platforms, Inc. and affiliates
# Backward compatibility stub - this module has been moved to _context_parallel/_attention.py

from ._context_parallel._attention import (
    _CausalBehavior,
    _context_parallel_shard,
    _ContextParallel,
    _cp_options,
    _disable_context_parallel_dispatcher,
    _enable_context_parallel_dispatcher,
    _is_causal_behavior,
    _RotateMethod,
    _templated_ring_attention,
    context_parallel,
    context_parallel_unshard,
    set_rotate_method,
)
from ._context_parallel._load_balancer import (
    _HeadTailLoadBalancer,
    _LoadBalancer,
    _PerDocumentHeadTailLoadBalancer,
    _PTRRLoadBalancer,
)


# TODO(fegin): add deprecation message once the final interfaces are concluded.
__all__ = [
    "_CausalBehavior",
    "_context_parallel_shard",
    "_ContextParallel",
    "_cp_options",
    "_disable_context_parallel_dispatcher",
    "_enable_context_parallel_dispatcher",
    "_is_causal_behavior",
    "_RotateMethod",
    "_templated_ring_attention",
    "context_parallel",
    "context_parallel_unshard",
    "set_rotate_method",
    "_HeadTailLoadBalancer",
    "_LoadBalancer",
    "_PerDocumentHeadTailLoadBalancer",
    "_PTRRLoadBalancer",
]
