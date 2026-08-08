"""
[not for land] Prototype: backward epilogue fusion across autograd.Function nodes.

A UX prototype for CODA-style backward epilogue fusion ("CODA: Rewriting
Transformer Blocks as GEMM-Epilogue Programs", arXiv:2605.19269). The problem:
the backward of an autograd.Function is not self-contained -- for a chain
mm -> epilogue -> mm, the epilogue's backward wants to fuse as the epilogue of
the *next* matmul's backward (grad_a = (grad_c @ W^T) * f'(a) is a matmul with a
pointwise epilogue). A single Function can't express that, since its backward
runs inside one node.

The approach: the user writes a "1-to-many" op that decomposes into two autograd
nodes (a matmul node and an epilogue marker node), plus a fusion rule. Before
backward, apply_epilogue_fusion() walks the graph and arms the matmul node to
defer its activation gradient into the previous epilogue's backward (no graph
rewriting; deferral rides a placeholder grad along the existing edge).

The user's forward receives TWO ctxs and saves into each explicitly:

    def forward(main_ctx, epilogue_ctx, x, w):
        a = x @ w
        main_ctx.save_for_backward(x, w)      # tensors the matmul backward needs
        epilogue_ctx.save_for_backward(a)     # tensors the pointwise epilogue needs
        return relu(a)

The framework saves each set on the corresponding autograd node, so:
  * `main_backward(main_ctx, grad)` reads `main_ctx.saved_tensors`, honors
    `main_ctx.needs_input_grad`, and returns a grad per forward input. When this
    node defers, the framework sets needs_input_grad[activation] = False, so the
    user simply returns None for that slot (as for any not-needed grad) and the
    fused kernel produces it instead;
  * `epilogue_backward(epilogue_ctx, ...)` reads `epilogue_ctx.saved_tensors`;
  * saved sets are partitioned per node (the epilogue tensor `a` lives only on the
    epilogue node), and the deferred producer's placeholder carries only its main
    set -- never the epilogue set.

The fused backward kernel receives the producer's main saved set (a plain tuple)
and the consumer ctx, each exposing only its own subset:

    fused_impl(grad_producer_out, main_saved_tensors, consumer_ctx) -> grad_consumer_main_out
        main_saved_tensors          # the producer op's main set (e.g. x, w)
        consumer_ctx.saved_tensors  # the consumer op's epilogue set (e.g. a)

Fusion rules are passed explicitly (no global registry) as a list of
(producer.main_backward, consumer.epilogue_backward, fused_impl) and applied in
a single step:

    apply_epilogue_fusion(loss.grad_fn, rules, expect_num_fusions=2)
    loss.backward()

Each op below is self-contained with its own backwards.

Run:
    python coda_backward_epilogue_fusion.py
"""

from collections import deque
from dataclasses import dataclass

import torch
from torch.autograd import Function


class _Log:
    def __init__(self):
        self.reset()

    def reset(self):
        self.c = dict(main_full=0, main_params_only=0, fused_impl=0, epilogue_unfused=0)

    def hit(self, k):
        self.c[k] += 1

    def __repr__(self):
        return repr(self.c)


LOG = _Log()


class DeferredGradTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, shape, dtype, device, main_grad_out, main_saved_tensors):
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            shape,
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        r._main_grad_out = main_grad_out
        r._main_saved_tensors = main_saved_tensors
        return r

    __torch_function__ = torch._C._disabled_torch_function_impl  # type: ignore[attr-defined]

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        raise RuntimeError(
            f"DeferredGradTensor is a metadata-only placeholder for a deferred "
            f"grad_input and must not be used in a real op (got {func}). The "
            f"consumer epilogue node should detect it and unwrap `._main_grad_out`; "
            f"reaching __torch_dispatch__ means the placeholder leaked into "
            f"computation."
        )


class _WrappedCtx:
    """Forwards every attribute to an inner ctx except the overridden ones, so a
    backward can be handed a ctx that differs in just one field. Mirrors the
    WrappedCtx pattern in torch/_functorch/autograd_function.py."""

    _reserved = ("_inner_ctx",)

    def __init__(self, ctx):
        self._inner_ctx = ctx

    def __getattr__(self, name):
        return getattr(self._inner_ctx, name)

    def __setattr__(self, name, value):
        if name in type(self)._reserved:
            self.__dict__[name] = value
        else:
            setattr(self._inner_ctx, name, value)


class _MainBackwardCtx(_WrappedCtx):
    """The ctx handed to main_backward: overrides needs_input_grad (deferred
    activation slot forced to False) and saved_tensors (served from a set read once
    in _MainNode.backward, since a checkpoint region errors on a second unpack)."""

    _reserved = ("_needs", "_saved", *_WrappedCtx._reserved)

    def __init__(self, ctx, needs_input_grad, saved_tensors):
        super().__init__(ctx)
        self._needs = needs_input_grad
        self._saved = saved_tensors

    @property
    def needs_input_grad(self):
        return self._needs

    @property
    def saved_tensors(self):
        return self._saved


class _StagingCtx:
    def __init__(self):
        self.saved = ()
        self.output_meta = None

    def save_for_backward(self, *tensors):
        self.saved = tensors

    def set_output_meta(self, like):
        # We need the user to explicitly set this because the forward is fused;
        # we don't know what the metadata for the intermediate is.
        self.output_meta = (like.shape, like.dtype, like.device)


_MAIN_BACKWARD = "_MainNodeBackward"
_EPILOGUE_BACKWARD = "_EpilogueNodeBackward"


def _is_main(node):
    return type(node).__name__ == _MAIN_BACKWARD


def _is_epilogue(node):
    return type(node).__name__ == _EPILOGUE_BACKWARD


class _MainNode(Function):
    @staticmethod
    def forward(ctx, cls, meta, main_saved, *inps):
        ctx.cls = cls
        ctx.in_metas = tuple((t.shape, t.dtype, t.device) for t in inps)
        ctx.save_for_backward(*main_saved)
        ctx.defer_input_idx = None  # set by the plan; None means do not defer
        shape, dtype, device = meta
        return torch.empty(shape, dtype=dtype, device=device)

    @staticmethod
    def backward(ctx, grad_main_out):
        cls = ctx.cls
        saved = ctx.saved_tensors
        needs_input_grad = list(ctx.needs_input_grad[3:])
        k = ctx.defer_input_idx
        if k is not None:
            needs_input_grad[k] = False
        bw_ctx = _MainBackwardCtx(ctx, tuple(needs_input_grad), saved)
        LOG.hit("main_params_only" if k is not None else "main_full")
        grads = list(cls.main_backward(bw_ctx, grad_main_out))
        if k is not None:
            shape, dtype, device = ctx.in_metas[k]
            grads[k] = DeferredGradTensor(shape, dtype, device, grad_main_out, saved)
        return (None, None, None) + tuple(grads)


class _EpilogueNode(Function):
    @staticmethod
    def forward(ctx, cls, epilogue_saved, out_holder, main_out):
        ctx.cls = cls
        ctx.save_for_backward(*epilogue_saved)
        ctx.fused_impl = None  # set by the plan when this node fuses
        (out,) = out_holder
        return out.view_as(out)

    @staticmethod
    def backward(ctx, grad_out):
        cls = ctx.cls
        if isinstance(grad_out, DeferredGradTensor):
            LOG.hit("fused_impl")
            grad_main_out = ctx.fused_impl(
                grad_out._main_grad_out, grad_out._main_saved_tensors, ctx
            )
        else:
            LOG.hit("epilogue_unfused")
            grad_main_out = cls.epilogue_backward(ctx, grad_out)
        return (None, None, None, grad_main_out)


class FusibleFunction:
    r"""A fusible op: one forward that decomposes into two autograd nodes -- a "main"
    node (the matmul) and an "epilogue" node (the pointwise tail) -- so the epilogue's
    backward can later be fused into the next op's matmul backward by
    :func:`apply_epilogue_fusion`. Used on its own (without fusion) it behaves like an
    ordinary autograd Function.

    Subclasses implement three staticmethods:

    ``forward(main_ctx, epilogue_ctx, *inputs) -> output``
        Runs the full forward. Saves the tensors each backward needs into the
        corresponding ctx (``main_ctx.save_for_backward(...)`` for the matmul backward,
        ``epilogue_ctx.save_for_backward(...)`` for the pointwise backward), and MUST
        call ``main_ctx.set_output_meta(a)`` to declare the intermediate ``a`` (the
        GEMM output flowing between the two nodes). That metadata cannot be inferred
        from the final output (e.g. SwiGLU is dim-reducing), and ``apply`` raises if
        it is missing.

    ``main_backward(main_ctx, grad_main_out) -> grads``
        The matmul backward; reads ``main_ctx.saved_tensors`` and returns one grad per
        forward input, honoring ``main_ctx.needs_input_grad``. Compute the weight grad
        (dW) unconditionally first and guard the activation grad (dx) with
        ``needs_input_grad``. When this op's activation grad is deferred into a fused
        kernel, the framework sets ``needs_input_grad`` to ``False`` at the activation
        slot, so ``main_backward`` returns ``None`` there and the fused kernel produces
        it instead.

    ``epilogue_backward(epilogue_ctx, grad_out) -> grad_main_out``
        The pointwise backward; reads ``epilogue_ctx.saved_tensors``.

    Examples::

        >>> import torch
        >>> class MMRelu(FusibleFunction):
        ...     @staticmethod
        ...     def forward(main_ctx, epilogue_ctx, x, w):
        ...         a = x @ w
        ...         main_ctx.save_for_backward(x, w)
        ...         main_ctx.set_output_meta(a)        # REQUIRED: boundary meta (else apply raises)
        ...         epilogue_ctx.save_for_backward(a)
        ...         return torch.relu(a)
        ...     @staticmethod
        ...     def main_backward(main_ctx, grad_a):   # the matmul backward
        ...         x, w = main_ctx.saved_tensors
        ...         # dW first (always local); dx (input 0) is guarded by needs_input_grad
        ...         # and skipped when deferred into the fused kernel.
        ...         gw = x.T @ grad_a if main_ctx.needs_input_grad[1] else None
        ...         gx = grad_a @ w.T if main_ctx.needs_input_grad[0] else None
        ...         return gx, gw
        ...     @staticmethod
        ...     def epilogue_backward(epilogue_ctx, grad_out):   # the pointwise backward
        ...         (a,) = epilogue_ctx.saved_tensors
        ...         return grad_out * (a > 0).to(a.dtype)
        >>>
        >>> x = torch.randn(4, 6, requires_grad=True)
        >>> w1 = torch.randn(6, 6, requires_grad=True)
        >>> w2 = torch.randn(6, 6, requires_grad=True)
        >>> out = MMRelu.apply(MMRelu.apply(x, w1), w2)   # works as a normal op
        >>> out.sum().backward()

    To fuse each epilogue's backward into the next matmul's backward, pass a rule to
    :func:`apply_epilogue_fusion` before calling ``backward``::

        >>> def mm_bw_relu_bw_fused(grad_producer_out, main_saved_tensors, consumer_ctx):
        ...     _x, w = main_saved_tensors             # producer's matmul weight
        ...     (a,) = consumer_ctx.saved_tensors      # consumer's preactivation
        ...     return (grad_producer_out @ w.T) * (a > 0).to(a.dtype)
        >>>
        >>> rules = [(MMRelu.main_backward, MMRelu.epilogue_backward, mm_bw_relu_bw_fused)]
        >>> out = MMRelu.apply(MMRelu.apply(x, w1), w2)
        >>> loss = out.sum()
        >>> apply_epilogue_fusion(loss, rules, expect_num_fusions=1)
        >>> loss.backward()   # relu1's backward runs fused into mm2's grad_input GEMM
    """

    @classmethod
    def apply(cls, *inputs):
        # Run the user's forward once, outside both nodes, against staging ctxs.
        main_staging = _StagingCtx()
        epilogue_staging = _StagingCtx()
        with torch.no_grad():
            out = cls.forward(main_staging, epilogue_staging, *inputs)
        # The intermediate may differ in shape from `out` (e.g. SwiGLU is
        # dim-reducing), so the user must declare its metadata; we can't infer it.
        if main_staging.output_meta is None:
            raise RuntimeError(
                f"{cls.__name__}.forward must call main_ctx.set_output_meta(...) to "
                f"declare the intermediate (main output) metadata"
            )
        # cls and the saved sets pass explicitly into the module-level nodes; `out`
        # rides in a 1-tuple to stay a non-autograd input to the epilogue.
        main_out = _MainNode.apply(
            cls, main_staging.output_meta, main_staging.saved, *inputs
        )
        return _EpilogueNode.apply(cls, epilogue_staging.saved, (out,), main_out)

    @staticmethod
    def forward(main_ctx, epilogue_ctx, *inputs):
        raise NotImplementedError

    @staticmethod
    def main_backward(main_ctx, grad_main_out):
        raise NotImplementedError

    @staticmethod
    def epilogue_backward(epilogue_ctx, grad_out):
        raise NotImplementedError


@dataclass
class _PlannedPair:
    producer: object
    consumer: object
    unfused_reason: str | None  # None when fusion is armed; else why no rule matched

    @property
    def fused(self):
        return self.unfused_reason is None

    def _label(self):
        return (
            f"{self.producer.cls.__name__}.main_backward -> "
            f"{self.consumer.cls.__name__}.epilogue_backward"
        )


class _InternalDebugFusionPlan:
    def __init__(self, fused, missing_rules):
        self._pairs_fused = fused
        self._pairs_missing_rules = missing_rules

    def assert_num_fusions(self, expected):
        got = len(self._pairs_fused)
        if got != expected:
            lines = [f"expected {expected} backward fusions, planned {got}"]
            if self._pairs_missing_rules:
                lines.append(
                    f"{len(self._pairs_missing_rules)} fusible main -> epilogue "
                    f"adjacency(ies) have no registered rule:"
                )
                lines += [f"  - {p._label()}" for p in self._pairs_missing_rules]
                lines.append(
                    "Did you forget to pass a rule for these to "
                    "apply_epilogue_fusion(rules=...)?"
                )
            raise AssertionError("\n".join(lines))
        return self

    def __repr__(self):
        all_pairs = self._pairs_fused + self._pairs_missing_rules
        name = type(self).__name__
        if not all_pairs:
            return f"{name}(no candidates)"
        return (
            f"{name}(\n  "
            + "\n  ".join(
                f"{p.producer.cls.__name__}.main -> "
                f"{p.consumer.cls.__name__}.epilogue: "
                f"{'FUSE' if p.fused else 'bail:' + p.unfused_reason}"
                for p in all_pairs
            )
            + "\n)"
        )


def apply_epilogue_fusion(
    root, rules, *, expect_num_fusions=None, _internal_debug=False
):
    r"""Applies epilogue fusion rules to :class:`FusibleFunction` nodes in the autograd graph.

    Each rule in :attr:`rules` specifies a pair of ``main_backward`` and
    ``epilogue_backward`` methods together with the fused implementation that should run
    in their place. This function traverses the autograd graph starting from
    :attr:`root`, and for each adjacent pair of nodes matching a rule, mutates those
    nodes in place so the fused implementation runs instead.

    This function should be called after the forward pass and before :meth:`backward`,
    on every iteration.

    It only operates on :class:`FusibleFunction` subclasses; see that class for more
    details.

    Args:
        root (Tensor or Node): the loss tensor (or its ``grad_fn``) to traverse
            back from.
        rules (list): fusion rules, each a tuple
            ``(producer_cls.main_backward, consumer_cls.epilogue_backward, fused_impl)``.
            ``fused_impl(grad_producer_out, main_saved_tensors, consumer_ctx)`` returns
            the grad of the consumer's main output, computing the producer's deferred
            ``grad_input`` GEMM with the consumer's epilogue fused on. Only registered
            pairs fuse.
        expect_num_fusions (int, optional): if given, assert exactly this many
            fusions were planned, raising a diagnostic that names any fusible
            adjacency lacking a rule. This is the supported way to check coverage.
            Default: ``None``.

    Returns:
        None. (For debugging only, passing ``_internal_debug=True`` returns an
        internal plan object describing the planned fusions; its shape is not stable
        and not part of the public API.)

    Examples::

        >>> import torch
        >>> class MMRelu(FusibleFunction):
        ...     @staticmethod
        ...     def forward(main_ctx, epilogue_ctx, x, w):
        ...         a = x @ w
        ...         main_ctx.save_for_backward(x, w)
        ...         main_ctx.set_output_meta(a)        # REQUIRED: boundary meta (else apply raises)
        ...         epilogue_ctx.save_for_backward(a)
        ...         return torch.relu(a)
        ...     @staticmethod
        ...     def main_backward(main_ctx, grad_a):   # the matmul backward
        ...         x, w = main_ctx.saved_tensors
        ...         gw = x.T @ grad_a if main_ctx.needs_input_grad[1] else None
        ...         gx = grad_a @ w.T if main_ctx.needs_input_grad[0] else None
        ...         return gx, gw
        ...     @staticmethod
        ...     def epilogue_backward(epilogue_ctx, grad_out):   # the pointwise backward
        ...         (a,) = epilogue_ctx.saved_tensors
        ...         return grad_out * (a > 0).to(a.dtype)
        >>>
        >>> def mm_bw_relu_bw_fused(grad_producer_out, main_saved_tensors, consumer_ctx):
        ...     _x, w = main_saved_tensors             # producer's matmul weight
        ...     (a,) = consumer_ctx.saved_tensors      # consumer's preactivation
        ...     return (grad_producer_out @ w.T) * (a > 0).to(a.dtype)
        >>>
        >>> rules = [(MMRelu.main_backward, MMRelu.epilogue_backward, mm_bw_relu_bw_fused)]
        >>> x = torch.randn(4, 6, requires_grad=True)
        >>> w1 = torch.randn(6, 6, requires_grad=True)
        >>> w2 = torch.randn(6, 6, requires_grad=True)
        >>> out = MMRelu.apply(MMRelu.apply(x, w1), w2)      # mm1 -> relu -> mm2 -> relu
        >>> loss = out.sum()
        >>> apply_epilogue_fusion(loss, rules, expect_num_fusions=1)
        >>> loss.backward()   # relu1's backward runs fused into mm2's grad_input GEMM
    """
    if isinstance(root, torch.Tensor):
        root = root.grad_fn

    rule_map = {(p, c): impl for p, c, impl in rules}

    nodes, in_degree, seen = [], {}, set()
    q = deque()
    if root is not None:
        seen.add(root)
        q.append(root)
    while q:
        node = q.popleft()
        nodes.append(node)
        for fn, _ in node.next_functions:
            if fn is None:
                continue
            in_degree[fn] = in_degree.get(fn, 0) + 1
            if fn not in seen:
                seen.add(fn)
                q.append(fn)

    fused, missing_rules = [], []
    for n in nodes:
        if not _is_main(n):
            continue
        candidates = [
            (i, c)
            for i, (c, _) in enumerate(n.next_functions)
            if c is not None and _is_epilogue(c)
        ]
        if len(candidates) > 1:
            raise RuntimeError(
                f"{type(n).__name__}: {len(candidates)} inputs feed epilogue nodes; "
                f"deferring more than one grad_input is not supported"
            )
        if not candidates:
            continue
        idx, consumer = candidates[0]
        impl = rule_map.get((n.cls.main_backward, consumer.cls.epilogue_backward))
        if impl is None:
            # Structural candidate (a main feeds an epilogue) but no rule given.
            missing_rules.append(_PlannedPair(n, consumer, "no rule registered"))
            continue
        if in_degree.get(consumer, 0) > 1:
            raise RuntimeError(
                f"Cannot fuse {n.cls.__name__}.main_backward into "
                f"{consumer.cls.__name__}.epilogue_backward: the epilogue output "
                f"feeds {in_degree[consumer]} downstream main nodes, so its backward "
                f"grad is accumulated across those branches. Deferral rides a single "
                f"placeholder along one edge and cannot represent that accumulation. "
                f"Backward epilogue fusion requires the epilogue output to have "
                f"exactly one consumer."
            )
        n.defer_input_idx = idx
        consumer.fused_impl = impl
        fused.append(_PlannedPair(n, consumer, None))  # None == fusion armed

    plan = _InternalDebugFusionPlan(fused, missing_rules)
    if expect_num_fusions is not None:
        plan.assert_num_fusions(expect_num_fusions)
    if _internal_debug:
        return plan
    return None


# ===========================================================================
# Example user code: two self-contained ops, each with its own backwards.
# ===========================================================================
class MMRelu(FusibleFunction):
    @staticmethod
    def forward(main_ctx, epilogue_ctx, x, w):
        a = x @ w
        main_ctx.save_for_backward(x, w)
        main_ctx.set_output_meta(a)  # REQUIRED (else apply raises): boundary metadata
        epilogue_ctx.save_for_backward(a)
        return torch.relu(a)

    @staticmethod
    def main_backward(main_ctx, grad_main_out):
        x, w = main_ctx.saved_tensors
        # dW first (always local); dx (input 0) is guarded and skipped when deferred.
        grad_w = (
            x.transpose(-1, -2) @ grad_main_out
            if main_ctx.needs_input_grad[1]
            else None
        )
        grad_x = (
            grad_main_out @ w.transpose(-1, -2)
            if main_ctx.needs_input_grad[0]
            else None
        )
        return grad_x, grad_w

    @staticmethod
    def epilogue_backward(epilogue_ctx, grad_out):
        (a,) = epilogue_ctx.saved_tensors
        return grad_out * (a > 0).to(a.dtype)


class MMTanh(FusibleFunction):
    @staticmethod
    def forward(main_ctx, epilogue_ctx, x, w):
        a = x @ w
        main_ctx.save_for_backward(x, w)
        main_ctx.set_output_meta(a)  # REQUIRED (else apply raises): boundary metadata
        epilogue_ctx.save_for_backward(a)
        return torch.tanh(a)

    @staticmethod
    def main_backward(main_ctx, grad_main_out):
        x, w = main_ctx.saved_tensors
        # dW first (always local); dx is guarded and skipped when deferred.
        grad_w = (
            x.transpose(-1, -2) @ grad_main_out
            if main_ctx.needs_input_grad[1]
            else None
        )
        grad_x = (
            grad_main_out @ w.transpose(-1, -2)
            if main_ctx.needs_input_grad[0]
            else None
        )
        return grad_x, grad_w

    @staticmethod
    def epilogue_backward(epilogue_ctx, grad_out):
        (a,) = epilogue_ctx.saved_tensors
        return grad_out * (1 - torch.tanh(a) ** 2)


def mm_relu_fused_backward(grad_producer_out, main_saved_tensors, consumer_ctx):
    _x_p, w_p = main_saved_tensors
    (a_c,) = consumer_ctx.saved_tensors
    grad_main_input = grad_producer_out @ w_p.transpose(-1, -2)
    return grad_main_input * (a_c > 0).to(a_c.dtype)


def mm_tanh_fused_backward(grad_producer_out, main_saved_tensors, consumer_ctx):
    _x_p, w_p = main_saved_tensors
    (a_c,) = consumer_ctx.saved_tensors
    grad_main_input = grad_producer_out @ w_p.transpose(-1, -2)
    return grad_main_input * (1 - torch.tanh(a_c) ** 2)


# One rule per (producer.main_backward, consumer.epilogue_backward) pair, passed
# explicitly to apply_epilogue_fusion. The fused impl depends only on the
# consumer's epilogue, so both producers reuse the same impl.
RULES = [
    (MMRelu.main_backward, MMRelu.epilogue_backward, mm_relu_fused_backward),
    (MMTanh.main_backward, MMRelu.epilogue_backward, mm_relu_fused_backward),
    (MMRelu.main_backward, MMTanh.epilogue_backward, mm_tanh_fused_backward),
    (MMTanh.main_backward, MMTanh.epilogue_backward, mm_tanh_fused_backward),
]


# ===========================================================================
# Verification.
# ===========================================================================
def _check(name, ins, refs, atol=1e-9):
    ok = True
    print(f"=== gradient check: {name} ===")
    labels = ["x"] + [f"w{i}" for i in range(1, len(ins))]
    for nm, t, r in zip(labels, ins, refs):
        err = (t.grad - r).abs().max().item()
        good = torch.allclose(t.grad, r, atol=atol)
        ok &= good
        print(f"  grad_{nm}: max_abs_err={err:.2e} ok={good}")
    assert ok, f"{name}: gradients do not match reference"


def scenario_mixed_chain():
    """x -> MMTanh -> MMRelu -> MMTanh -> sum.

    Fusions:
      ep1 (MMTanh) <- main2 (MMRelu): key (MMRelu.main, MMTanh.epilogue) -> tanh rule
      ep2 (MMRelu) <- main3 (MMTanh): key (MMTanh.main, MMRelu.epilogue) -> relu rule
    """
    print("\n########## scenario: mixed chain ##########")
    torch.manual_seed(0)
    B, K = 4, 6

    def make():
        return [
            torch.randn(B if i == 0 else K, K, dtype=torch.double, requires_grad=True)
            for i in range(4)
        ]

    ref = make()
    x, w1, w2, w3 = ref
    torch.tanh(torch.relu(torch.tanh(x @ w1) @ w2) @ w3).sum().backward()
    refs = [t.grad.clone() for t in ref]

    ins = make()
    for t, r in zip(ins, ref):
        t.data.copy_(r.data)
    x, w1, w2, w3 = ins

    LOG.reset()
    loss = MMTanh.apply(MMRelu.apply(MMTanh.apply(x, w1), w2), w3).sum()
    plan = apply_epilogue_fusion(
        loss.grad_fn, RULES, expect_num_fusions=2, _internal_debug=True
    )
    print(plan)
    loss.backward()

    _check("mixed", ins, refs)
    print("kernel paths:", LOG)
    assert LOG.c["fused_impl"] == 2
    assert LOG.c["epilogue_unfused"] == 1
    assert LOG.c["main_params_only"] == 2
    assert LOG.c["main_full"] == 1
    print("PASS: both epilogues fused across the mixed chain.")


if __name__ == "__main__":
    scenario_mixed_chain()
    print("\nALL SCENARIOS PASSED")
