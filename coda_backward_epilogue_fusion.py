"""
[not for land] Prototype: backward epilogue fusion across autograd.Function nodes.

A UX prototype for CODA-style backward epilogue fusion ("CODA: Rewriting
Transformer Blocks as GEMM-Epilogue Programs", arXiv:2605.19269). The problem:
the backward of an autograd.Function is not self-contained -- for a chain
mm -> epilogue -> mm, the epilogue's backward wants to fuse as the epilogue of
the *next* matmul's backward (grad_a = (grad_c @ W^T) * f'(a) is a matmul with a
pointwise epilogue). A single Function can't express that, since its backward
runs inside one node. A multi-use activation adds another boundary: the autograd
engine normally accumulates its branch gradients in the consumer's InputBuffer
before running the epilogue backward.

The approach: the user writes a "1-to-many" op that decomposes into two autograd
nodes (a matmul node and an epilogue marker node), plus a fusion rule. Before
backward, apply_epilogue_fusion() walks the graph and arms the matmul node to
defer its activation gradient into the previous epilogue's backward (no graph
rewriting; deferral rides a placeholder grad along the existing edge). It can
also fuse a main-backward output with a leaf AccumulateGrad, so the GEMM writes
or adds directly into the leaf's .grad instead of materializing a separate grad.
For multi-use activations, split_multi_use() explicitly writes out one aliased
output per use. Their gradients occupy distinct splitter InputBuffer slots, so
the splitter can either fuse their accumulation directly or package all deferred
GEMM terms for a rule that also replaces an upstream epilogue.

The user's forward receives TWO ctxs and saves into each explicitly:

    def forward(main_ctx, epilogue_ctx, x, w):
        a = x @ w
        main_ctx.save_for_backward(x, w)      # tensors the matmul backward needs
        epilogue_ctx.save_for_backward(a)     # tensors the pointwise epilogue needs
        return relu(a)

The framework saves each set on the corresponding autograd node, so:
  * `main_backward(main_ctx, grad)` reads `main_ctx.saved_tensors`, honors
    `main_ctx.needs_input_grad`, and returns a grad per forward input. When an
    output is deferred, the framework sets needs_input_grad[input] = False, so
    the user simply returns None for that slot (as for any not-needed grad) and
    the fused kernel produces it instead;
  * `epilogue_backward(epilogue_ctx, ...)` reads `epilogue_ctx.saved_tensors`;
  * saved sets are partitioned per node (the epilogue tensor `a` lives only on the
    epilogue node), and the deferred producer's placeholder carries only its main
    set -- never the epilogue set.

The fused backward kernel receives the producer's main saved set (a plain tuple)
and the consumer ctx, each exposing only its own subset:

    fused_impl(grad_producer_out, main_saved_tensors, consumer_ctx) -> grad_consumer_main_out
        main_saved_tensors          # the producer op's main set (e.g. x, w)
        consumer_ctx.saved_tensors  # the consumer op's epilogue set (e.g. a)

A fused accumulation kernel instead receives the destination leaf and updates
its grad directly:

    fused_accumulate(grad_producer_out, main_saved_tensors, variable) -> None
        variable.grad = ...                         # first backward
        addmm(variable.grad, ..., out=variable.grad) # subsequent accumulation

A splitter-only replacement receives all ordered branch terms and returns the
accumulated activation gradient:

    fused_multiuse_accumulate(((grad1, saved1), (grad2, saved2)))

A multi-use epilogue replacement additionally receives the consumer context:

    fused_multiuse(((grad1, saved1), (grad2, saved2)), consumer_ctx)

Fusion rules are passed explicitly (no global registry) as ``(pattern,
replacement)`` pairs and applied in a single step. AccumulateGrad patterns
separately identify the main-backward output by forward input index:

    accumulate_grad_rules = [
        ((producer.main_backward, input_idx), fused_accumulate),
    ]
    rules.append(
        (
            (
                (producer1.main_backward, producer2.main_backward),
                split_multi_use,
                consumer.epilogue_backward,
            ),
            fused_multiuse,
        )
    )
    apply_epilogue_fusion(
        loss.grad_fn,
        rules,
        accumulate_grad_rules=accumulate_grad_rules,
        expect_num_fusions=2,
    )
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
        self.c = dict(
            main_full=0,
            main_params_only=0,
            main_partial=0,
            main_fully_deferred=0,
            fused_impl=0,
            multiuse_fused=0,
            splitter_fused=0,
            splitter_deferred=0,
            splitter_unfused=0,
            accumulate_fused=0,
            epilogue_unfused=0,
        )

    def hit(self, k):
        self.c[k] += 1

    def __repr__(self):
        return repr(self.c)


LOG = _Log()


class DeferredGradTensor(torch.Tensor):
    @classmethod
    def _new_wrapper(cls, shape, dtype, device):
        return torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            shape,
            dtype=dtype,
            device=device,
            requires_grad=False,
        )

    @staticmethod
    def __new__(cls, shape, dtype, device, main_grad_out, main_saved_tensors):
        r = cls._new_wrapper(shape, dtype, device)
        r._main_grad_out = main_grad_out
        r._main_saved_tensors = main_saved_tensors
        r._main_grad_terms = ((main_grad_out, main_saved_tensors),)
        return r

    @classmethod
    def from_terms(cls, shape, dtype, device, terms):
        r = cls._new_wrapper(shape, dtype, device)
        r._main_grad_terms = tuple(terms)
        return r

    __torch_function__ = torch._C._disabled_torch_function_impl  # type: ignore[attr-defined]

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        raise RuntimeError(
            f"DeferredGradTensor is a metadata-only placeholder for a deferred "
            f"grad_input and must not be used in a real op (got {func}). The "
            f"consumer epilogue or explicit splitter should detect and unwrap it; "
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
    """The ctx handed to main_backward: overrides needs_input_grad (deferred input
    slots forced to False) and saved_tensors (served from a set read once in
    _MainNode.backward, since a checkpoint region errors on a second unpack)."""

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
_SPLITTER_BACKWARD = "_SplitterNodeBackward"
_ACCUMULATE_GRAD = "AccumulateGrad"


def _is_main(node):
    return type(node).__name__ == _MAIN_BACKWARD


def _is_epilogue(node):
    return type(node).__name__ == _EPILOGUE_BACKWARD


def _is_splitter(node):
    return type(node).__name__ == _SPLITTER_BACKWARD


def _is_accumulate_grad(node):
    return type(node).__name__ == _ACCUMULATE_GRAD


class _MainNode(Function):
    @staticmethod
    def forward(ctx, cls, meta, main_saved, *inps):
        ctx.cls = cls
        ctx.in_metas = tuple((t.shape, t.dtype, t.device) for t in inps)
        ctx.save_for_backward(*main_saved)
        ctx.defer_input_idx = None  # set by the plan; None means do not defer
        ctx.defer_accumulate_input_idxs = ()
        shape, dtype, device = meta
        return torch.empty(shape, dtype=dtype, device=device)

    @staticmethod
    def backward(ctx, grad_main_out):
        cls = ctx.cls
        saved = ctx.saved_tensors
        needs_input_grad = list(ctx.needs_input_grad[3:])
        k = ctx.defer_input_idx
        deferred_input_idxs = set(ctx.defer_accumulate_input_idxs)
        if k is not None:
            deferred_input_idxs.add(k)
        for input_idx in deferred_input_idxs:
            needs_input_grad[input_idx] = False
        bw_ctx = _MainBackwardCtx(ctx, tuple(needs_input_grad), saved)
        if ctx.defer_accumulate_input_idxs:
            LOG.hit("main_partial" if any(needs_input_grad) else "main_fully_deferred")
        else:
            LOG.hit("main_params_only" if k is not None else "main_full")
        grads = list(cls.main_backward(bw_ctx, grad_main_out))
        for input_idx in deferred_input_idxs:
            shape, dtype, device = ctx.in_metas[input_idx]
            grads[input_idx] = DeferredGradTensor(
                shape, dtype, device, grad_main_out, saved
            )
        return (None, None, None) + tuple(grads)


class _EpilogueNode(Function):
    @staticmethod
    def forward(ctx, cls, epilogue_saved, out_holder, main_out):
        ctx.cls = cls
        ctx.save_for_backward(*epilogue_saved)
        ctx.fused_impl = None  # set by the plan when this node fuses
        ctx.multiuse_fused_impl = None
        (out,) = out_holder
        return out.view_as(out)

    @staticmethod
    def backward(ctx, grad_out):
        cls = ctx.cls
        if isinstance(grad_out, DeferredGradTensor):
            terms = grad_out._main_grad_terms
            if len(terms) == 1:
                LOG.hit("fused_impl")
                grad_main_out = ctx.fused_impl(*terms[0], ctx)
            else:
                if ctx.multiuse_fused_impl is None:
                    raise RuntimeError(
                        "A multi-use DeferredGradTensor reached an unarmed epilogue"
                    )
                LOG.hit("multiuse_fused")
                grad_main_out = ctx.multiuse_fused_impl(terms, ctx)
        else:
            LOG.hit("epilogue_unfused")
            grad_main_out = cls.epilogue_backward(ctx, grad_out)
        return (None, None, None, grad_main_out)


class _SplitterNode(Function):
    @staticmethod
    def forward(ctx, tensor, num_uses):
        if num_uses < 2:
            raise RuntimeError("split_multi_use requires at least two uses")
        ctx.input_meta = (tensor.shape, tensor.dtype, tensor.device)
        ctx.num_uses = num_uses
        ctx.defer_fusion = False
        ctx.fused_impl = None
        ctx.set_materialize_grads(False)
        return tuple(tensor.view_as(tensor) for _ in range(num_uses))

    @staticmethod
    def backward(ctx, *grads):
        deferred = [isinstance(grad, DeferredGradTensor) for grad in grads]
        if ctx.defer_fusion:
            if not all(deferred):
                raise RuntimeError(
                    "A fused splitter expected one DeferredGradTensor per explicit use"
                )
            LOG.hit("splitter_deferred")
            terms = [term for grad in grads for term in grad._main_grad_terms]
            if ctx.fused_impl is None:
                shape, dtype, device = ctx.input_meta
                grad_input = DeferredGradTensor.from_terms(shape, dtype, device, terms)
            else:
                LOG.hit("splitter_fused")
                grad_input = ctx.fused_impl(terms)
        else:
            if any(deferred):
                raise RuntimeError(
                    "A DeferredGradTensor reached an unarmed explicit splitter"
                )
            LOG.hit("splitter_unfused")
            present_grads = [grad for grad in grads if grad is not None]
            grad_input = (
                None if not present_grads else sum(present_grads[1:], present_grads[0])
            )
        return grad_input, None


def split_multi_use(tensor, num_uses):
    """Returns one aliased output per downstream use of ``tensor``.

    In backward, each output has a distinct InputBuffer slot. A matching rule can
    replace just the branch-gradient accumulation or compose it with an upstream
    epilogue. Without one, the splitter explicitly sums the ordinary gradients.
    """
    return _SplitterNode.apply(tensor, num_uses)


def _make_fused_accumulate_grad_prehook(impl, variable):
    def prehook(grads):
        (grad,) = grads
        if not isinstance(grad, DeferredGradTensor):
            raise RuntimeError(
                "A fused AccumulateGrad expected a DeferredGradTensor from its "
                f"producer, but got {type(grad).__name__}"
            )
        if len(grad._main_grad_terms) != 1:
            raise RuntimeError(
                "A fused AccumulateGrad cannot consume a multi-use deferred gradient"
            )
        LOG.hit("accumulate_fused")
        result = impl(*grad._main_grad_terms[0], variable)
        if result is not None:
            raise RuntimeError(
                "An AccumulateGrad fused implementation must update variable.grad "
                "and return None"
            )
        return (None,)

    return prehook


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
        (dW) and activation grad (dx) only when their corresponding
        ``needs_input_grad`` entries are true. The framework sets an entry to ``False``
        when that output is deferred into either an epilogue or AccumulateGrad fused
        kernel, so ``main_backward`` returns ``None`` there and the fused kernel
        produces it instead.

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
        ...         # Both outputs honor needs_input_grad and may be deferred.
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
        >>> pattern = (MMRelu.main_backward, MMRelu.epilogue_backward)
        >>> rules = [(pattern, mm_bw_relu_bw_fused)]
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


@dataclass
class _PlannedAccumulateGrad:
    producer: object
    input_idx: int
    accumulator: object

    def _label(self):
        return (
            f"{self.producer.cls.__name__}.main_backward input {self.input_idx} -> "
            "AccumulateGrad"
        )


@dataclass
class _PlannedMultiUsePair:
    producers: tuple
    splitter: object
    consumer: object
    unfused_reason: str | None
    includes_epilogue: bool = True

    @property
    def fused(self):
        return self.unfused_reason is None

    def _label(self):
        producers = ", ".join(
            f"{producer.cls.__name__}.main_backward" for producer in self.producers
        )
        label = f"({producers}) -> split_multi_use"
        if self.includes_epilogue:
            label += f" -> {self.consumer.cls.__name__}.epilogue_backward"
        return label


class _InternalDebugFusionPlan:
    def __init__(
        self,
        fused,
        missing_rules,
        multiuse_fused,
        multiuse_unfused,
        accumulate_grad_fused,
    ):
        self._pairs_fused = fused
        self._pairs_missing_rules = missing_rules
        self._multiuse_pairs_fused = multiuse_fused
        self._multiuse_pairs_unfused = multiuse_unfused
        self._accumulate_grad_fused = accumulate_grad_fused

    def assert_num_fusions(self, expected):
        got = len(self._pairs_fused) + len(self._multiuse_pairs_fused)
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
            if self._multiuse_pairs_unfused:
                lines.append("Explicit multi-use candidate(s) did not fuse:")
                lines += [
                    f"  - {pair._label()}: {pair.unfused_reason}"
                    for pair in self._multiuse_pairs_unfused
                ]
            raise AssertionError("\n".join(lines))
        return self

    def assert_num_accumulate_grad_fusions(self, expected):
        got = len(self._accumulate_grad_fused)
        if got != expected:
            raise AssertionError(
                f"expected {expected} AccumulateGrad fusions, planned {got}"
            )
        return self

    def __repr__(self):
        all_pairs = self._pairs_fused + self._pairs_missing_rules
        name = type(self).__name__
        all_multiuse = self._multiuse_pairs_fused + self._multiuse_pairs_unfused
        if not all_pairs and not all_multiuse and not self._accumulate_grad_fused:
            return f"{name}(no candidates)"
        lines = [
            f"{p.producer.cls.__name__}.main -> "
            f"{p.consumer.cls.__name__}.epilogue: "
            f"{'FUSE' if p.fused else 'bail:' + p.unfused_reason}"
            for p in all_pairs
        ]
        lines += [
            f"{pair._label()}: "
            f"{'FUSE' if pair.fused else 'bail:' + pair.unfused_reason}"
            for pair in all_multiuse
        ]
        lines += [f"{p._label()}: FUSE" for p in self._accumulate_grad_fused]
        return f"{name}(\n  " + "\n  ".join(lines) + "\n)"


def apply_epilogue_fusion(
    root,
    rules,
    *,
    accumulate_grad_rules=(),
    expect_num_fusions=None,
    expect_num_accumulate_grad_fusions=None,
    _internal_debug=False,
):
    r"""Applies epilogue fusion rules to :class:`FusibleFunction` nodes in the autograd graph.

    Each rule in :attr:`rules` is a ``(pattern, replacement)`` pair. A pattern can
    replace a main-backward/epilogue pair, the accumulation at an explicit splitter,
    or the splitter accumulation composed with an upstream epilogue. This function
    traverses the autograd graph starting from :attr:`root` and arms matching nodes in
    place.

    This function should be called after the forward pass and before :meth:`backward`,
    on every iteration.

    AccumulateGrad fusion requires the matched leaf to have exactly one incoming edge
    in the traversed graph. It is intended for ``backward()`` calls that execute leaf
    accumulators, not ``autograd.grad()`` calls that return gradients directly. Leaf
    Tensor backward hooks are rejected because they run before the accumulator's
    prehooks and would observe the placeholder; post-accumulate hooks remain supported.

    Multi-use fusion requires callers to pass each use through :func:`split_multi_use`.
    Every declared output must feed exactly one main backward. The output order selects
    the ordered producer tuple in the rule. A splitter-only replacement returns a real
    accumulated gradient to any upstream autograd node. A replacement that also
    includes an epilogue requires the splitter to be that epilogue output's only
    consumer.

    Matched main-backward producers must be :class:`FusibleFunction` subclasses, but a
    splitter-only replacement can return its gradient to any upstream autograd node.

    Args:
        root (Tensor or Node): the loss tensor (or its ``grad_fn``) to traverse
            back from.
        rules (list): fusion rules expressed as ``(pattern, replacement)`` pairs.
            Supported patterns are ``(producer.main_backward,
            consumer.epilogue_backward)``, ``((producer1.main_backward, ...),
            split_multi_use)``, and ``((producer1.main_backward, ...),
            split_multi_use, consumer.epilogue_backward)``. A single-use replacement
            receives ``(grad_producer_out, main_saved_tensors, consumer_ctx)``. A
            splitter-only replacement receives ``terms`` and returns the accumulated
            activation gradient. A splitter-plus-epilogue replacement receives
            ``(terms, consumer_ctx)``. Terms are ordered ``(grad_producer_out,
            main_saved_tensors)`` pairs. Only registered patterns fuse.
        accumulate_grad_rules (list): fusion rules expressed as ``(pattern,
            replacement)`` pairs, where the pattern is
            ``(producer_cls.main_backward, input_idx)``. When that main backward input
            feeds a unique leaf ``AccumulateGrad``, the normal gradient is deferred and
            the replacement receives ``(grad_producer_out, main_saved_tensors,
            variable)``. It must accumulate directly into ``variable.grad`` and return
            ``None``. Default: ``()``.
        expect_num_fusions (int, optional): if given, assert exactly this many
            fusions were planned, raising a diagnostic that names any fusible
            adjacency lacking a rule. This is the supported way to check coverage.
            Default: ``None``.
        expect_num_accumulate_grad_fusions (int, optional): if given, assert exactly
            this many main-backward to ``AccumulateGrad`` fusions were planned.
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
        >>> pattern = (MMRelu.main_backward, MMRelu.epilogue_backward)
        >>> rules = [(pattern, mm_bw_relu_bw_fused)]
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

    rule_map = dict(rules)
    accumulate_grad_rule_map = dict(accumulate_grad_rules)

    nodes, in_degree, edge_in_degree, seen = [], {}, {}, set()
    q = deque()
    if root is not None:
        seen.add(root)
        q.append(root)
    while q:
        node = q.popleft()
        nodes.append(node)
        for fn, output_nr in node.next_functions:
            if fn is None:
                continue
            in_degree[fn] = in_degree.get(fn, 0) + 1
            edge = (fn, output_nr)
            edge_in_degree[edge] = edge_in_degree.get(edge, 0) + 1
            if fn not in seen:
                seen.add(fn)
                q.append(fn)

    splitter_producers = {}
    for node in nodes:
        if not _is_main(node):
            continue
        for input_idx, (next_node, output_nr) in enumerate(node.next_functions):
            if next_node is not None and _is_splitter(next_node):
                splitter_producers.setdefault(next_node, []).append(
                    (output_nr, node, input_idx)
                )

    multiuse_fused, multiuse_unfused = [], []
    for splitter, entries in splitter_producers.items():
        consumer = splitter.next_functions[0][0]
        has_epilogue = consumer is not None and _is_epilogue(consumer)

        by_output = {}
        for output_nr, producer, input_idx in entries:
            by_output.setdefault(output_nr, []).append((producer, input_idx))
        ordered_entries = []
        invalid_outputs = False
        for output_nr in range(splitter.num_uses):
            matches = by_output.get(output_nr, [])
            if len(matches) != 1 or edge_in_degree.get((splitter, output_nr), 0) != 1:
                invalid_outputs = True
                break
            producer, input_idx = matches[0]
            ordered_entries.append((producer, input_idx))

        producers = tuple(
            producer for _, producer, _ in sorted(entries, key=lambda entry: entry[0])
        )
        if invalid_outputs or len(entries) != splitter.num_uses:
            multiuse_unfused.append(
                _PlannedMultiUsePair(
                    producers,
                    splitter,
                    consumer,
                    "each splitter output must feed exactly one main backward",
                    includes_epilogue=has_epilogue,
                )
            )
            continue
        if len({producer for producer, _ in ordered_entries}) != len(ordered_entries):
            multiuse_unfused.append(
                _PlannedMultiUsePair(
                    producers,
                    splitter,
                    consumer,
                    "one main backward consumes multiple splitter outputs",
                    includes_epilogue=has_epilogue,
                )
            )
            continue
        if any(producer.defer_input_idx is not None for producer, _ in ordered_entries):
            multiuse_unfused.append(
                _PlannedMultiUsePair(
                    producers,
                    splitter,
                    consumer,
                    "a main backward already participates in another fusion",
                    includes_epilogue=has_epilogue,
                )
            )
            continue
        producer_methods = tuple(
            producer.cls.main_backward for producer, _ in ordered_entries
        )
        splitter_impl = rule_map.get((producer_methods, split_multi_use))
        epilogue_impl = None
        if has_epilogue:
            epilogue_impl = rule_map.get(
                (
                    producer_methods,
                    split_multi_use,
                    consumer.cls.epilogue_backward,
                )
            )
        if (
            epilogue_impl is not None
            and in_degree.get(consumer, 0) != 1
            and splitter_impl is None
        ):
            multiuse_unfused.append(
                _PlannedMultiUsePair(
                    producers,
                    splitter,
                    consumer,
                    "the splitter must be the epilogue output's only consumer",
                    includes_epilogue=True,
                )
            )
            continue

        includes_epilogue = (
            epilogue_impl is not None and in_degree.get(consumer, 0) == 1
        )
        impl = epilogue_impl if includes_epilogue else splitter_impl
        pair = _PlannedMultiUsePair(
            producers,
            splitter,
            consumer,
            None,
            includes_epilogue=includes_epilogue,
        )
        if impl is None:
            pair.unfused_reason = "no rule registered"
            pair.includes_epilogue = has_epilogue
            multiuse_unfused.append(pair)
            continue

        splitter.defer_fusion = True
        if includes_epilogue:
            consumer.multiuse_fused_impl = impl
        else:
            splitter.fused_impl = impl
        for producer, input_idx in ordered_entries:
            producer.defer_input_idx = input_idx
        multiuse_fused.append(pair)

    fused, missing_rules, accumulate_grad_fused = [], [], []
    for n in nodes:
        if not _is_main(n):
            continue
        for input_idx, (accumulator, _) in enumerate(n.next_functions):
            impl = accumulate_grad_rule_map.get((n.cls.main_backward, input_idx))
            if (
                impl is None
                or accumulator is None
                or not _is_accumulate_grad(accumulator)
            ):
                continue
            num_producers = in_degree.get(accumulator, 0)
            if num_producers > 1:
                raise RuntimeError(
                    f"Cannot fuse {n.cls.__name__}.main_backward input {input_idx} "
                    f"into AccumulateGrad: the leaf receives gradients from "
                    f"{num_producers} backward edges. Deferred accumulation requires "
                    f"exactly one producer."
                )
            if accumulator.variable._backward_hooks:
                raise RuntimeError(
                    f"Cannot fuse {n.cls.__name__}.main_backward input {input_idx} "
                    "into AccumulateGrad: leaf Tensor backward hooks would observe "
                    "the deferred gradient placeholder"
                )
            n.defer_accumulate_input_idxs = (
                *n.defer_accumulate_input_idxs,
                input_idx,
            )
            accumulator.register_prehook(
                _make_fused_accumulate_grad_prehook(impl, accumulator.variable)
            )
            accumulate_grad_fused.append(
                _PlannedAccumulateGrad(n, input_idx, accumulator)
            )
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
        if candidates and n.defer_input_idx is not None:
            raise RuntimeError(
                f"{type(n).__name__}: multiple inputs feed fusible nodes; "
                "deferring more than one grad_input is not supported"
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

    plan = _InternalDebugFusionPlan(
        fused,
        missing_rules,
        multiuse_fused,
        multiuse_unfused,
        accumulate_grad_fused,
    )
    if expect_num_fusions is not None:
        plan.assert_num_fusions(expect_num_fusions)
    if expect_num_accumulate_grad_fusions is not None:
        plan.assert_num_accumulate_grad_fusions(expect_num_accumulate_grad_fusions)
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
        # Both outputs honor needs_input_grad and may be deferred.
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
        # Both outputs honor needs_input_grad and may be deferred.
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


def mm_multiuse_accumulate_backward(terms):
    grad_main_input = None
    for grad_producer_out, main_saved_tensors in terms:
        _x, w = main_saved_tensors
        branch_grad = grad_producer_out @ w.transpose(-1, -2)
        grad_main_input = (
            branch_grad if grad_main_input is None else grad_main_input + branch_grad
        )
    return grad_main_input


def mm_relu_multiuse_fused_backward(terms, consumer_ctx):
    grad_main_input = mm_multiuse_accumulate_backward(terms)
    (a,) = consumer_ctx.saved_tensors
    return grad_main_input * (a > 0).to(a.dtype)


def _mm_accumulate_grad(variable, mat1, mat2):
    if variable.grad is None:
        variable.grad = mat1 @ mat2
    else:
        torch.addmm(variable.grad, mat1, mat2, out=variable.grad)


def mm_input_accumulate_fused_backward(grad_producer_out, main_saved_tensors, variable):
    _x, w = main_saved_tensors
    _mm_accumulate_grad(variable, grad_producer_out, w.transpose(-1, -2))


def mm_weight_accumulate_fused_backward(
    grad_producer_out, main_saved_tensors, variable
):
    x, _w = main_saved_tensors
    _mm_accumulate_grad(variable, x.transpose(-1, -2), grad_producer_out)


# Rules map graph patterns to replacements. The splitter marker distinguishes
# activation-gradient accumulation from ordinary main-to-epilogue fusion.
RULES = [
    (
        (MMRelu.main_backward, MMRelu.epilogue_backward),
        mm_relu_fused_backward,
    ),
    (
        (MMTanh.main_backward, MMRelu.epilogue_backward),
        mm_relu_fused_backward,
    ),
    (
        (MMRelu.main_backward, MMTanh.epilogue_backward),
        mm_tanh_fused_backward,
    ),
    (
        (MMTanh.main_backward, MMTanh.epilogue_backward),
        mm_tanh_fused_backward,
    ),
    (
        (
            (MMRelu.main_backward, MMRelu.main_backward),
            split_multi_use,
        ),
        mm_multiuse_accumulate_backward,
    ),
    (
        (
            (MMRelu.main_backward, MMRelu.main_backward),
            split_multi_use,
            MMRelu.epilogue_backward,
        ),
        mm_relu_multiuse_fused_backward,
    ),
]

# These rules identify a main-backward output by its forward input index. They
# apply only when that edge leads directly to a unique leaf AccumulateGrad.
ACCUMULATE_GRAD_RULES = [
    ((MMRelu.main_backward, 0), mm_input_accumulate_fused_backward),
    ((MMRelu.main_backward, 1), mm_weight_accumulate_fused_backward),
    ((MMTanh.main_backward, 0), mm_input_accumulate_fused_backward),
    ((MMTanh.main_backward, 1), mm_weight_accumulate_fused_backward),
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
    assert ok, f"{name}: gradients do not match reference"  # noqa: S101


def scenario_mixed_chain():
    """x -> MMTanh -> MMRelu -> MMTanh -> sum.

    Fusions:
      ep1 (MMTanh) <- main2 (MMRelu): key (MMRelu.main, MMTanh.epilogue) -> tanh rule
      ep2 (MMRelu) <- main3 (MMTanh): key (MMTanh.main, MMRelu.epilogue) -> relu rule
      each leaf edge (x, w1, w2, w3) fuses its GEMM with AccumulateGrad
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
        loss.grad_fn,
        RULES,
        accumulate_grad_rules=ACCUMULATE_GRAD_RULES,
        expect_num_fusions=2,
        expect_num_accumulate_grad_fusions=4,
        _internal_debug=True,
    )
    print(plan)
    loss.backward()

    _check("mixed", ins, refs)
    print("kernel paths:", LOG)
    assert LOG.c["fused_impl"] == 2  # noqa: S101
    assert LOG.c["accumulate_fused"] == 4  # noqa: S101
    assert LOG.c["epilogue_unfused"] == 1  # noqa: S101
    assert LOG.c["main_fully_deferred"] == 3  # noqa: S101
    print("PASS: epilogues and leaf gradient accumulation fused across the chain.")


def scenario_activation_grad_accumulation():
    """Fuse branch gradient accumulation at a split native activation."""
    print("\n########## scenario: activation gradient accumulation ##########")
    torch.manual_seed(1)

    def make():
        return [
            torch.randn(4 if i == 0 else 6, 6, dtype=torch.double, requires_grad=True)
            for i in range(3)
        ]

    ref = make()
    x, wa, wb = ref
    h = torch.sigmoid(x)
    (torch.relu(h @ wa).sum() + torch.relu(h @ wb).sum()).backward()
    refs = [tensor.grad.clone() for tensor in ref]

    ins = make()
    for tensor, reference in zip(ins, ref):
        tensor.data.copy_(reference.data)
    x, wa, wb = ins

    h = torch.sigmoid(x)
    ha, hb = split_multi_use(h, 2)
    loss = MMRelu.apply(ha, wa).sum() + MMRelu.apply(hb, wb).sum()
    pattern = (
        (MMRelu.main_backward, MMRelu.main_backward),
        split_multi_use,
    )
    replacement = mm_multiuse_accumulate_backward
    LOG.reset()
    plan = apply_epilogue_fusion(
        loss,
        [(pattern, replacement)],
        expect_num_fusions=1,
        _internal_debug=True,
    )
    print(plan)
    loss.backward()

    _check("activation accumulation", ins, refs)
    print("kernel paths:", LOG)
    assert LOG.c["splitter_fused"] == 1  # noqa: S101
    assert LOG.c["splitter_deferred"] == 1  # noqa: S101
    assert LOG.c["multiuse_fused"] == 0  # noqa: S101
    assert LOG.c["fused_impl"] == 0  # noqa: S101
    assert LOG.c["accumulate_fused"] == 0  # noqa: S101
    print("PASS: only the two branch activation gradients were fused and accumulated.")


def scenario_multi_use_end_to_end():
    """Compose splitter accumulation, an upstream epilogue, and AccumulateGrad."""
    print("\n########## scenario: explicit multi-use end to end ##########")
    torch.manual_seed(2)

    def make():
        return [
            torch.randn(4 if i == 0 else 6, 6, dtype=torch.double, requires_grad=True)
            for i in range(4)
        ]

    ref = make()
    x, w1, wa, wb = ref
    h = torch.relu(x @ w1)
    (torch.relu(h @ wa).sum() + torch.relu(h @ wb).sum()).backward()
    refs = [tensor.grad.clone() for tensor in ref]

    ins = make()
    for tensor, reference in zip(ins, ref):
        tensor.data.copy_(reference.data)
    x, w1, wa, wb = ins

    h = MMRelu.apply(x, w1)
    ha, hb = split_multi_use(h, 2)
    loss = MMRelu.apply(ha, wa).sum() + MMRelu.apply(hb, wb).sum()
    LOG.reset()
    plan = apply_epilogue_fusion(
        loss,
        RULES,
        accumulate_grad_rules=ACCUMULATE_GRAD_RULES,
        expect_num_fusions=1,
        expect_num_accumulate_grad_fusions=4,
        _internal_debug=True,
    )
    print(plan)
    loss.backward()

    _check("multi-use end to end", ins, refs)
    print("kernel paths:", LOG)
    assert LOG.c["multiuse_fused"] == 1  # noqa: S101
    assert LOG.c["splitter_deferred"] == 1  # noqa: S101
    assert LOG.c["splitter_fused"] == 0  # noqa: S101
    assert LOG.c["accumulate_fused"] == 4  # noqa: S101
    print("PASS: branch GEMMs, InputBuffer accumulation, and epilogue fused.")


if __name__ == "__main__":
    scenario_mixed_chain()
    scenario_activation_grad_accumulation()
    scenario_multi_use_end_to_end()
    print("\nALL SCENARIOS PASSED")
