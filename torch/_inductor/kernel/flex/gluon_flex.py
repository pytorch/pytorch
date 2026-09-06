# mypy: allow-untyped-defs
"""Gluon FlexAttention templates for Inductor.

Offers hand-tuned Gluon flash-attention forward kernels as flex-attention
autotuning candidates through ``InductorChoices.append_flex_attention_choices``
(the extension seam TLX uses), installed via ``config.inductor_choices_class``.

A template is a ``GluonTemplate`` -- a ``TritonTemplate`` that emits a
``@gluon.jit`` body -- so ``score_mod``/``mask_mod`` are rendered by Inductor's
existing ``modification()`` machinery and the kernel runs through the standard
Triton scheduling/compile path.

Targets are described by ``GluonFlexTarget``: matmul tile geometry, which
template bodies exist, and the DMA staging ladder. Kernel *bodies* are per target
because Gluon's primitives differ by family, but everything here -- config
filtering, subgraph rendering, autotuning -- is shared. Adding a target is a new
descriptor plus its template files.
"""

import dataclasses
import functools
import hashlib
import logging
from typing import Any, NamedTuple
from typing_extensions import override

import torch
from torch._inductor import config
from torch._inductor.choices import InductorChoices
from torch._inductor.codegen.gluon.cdna4 import Cdna4GluonTemplate
from torch._inductor.codegen.gluon.gluon_template import GluonTemplate

from .common import load_flex_template
from .gluon_dma_layouts import as_template_options, CDNA4_DMA_LADDER, DmaLayouts


log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, eq=False)
class GluonFlexTarget:
    """Everything target-specific about offering a Gluon flex-attention body.

    ``eq=False`` keeps the default identity hash, so a target can be used as a
    cache key. Field-based equality would hash ``dma_ladder`` and fail, which is
    also why this is not a NamedTuple -- being a tuple implies hashability that a
    mapping field does not provide.
    """

    name: str
    # gfx target this descriptor's bodies were written and measured against,
    # matched exactly rather than as a family prefix
    arch_prefix: str
    template_cls: type[GluonTemplate]
    sync_template: str
    # None when the target has no async body yet; only the sync one is offered.
    async_template: str | None
    # Matmul instruction tile (MFMA on CDNA, WMMA on RDNA/gfx1250, MMA on NVIDIA).
    mma_m: int
    mma_n: int
    max_warps: int
    # Wavefront width the template's layouts are declared against. GluonASTSource
    # stamps ttg.threads-per-warp from this, so a wave32 target that inherited 64
    # would fail layout verification rather than run slowly.
    threads_per_warp: int
    # (head_dim, block_n, num_warps) -> staging layouts for the async body.
    dma_ladder: dict[tuple[int, int, int], DmaLayouts]

    def warps_for(self, block_m: int) -> int:
        """The matmul layout splits BLOCK_M across waves."""
        return min(self.max_warps, max(1, block_m // self.mma_m))

    def tiles_evenly(self, block_m: int, block_n: int) -> bool:
        """Blocks smaller than one matmul tile give a degenerate layout."""
        return block_m % self.mma_m == 0 and block_n % self.mma_n == 0


CDNA4_TARGET = GluonFlexTarget(
    name="cdna4",
    arch_prefix="gfx950",
    template_cls=Cdna4GluonTemplate,
    sync_template="gluon_flex_attention",
    async_template="gluon_flex_attention_async",
    mma_m=32,
    mma_n=32,
    max_warps=8,
    threads_per_warp=64,
    dma_ladder=CDNA4_DMA_LADDER,
)

# Every target the module knows about. Both the arch lookup and the template
# digest iterate this, so adding one cannot leave either behind.
TARGETS: tuple[GluonFlexTarget, ...] = (CDNA4_TARGET,)


@functools.cache
def _target_for_arch(arch: str) -> GluonFlexTarget | None:
    """The descriptor for a bare gfx target name, or None.

    Matched exactly rather than as a gfx95 family prefix: the bodies below are
    written against CDNA4's matmul geometry and wavefront width, so a later gfx95*
    part should fall back to Triton until someone measures it, not inherit layouts
    silently.
    """
    for target in TARGETS:
        if arch.startswith(target.arch_prefix):
            return target
    return None


def _active_target(device: Any = None) -> GluonFlexTarget | None:
    """The descriptor for this device, or None if no Gluon body targets it."""
    if not torch.version.hip:
        return None
    try:
        # The compiling device, not device 0: a host can mix architectures, and
        # Inductor compiles for wherever the inputs live.
        index = device.index if getattr(device, "index", None) is not None else None
        arch = torch.cuda.get_device_properties(index).gcnArchName
    except Exception:
        return None
    return _target_for_arch(arch.split(":", 1)[0])


@functools.cache
def _get_gluon_flex_template(target: GluonFlexTarget, name: str) -> GluonTemplate:
    from .flex_attention import flex_attention_grid

    return target.template_cls(
        name=name,
        grid=flex_attention_grid,
        source=load_flex_template(name) + load_flex_template("utilities"),
        always_freeze_layout=True,
    )


def _async_body(
    target: GluonFlexTarget, conf: Any, kernel_options: dict[str, Any], kv_len: Any
) -> tuple[str, DmaLayouts] | None:
    """This config's async body and its staging layouts, or None if unsupported.

    This *is* the gate: the async body renders its layout declarations from these
    values, so a config is offered exactly when the ladder has an entry for it.
    """
    from torch._inductor.virtualized import V

    name = target.async_template
    if name is None:
        return None
    qk_rounded = kernel_options.get("QK_HEAD_DIM_ROUNDED")
    v_rounded = kernel_options.get("V_HEAD_DIM_ROUNDED")
    # The ladder is keyed on one head dim; K^T and V are staged with the same one.
    if qk_rounded is None or qk_rounded != v_rounded:
        return None
    # The DMA spans the rounded head dim with no mask, so a head dim that had to be
    # rounded up would stage past the end of K/V. The sync body masks and handles it.
    if not kernel_options.get("SAFE_HEAD_DIM"):
        return None
    # The async body stages K/V by unmasked DMA, so it clamps a prefetch that would
    # run past the end of the tensor to KV_LEN - BLOCK_N. That is a block start only
    # when BLOCK_N divides KV_LEN; otherwise the final block stages rows other than
    # the ones mask_mod/score_mod are told it staged, and the results are wrong
    # rather than imprecise. A dynamic KV_LEN cannot prove the property, so it also
    # takes the sync body.
    if not V.graph.sizevars.statically_known_multiple_of(kv_len, conf.block_n):
        return None
    key = (qk_rounded, conf.block_n, kernel_options["num_warps"])
    layouts = target.dma_ladder.get(key)
    return None if layouts is None else (name, layouts)


@functools.cache
def _template_digest() -> str:
    """Short digest of every Gluon template body, for cache keying."""
    h = hashlib.sha256()
    for target in TARGETS:
        for name in (target.sync_template, target.async_template):
            if name is not None:
                h.update(load_flex_template(name).encode())
    return h.hexdigest()[:16]


class _ExtraConfig(NamedTuple):
    """A block shape offered on top of the ones flex proposes."""

    block_m: int
    block_n: int


# flex's own config list stops at BLOCK_M=128, but a taller query tile is what the
# hand-tuned gfx950 kernel runs: one K/V block then serves twice the query rows, so
# the DMA traffic per unit of math halves. Only usable when the sparse Q block is a
# multiple of it, which the loop below already checks.
EXTRA_CONFIGS = (_ExtraConfig(256, 64), _ExtraConfig(256, 32))


def _score_mod_is_identity(subgraphs, sm_scale: Any) -> bool:
    """True when score_mod passes the score through unchanged.

    Inductor lowers an identity score_mod to a buffer that just reads the
    ``score`` input, so the template can then fold sm_scale into the same FMA
    that does the exp2 change of base instead of scaling the tile separately.

    Two invariants this depends on. The detection reads Inductor's lowering of an
    identity subgraph, so a *false* answer only costs the fold while a *true* one
    makes the template skip score_mod entirely -- if that lowering ever stops
    producing an InputBuffer for the identity, this has to fail closed. And the
    fold scales the row max instead of the tile, which reorders max and multiply,
    so it holds only for a positive scale; a negative one would turn the row max
    into a row min and overflow the exp2. The Triton template scales the tile
    first and so does not care about the sign.
    """
    from torch._inductor import ir

    if not subgraphs:
        return False
    if not isinstance(sm_scale, (int, float)) or sm_scale <= 0:
        return False
    return isinstance(getattr(subgraphs[0], "data", None), ir.InputBuffer)


class GluonInductorChoices(InductorChoices):
    """InductorChoices that offers the Gluon flash-attention templates."""

    def uuid(self) -> str:
        # Derived from the template sources so editing a body invalidates cached
        # choices without anyone remembering to bump a version string.
        return f"gluon-flex-attention-{_template_digest()}"

    @override
    def append_flex_attention_choices(
        self,
        choices: list[Any],
        configs: list[Any],
        input_nodes: list[Any],
        subgraphs: list[Any],
        layout: Any,
        kernel_options: dict[str, Any],
        sparse_q_block_size: int,
        sparse_kv_block_size: int,
    ) -> list[Any]:
        query, _key, _value, logsumexp, max_scores = input_nodes[:5]
        target = _active_target(query.get_device())
        if not config.gluon_flex_attention or target is None:
            return choices

        if query.get_dtype() not in (torch.float16, torch.bfloat16):
            return choices

        # These bodies pick their own block shape and schedule, so offering one
        # would quietly hand back a kernel built with something other than what
        # the caller pinned. Leave the Triton candidates to honour it instead.
        if any(
            k in kernel_options
            for k in ("num_warps", "num_stages", "fwd_num_warps", "fwd_num_stages")
        ):
            return choices

        # None of this depends on the block shape, so build it once rather than
        # per config.
        base_opts = kernel_options.copy()
        for k in list(base_opts.keys()):
            if k.startswith("fwd_"):
                base_opts[k[4:]] = base_opts.pop(k)
            elif k.startswith("bwd_"):
                base_opts.pop(k)
        base_opts["USE_TMA"] = False
        base_opts["GLUON_THREADS_PER_WARP"] = target.threads_per_warp
        score_mod_is_identity = _score_mod_is_identity(
            subgraphs, base_opts.get("SM_SCALE")
        )
        # The Gluon body schedules its own loop; keep Triton's software pipeliner
        # out of it.
        base_opts["num_stages"] = 1
        base_opts.setdefault("SPARSE_Q_BLOCK_SIZE", sparse_q_block_size)
        base_opts.setdefault("SPARSE_KV_BLOCK_SIZE", sparse_kv_block_size)
        # Validate against the effective sizes rather than the arguments: the
        # setdefaults above leave a caller-pinned value in place, and that is what
        # the kernel is built with. Same reasoning as the Triton path.
        sparse_q = base_opts["SPARSE_Q_BLOCK_SIZE"]
        sparse_kv = base_opts["SPARSE_KV_BLOCK_SIZE"]

        kv_len = _key.get_size()[2]
        for conf in tuple(configs) + EXTRA_CONFIGS:
            if not target.tiles_evenly(conf.block_m, conf.block_n):
                continue
            if sparse_kv % conf.block_n != 0 or sparse_q % conf.block_m != 0:
                continue

            opts = dict(base_opts)
            opts["BLOCK_M"] = conf.block_m
            opts["BLOCK_N"] = conf.block_n
            opts["num_warps"] = target.warps_for(conf.block_m)

            # Offer the synchronous body always, and the async one wherever the
            # ladder has staging layouts, so autotuning picks between them.
            bodies: list[tuple[str, DmaLayouts | None]] = [(target.sync_template, None)]
            async_body = _async_body(target, conf, opts, kv_len)
            if async_body is not None:
                bodies.append(async_body)

            for template_name, dma_layouts in bodies:
                body_opts = dict(opts)
                is_async = template_name == target.async_template
                if is_async:
                    body_opts.update(as_template_options(dma_layouts))
                    # LDS ring depth: measured 2 > 3 > 4 on gfx950 (deeper rings
                    # cost more occupancy at D=128 than the extra overlap buys:
                    # 608 -> 559 TFLOPS non-causal).
                    body_opts["GLUON_NUM_BUF"] = 2
                    body_opts["SCORE_MOD_IS_IDENTITY"] = score_mod_is_identity
                # Unrolling the KV loop pays off on some shapes and costs on
                # others (it helps D=64 causal, hurts D=128), so offer both and
                # let autotuning decide per shape. Only the async body reads it.
                for unroll in (1, 2) if is_async else (None,):
                    if unroll is not None:
                        body_opts["GLUON_UNROLL"] = unroll
                    error = _get_gluon_flex_template(
                        target, template_name
                    ).maybe_append_choice(
                        choices=choices,
                        input_nodes=input_nodes,
                        layout=layout,
                        subgraphs=subgraphs,
                        mutated_inputs=[logsumexp, max_scores],
                        call_sizes=query.get_size(),
                        **body_opts,
                    )
                    if error is not None:
                        log.debug(
                            "gluon flex choice skipped: target=%s body=%s "
                            "BLOCK_M=%s BLOCK_N=%s unroll=%s: %s: %s",
                            target.name,
                            template_name,
                            conf.block_m,
                            conf.block_n,
                            unroll,
                            type(error).__name__,
                            error,
                        )
        return choices


def enable_gluon_flex_attention() -> None:
    """Turn on the Gluon flex-attention choices (currently AMD gfx950 only).

    Equivalent to setting both config values, and provided so callers do not have
    to know which choices class implements the hook.
    """
    config.gluon_flex_attention = True
    config.inductor_choices_class = GluonInductorChoices
