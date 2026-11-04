from dataclasses import dataclass
from typing import Literal


__all__ = ["FSDPPolicy"]


@dataclass(frozen=True)
class FSDPPolicy:
    r"""FSDP parameter residency and gradient reduction for a pipeline schedule.

    With the default ``FSDPPolicy()``, each stage keeps its parameters unsharded
    and its gradients unreduced, in the reduce dtype, until the stage's final
    gradient reduction. This minimizes FSDP communication, but each device holds
    the unsharded parameters and gradients of all of its stages for the whole
    step, and FSDP sharding does not reduce that memory.

    The other settings trade more FSDP collectives for lower peak memory. They
    require a schedule that runs one full backward per stage and microbatch:
    :class:`~torch.distributed.pipelining.schedules.ScheduleGPipe`,
    :class:`~torch.distributed.pipelining.schedules.Schedule1F1B`,
    :class:`~torch.distributed.pipelining.schedules.ScheduleInterleaved1F1B`, or
    :class:`~torch.distributed.pipelining.schedules.ScheduleLoopedBFS`.
    Schedules that split the backward into input and weight passes raise
    ``ValueError``.

    Pass the policy to a schedule as ``fsdp_policy``. Each FSDP module is one
    unit of the policy, so with one ``fully_shard`` call per transformer block,
    ``reshard="layer"`` releases one block at a time.

    The policy sets the FSDP flags for the stage backward of each microbatch. A
    backward that ``loss_fn`` runs itself, for example in a chunked loss,
    manages its own FSDP settings.

    Args:
        reshard: When to release unsharded parameters.

            - ``"schedule"``: The schedule decides. Runtime schedules reshard a
              whole stage when it leaves the active-stage window (see
              ``max_active_stages``). Other schedules keep each stage unsharded
              until its final gradient reduction. The construction-time
              ``reshard_after_forward`` of each FSDP module still applies to
              forward.
            - ``"layer"``: FSDP reshards each module after its forward and
              after its backward, and unshards it again before its next use.
              This sets ``reshard_after_forward=True`` on every FSDP module of
              the stage. Runtime schedules emit no ``UNSHARD`` or ``RESHARD``
              actions.
        reduce: When to reduce-scatter gradients.

            - ``"step"``: Once per stage, after its last microbatch.
            - ``"microbatch"``: After the backward of microbatches
              ``k - 1, 2k - 1, ...`` of the stage, where
              ``k = k_microbatch_reduce``. During that backward, FSDP
              reduce-scatters each module's accumulated gradients as soon as
              the module finishes, and adds the result to the sharded
              gradients. The final reduction covers the remaining microbatches.
              With HSDP, each of these reductions also all-reduces across
              replicas.
        k_microbatch_reduce: The number of microbatches per reduction for
            ``reduce="microbatch"``. Only ``1`` frees the unsharded gradients
            between microbatches. A larger value keeps them until the group's
            reduction, but moves reduce-scatters earlier in the step, where
            they can overlap with compute.
    """

    reshard: Literal["schedule", "layer"] = "schedule"
    reduce: Literal["step", "microbatch"] = "step"
    k_microbatch_reduce: int = 1

    def __post_init__(self) -> None:
        if self.reshard not in ("schedule", "layer"):
            raise ValueError(
                f"reshard must be 'schedule' or 'layer', got {self.reshard!r}"
            )
        if self.reduce not in ("step", "microbatch"):
            raise ValueError(
                f"reduce must be 'step' or 'microbatch', got {self.reduce!r}"
            )
        if (
            isinstance(self.k_microbatch_reduce, bool)
            or not isinstance(self.k_microbatch_reduce, int)
            or self.k_microbatch_reduce < 1
        ):
            raise ValueError(
                "k_microbatch_reduce must be a positive integer, got "
                f"{self.k_microbatch_reduce!r}"
            )
        if self.k_microbatch_reduce != 1 and self.reduce != "microbatch":
            raise ValueError(
                "k_microbatch_reduce applies only to reduce='microbatch', got "
                f"reduce={self.reduce!r}"
            )

    @property
    def _requires_full_backward(self) -> bool:
        """Whether the policy needs one full backward per stage and microbatch."""
        # TODO: Support split-backward schedules (BACKWARD_INPUT and
        # BACKWARD_WEIGHT), as proposed in
        # https://github.com/pytorch/pytorch/issues/202853. FSDP's post-backward
        # hook fires in the input pass, before the weight gradients exist, and
        # the weight pass fires no pre-backward hooks, so FSDP's hooks cannot
        # drive either setting there.
        # - reduce="microbatch": add an FSDP method that reduce-scatters the
        #   pending unsharded gradients and keeps the backward iteration open
        #   (finalize_backward ends it and must not run between forward and
        #   backward). Then lower a partial REDUCE_GRAD after every
        #   k_microbatch_reduce-th BACKWARD_WEIGHT of a stage.
        # - reshard="layer": reshard after the input pass only when the weight
        #   pass reads no parameters, and keep the weight gradients reachable
        #   for the partial reduction after the parameters are resharded.
        # - Compiled regions: with the input/weight split for compiled regions
        #   (https://github.com/pytorch/pytorch/issues/202127), their weight
        #   gradients also land in the weight pass, after the deferred dW runs.
        #   The partial reduction must run after it.
        return self.reshard != "schedule" or self.reduce != "step"
