import logging
import weakref
from copy import deepcopy
from enum import auto, Enum
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch
from torch.distributed._tools.ilp_utils import (
    aggregate_stats,
    collect_stats,
    parse_module_info,
)
from torch.distributed._tools.sac_estimator import (
    OPS_TO_ALWAYS_SKIP,
    SACGreedyOrderMeta,
    SACStats,
)
from torch.distributed._tools.sac_ilp import (
    get_optimal_checkpointing_policy_per_module,
    sac_milp,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)


# Create a logger object
logger = logging.getLogger(__name__)

# Set the logging level to INFO
logger.setLevel(logging.INFO)


class AutoSACResult(NamedTuple):
    """
    Represents the results of a SAC (Selective Activation Checkpointing) optimization.

    Attributes:
        sac_policies (Dict[str, List[int]]):
            A dictionary mapping each module's Fully Qualified Name (FQN) to its SAC policy.
            The policy determines whether each operator in the module should be saved (1)
            or recomputed (0).

        ac_decisions (Dict[str, float]):
            A dictionary mapping each module's FQN to the percentage of activation
            memory to discard in the optimal SAC solution.

        recomputation_time (float):
            The total recomputation time (in ms) for the optimal SAC solution.

        peak_mem (int):
            The upper bound on the peak memory (in Bytes) of the optimal SAC solution. A value of -1
            indicates that the ILP solver failed to find a solution.
    """

    sac_policies: Dict[str, List[int]]
    ac_decisions: Dict[str, float]
    recomputation_time: float
    peak_mem: int


class SACAlgorithm(Enum):
    """
    Enum representing the Selective Activation Checkpointing (SAC) algorithms.

    Attributes:
        GREEDY: Represents the greedy algorithm for SAC.
        OPTIMAL: Represents the optimal algorithm for SAC.
    """

    GREEDY = auto()
    OPTIMAL = auto()


class _SACPolicy:
    """
    Represents a Selective Activation Checkpointing (SAC) policy for managing
    operator save or recompute decisions.

    Args:
        policy_output (List[int]):
            A list of integers where 1 indicates saving an operator and
            0 indicates recomputing it.

    Attributes:
        forward_counter (int):
            Tracks the number of forward calls made for policy evaluation.

        recompute_counter (int):
            Tracks the number of recomputation calls made for policy evaluation.

        policy_output (List[int]):
            The policy decision list provided during initialization.

    Methods:
        __call__(ctx, func, *args, **kwargs) -> CheckpointPolicy:
            Evaluates the checkpoint policy for a given function based on
            the SAC policy and context.
    """

    def __init__(self, policy_output: List[int]):
        self.forward_counter = 0
        self.recompute_counter = 0
        self.policy_output = policy_output

    def __call__(self, ctx, func, *args, **kwargs) -> CheckpointPolicy:  # type: ignore[no-untyped-def]
        if func in OPS_TO_ALWAYS_SKIP:
            return CheckpointPolicy.MUST_RECOMPUTE

        if ctx.is_recompute:
            count = self.recompute_counter
            self.recompute_counter += 1
        else:
            count = self.forward_counter
            self.forward_counter += 1

        if self.policy_output[count] == 1:
            return CheckpointPolicy.PREFER_SAVE
        else:
            return CheckpointPolicy.MUST_RECOMPUTE


def get_greedy_checkpointing_policy_per_module(
    sac_stats: SACStats, sac_greedy_order_meta: SACGreedyOrderMeta, memory_budget: float
) -> List[int]:
    """
    Compute greedy checkpoint policy per module.

    Args:
        sac_stats (SACStats): SAC statistics.
        sac_greedy_order_meta (SACGreedyOrderMeta): SAC greedy order metadata.
        memory_budget (float): Memory budget as a fraction of total memory (0 <= memory_budget <= 1).

    Returns:
        List[int]: Policy output as a list of integers (1: save, 0: discard).

    Raises:
        ValueError: If memory_budget is not within the valid range (0 <= memory_budget <= 1).
    """

    # Validate memory budget range
    if not (0 <= memory_budget <= 1):
        raise ValueError(
            f"`memory_budget` must be a float between 0 and 1. Got {memory_budget}."
        )

    # Initialize policy output with all ops saved
    policy_output = [1 for _ in range(len(sac_stats.memory))]

    sac_memory = sum(sac_stats.memory)
    sac_memory_budget = memory_budget * sac_memory

    stored_ops, recomputed_ops, inplace_op_groups, random_inplace_ops, msps_meta = (
        sac_greedy_order_meta.stored_ops,
        sac_greedy_order_meta.recomputed_ops,
        sac_greedy_order_meta.inplace_op_groups,
        sac_greedy_order_meta.random_inplace_ops,
        sac_greedy_order_meta.msps_meta,
    )

    stored_indices: Set[int] = set()
    for s_idx in stored_ops:
        stored_indices.add(s_idx)
        if s_idx in inplace_op_groups:
            stored_indices.update(inplace_op_groups[s_idx])
        if s_idx in random_inplace_ops:
            stored_indices.update(random_inplace_ops)

    saved_memory = sum(sac_stats.memory[op_idx] for op_idx in stored_indices)

    # Check if saved ops exceed memory budget
    if saved_memory > sac_memory_budget:
        logger.error(
            "Ops that need to be saved already exceed the given memory budget.\n"
            "Ops: %s\n"
            "Budget: %s Saved Ops Memory: %s",
            [sac_stats.func_names[i] for i in stored_ops],
            sac_memory_budget,
            saved_memory,
        )
        return [
            1 if idx in stored_indices else 0 for idx in range(len(sac_stats.memory))
        ]

    recompute_indices = set(recomputed_ops)
    discarded_memory = sum(sac_stats.memory[i] for i in recompute_indices)

    sac_memory_budget -= saved_memory
    msps_meta = deepcopy(msps_meta)

    # Discard ops until memory budget is met
    while (sac_memory - discarded_memory) > sac_memory_budget:
        try:
            msps = msps_meta.pop(0)
        except IndexError:
            logger.error("Exhausted the Ops to recompute, cannot satisfy budget.")
            return [
                1 if idx in stored_indices else 0
                for idx in range(len(sac_stats.memory))
            ]
        recompute_indices.add(msps.op_idx)
        if msps.op_idx in random_inplace_ops:
            recompute_indices.update(random_inplace_ops)
        if inplace_op_group := inplace_op_groups.get(msps.op_idx, None):
            recompute_indices.update(inplace_op_group)
        discarded_memory += msps.memory

    # Update policy output with recompute ops
    for i in recompute_indices:
        policy_output[i] = 0

    return policy_output


def get_auto_sac_policies(
    train_step: Callable,
    models: List[torch.nn.Module],
    optimizers: List[torch.optim.Optimizer],
    inputs: Any,
    dev: torch.device,
    memory_budget: float,
    sac_algo: SACAlgorithm = SACAlgorithm.GREEDY,
    shard_degree: int = 1,
    ac_units: Optional[Set[str]] = None,
    fsdp_units: Optional[Set[str]] = None,
    runtime_kwargs: Optional[Dict[str, Any]] = None,
) -> AutoSACResult:
    """
    Computes auto-SAC (Selective Activation Checkpointing) policies for the given model.

    Args:
        train_step (Callable):
            A function that executes a single training step for the given models, optimizers, and inputs.
            It should have the signature `train_step(models, optimizers, inputs)`.
        models (List[torch.nn.Module]):
            A list of PyTorch root modules whose statistics are to be collected, initialized under `FakeTensorMode`.
        optimizers (List[torch.optim.Optimizer]):
            A list of optimizers corresponding to the root modules, initialized under `FakeTensorMode`.
            For root modules that don't require an optimizer, `None` should be passed.
        inputs (Any):
            The inputs required for the training step, initialized under `FakeTensorMode`.
        dev (torch.device):
            The device on which the model, inputs, and optimizer are initialized.
        memory_budget (float):
            The memory budget in GiB for which SAC policies are optimized.
            Recommended max budget is 80% of the device memory.
        sac_algo (SACAlgorithm, optional):
            The SAC algorithm to use for policy computation. Defaults to `SACAlgorithm.GREEDY`.
        shard_degree (int, optional):
            The number of GPUs across which the model is sharded. Used to calculate
            parameter, gradient, and optimizer memory for FSDP. Defaults to 1.
        ac_units (Optional[Set[str]], optional):
            A set of user-specified Activation Checkpointing (AC) unit Fully Qualified Names (FQNs).
            Defaults to `None`.
        fsdp_units (Optional[Set[str]], optional):
            A set of Fully Sharded Data Parallel (FSDP) units. AC units cannot be supermodules
            of FSDP unit FQNs. Defaults to `None`.
        runtime_kwargs (Optional[Dict[str, Any]], optional):
            A dictionary of runtime-related configuration parameters. Supported keys:
                - `"estimate_mode"` (str): The runtime estimation mode to use. Supported modes:
                    - `"operator-level-benchmark"`: Estimates runtime using operator benchmarking.
                    - `"operator-level-cost-model"`: Estimates runtime using a roofline cost model.
                      Defaults to `"operator-level-cost-model"`.
                - `"gpu_type"` (str): The GPU type to configure specific settings (e.g., `"H100_SXM_80GB"`).
                - `"custom_config"` (Tuple[Dict[torch.dtype, float], Dict[torch.dtype, float], float]):
                  A tuple containing:
                    - A dictionary mapping `torch.dtype` to peak FLOPS (in GFLOPS/s).
                    - A dictionary mapping `torch.dtype` to peak FLOPS factors.
                    - The peak bandwidth (in GB/s).

    Returns:
        AutoSACResult:
            The computed SAC policies, activation memory decisions, recomputation time,
            and peak memory estimates.

    Example usage:

        .. code-block:: python

            dev = torch.device('cuda:0')
            torch.set_default_device(dev)

            def train_step(models, optimizers, inputs):
                # Abstract training step implementation
                ...

            with FakeTensorMode():
                models = [...]  # List of PyTorch models
                optimizers = [...]  # List of optimizers
                inputs = [...]  # Inputs for the training step

                auto_sac_result = get_auto_sac_policies(
                    train_step=train_step,
                    models=models,
                    optimizers=optimizers,
                    inputs=inputs,
                    dev=dev,
                    memory_budget=60.0,
                    sac_algo=SACAlgorithm.OPTIMAL,
                )
    """
    assert len(models) == len(
        optimizers
    ), "Expect the number of root modules and optimizers to be same"
    # Collect model statistics
    mem_tracker, runtime_estimator, sac_estimator = collect_stats(
        train_step, models, optimizers, inputs, runtime_kwargs
    )
    # Aggregate model statistics
    mod_info = aggregate_stats(
        models, optimizers, mem_tracker, runtime_estimator, sac_estimator, dev
    )
    # Parse module information into a graph
    graph = parse_module_info(mod_info)
    # Solve SAC MILP problem
    ac_decisions, est_recomp_time, est_peak_mem = sac_milp(
        graph, memory_budget, shard_degree, ac_units, fsdp_units
    )

    sac_policies: Dict[str, List[int]] = {}
    if est_peak_mem != -1:
        # Solver succeeded Compute SAC policies for each module
        for mod_name, discard_ratio in ac_decisions.items():
            sac_stats = sac_estimator.sac_mod_stats[mod_name]
            budget = 1 - discard_ratio
            sac_greedy_order_meta = sac_estimator.sac_mod_greedy_order_meta[mod_name]
            if sac_algo == SACAlgorithm.GREEDY:
                policy_output = get_greedy_checkpointing_policy_per_module(
                    sac_stats, sac_greedy_order_meta, budget
                )
            else:
                policy_output = get_optimal_checkpointing_policy_per_module(
                    sac_stats, budget
                )
            # Create and store SAC policy in dictionary
            sac_policies[mod_name] = policy_output

    return AutoSACResult(sac_policies, ac_decisions, est_recomp_time, est_peak_mem)


def apply_auto_sac(
    model: torch.nn.Module,
    sac_policies: Dict[str, List[int]],
    checkpoint_impl: CheckpointImpl = CheckpointImpl.NO_REENTRANT,
    checkpoint_fn: Optional[Callable] = None,
    **checkpoint_fn_kwargs: Dict[str, Any],
) -> None:
    """
    Applies auto-SAC (Selective Activation Checkpointing) policies to the specified model.

    Args:
        model (torch.nn.Module):
            The target model to which SAC policies are to be applied.

        sac_policies (Dict[str, List[int]]):
            A dictionary mapping module Fully Qualified Names (FQNs) to SAC policies.
            Each policy is a list of integers where 1 indicates saving an operator
            and 0 indicates recomputation.

        checkpoint_impl (CheckpointImpl, optional):
            The checkpointing implementation to use. Defaults to `CheckpointImpl.NO_REENTRANT`.

        checkpoint_fn (callable, optional):
            Functional checkpoint implementation to use. If this is specified,
            it will be used over the default ``torch.utils.checkpoint.checkpoint``
            implementation and the `checkpoint_impl` argument will be ignored.

        **checkpoint_fn_kwargs (Dict[str, Any]):
            Keyword arguments to pass into `checkpoint_fn`.

            Note:
                The `context_fn` key is not allowed in `checkpoint_fn_kwargs` as it is
                automatically populated by the Auto SAC wrapper. If you require this
                functionality, please file a GitHub issue.

    Notes:
        - This function modifies the model in place, wrapping applicable modules with
          checkpointing contexts based on the provided SAC policies.
        - Modules without a corresponding policy in `sac_policies` are left unchanged.

    Returns:
        None
    """

    checkpoint_fn_kwargs = checkpoint_fn_kwargs if checkpoint_fn_kwargs else {}
    if checkpoint_fn_kwargs and "context_fn" in checkpoint_fn_kwargs:
        raise ValueError(
            "`context_fn` will be automatically populated by the Auto SAC wrapper. "
            "If you require this functionality, please file a GitHub issue."
        )
    mod_names = get_module_name_dict(model)

    def _sac_wrapper(mod: torch.nn.Module) -> Union[torch.nn.Module, None]:
        """
        Wraps a module with selective checkpointing based on its SAC policy.

        Args:
            mod (torch.nn.Module): The module to be wrapped.

        Returns:
            torch.nn.Module or None: The wrapped module if a policy exists, otherwise None.
        """
        if policy_output := sac_policies.get(mod_names[mod], None):

            def selective_checkpointing_context_fn() -> (
                Tuple[ContextManager, ContextManager]
            ):
                return create_selective_checkpoint_contexts(_SACPolicy(policy_output))

            return checkpoint_wrapper(
                mod,
                checkpoint_impl=checkpoint_impl,
                checkpoint_fn=checkpoint_fn,
                context_fn=selective_checkpointing_context_fn,
                **checkpoint_fn_kwargs,
            )
        else:
            return None

    from torch.distributed.fsdp._wrap_utils import _post_order_apply

    _post_order_apply(model, fn=_sac_wrapper)


def get_module_name_dict(root_module: torch.nn.Module) -> weakref.WeakKeyDictionary:
    """
    Create a weak key dictionary of modules and their fully qualified names.

    Args:
        root_module (torch.nn.Module): Root module to start traversal.

    Returns:
        weakref.WeakKeyDictionary: Dictionary of modules and their names.
    """
    module_dict: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def _get_mod_name(mod: torch.nn.Module, fqn: str = "") -> str:
        if mod in module_dict:
            return module_dict[mod]
        mod_name = fqn or type(mod).__name__
        module_dict[mod] = mod_name
        for name, submod in mod.named_children():
            _get_mod_name(submod, f"{mod_name}.{name}")
        return mod_name

    _get_mod_name(root_module)
    return module_dict
