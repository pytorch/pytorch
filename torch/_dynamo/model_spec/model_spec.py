"""
ModelSpec: Declarative DSL for torch.compile dispatching.

This module provides the user-facing API for specifying compilation rules based on
input properties, execution contexts, and custom dispatch logic.

ModelSpec uses a three-layer architecture that compiles user-specified conditions
into runtime guards and compiler checks:

1. **Conditions** (User-facing API):
   Users specify constraints using the Arg/KwArg builder pattern:
   - Arg(0).rank(4).dtype("float32")
   - KwArg("batch_size").type_(int)

2. **Constraints** (Internal IR):
   Conditions compile down to typed constraint objects (defined in types.py):
   - ShapeConstraint, DtypeConstraint, RankConstraint
   - NoneConstraint, TypeConstraint, LayoutConstraint
   - DeviceConstraint, TensorSubclassConstraint

3. **Guards & Checks** (Output):
   Each constraint generates two outputs:
   - Guards: Runtime conditions for selecting which compiled variant to use
     (via to_guard_expression())
   - Checks: torch._check assertions that inform downstream compilers about
     assumptions they can safely make (via to_check())

Example flow:
    Arg(0).rank(4).dtype("float32")
        ↓
    [RankConstraint(rank=4), DtypeConstraint(dtype=torch.float32)]
        ↓
    ┌────────────────────┴────────────────────┐
    ↓                                         ↓
Guard: "x.dim() == 4 and x.dtype == float32"  Check: "torch._check(x.dim() == 4, ...)"
(runtime dispatch)                             (compiler assumptions)

This separation allows ModelSpec to provide a clean user API while generating efficient
runtime dispatch logic and preserving correctness through compiler checks.
"""

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, TYPE_CHECKING, Union

import torch
from torch._subclasses.fake_tensor import (
    _MetadataIntLike,
    extract_tensor_metadata,
    TensorMetadata,
)

from .types import (
    Constraint,
    DeviceConstraint,
    DtypeConstraint,
    LayoutConstraint,
    NoneConstraint,
    RankConstraint,
    ShapeConstraint,
    TensorSubclassConstraint,
    TypeConstraint,
)


if TYPE_CHECKING:
    from .dispatcher import DispatcherFunction


# ============================================================================
# Enums
# ============================================================================


class Context(Enum):
    """Execution context for compilation rules."""

    GRAD = "grad"
    NO_GRAD = "no_grad"
    INFERENCE = "inference"
    AUTOCAST = "autocast"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"Context.{self.name}"


class CompileAction(Enum):
    """Action to take when condition matches in a custom dispatcher."""

    COMPILE = "compile"
    RAISE_ERROR = "raise_error"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"CompileAction.{self.name}"


# Export for user convenience
COMPILE = CompileAction.COMPILE
RAISE_ERROR = CompileAction.RAISE_ERROR


# ============================================================================
# Argument Specification Classes (Arg, KwArg)
# ============================================================================


# Mapping from dtype string to torch.dtype
_DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


class _ArgBase:
    """
    Base class for Arg and KwArg containing shared implementation.

    This class contains all the common builder methods and constraint logic
    that are identical between positional and keyword arguments.
    """

    def __init__(self, example_input: Optional[torch.Tensor] = None):
        # Extract and store metadata instead of holding reference to tensor
        self._metadata: Optional[TensorMetadata] = (
            extract_tensor_metadata(example_input)
            if example_input is not None
            else None
        )
        self._constraints: list[Constraint] = []

        # Track dimension constraints separately to detect conflicts
        # Maps dimension index -> either _MetadataIntLike (static size) or "dynamic"
        self._dim_specs: dict[int, Union[_MetadataIntLike, str]] = {}
        # Track rank if specified
        self._rank: Optional[int] = None

    def static(
        self,
        shape: Optional[tuple[_MetadataIntLike, ...]] = None,
        idx: Optional[int] = None,
    ) -> "_ArgBase":
        """
        Mark argument as having a static (fixed) shape, or mark specific dimension as static.

        Args:
            shape: The fixed shape. If None, derives from example_input. Cannot be used with idx.
            idx: Specific dimension index to mark as static. Cannot be used with shape.

        Returns:
            self for method chaining
        """
        if shape is not None and idx is not None:
            raise ValueError("Cannot specify both shape and idx for static()")

        if idx is not None:
            # Mark a specific dimension as static
            if idx in self._dim_specs:
                if self._dim_specs[idx] == "dynamic":
                    raise ValueError(
                        f"Dimension {idx} is already marked as dynamic, cannot also mark as static"
                    )
            # For now, mark as static without a size (we'll need metadata or shape later)
            # We use a placeholder to indicate static-without-size
            if idx in self._dim_specs and self._dim_specs[idx] not in (
                "dynamic",
                "static_unspecified",
            ):
                # Already has a static size, keep it
                pass
            else:
                # Mark as needing a size
                self._dim_specs[idx] = "static_unspecified"
            return self

        # Mark all dimensions as static with a specific shape
        if shape is None:
            if self._metadata is None:
                raise ValueError(
                    "Must provide either shape or example_input for static()"
                )
            shape = tuple(self._metadata.shape)

        # Mark all dimensions as static and store their sizes
        for i, size in enumerate(shape):
            if i in self._dim_specs and self._dim_specs[i] == "dynamic":
                raise ValueError(
                    f"Dimension {i} is already marked as dynamic, cannot mark as static"
                )
            self._dim_specs[i] = size

        # Also set rank
        if self._rank is None:
            self._rank = len(shape)
        elif self._rank != len(shape):
            raise ValueError(
                f"Rank mismatch: rank() specified {self._rank}, but shape has {len(shape)} dimensions"
            )

        return self

    def dynamic(self, idx: Optional[int] = None) -> "_ArgBase":
        """
        Mark dimension(s) as dynamic.

        Args:
            idx: Dimension index to mark dynamic. If None, all dims are dynamic.

        Returns:
            self for method chaining
        """
        if idx is not None:
            # Mark a specific dimension as dynamic
            if idx in self._dim_specs:
                spec = self._dim_specs[idx]
                if spec != "dynamic":
                    raise ValueError(
                        f"Dimension {idx} is already marked as static, cannot also mark as dynamic"
                    )
            self._dim_specs[idx] = "dynamic"
            return self

        # Mark all dimensions as dynamic
        if self._metadata is None:
            raise ValueError(
                "Must provide example_input to use dynamic() without idx parameter"
            )

        # Mark all dimensions as dynamic
        for i in range(len(self._metadata.shape)):
            if i in self._dim_specs:
                spec = self._dim_specs[i]
                if spec != "dynamic":
                    raise ValueError(
                        f"Dimension {i} is already marked as static, cannot mark as dynamic"
                    )
            self._dim_specs[i] = "dynamic"

        # Also set rank if not already set
        if self._rank is None:
            self._rank = len(self._metadata.shape)

        return self

    def dtype(self, dtype: Union[str, torch.dtype]) -> "_ArgBase":
        """
        Constrain the tensor dtype.

        Args:
            dtype: Expected dtype (e.g., "float32" or torch.float32)

        Returns:
            self for method chaining
        """
        if isinstance(dtype, str):
            if dtype not in _DTYPE_MAP:
                raise ValueError(f"Unknown dtype string: {dtype}")
            dtype = _DTYPE_MAP[dtype]

        self._constraints.append(DtypeConstraint(dtype=dtype))
        return self

    def rank(self, rank: int) -> "_ArgBase":
        """
        Constrain the number of dimensions.

        Args:
            rank: Expected number of dimensions

        Returns:
            self for method chaining
        """
        if rank < 0:
            raise ValueError(f"Rank must be >= 0, got {rank}")

        if self._rank is not None and self._rank != rank:
            raise ValueError(
                f"Rank already specified as {self._rank}, cannot change to {rank}"
            )

        self._rank = rank
        return self

    def notNone(self) -> "_ArgBase":
        """
        Assert that argument is not None.

        Returns:
            self for method chaining
        """
        self._constraints.append(NoneConstraint(is_none=False))
        return self

    def isNone(self) -> "_ArgBase":
        """
        Assert that argument is None.

        Returns:
            self for method chaining
        """
        self._constraints.append(NoneConstraint(is_none=True))
        return self

    def layout(self, layout: torch.layout) -> "_ArgBase":
        """
        Constrain the tensor layout.

        Args:
            layout: Expected layout (e.g., torch.strided, torch.sparse_coo)

        Returns:
            self for method chaining
        """
        self._constraints.append(LayoutConstraint(layout=layout))
        return self

    def device(self, device: Union[str, torch.device]) -> "_ArgBase":
        """
        Constrain the tensor device.

        Args:
            device: Expected device (e.g., "cuda", "cpu", torch.device("cuda:0"))

        Returns:
            self for method chaining
        """
        if isinstance(device, str):
            device = torch.device(device)

        self._constraints.append(DeviceConstraint(device=device))
        return self

    def type_(self, t: type) -> "_ArgBase":
        """
        Constrain the type of non-tensor argument.

        Args:
            t: Expected Python type (e.g., int, bool, str)

        Returns:
            self for method chaining
        """
        self._constraints.append(TypeConstraint(expected_type=t))
        return self

    def tensor_subclass(self, t: type) -> "_ArgBase":
        """
        Constrain the tensor subclass type.

        Args:
            t: Expected tensor subclass (e.g., DTensor)

        Returns:
            self for method chaining
        """
        self._constraints.append(TensorSubclassConstraint(subclass_type=t))
        return self

    def _build_constraints(self) -> list[Constraint]:
        """Build the list of constraints from tracked state."""
        result = list(self._constraints)

        # Add RankConstraint if rank was specified
        if self._rank is not None:
            result.append(RankConstraint(rank=self._rank))

        # Build ShapeConstraint if we have dimension constraints
        if self._dim_specs:
            # We need to know the rank to build a proper ShapeConstraint
            rank = self._rank
            if rank is None and self._metadata is not None:
                rank = len(self._metadata.shape)

            if rank is not None:
                # Build a shape tuple and dynamic_dims list
                shape: list[_MetadataIntLike] = []
                dynamic_dims: list[int] = []

                for i in range(rank):
                    if i in self._dim_specs:
                        spec = self._dim_specs[i]
                        if spec not in ("dynamic", "static_unspecified"):
                            # Static with a size
                            assert isinstance(spec, int)
                            shape.append(spec)
                        elif spec == "static_unspecified":
                            # Static but no size yet - use metadata if available
                            if self._metadata is not None and i < len(
                                self._metadata.shape
                            ):
                                shape.append(self._metadata.shape[i])
                            else:
                                # Can't determine size, skip creating ShapeConstraint
                                return result
                        else:  # "dynamic"
                            # For dynamic dims, we still need a placeholder shape value
                            if self._metadata is not None and i < len(
                                self._metadata.shape
                            ):
                                shape.append(self._metadata.shape[i])
                            else:
                                # Can't determine shape, skip creating ShapeConstraint
                                return result
                            dynamic_dims.append(i)
                    else:
                        # No constraint on this dimension
                        # Use metadata if available
                        if self._metadata is not None and i < len(self._metadata.shape):
                            shape.append(self._metadata.shape[i])
                        else:
                            # Can't determine shape, skip creating ShapeConstraint
                            return result

                result.append(
                    ShapeConstraint(
                        shape=tuple(shape),
                        dynamic_dims=dynamic_dims if dynamic_dims else None,
                    )
                )

        return result


class Arg(_ArgBase):
    """
    Represents constraints on a positional argument.

    This class uses a builder pattern to allow chaining multiple constraints:
        Arg(0).rank(4).dynamic(idx=0).dtype("float32")

    Args:
        position: 0-based index of the positional argument
        example_input: Optional example tensor to derive properties from
    """

    def __init__(self, position: int, example_input: Optional[torch.Tensor] = None):
        if position < 0:
            raise ValueError(f"Argument position must be >= 0, got {position}")

        super().__init__(example_input)
        self.position = position

    # ========================================================================
    # Builder Methods
    # ========================================================================
    # All builder methods (static, dynamic, dtype, rank, notNone, isNone,
    # layout, device, type_, tensor_subclass) inherited from _ArgBase

    # ========================================================================
    # Public Interface
    # ========================================================================

    @property
    def constraints(self) -> list[Constraint]:
        """Get the list of constraints for this argument."""
        return self._build_constraints()

    def __repr__(self) -> str:
        constraints_repr = ", ".join(
            c.__class__.__name__.replace("Constraint", "") for c in self._constraints
        )
        if constraints_repr:
            return f"Arg({self.position}, constraints=[{constraints_repr}])"
        else:
            return f"Arg({self.position})"


class KwArg(_ArgBase):
    """
    Represents constraints on a keyword argument.

    This class uses the same builder pattern as Arg:
        KwArg("batch_size").type_(int)

    Args:
        name: Name of the keyword argument
        example_input: Optional example value to derive properties from
    """

    def __init__(self, name: str, example_input: Optional[torch.Tensor] = None):
        if not name:
            raise ValueError("Keyword argument name cannot be empty")

        super().__init__(example_input)
        self.name = name

    # ========================================================================
    # Public Interface
    # ========================================================================

    @property
    def constraints(self) -> list[Constraint]:
        """Get the list of constraints for this argument."""
        return self._build_constraints()

    def __repr__(self) -> str:
        constraints_repr = ", ".join(
            c.__class__.__name__.replace("Constraint", "") for c in self._constraints
        )
        if constraints_repr:
            return f"KwArg({self.name!r}, constraints=[{constraints_repr}])"
        else:
            return f"KwArg({self.name!r})"


# ============================================================================
# ModelSpec and Rules
# ============================================================================


@dataclass
class CompilationRule:
    """
    Represents a single compilation rule in the ModelSpec.

    A rule consists of:
    - conditions: List of argument constraints (Arg/KwArg instances)
    - dispatcher: Optional custom dispatcher function for complex logic
    - contexts: Execution contexts this rule applies to
    - rule_id: Unique identifier for this rule
    """

    conditions: list[Union[Arg, KwArg]]
    dispatcher: Optional[Callable[..., Any]]
    contexts: list[Context]
    rule_id: str

    def __repr__(self) -> str:
        ctx_str = ", ".join(str(c) for c in self.contexts) if self.contexts else "any"
        dispatcher_str = (
            f", dispatcher={self.dispatcher.__name__}" if self.dispatcher else ""
        )
        return f"Rule({len(self.conditions)} conditions, contexts=[{ctx_str}]{dispatcher_str}, id={self.rule_id})"


@dataclass
class DefaultRule:
    """
    Represents the default fallback rule.

    This rule is used when no other rules match.
    """

    contexts: list[Context]

    def __repr__(self) -> str:
        ctx_str = ", ".join(str(c) for c in self.contexts) if self.contexts else "any"
        return f"DefaultRule(contexts=[{ctx_str}])"


class ModelSpec:
    """
    Main specification class for model compilation rules.

    ModelSpec allows users to declaratively specify different compilation
    strategies based on input properties, execution contexts, and custom
    dispatch logic.

    Example:
        spec = ModelSpec(model)
        spec.add(Arg(0).static((3, 4)), Arg(1).static((3, 4)))
        spec.add(Arg(0).rank(4).dynamic(idx=0), ctxs=[Context.GRAD])
        spec.default()

        dispatcher = spec.create_dispatcher(torch.compile)
        result = dispatcher(inputs)
    """

    def __init__(self, model: Union[torch.nn.Module, Callable[..., Any]]):
        """
        Initialize ModelSpec with a model or function.

        Args:
            model: The model or function to compile. Can be:
                   - torch.nn.Module
                   - Callable (regular Python function)
                   - torch.fx.GraphModule (for post-dynamo dispatch)
        """
        self.model = model
        self._rules: list[CompilationRule] = []
        self._default_rule: Optional[DefaultRule] = None

    def add(
        self,
        *conditions: Union[Arg, KwArg],
        dispatcher: Optional[Callable[..., Any]] = None,
        ctxs: Optional[list[Context]] = None,
    ) -> None:
        """
        Add a compilation rule to the spec.

        Args:
            *conditions: One or more Arg or KwArg objects
            dispatcher: Optional custom dispatcher function for complex symbolic logic
            ctxs: Optional list of execution contexts this rule applies to
                  (e.g., [Context.GRAD, Context.NO_GRAD])

        Example:
            spec.add(
                Arg(0).rank(4).dynamic(idx=0).dtype("float32"),
                Arg(1).static((4, 8)),
                ctxs=[Context.GRAD]
            )
        """
        if not conditions:
            raise ValueError("Must provide at least one condition")

        if ctxs is None:
            ctxs = []

        # Generate unique rule ID
        rule_id = f"rule_{uuid.uuid4().hex[:8]}"

        rule = CompilationRule(
            conditions=list(conditions),
            dispatcher=dispatcher,
            contexts=ctxs,
            rule_id=rule_id,
        )

        self._rules.append(rule)

    def default(self, ctxs: Optional[list[Context]] = None) -> None:
        """
        Add a default fallback compilation rule.

        This rule is used when no other rules match. If not specified,
        an error will be raised when no rules match.

        Args:
            ctxs: Optional list of execution contexts for the default rule

        Example:
            spec.default(ctxs=[Context.NO_GRAD])
        """
        if self._default_rule is not None:
            raise ValueError("Default rule already specified")

        if ctxs is None:
            ctxs = []

        self._default_rule = DefaultRule(contexts=ctxs)

    def create_dispatcher(self, compile_fn: Callable[..., Any]) -> "DispatcherFunction":
        """
        Generate a dispatcher function from this spec.

        The dispatcher will:
        1. Check guards for each rule in order
        2. Select the first matching rule
        3. Compile using compile_fn if needed
        4. Execute the compiled function with checks

        Args:
            compile_fn: The compiler function to use (e.g., torch.compile,
                       torch._inductor.compile_fx, custom compiler)

        Returns:
            DispatcherFunction: An executable dispatcher

        Example:
            # JIT compilation
            dispatcher = spec.create_dispatcher(torch.compile)

            # AOT compilation
            dispatcher = spec.create_dispatcher(
                lambda m, *args: torch._dynamo.aot_compile.aot_compile_fullgraph(m, args)
            )

            # Post-dynamo dispatch
            dispatcher = spec.create_dispatcher(torch._inductor.compile_fx)
        """
        # Import here to avoid circular dependency
        from torch._dynamo.model_spec.dispatcher import DispatcherFunction

        return DispatcherFunction(
            spec=self,
            compile_fn=compile_fn,
            compiled_variants={},
        )

    @staticmethod
    def custom_dispatcher(fn: Callable[..., Any]) -> Callable[..., Any]:
        """
        Decorator for marking a function as a custom dispatcher.

        Custom dispatcher functions should follow this pattern:
        - Single (possibly nested) if statement
        - Each branch returns COMPILE or RAISE_ERROR
        - Conditions only on fx-proxyable operations (shapes, strides, etc.)

        Example:
            @ModelSpec.custom_dispatcher
            def my_dispatcher(x, y):
                if x.shape[0] % 8 == 0:
                    return COMPILE
                else:
                    return RAISE_ERROR

            spec.add(Arg(0).notNone(), dispatcher=my_dispatcher)

        TODO: Actually implement this decorator to parse the function body and create
        a custom dispatcher function.
        """
        # Mark the function as a custom dispatcher
        fn._is_model_spec_dispatcher = True  # type: ignore[attr-defined]
        return fn

    def addWithDispatch(
        self,
        *conditions: Union[Arg, KwArg],
        ctxs: Optional[list[Context]] = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Decorator for adding a rule with a custom dispatcher.

        This is syntactic sugar for combining add() and @dispatcher.

        Example:
            @spec.addWithDispatch(Arg(0).notNone(), ctxs=[Context.GRAD])
            def my_dispatcher(x):
                if x.shape[0] > 10:
                    return COMPILE
                else:
                    return RAISE_ERROR
        """

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            # Mark as dispatcher
            fn = self.custom_dispatcher(fn)
            # Add the rule
            self.add(*conditions, dispatcher=fn, ctxs=ctxs)
            return fn

        return decorator

    def __repr__(self) -> str:
        model_name = (
            self.model.__class__.__name__
            if isinstance(self.model, torch.nn.Module)
            else getattr(self.model, "__name__", str(self.model))
        )

        rules_repr = "\n  ".join(repr(r) for r in self._rules)
        default_repr = f"\n  {self._default_rule}" if self._default_rule else ""

        return (
            f"ModelSpec(model={model_name}, "
            f"num_rules={len(self._rules)})\n"
            f"Rules:\n  {rules_repr}{default_repr}"
        )
