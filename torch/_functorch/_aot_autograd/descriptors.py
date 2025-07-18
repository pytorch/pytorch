import dataclasses


"""
AOTAutograd descriptors are a path-like data structure (similar to pytree
paths and sources) that describe the semantic meaning of an input/output to FX
graphs.  Although you may know the input/output meaning at the top level of the
original function you traced, because we have many graph capture wrappers that
change the calling convention, it can be difficult to tell how these correspond
to the actual FX graph you get back, to say nothing about the extra arguments/outputs
for tangents, gradients, etc.

Intuitively, suppose we have:

def wrapped_graph(*args):
    ret = graph(*in_transform(args))
    return out_transform(ret)

Then the descriptor for input[i] to graph describes a function fin_i such that:

    fin_i(args) == in_transform(args)[i],

TODO: output descriptor doesn't make sense

and the descriptor for output[j] from graph describes a function fout_j such that:

    fout_j(out_transform(ret)) == ret[j]

AKA input descriptors tell you how to get from outer inputs to inner inputs, while
output descriptors tell you how to get from outer outputs to inner outputs (inverse data flow!)
"""


@dataclasses.dataclass(frozen=True)
class AOTInput:
    """Describes where an input from an AOTAutograd produced FX graph comes from"""

    def expr(self) -> str:
        raise NotImplementedError("Subclasses must implement expr()")


@dataclasses.dataclass(frozen=True)
class AOTOutput:
    """Describes where an output from an AOTAutograd produced FX graph will
    eventually be bundled into the final output"""

    def expr(self) -> str:
        raise NotImplementedError("Subclasses must implement expr()")


# ------------

# AOTInput

# ------------


@dataclasses.dataclass(frozen=True)
class ParamAOTInput(AOTInput):
    """The input is a parameter, whose FQN is target"""

    target: str

    def expr(self) -> str:
        return f"self.get_parameter({self.target!r})"


@dataclasses.dataclass(frozen=True)
class DummyAOTInput(AOTInput):
    """In some circumstances, we want to call into a function that expects AOTInput, but
    we don't actually care about that logic (most typically, because some code is being used
    for both compile-time and run-time; AOTInput processing is not needed in this situation.
    Pass a dummy in this situation; but it is better to just have a version of the function
    that doesn't have this at all."""

    idx: int

    def expr(self) -> str:
        return f"__dummy{self.idx}"


@dataclasses.dataclass(frozen=True)
class InputAOTInput(AOTInput):
    """The input is a plain input, corresponding to a particular positional index.

    Note that AOTInput is always relative to a function with a *flat* calling convention,
    e.g., as accepted by `aot_module_simplified`.  There are some AOTAutograd APIs that
    flatten pytrees, and we don't record PyTree key paths from the flattening (but we
    could and should!)
    """

    idx: int

    def expr(self) -> str:
        return f"args[{self.idx}]"


@dataclasses.dataclass(frozen=True)
class SubclassGetAttrAOTInput(AOTInput):
    """Subclass inputs get unpacked into their constituent pieces before going into an FX
    graph.  This tells you which particular attribute of the subclass this particular
    input corresponds to (of the 'base' originally subclass argument.)
    """

    base: AOTInput
    attr: str

    def expr(self) -> str:
        return f"{self.base.expr()}.{self.attr}"


@dataclasses.dataclass(frozen=True)
class SubclassSizeAOTInput(AOTInput):
    """Which subclass this particular outer size SymInt input (at dim idx) came from."""

    base: AOTInput
    idx: int

    def expr(self) -> str:
        return f"{self.base.expr()}.size({self.idx})"


@dataclasses.dataclass(frozen=True)
class SubclassStrideAOTInput(AOTInput):
    """Which subclass this particular outer stride SymInt input (at dim idx) came from."""

    base: AOTInput
    idx: int

    def expr(self) -> str:
        return f"{self.base.expr()}.stride({self.idx})"


@dataclasses.dataclass(frozen=True)
class ViewBaseAOTInput(AOTInput):
    """
    When multiple differentiable inputs are views of the same input, AOTAutograd will replace all of these
    views with a single input representing the base.  If this is undesirable, you can clone the views
    example inputs before passing them into AOTAutograd.

    TODO: In principle we could report ALL of the inputs who this is a base of.
    """

    base_of: AOTInput

    def expr(self) -> str:
        return f"{self.base_of.expr()}._base"


@dataclasses.dataclass(frozen=True)
class SyntheticBaseAOTInput(AOTInput):
    """This is similar to ViewBaseAOTInput, but this happens when none of the views were differentiable, so
    we weren't able to get our hands on the true original view and constructed a synthetic one instead
    for the sake of autograd.
    """

    base_of: AOTInput

    def expr(self) -> str:
        return f"__make_synthetic_base({self.base_of.expr()})"


@dataclasses.dataclass(frozen=True)
class PhiloxForwardSeedAOTInput(AOTInput):
    """The seed for functionalized Philox RNG calls, specifically for forward graph."""

    def expr(self) -> str:
        return "__philox_forward_seed"


@dataclasses.dataclass(frozen=True)
class PhiloxForwardBaseOffsetAOTInput(AOTInput):
    """The offset for functionalized Philox RNG calls, specifically for forward graph."""

    def expr(self) -> str:
        return "__philox_forward_base_offset"


@dataclasses.dataclass(frozen=True)
class PhiloxBackwardSeedAOTInput(AOTInput):
    """The seed for functionalized Philox RNG calls, specifically for backward graph."""

    def expr(self) -> str:
        return "__philox_backward_seed"


@dataclasses.dataclass(frozen=True)
class PhiloxBackwardBaseOffsetAOTInput(AOTInput):
    """The offset for functionalized Philox RNG calls, specifically for backward graph."""

    def expr(self) -> str:
        return "__philox_backward_base_offset"


@dataclasses.dataclass(frozen=True)
class ForwardTokenAOTInput(AOTInput):
    """The world token which is threaded through side-effectful operations"""

    def expr(self) -> str:
        return "__forward_token"


@dataclasses.dataclass(frozen=True)
class InputMutationTangentAOTInput(AOTInput):
    """An input to the joint graph representing the tangent of a mutated input
    (which is therefore implicitly an output)"""

    base: AOTInput

    def expr(self) -> str:
        # these are not "real" functions, in that they can't actually be executed;
        # they truly are new inputs, but they're parameterized by other sources
        return f"__input_mutation_tangent({self.base.expr()})"


# Technically the "output" here is redundant, tangents always correspond to
# outputs
@dataclasses.dataclass(frozen=True)
class OutputTangentAOTInput(AOTInput):
    """An input to the joint graph representing the tangent of an output."""

    output: "AOTOutput"

    def expr(self) -> str:
        return f"__output_tangent({self.output.expr()})"


@dataclasses.dataclass(frozen=True)
class OutputIntermediateBaseTangentAOTInput(AOTInput):
    """An input to the joint graph representing the tangent of an
    'intermediate base' output that was created to deal with aliasing
    outputs."""

    output: "AOTOutput"

    def expr(self) -> str:
        return f"__output_intermediate_base_tangent({self.output.expr()})"


# ------------

# AOTOutput

# ------------


@dataclasses.dataclass(frozen=True)
class OutputAOTOutput(AOTOutput):
    """A plain tensor output at position idx of the output tuple"""

    idx: int

    def expr(self) -> str:
        return f"output[{self.idx}]"


@dataclasses.dataclass(frozen=True)
class InputMutationAOTOutput(AOTOutput):
    """The mutated value of an input tensor, returned so we can appropriately propagate autograd."""

    mutated_input: AOTInput

    def expr(self) -> str:
        return f"__input_mutation({self.mutated_input.expr()})"


@dataclasses.dataclass(frozen=True)
class IntermediateBaseAOTOutput(AOTOutput):
    """An intermediate base of multiple outputs which alias each other.  We only report ONE of
    the outputs that contributed to this base"""

    base_of: "AOTOutput"

    def expr(self) -> str:
        return f"__intermediate_base({self.base_of.expr()})"


@dataclasses.dataclass(frozen=True)
class AliasedArgWithMetadataMutationAOTOutput(AOTOutput):
    # TODO: we're not recording detailed information about this

    def expr(self) -> str:
        return "__aliased_arg_with_metadata_mutation"


@dataclasses.dataclass(frozen=True)
class GradAOTOutput(AOTOutput):
    """An output representing the computed gradient for a differentiable input, in the joint graph"""

    grad_of: AOTInput

    def expr(self) -> str:
        return f"__grad({self.grad_of.expr()})"


@dataclasses.dataclass(frozen=True)
class PhiloxUpdatedForwardOffsetAOTOutput(AOTOutput):
    """The final offset from the functionalized RNG calls, forward only"""

    def expr(self) -> str:
        return "__philox_updated_forward_offset"


@dataclasses.dataclass(frozen=True)
class PhiloxUpdatedBackwardOffsetAOTOutput(AOTOutput):
    """The final offset from the functionalized RNG calls, backward only"""

    def expr(self) -> str:
        return "__philox_updated_backward_offset"


@dataclasses.dataclass(frozen=True)
class ForwardTokenAOTOutput(AOTOutput):
    """The world token output for side-effectful calls, returned so we cannot DCE it, forward only"""

    def expr(self) -> str:
        return "__forward_token"


@dataclasses.dataclass(frozen=True)
class BackwardTokenAOTOutput(AOTOutput):
    """The world token output for side-effectful calls, returned so we cannot DCE it, backward only"""

    def expr(self) -> str:
        return "__backward_token"


# These are seemingly symmetric with their AOTInput counterparts.  The way to
# think about it is that a subclass could be an input or an output, and they
# get exploded into plain tensors on the way in and out.  So we need
# descriptors for both.
@dataclasses.dataclass(frozen=True)
class SubclassGetAttrAOTOutput(AOTOutput):
    """This output will be bundled into a subclass at this location"""

    base: AOTOutput
    attr: str

    def expr(self) -> str:
        return f"{self.base.expr()}.{self.attr}"


@dataclasses.dataclass(frozen=True)
class SubclassSizeAOTOutput(AOTOutput):
    """This output size will be bundled into a subclass at this location"""

    base: AOTOutput
    idx: int

    def expr(self) -> str:
        return f"{self.base.expr()}.size({self.idx})"


@dataclasses.dataclass(frozen=True)
class SubclassStrideAOTOutput(AOTOutput):
    """This output stride will be bundled into a subclass at this location"""

    base: AOTOutput
    idx: int

    def expr(self) -> str:
        return f"{self.base.expr()}.stride({self.idx})"


@dataclasses.dataclass(frozen=True)
class DummyAOTOutput(AOTOutput):
    """For cases when you don't actually care about descriptor propagation, do not use under normal
    circumstances."""

    idx: int

    def expr(self) -> str:
        return f"__dummy{self.idx}"
