import dataclasses


"""
AOTAutograd descriptors are a path-like data structure (similar to pytree
paths and sources) that describe the semantic meaning of an input/output to FX
graphs.  Although you may know the input/output meaning at the top level of
the original function you traced, because we have many graph capture wrappers
that change the calling convention, it can be difficult to tell how these
correspond to the actual FX graph you get back, to say nothing about the extra
arguments/outputs for tangents, gradients, etc.

Intuitively, suppose we have:

def wrapped_graph(*args):
    ret = graph(*in_transform(args))
    return out_transform(ret)

Then the descriptor for input[i] to graph describes a function fin_i such that:

    fin_i(args) == in_transform(args)[i],

and the descriptor for output[j] from graph describes a function fout_j such that:

    fout_j(out_transform(ret)) == ret[j]

AKA input descriptors tell you how to get from outer inputs to inner inputs,
while output descriptors tell you how to get from outer outputs to inner
outputs (inverse data flow!)

Example inputs:

- PlainAOTInput(idx=0) - the first input from the original callable, as is

- ParamAOTInput(target="mod.weight") - the parameter with FQN mod.weight

- TangentAOTInput(output=PlainAOTOutput(idx=1)) - the input tangent
  corresponding to the gradients for the second output in the forward graph

- ViewBaseAOTInput(base_of=PlainAOTInput(idx=0)) - it turned out the first
  input was actually a (differentiable) view of a tensor which aliased with
  another input tensor.  We replaced this input with a single input for the
  base of all of these inputs, replacing the original inputs (one of which is
  mentioned in base_of).  We would generate a GradAOTOutput for *this* input
  (and not the original PlainAOTInputs!)  If you have a joint graph where a
  view base like this is undesirable, you can eliminate this by cloning
  the views outside of the compiled region (assuming you aren't mutating this
  tensor).

- SubclassGetAttrAOTInput(base=AOTInput(idx=0), attr="inner") - this tensor
  corresponds to the "inner" tensor from the tensor subclass that is at the
  first index.  In general, joint graphs from AOTAutograd never take tensor
  subclasses as inputs; they are always unpacked into their constituent plain
  tensor pieces; use the descriptors to identify the parts of the tensor that
  are related.  Note that this can be nested (if you have nested tensor
  subclasses!)

Example outputs:

- PlainAOTOutput(idx=0) - the first output from the original forward function,
  as is

- GradAOTOutput(grad_of=PlainAOTInput(idx=1)) - the computed gradient for the
  second input to the graph, an output of the backward graph

- InputMutationAOTOutput(mutated_input=PlainAOTInput(idx=0)) - when the first
  input is mutated, the new value to be copied into the first input of the
  graph.  Sometimes, these outputs can be elided and the copy_ is done directly
  in the graph (controlled by keep_input_mutations), but if the input
  mutation must be differentiated through we always generate an output like this

- IntermediateBaseAOTOutput(base_of=PlainAOTOutput(idx=0)) - if we return
  multiple outputs which alias each other, we instead replace them with a single
  output tensor representing the base of all the aliases.  This output indicates
  it is the base for /one/ of those original outputs.  If this is undesirable in
  the joint graph, clone all outputs before returning from the graph.

- SubclassGetAttrAOTOutput(base=PlainAOTOutput(idx=0), idx="inner") - this
  tensor correspondings to the inner tensor of the first original output which
  is a tensor subclass.  This and other subclass components of that output will
  get repacked into a tensor subclass.

The full set of descriptors that are possible to annotate a graph can be
understood by looking at the overall ordering of transformations in
AOTAutograd.  First, we present a highly stylized account which is semantically
equivalent to what AOTAutograd does, but doesn't follow the implementation.  Then,
we describe how the implementation works.





At a high level, AOTAutograd is structured as a series of
wrappers on the original user function, which are composed together to form
the final function to trace.  As a result of this, AOTAutograd ends up first
building the full AOTInputs for a function to be traced, and then in reverse
buildings up the AOTOutput as it is tracing.  However, there is one important
exception here: after the precompile steps (dedup and synthetic base
handling), we do an initial pass to collect forward metadata that produces the
initial set of AOTOutput that we use to create the TangentAOTInputs.

As of July 2025, here is an exhaustive description of how inputs/outputs
traverse the wrappers from AOTAutograd, and what descriptors can be introduced
at these phases.

```
                            Build wrappers (FLOWS DOWN)         Run trace (FLOWS UP)
-------------------------------------------------------------------------------------------------
Begin                       PlainAOTInput                       (n/a)
                            ParamAOTInput

Precompile dedupe           (remove dupes)                      (nothing)

Precompile synthetic base   SyntheticBaseAOTInput               MetadataMutationAOTOutput
                            ViewBaseAOTInput

Forward metadata trace      PlainAOTOutput                      (n/a)
                            MetadataMutationAOTOutput

Prepare for autograd        (nothing)                           InputMutationAOTOutput
                                                                IntermediateBaseAOTOutput

Create joint                TangentAOTInput                     GradAOTOutput
                            above w/ InputMutationAOTOutput
                            above w/ IntermediateBaseAOTOutput

Precompile subclass         SubclassGetAttrAOTInput et al.      SubclassGetAttrAOTOutput et al.

Effect tokens               ForwardTokenAOTInput                ForwardTokenAOTOutput
                            BackwardTokenAOTInput               BackwardTokenAOTOutput

End                         (n/a)                               PlainAOTOutput
```

We can unroll the phases for ease of understanding the data flow:

- Input desc propagation (happens as we build wrappers)
  - [IN] Begin with original calling convention (PlainAOTInput, ParamAOTInput)
  - [IN] Precompile dedupe: (removes duplicate AOTInputs)
  - [IN] Precompile synthetic base: SyntheticBaseAOTInput, ViewBaseAOTInput
  - Forward metadata trace (mini output desc propagation)
    - [OUT] Original output convention: PlainAOTOutput
    - [OUT] Precompile synthetic base: MetadataMutationAOTOutput
  - [IN] Prepare for autograd: (nothing)
  - [IN] Create joint: TangentAOTInput (potentially w/
         IntermediateBaseAOTOutput, InputMutationAOTOutput)
  - [IN] Precompile subclass: SubclassGetAttrAOTInput et al.
  - [IN] Effect tokens: ForwardTokenAOTInput, BackwardTokenAOTInput
- Trigger a trace with the modified inputs on the wrapper
- Output desc propagation (happens as we unwind from the user function call in trace)
  - [OUT] Begin with original calling convention: PlainAOTOutput
  - [OUT] Effect tokens: ForwardTokenAOTOutput, BackwardTokenAOTOutput
  - [OUT] Precompile subclass: SubclassGetAttrAOTOutput et al.
  - [OUT] Create joint: GradAOTOutput
  - [OUT] Prepare for autograd: InputMutationAOTOutput, IntermediateBaseAOTOutput
  - [OUT] Precompile synthetic base: MetadataMutationAOTOutput
  - [OUT] Precompile dedupe: (nothing)

The important thing to know about this list is that descriptors can only be
created from top-to-bottom.  So for example, you can have a
SubclassGetAttrAOTInput(TangentAOTInput(PlainAOTOutput(...))) (you can see
that PlainAOTOutput -> TangentAOTInput -> SubclassGetAttrAOTInput is
consistent with the pipeline ordering), but you can NEVER have
TangentAOTInput(SubclassGetAttrAOTOutput(PlainAOTOutput(...)) (this is
inconsistent; we always do autograd BEFORE we process subclasses!)

Another important thing to note is that some AOTOutputs reference AOTInputs
(InputMutationAOTOutput and GradAOTOutput), but they can only reference
AOTInputs that were constructed at the time the wrapper was built (it's easier
to tell by looking at the table above).  So for example,
GradAOTOutput(SubclassGetAttrAOTInput(PlainAOTInput(...))) is illegal, because
subclasses are handled *after* create joint during wrapper construction.
Instead, you would have
SubclassGetAttrAOTOutput(GradAOTOutput(PlainAOTInput(...))); this intuitively
captures the fact that we always to autograd directly on the subclass, rather
than after desugaring the subclass into its inner tensors.

In conclusion, while it seems like there is a lot going on here, there are
only really three major phases in AOTAutograd (in order):

- View/mutation handling
- Autograd
- Subclasses

Remembering that they get handled in this order (and in particular, that
subclasses get handled very LATE) will get you most of the intuition.

For reference, here is a list of all AOTInput/AOTOutput, organized by how
likely you need to handle them:

- AOTInput
  - Important:
    - PlainAOTInput
    - ParamAOTInput
    - SubclassGetAttrAOTInput et al. (if you use subclasses)
    - TangentAOTInput
  - Less important, can be eliminated by cloning inputs of graph:
    - ViewBaseAOTInput
    - SyntheticBaseAOTInput
  - Non-tensor, mostly just ignore them:
    - DummyAOTInput
    - PhiloxForwardSeedAOTInput
    - PhiloxForwardBaseOffsetAOTInput
    - PhiloxBackwardSeedAOTInput
    - PhiloxBackwardBaseOffsetAOTInput
    - ForwardTokenAOTInput
    - BackwardTokenAOTInput
- AOTOutput
  - Important:
    - PlainAOTOutput
    - GradAOTOutput
    - SubclassGetAttrAOTOutput et al. (if you use subclasses)
    - InputMutationAOTOutput
  - Less important, can be eliminated by cloning outputs of graph:
    - IntermediateBaseAOTOutput
    - MetadataMutationAOTOutput
  - Non-tensor, mostly just ignore them:
    - PhiloxUpdatedForwardOffsetAOTOutput
    - PhiloxUpdatedBackwardOffsetAOTOutput
    - ForwardTokenAOTOutput
    - BackwardTokenAOTOutput
    - DummyAOTOutput
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
class PlainAOTInput(AOTInput):
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
class BackwardTokenAOTInput(AOTInput):
    """The world token which is threaded through side-effectful operations, for backwards"""

    def expr(self) -> str:
        return "__backward_token"


# Technically the "output" here is redundant, tangents always correspond to
# outputs
@dataclasses.dataclass(frozen=True)
class TangentAOTInput(AOTInput):
    """An input to the joint graph representing the tangent of an output."""

    output: "AOTOutput"

    def expr(self) -> str:
        return f"__output_tangent({self.output.expr()})"


# ------------

# AOTOutput

# ------------


@dataclasses.dataclass(frozen=True)
class PlainAOTOutput(AOTOutput):
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
class MetadataMutationAOTOutput(AOTOutput):
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
