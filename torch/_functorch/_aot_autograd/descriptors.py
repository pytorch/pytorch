import dataclasses
from typing import Literal


"""
AOTAutograd descriptors are a path-like data structure (similar to pytree
paths and sources) that describe the semantic meaning of an input/output to FX
graphs.  Although you may know the input/output meaning at the top level of the
original function you traced, because we have many graph capture wrappers that
change the calling convention, it can be difficult to tell how these correspond
to the actual function you get back, to say nothing about the extra arguments/outputs
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
    pass


@dataclasses.dataclass(frozen=True)
class ParamAOTInput(AOTInput):
    target: str

    def expr(self):
        return f"self.get_parameter({self.target!r})"


@dataclasses.dataclass(frozen=True)
class InputAOTInput(AOTInput):
    idx: int

    def expr(self):
        return f"args[{self.idx}]"


@dataclasses.dataclass(frozen=True)
class FunctionalizedRNGWrapperAOTInput(AOTInput):
    name: Literal["seed", "offset"]

    def expr(self):
        return f"__functionalized_rng_wrapper_{self.name}"


@dataclasses.dataclass(frozen=True)
class SubclassGetAttrAOTInput(AOTInput):
    base: AOTInput
    attr: str

    def expr(self):
        return f"{self.base.expr()}.{self.attr}"


@dataclasses.dataclass(frozen=True)
class SubclassSizeAOTInput(AOTInput):
    base: AOTInput
    idx: int

    def expr(self):
        return f"{self.base.expr()}.size({self.idx})"


@dataclasses.dataclass(frozen=True)
class SubclassStrideAOTInput(AOTInput):
    base: AOTInput
    idx: int

    def expr(self):
        return f"{self.base.expr()}.stride({self.idx})"


@dataclasses.dataclass(frozen=True)
class ViewBaseAOTInput(AOTInput):
    base: AOTInput

    def expr(self):
        return f"{self.base.expr()}._base"


@dataclasses.dataclass(frozen=True)
class SyntheticBaseAOTInput(AOTInput):
    base: AOTInput

    def expr(self):
        return f"__make_synthetic_base({self.base.expr()})"


@dataclasses.dataclass(frozen=True)
class PhiloxForwardSeedAOTInput(AOTInput):
    pass

    # TODO: expr


@dataclasses.dataclass(frozen=True)
class PhiloxForwardBaseOffsetAOTInput(AOTInput):
    pass

    # TODO: expr


@dataclasses.dataclass(frozen=True)
class PhiloxBackwardSeedAOTInput(AOTInput):
    pass

    # TODO: expr


@dataclasses.dataclass(frozen=True)
class PhiloxBackwardBaseOffsetAOTInput(AOTInput):
    pass

    # TODO: expr


@dataclasses.dataclass(frozen=True)
class ForwardTokenAOTInput(AOTInput):
    pass


# Define AOTOutput first since it's referenced by other classes
@dataclasses.dataclass(frozen=True)
class AOTOutput:
    pass


# these are not "real" functions, in that they can't actually be executed;
# they truly are new inputs, but they're parameterized by other sources


@dataclasses.dataclass(frozen=True)
class InputMutationTangentAOTInput(AOTInput):
    base: AOTInput

    def expr(self):
        return f"__input_mutation_tangent({self.base.expr()})"


# Technically the "output" here is redundant, tangents always correspond to
# outputs
@dataclasses.dataclass(frozen=True)
class OutputTangentAOTInput(AOTInput):
    output: AOTOutput

    def expr(self):
        return f"__output_tangent({self.output.expr()})"


@dataclasses.dataclass(frozen=True)
class OutputIntermediateBaseTangentAOTInput(AOTInput):
    output: AOTOutput


# TODO: figure out repr for this


@dataclasses.dataclass(frozen=True)
class OutputAOTOutput(AOTOutput):
    idx: int


@dataclasses.dataclass(frozen=True)
class InputMutationAOTOutput(AOTOutput):
    mutated_input: AOTInput


@dataclasses.dataclass(frozen=True)
class IntermediateBaseAOTOutput(AOTOutput):
    base_of: AOTOutput


@dataclasses.dataclass(frozen=True)
class AliasedArgWithMetadataMutationAOTOutput(AOTOutput):
    # TODO: we're not recording detailed information about this
    pass


@dataclasses.dataclass(frozen=True)
class GradAOTOutput(AOTOutput):
    grad_of: AOTInput


@dataclasses.dataclass(frozen=True)
class PhiloxUpdatedForwardOffset(AOTOutput):
    pass


@dataclasses.dataclass(frozen=True)
class PhiloxUpdatedBackwardOffset(AOTOutput):
    pass


@dataclasses.dataclass(frozen=True)
class ForwardTokenAOTOutput(AOTOutput):
    pass


@dataclasses.dataclass(frozen=True)
class BackwardTokenAOTOutput(AOTOutput):
    pass
