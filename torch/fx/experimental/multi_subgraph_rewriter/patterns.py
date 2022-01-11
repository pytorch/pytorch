from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import ModuleType
from typing import Callable, List, Union

from torch import nn, Tensor
from torch.fx import symbolic_trace
from torch.fx.graph_module import GraphModule


@dataclass
class Pattern:
    """
    Named container for a pattern subgraph and its replacement.

    Attributes:
        name (str): The name of the pattern.
        pattern (GraphModule): The pattern subgraph to find and replace with `replacement`.
        replacement (GraphModule): The subgraph to replace `pattern` with.
    """

    name: str
    pattern: GraphModule
    replacement: GraphModule


class PatternVerificationError(Exception):
    """
    Raise to indicate a verification job failed.

    See abstract method `verify` in `PatternLoader`.
    """
    pass


class PatternLoader(ABC):
    """
    A base class for defining a subgraph subtitution pattern and verification tasks.

    Subclass this class and define all the abstract methods to define a pattern.
    """

    def __init__(self) -> None:
        self.candidate_traced: GraphModule = symbolic_trace(self.pattern)
        self.replacement_traced: GraphModule = symbolic_trace(self.replacement)
        for verification_method in self.verify:
            verification_method()
        self.input: Pattern = Pattern(
            name=self.name,
            pattern=self.candidate_traced,
            replacement=self.replacement_traced,
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Specify the name of the pattern object.
        """

    @property
    @abstractmethod
    def pattern(self) -> Union[Callable[..., Tensor], nn.Module]:
        """
        Specify the pattern subgraph as a PyTorch module.

        This method should return either an instantiated `nn.Module` object or a PyTorch forward function.
        Note that the torch.fx symbolic trace results of a forward function `f` and an `nn.Module` object
        whose forward function is `f` are equivalent.
        """
        pass

    @property
    @abstractmethod
    def replacement(self) -> Union[Callable[..., Tensor], nn.Module]:
        """
        Specify the replacement subgraph as a PyTorch module.

        This method should return either an instantiated `nn.Module` object or a PyTorch forward function.
        Note that the torch.fx symbolic trace results of a forward function `f` and an `nn.Module` object
        whose forward function is `f` are equivalent.
        """
        pass

    @property
    @abstractmethod
    def verify(self) -> List[Callable[[], None]]:
        """
        Specify the collection of verification tasks to run on the pattern-replacement pair.

        This method should return a list of methods that do not take any input and return nothing,
        instead raising a `PatternVerificationError` to indicate a verification job failed. We impose
        the restriction on input to force the verification tasks to rely only on the available
        attributes, e.g., `self.candidate_traced` and `self.replacement_traced`.
        """
        pass


def load_all_patterns_from_a_module(module: ModuleType) -> List[Pattern]:
    """
    Gather all `PatternLoader` objects from a module and return the `Pattern` objects therein.

    Since each `PatternLoader` object runs its `verfy` method upon instantiation, collecting
    `PatternLoader` objects first ensures that we end up with `Pattern` objects that satisfy
    the user-defined checks.
    """
    patterns: List[Pattern] = []
    for obj_name in dir(module):
        obj = getattr(module, obj_name)
        if isinstance(obj, PatternLoader):
            patterns.append(obj.input)
    return patterns
