import importlib
import inspect
import pkgutil
from collections import defaultdict
from typing import Any, Dict, List, Optional

from torch._inductor.autoheuristic.autoheuristic_utils import (
    AHContext,
    AHMetadata,
    Choice,
)
from torch._inductor.autoheuristic.learnedheuristic_interface import LearnedHeuristic


def find_and_instantiate_subclasses(
    package_name: str, base_class: Any
) -> List[LearnedHeuristic]:
    instances = []

    package = importlib.import_module(package_name)
    for _, module_name, _ in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        try:
            module_basename = module_name.split(".")[-1]
            if not module_basename.startswith("_"):
                # learned heuristics start with an underscore
                continue
            module = importlib.import_module(module_name)

            # look for classes that are subclasses of base_class
            for _, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, base_class)
                    and obj != base_class
                ):
                    instance = obj()
                    instances.append(instance)
        except Exception as e:
            print(f"Error processing module {module_name}: {e}")

    return instances


class LearnedHeuristicController:
    """
    Class that finds and instantiates all learned heuristics. It also provides
    a way to get the decision of a learned heuristic.
    """

    existing_heuristics: Dict[str, List[LearnedHeuristic]] = defaultdict(list)
    """
    A dictionary that stores all the learned heuristics for each optimization.
    The key is the optimization name, and the value is a list of LearnedHeuristic objects.
    """

    heuristics_initialized: bool = False
    """
    A flag that indicates whether the learned heuristics have been initialized.
    Set to true when the get_decision() function is called for the first time.
    """

    def __init__(
        self,
        metadata: AHMetadata,
        context: AHContext,
    ) -> None:
        self.metadata = metadata
        self.context = context

    def get_heuristics(self, name: str) -> List[LearnedHeuristic]:
        """
        Returns a list of learned heuristics for the given optimization name.
        """

        if not LearnedHeuristicController.heuristics_initialized:
            # learned heuristics are generated into the following package
            learned_heuristics_package = "torch._inductor.autoheuristic.artifacts"

            # learned heuristics have to be of type LearnedHeuristic
            base_class = LearnedHeuristic
            found_heuristics = find_and_instantiate_subclasses(
                learned_heuristics_package, base_class
            )

            for learned_heuristic in found_heuristics:
                opt_name = learned_heuristic.get_name()
                LearnedHeuristicController.existing_heuristics[opt_name].append(
                    learned_heuristic
                )
            LearnedHeuristicController.heuristics_initialized = True

        return LearnedHeuristicController.existing_heuristics[name]

    def get_decision(self) -> Optional[Choice]:
        """
        Returns the decision made by the learned heuristic or None if no heuristic was found or the heuristic is unsure
        which choice to make.
        """

        heuristics = self.get_heuristics(self.metadata.name)
        for heuristic in heuristics:
            if heuristic.check_precondition(self.metadata, self.context):
                return heuristic.get_decision(self.context, self.metadata.choices)
        return None

    def get_decisions_ranked(self, top_k: int) -> Optional[List[Choice]]:
        heuristics = self.get_heuristics(self.metadata.name)
        for heuristic in heuristics:
            if heuristic.check_precondition(self.metadata, self.context):
                choices = heuristic.get_decisions_ranked(self.context)
                if choices is None:
                    return None
                avail_choices = [
                    choice for choice in choices if choice in self.metadata.choices
                ]
                return avail_choices[:top_k]
        return None
