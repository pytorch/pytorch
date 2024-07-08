import json
import os

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from torch._inductor.autoheuristic.autoheuristic_utils import (
    AHContext,
    AHMetadata,
    AHOperation,
    Choice,
    CHOICE_COL,
    Feedback,
    FEEDBACK_COL,
)
from torch._inductor.autoheuristic.learned_heuristic_controller import (
    LearnedHeuristicController,
)
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.utils import get_gpu_shared_memory


def deserialize_data(log_path: str) -> Tuple[Any, Dict[str, Any]]:
    json_string = get_metadata_str_from_log(log_path)
    metadata = deserialize_metadata(json_string)
    import pandas as pd  # type: ignore[import-untyped]

    df = pd.read_csv(log_path, skiprows=1)
    return (df, metadata)


def deserialize_metadata(json_string: str) -> Dict[str, Any]:
    return json.loads(json_string)


def get_metadata_str_from_log(log_path: str) -> str:
    with open(log_path, newline="") as file:
        json_string = file.readline().strip()
        return json_string


class _Feedback:
    """
    This is a base class for Feedback objects. It takes a function that calculates the feedback for a given choice.
    """

    def __init__(self, feedback_fn: Callable[[Choice], Feedback]) -> None:
        self.feedback_fn = feedback_fn

    def __call__(self, choice: Choice) -> Feedback:
        return self.feedback_fn(choice)


class LocalFeedback(_Feedback):
    """
    To be able to collect data for a choice, a function providing feedback given a choice has to be provided.
    LocalFeedback can be used when AutoHeuristic should immediately run the function to collect feedback for each choice
    (see pad_mm.py, where the autotuning happens locally, for an example).
    """

    def __init__(self, feedback_fn: Callable[[Choice], Feedback]) -> None:
        super().__init__(feedback_fn)


class GlobalFeedback(_Feedback):
    """
    In contrast to LocalFeedback, GlobalFeedback can be used when it is not possible to immediately collect feedback for
    the provided choices. GlobalFeedback will be required for example for kernel choice selection, where the feedback
    will be provided later after autotuning has happened in select_algorithm.py.
    """

    # TODO: will be supported later
    def __init__(self, feedback_fn: Callable[[Choice], Feedback]) -> None:
        super().__init__(feedback_fn)


class InconsistentMetadata(Exception):
    """
    Exception that is thrown when AutoHeuristic tries to log data to a file where the metadata stored in the file does
    not match the metadata it would store if the file didn't exist.
    """

    pass


class AutoHeuristic:
    """
    AutoHeuristic is a framework that allows one to collect data, learn a heuristic (i.e. a regression tree) and
    generate the heuristic to code. This class allows one to collect data. The collected data can then be used to train
    a heuristic (see torchgen/autoheuristic/).
    """

    collected_feedback: Dict[Choice, Feedback]

    def __init__(
        self,
        fallback: Callable[[], Choice],
        choices: List[Choice],
        feedback: Union[LocalFeedback, GlobalFeedback],
        context: AHContext,
        name: str,
        augment_context: Optional[List[AHOperation]] = None,
        precondition: Optional[Callable[[AHMetadata, AHContext], bool]] = None,
    ) -> None:
        """
        Initializes an instance of the AutoHeuristic class.

        Args:
            fallback: A callable that returns a Choice when the heuristic is unsure which choice to make, or
            AutoHeuristic is in data collection mode.
            choices: A list of possible choices the heuristic can make.
            feedback: An instance of LocalFeedback or GlobalFeedback that provides feedback for a given choice.
            context: Context to store with each choice and feedback.
            name: A string that identifies the heuristic.
            augment_context: An optional list of AHOperation instances that augment the context.
            precondition: A callable that returns a boolean indicating whether AutoHeuristic should run.
        """
        self.fallback = fallback
        self.choices = choices
        self.feedback = feedback
        self.context = context
        self.name = name
        self.collected_feedback = {}
        self.augment_context = augment_context
        self.metadata = AHMetadata(
            get_gpu_shared_memory(),
            torch.cuda.get_device_capability(),
            self.choices,
            self.name,
        )
        self.precondition = precondition

        if not self.satisfies_precondition():
            return

        if torch._inductor.config.autoheuristic_log_path == "DEFAULT":
            self.log_path = self.get_default_log_path()
        else:
            self.log_path = torch._inductor.config.autoheuristic_log_path

        # TODO(AlnisM): Allow something like AUTOHEURISTIC_MODE="collect:pad_mm,foo,bar"
        # to be able to collect data only for specific heuristics
        if torch._inductor.config.collect_autoheuristic(self.name) and isinstance(
            self.feedback, LocalFeedback
        ):
            for choice in self.choices:
                feedback_val = self.feedback(choice)
                self.save_data(choice, feedback_val)

    def satisfies_precondition(self) -> bool:
        return self.precondition is None or self.precondition(
            self.metadata, self.context
        )

    def get_choice(self) -> Choice:
        """
        Returns the chosen option based on the value of autoheuristic_use.
        If self.name is one of the comma separated strings in autoheuristic_use,
        it queries a learned heuristic to make a decision. Otherwise, it returns the fallback option.
        """

        if not self.satisfies_precondition():
            return self.fallback()

        if torch._inductor.config.use_autoheuristic(self.name):
            if self.augment_context is not None:
                self.context.apply_operations(self.augment_context)
            controller = LearnedHeuristicController(
                self.metadata,
                self.context,
            )
            decision = controller.get_decision()
            if decision is not None:
                return decision
        return self.fallback()

    def get_collected_feedback(self, choice: Choice) -> Any:
        return self.collected_feedback.get(choice, None)

    @staticmethod
    def get_device_identifier() -> str:
        # a heuristic might work well for one GPU, but not for another
        # we store the collected data per GPU model and learn a heuristic per GPU model

        # TODO(AlnisM): just using the device name for now, but the same GPU model can have different names
        device_name = torch.cuda.get_device_name().replace(" ", "_")
        return device_name

    def get_default_log_path(self) -> str:
        device_name = self.get_device_identifier()
        path = f"{cache_dir()}/autoheuristic/{device_name}/"
        os.makedirs(path, exist_ok=True)
        path += f"{self.name}.txt"
        return path

    def serialize_metadata(self) -> str:
        metadata_dict = self.metadata.to_dict()
        (
            num_features,
            cat_features,
        ) = self.context.get_numerical_and_categorical_features()
        metadata_dict["numerical_features"] = num_features
        metadata_dict["categorical_features"] = cat_features
        return json.dumps(metadata_dict)

    def save_data(self, choice: Choice, feedback_val: Feedback) -> None:
        self.collected_feedback[choice] = feedback_val
        log_path = self.log_path

        lines = []
        log_exists = os.path.exists(log_path)
        if log_exists:
            # if log already exists, make sure it is consistent
            metadata = self.serialize_metadata()
            existing_metadata = get_metadata_str_from_log(self.log_path)
            if existing_metadata != metadata:
                raise InconsistentMetadata(
                    "Given metadata does not match existing metadata"
                )
        else:
            lines.append(self.serialize_metadata())
            feature_header = self.context.get_feature_names_csv()
            header = feature_header + "," + CHOICE_COL + "," + FEEDBACK_COL
            lines.append(header)

        line = ""
        feature_values = self.context.get_feature_values_csv()
        line += feature_values + "," + choice + "," + str(feedback_val)
        lines.append(line)

        with open(log_path, "a") as f:
            f.write("\n".join(lines) + "\n")
