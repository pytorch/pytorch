import json
import os

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from torch._inductor.autoheuristic_utils import (
    AHOperation,
    Choice,
    CHOICE_COL,
    Feedback,
    FEEDBACK_COL,
    Value,
)
from torch._inductor.fx_passes.learned_heuristics.learned_heuristic_controller import (
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
    def __init__(self, feedback_fn: Callable[[Choice], Feedback]) -> None:
        self.feedback_fn = feedback_fn

    def __call__(self, choice: Choice) -> Feedback:
        return self.feedback_fn(choice)


class LocalFeedback(_Feedback):
    def __init__(self, feedback_fn: Callable[[Choice], Feedback]) -> None:
        super().__init__(feedback_fn)


class GlobalFeedback(_Feedback):
    # TODO: will be supported later
    def __init__(self, feedback_fn: Callable[[Choice], Feedback]) -> None:
        super().__init__(feedback_fn)


class AHFeature:
    def __init__(self, name: str, value: Value, is_categorical: bool = False) -> None:
        self.name = name
        self.value = value
        self.is_categorical = is_categorical


class AHContext:
    features: List[AHFeature]

    def __init__(self) -> None:
        self.features = []

    def add_feature(
        self, name: str, value: Value, is_categorical: bool = False
    ) -> None:
        self.features.append(AHFeature(name, value, is_categorical=is_categorical))

    def to_dict(self) -> Dict[str, Value]:
        return {f.name: f.value for f in self.features}


class InconsistentMetadata(Exception):
    pass


class AutoHeuristic:
    collected_feedback: Dict[Choice, Feedback]

    def __init__(
        self,
        fallback: Callable[[], Choice],
        choices: List[Choice],
        feedback: Union[LocalFeedback, GlobalFeedback],
        context: AHContext,
        name: str,
        augment_context: Optional[List[AHOperation]] = None,
    ) -> None:
        self.fallback = fallback
        self.choices = choices
        self.feedback = feedback
        self.context = context
        self.name = name
        self.features = context.features
        self.collected_feedback = {}
        self.augment_context = augment_context

        if torch._inductor.config.autoheuristic_log_path == "DEFAULT":
            self.log_path = self.get_default_log_path()
        else:
            self.log_path = torch._inductor.config.autoheuristic_log_path

        # TODO(AlnisM): Allow something like AUTOHEURISTIC_MODE="collect:pad_mm,foo,bar"
        # to be able to collect data only for specific heuristics
        if torch._inductor.config.autoheuristic_mode == "COLLECT_DATA" and isinstance(
            self.feedback, LocalFeedback
        ):
            for choice in self.choices:
                feedback_val = self.feedback(choice)
                self.save_data(choice, feedback_val)

    def get_choice(self) -> Choice:
        if torch._inductor.config.autoheuristic_mode == "USE_HEURISTIC":
            context_dict = self.context.to_dict()
            if self.augment_context is not None:
                for op in self.augment_context:
                    op.apply_operation(context_dict)
            controller = LearnedHeuristicController(
                self.name,
                context_dict,
                self.choices,
                get_gpu_shared_memory(),
                torch.cuda.get_device_capability(),
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
        numerical_features = [f.name for f in self.features if not f.is_categorical]
        categorical_features = [f.name for f in self.features if f.is_categorical]

        # use amount of shared_memory and device_capability to identify GPU
        # TODO(AlnisM): there might be a better way to do this
        shared_memory = get_gpu_shared_memory()
        device_capability = torch.cuda.get_device_capability()

        metadata = {
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
            "choices": self.choices,
            "shared_memory": shared_memory,
            "device_capa": device_capability,
            "name": self.name,
        }
        return json.dumps(metadata)

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
            feature_header = ",".join([f.name for f in self.features])
            header = feature_header + "," + CHOICE_COL + "," + FEEDBACK_COL
            lines.append(header)

        line = ""
        feature_values = ",".join([str(f.value) for f in self.features])
        line += feature_values + "," + choice + "," + str(feedback_val)
        lines.append(line)

        with open(log_path, "a") as f:
            f.write("\n".join(lines) + "\n")
