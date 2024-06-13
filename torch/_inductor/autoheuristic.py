import json
import os

from typing import Any, Callable, Dict, List, Tuple, Union

import torch

from torch._inductor.runtime.runtime_utils import cache_dir

Feedback = Union[int, float]
Choice = str


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
    def __init__(
        self, name: str, value: Union[int, float, str], is_categorical: bool = False
    ) -> None:
        self.name = name
        self.value = value
        self.is_categorical = is_categorical


class AHContext:
    features: List[AHFeature]

    def __init__(self) -> None:
        self.features = []

    def add_feature(
        self, name: str, value: Union[int, float, str], is_categorical: bool = False
    ) -> None:
        self.features.append(AHFeature(name, value, is_categorical=is_categorical))


class InconsistentMetadata(Exception):
    pass


class AutoHeuristic:
    def __init__(
        self,
        fallback: Choice,
        choices: List[Choice],
        feedback: Union[LocalFeedback, GlobalFeedback],
        context: AHContext,
        name: str,
    ) -> None:
        self.fallback = fallback
        self.choices = choices
        self.feedback = feedback
        self.context = context
        self.name = name
        self.features = context.features

        if torch._inductor.config.autoheuristic_mode == "COLLECT_DATA" and isinstance(
            self.feedback, LocalFeedback
        ):
            for choice in self.choices:
                feedback_val = self.feedback(choice)
                self.save_data(choice, feedback_val)

    def get_choice(self) -> Choice:
        return self.fallback

    @staticmethod
    def get_device_identifier() -> str:
        # a heuristic might work well for one GPU, but not for another
        # we store the collected data per GPU model and learn a heuristic per GPU model

        # TODO: just using the device name for now, but the same GPU model can have different names
        device_name = torch.cuda.get_device_name().replace(" ", "_")
        return device_name

    def get_log_path(self) -> str:
        device_name = self.get_device_identifier()
        path = f"{cache_dir()}/autoheuristic/{device_name}/"
        os.makedirs(path, exist_ok=True)
        path += f"{self.name}.txt"
        return path

    def serialize_metadata(self) -> str:
        numerical_features = [f.name for f in self.features if not f.is_categorical]
        categorical_features = [f.name for f in self.features if f.is_categorical]
        metadata = {
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
            "choices": self.choices,
        }
        return json.dumps(metadata)

    def deserialize_metadata(self, json_string: str) -> Dict[str, Any]:
        return json.loads(json_string)

    def save_data(self, choice: Choice, feedback_val: Feedback) -> None:
        log_path = self.get_log_path()

        lines = []
        log_exists = os.path.exists(log_path)
        if log_exists:
            # if log already exists, make sure it is consistent
            metadata = self.serialize_metadata()
            existing_metadata = self.get_metadata_str_from_log()
            if existing_metadata != metadata:
                raise InconsistentMetadata(
                    "Given metadata does not match existing metadata"
                )
        else:
            lines.append(self.serialize_metadata())
            feature_header = ",".join([f.name for f in self.features])
            header = feature_header + ",choice,feedback"
            lines.append(header)

        line = ""
        feature_values = ",".join([str(f.value) for f in self.features])
        line += feature_values + "," + choice + "," + str(feedback_val)
        lines.append(line)

        with open(log_path, "a") as f:
            f.write("\n".join(lines) + "\n")

    def get_metadata_str_from_log(self) -> str:
        log_path = self.get_log_path()
        with open(log_path, newline="") as file:
            json_string = file.readline().strip()
            return json_string

    def deserialize_data(self) -> Tuple[Any, Dict[str, Any]]:
        log_path = self.get_log_path()
        json_string = self.get_metadata_str_from_log()
        metadata = self.deserialize_metadata(json_string)

        import pandas as pd  # type: ignore[import-untyped]

        df = pd.read_csv(log_path, skiprows=1)
        return (df, metadata)
