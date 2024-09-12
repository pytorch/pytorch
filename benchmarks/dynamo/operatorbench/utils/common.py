
from operators import load_operator_by_name, OperatorNotFoundError
from typing import Optional, List, Dict
import pathlib
import dataclasses

@dataclasses.dataclass
class OperatorConfig:
    name: str
    variant: str
    device: str
    extra_args: List[str]
    extra_env: Optional[Dict[str, str]] = None
    output_dir: Optional[pathlib.Path] = None


class BenchmarkResults:
    def __init__(self, durations: List[float]):
        self.durations = durations

    def median(self) -> float:
        return np.median(self.durations)