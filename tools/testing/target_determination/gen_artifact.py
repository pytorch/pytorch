from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
TD_SOURCE = "td"
LLM_SELECTION_SOURCE = "llm_selection"


def gen_ci_artifact(
    included: list[Any], excluded: list[Any], *, source: str, td_excluded: list[Any]
) -> None:
    """included and excluded are the split that ran, which the LLM test selection
    made when source is LLM_SELECTION_SOURCE; td_excluded is what TD's split excluded."""
    file_name = f"td_exclusions-{os.urandom(10).hex()}.json"
    artifact = {"included": included, "excluded": excluded, "source": source}
    with open(REPO_ROOT / "test" / "test-reports" / file_name, "w") as f:
        json.dump({**artifact, "td_excluded": td_excluded}, f)
