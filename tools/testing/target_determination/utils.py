# Not to be confused with the heuristics/utils
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def gen_ci_artifact(included: list[Any], excluded: list[Any]) -> None:
    file_name = f"td_exclusions-{os.urandom(10).hex()}.json"
    with open(REPO_ROOT / "test" / "test-reports" / file_name, "w") as f:
        json.dump({"included": included, "excluded": excluded}, f)


def get_percent_to_run(enable_td: bool) -> int:
    # Returns the percent of tests to run, as an int.  If enable_td is True, run
    # everything.  Otherwise, the default is to run 25% of tests.  Sometimes we
    # run 50% of tests in order to gather some data about how TD performs, but
    # it is still not 100% because we still want to reduce TTS
    if not enable_td:
        return 100
    # Use SystemRandom instead of normal random because we frequently set seed
    # in testing, which would affect random. SystemRandom uses os.urandom, which
    # takes randomness from unaffected sources.
    run_extra = random.SystemRandom().random() < 0.25
    default = 25
    extra = 50
    return extra if run_extra else default
