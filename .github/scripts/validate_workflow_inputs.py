import sys
import json


def print_err(title: str, msg: str) -> None:
    print(f"::error title={title}::{msg}")


if __name__ == "__main__":
    for line in sys.stdin:
        print(f"Processing line: {line}")
        inputs = json.loads(line)
        print("json:")
        print(json.dumps(inputs, indent=2))

        def is_set(key: str) -> bool:
            return inputs.get(key) is not None and inputs.get(key) != ""

        is_custom_build = is_set("env-variables") or is_set("script")
        if is_custom_build and not is_set("artifact-suffix"):
            print_err(
                "Missing artifact-suffix",
                "Custom builds must have an artifact-suffix; "
                "see the workflow input descriptions for more info.",
            )
            sys.exit(1)
