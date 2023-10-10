import argparse
import os
from dataclasses import fields

from torch.onnx._internal import diagnostics
from torch.onnx._internal.diagnostics import infra


def gen_docs(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for field in fields(diagnostics.rules):
        rule = getattr(diagnostics.rules, field.name)
        if not isinstance(rule, infra.Rule):
            continue
        if not rule.id.startswith("FXE"):
            # Only generate docs for `dynamo_export` rules. Excluding rules for TorchScript
            # ONNX exporter.
            continue
        title = f"{rule.id}:{rule.name}"
        full_description_markdown = rule.full_description_markdown
        assert (
            full_description_markdown is not None
        ), f"Expected {title} to have a full description in markdown"
        with open(f"{out_dir}/{title}.md", "w") as f:
            f.write(f"# {title}\n")
            f.write(full_description_markdown)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ONNX diagnostics rules doc in markdown."
    )
    parser.add_argument(
        "out_dir", metavar="OUT_DIR", help="path to output directory for docs"
    )
    args = parser.parse_args()
    gen_docs(args.out_dir)


if __name__ == "__main__":
    main()
