from __future__ import annotations

import argparse
import collections
import dataclasses

import io
import logging
import pathlib
import random
import re

from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from torch.onnx._internal.fx import diagnostics

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_COMPACT_ERROR_GROUP = False


class ErrorAggregator:
    """
    Collect and group error messages for report at the end.

    Error messages are grouped if they are considered similar enough by inspecting the
    shared bigrams. If any keyword in `_EXACT_MATCH_KEYWORDS` are found in the error
    message, it will require exact match to be grouped.

    Copied and modified from pytorch-jit-paritybench/paritybench/reporting.py
    """

    # If these keywords are found in an error message, it won't be aggregated with other errors,
    # unless it is an exact match.
    # NOTE: When adding or updating keywords, please also comment the source of the keyword
    # and the reason why it is added.
    _EXACT_MATCH_KEYWORDS = [
        # Defined at https://github.com/pytorch/pytorch/blob/0eb4f072825494eda8d6a4711f7ef10163342937/torch/onnx/_internal/fx/function_dispatcher.py#L268  # noqa: B950
        # Show individual missing symbolic functions.
        "Cannot find symbolic function",
        # Defined at https://github.com/pytorch/pytorch/blob/773f6b626d2f6d44a9a875d08991627ec631dc01/torch/onnx/_internal/diagnostics/rules.yaml#L310  # noqa: B950
        # Show individual unsupported FX node.
        "Unsupported FX nodes",
    ]

    def __init__(self, log: Optional[logging.Logger] = None):
        super().__init__()
        self.error_groups = []
        self.bigram_to_group_ids = collections.defaultdict(list)
        self.log = log or logging.getLogger(__name__)

    def record(self, e: str, module: str):
        # NOTE: Hack since we don't have actual `Exception` object, just the message.
        # Original implementation was expecting an `Exception` object.
        error_msg = e
        full_msg = e
        return self._add(error_msg, [(error_msg, module, full_msg)])

    def update(self, other: ErrorAggregator):
        for errors in other.error_groups:
            self._add(errors[0][0], errors)

    def _add(self, error_msg: str, errors: List[Tuple[str, str, str]]):
        msg_words = list(re.findall(r"[a-zA-Z]+", error_msg))
        if any(keyword in error_msg for keyword in self._EXACT_MATCH_KEYWORDS):
            msg_bigrams = [error_msg]  # need exact match
        else:
            msg_bigrams = [
                f"{a}_{b}" for a, b in zip(msg_words, msg_words[1:])
            ] or msg_words

        shared_bigrams = collections.Counter()
        for bigram in msg_bigrams:
            shared_bigrams.update(self.bigram_to_group_ids[bigram])

        if shared_bigrams:
            best_match, count = shared_bigrams.most_common(1)[0]
            if count > len(msg_bigrams) // 2:
                self.error_groups[best_match].extend(errors)
                return False

        # No match, create a new error group
        group_id = len(self.error_groups)
        self.error_groups.append(errors)
        for bigram in msg_bigrams:
            self.bigram_to_group_ids[bigram].append(group_id)

        return True

    @staticmethod
    def format_error_group(errors: List[Tuple[str, str, str]]):
        if _COMPACT_ERROR_GROUP:
            # Compress each error group into a single line.
            contexts = [
                context
                for context, _ in random.choices(
                    list(
                        collections.Counter(
                            context for msg, context, _ in errors
                        ).items()
                    ),
                    k=3,
                )
            ]
            return f"  - {len(errors)} like: `{errors[0][0]}` (examples {', '.join(contexts)})"
        else:
            # Print test cases in error group on individual lines.
            indent = " " * 8
            indented_contexts = []
            unique_contexts = {context for _, context, _ in errors}

            for context in unique_contexts:
                indented_contexts.extend(
                    f"{indent}{line}" for line in context.split("\n")
                )
            # Handle multiline error messages. Put it into code block.
            error_str = errors[0][0]
            if error_str.find("\n") != -1:
                error_str = f"\n```\n{error_str}\n```"
            else:
                error_str = f"`{error_str}`"
            title = f"  - {len(errors)} like: {error_str}"
            joined_context = "\n".join(indented_contexts)
            joined_context = f"```\n{joined_context}\n```"

            return "\n".join([title, joined_context])

    def __str__(self):
        # For each error group, sort based on unique model.
        errors = sorted(
            self.error_groups,
            key=lambda error_group: len({context for _, context, _ in error_group}),
            reverse=True,
        )
        return "\n".join(map(self.format_error_group, errors))

    def __len__(self):
        return sum(map(len, self.error_groups))


class ErrorAggregatorDict:
    """
    Collect error types and individually group their error messages for a debug report at the end.

    For each error type, create an `ErrorAggregator` object to collect and group
    error messages.

    Copied and modified from pytorch-jit-paritybench/paritybench/reporting.py
    """

    def __init__(self):
        super().__init__()
        self.aggregator: Dict[str, ErrorAggregator] = dict()

    def __getitem__(self, item: str):
        if item not in self.aggregator:
            self.aggregator[item] = ErrorAggregator(logging.getLogger(f"{item}"))
        return self.aggregator[item]

    def update(self, other):
        for key, value in other.aggregator.items():
            self[key].update(other=value)

    def format_report(self) -> str:
        return "\n".join(
            f"\n#### {name} ({len(self[name])} total):\n{self[name]}"
            for name, _ in sorted(
                [(k, len(v)) for k, v in self.aggregator.items()],
                key=lambda x: x[1],
                reverse=True,
            )
        )

    def record(self, error_type: str, error: str, module: str):
        if self[error_type].record(error, module):
            log.exception("%s error from %s", error_type, module)


class ExportErrorCsvParser:
    """Parses `*_export_error.csv` produced by onnxbench, aggregates errors and produces report.

    Two types of aggregations are performed.
    - Per error type: For each error type, group affected models by similar error messages.
        Sorted by number of affected models. Helps identifying critical errors that affect
        models the most.
    - Per model: For each model, group error messages by similar error type.
        Sorted by number of errors. This is typically showing all errors reported for
        each model.
    """

    def __init__(
        self,
        output_dir: pathlib.Path,
        compiler: str,
        suites: Sequence[str],
        dtype: str,
        mode: str,
        device: str,
        testing: str,
    ):
        self.output_dir = output_dir
        self.compiler = compiler
        self.suites = suites
        self.dtype = dtype
        self.mode = mode
        self.device = device
        self.testing = testing

    def get_output_filename(self, suite: str) -> pathlib.Path:
        return (
            self.output_dir
            / f"{self.compiler}_{suite}_{self.dtype}_{self.mode}_{self.device}_{self.testing}_export_error.csv"
        )

    def initialize_summary(self) -> None:
        self._per_error_summary: ErrorAggregatorDict = ErrorAggregatorDict()
        self._per_model_summary: ErrorAggregatorDict = ErrorAggregatorDict()

    def read_csv(self, output_filename: pathlib.Path) -> pd.DataFrame:
        return pd.read_csv(output_filename).replace(float("nan"), None)

    def parse_csv(self, output_filename: pathlib.Path) -> Sequence[ExportErrorRow]:
        try:
            df = self.read_csv(output_filename)
        except FileNotFoundError as e:
            # Could be no error.
            # TODO: Create empty file if no error.
            log.warning("File not found: %s", output_filename)
            return
        for _, row in df.iterrows():
            yield ExportErrorRow(**row)

    def summarize_error_row(self, error_row: ExportErrorRow) -> None:
        if error_row.rule_id and error_row.rule_name:
            error_type = f"{error_row.rule_id}: {error_row.rule_name}[{error_row.diagnostic_level}]"
            error_message = error_row.diagnostic_message
        else:
            error_type = error_row.exception_type_name
            error_message = error_row.exception_message

        self._per_error_summary.record(error_type, error_message, error_row.model_name)

        self._per_model_summary.record(error_row.model_name, error_type, error_message)

    def parse_and_summarize(self):
        self.initialize_summary()
        for suite in self.suites:
            output_filename = self.get_output_filename(suite)
            for error_row in self.parse_csv(output_filename):
                self.summarize_error_row(error_row)

    def gen_summary_files(self):
        self.parse_and_summarize()

        str_io = io.StringIO()

        str_io.write(f"# Export Error Summary Dashboard for {self.compiler} ##\n")
        str_io.write("\n")

        str_io.write("## Summary Grouped by Error ##\n")
        str_io.write("<details>\n<summary>See more</summary>\n")
        str_io.write(self._per_error_summary.format_report())
        str_io.write("\n")
        str_io.write("</details>\n")
        str_io.write("\n")

        str_io.write("## Summary Grouped by Model ##\n")
        str_io.write("<details>\n<summary>See more</summary>\n")
        str_io.write(self._per_model_summary.format_report())
        str_io.write("\n")
        str_io.write("</details>\n")
        str_io.write("\n")

        with open(f"{self.output_dir}/gh_{self.compiler}_error_summary.log", "w") as f:
            f.write(str_io.getvalue())


@dataclasses.dataclass
class ExportErrorRow:
    device: str
    model_name: str
    batch_size: int
    rule_id: Optional[str] = None
    rule_name: Optional[str] = None
    diagnostic_level: Optional[str] = None
    diagnostic_message: Optional[str] = None
    exception_type_name: Optional[str] = None
    exception_message: Optional[str] = None

    def __post_init__(self):
        assert (
            self.rule_id is not None
            and self.rule_name is not None
            and self.diagnostic_level is not None
            and self.diagnostic_message is not None
        ) or self.exception_type_name, (
            "Either rule_id, rule_name, diagnostic_level and diagnostic_message "
            "must be set or exception_type_name must be set"
        )

    @property
    def headers(self) -> List[str]:
        return [field.name for field in dataclasses.fields(self)]

    @property
    def row(self) -> List[str]:
        return [getattr(self, field.name) for field in dataclasses.fields(self)]


class ExportErrorParser:
    def __init__(self, device: str, model_name: str, batch_size: int):
        self.device = device
        self.model_name = model_name
        self.batch_size = batch_size

    def _qualified_exception_class_name(self, exception: Exception) -> str:
        if exception.__class__.__module__ == "builtins":
            return exception.__class__.__name__
        return f"{exception.__class__.__module__}.{exception.__class__.__name__}"

    def parse_diagnostic_context(
        self,
        diagnostic_context: diagnostics.DiagnosticContext,
    ) -> Sequence[ExportErrorRow]:
        for diagnostic in diagnostic_context.diagnostics:
            if diagnostic.level >= diagnostics.levels.ERROR:
                yield ExportErrorRow(
                    device=self.device,
                    model_name=self.model_name,
                    batch_size=self.batch_size,
                    rule_id=diagnostic.rule.id,
                    rule_name=diagnostic.rule.name,
                    diagnostic_level=diagnostic.level.name,
                    diagnostic_message=diagnostic.message,
                )

    def parse_exception(self, exception: Exception) -> ExportErrorRow:
        return ExportErrorRow(
            device=self.device,
            model_name=self.model_name,
            batch_size=self.batch_size,
            exception_type_name=self._qualified_exception_class_name(exception),
            exception_message=str(exception),
        )


def summarize_log_from_csv(
    output_dir: pathlib.Path,
    compilers: Sequence[str],
    suites: Sequence[str],
    dtype: str,
    mode: str,
    device: str,
    testing: str,
):
    for compiler in compilers:
        csv_parser = ExportErrorCsvParser(
            output_dir, compiler, suites, dtype, mode, device, testing
        )
        csv_parser.gen_summary_files()


def merge_logs(
    output_dir: pathlib.Path,
    compilers: Sequence[str],
    mode: str,
):
    logs_to_merge = [
        "gh_title.txt",
        "gh_build_summary.txt",
        "gh_executive_summary.txt",
        "gh_metric_regression.txt",
        f"gh_{mode}.txt",
    ]
    logs_to_merge.extend(f"gh_{compiler}_error_summary.log" for compiler in compilers)

    with open(output_dir / "gh_report.txt", "w") as f:
        for log in logs_to_merge:
            with open(output_dir / log) as log_file:
                f.write(log_file.read())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        help="The output directory to find the csv logs and save summary logs.",
    )
    parser.add_argument("--suites", action="append", help="huggingface/torchbench/timm")
    parser.add_argument(
        "--compilers", action="append", help="torchscript-onnx/dynamo-onnx"
    )
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda"], default="cuda", help="cpu/cuda"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output_dir = pathlib.Path(args.output_dir)
    compilers = args.compilers
    assert compilers, "Must specify at least one compiler"
    suites = args.suites
    assert suites, "Must specify at least one suite"
    device = args.device

    # TODO(bowbao): support different dtype, mode, testing
    dtype = "float32"
    mode = "inference"
    testing = "accuracy"

    summarize_log_from_csv(output_dir, compilers, suites, dtype, mode, device, testing)
    merge_logs(output_dir, compilers, mode)
