"""Utilities for truncating assertion output.

Current default behaviour is to truncate assertion explanations at
terminal lines, unless running with an assertions verbosity level of at least 2 or running on CI.
"""

from __future__ import annotations

from _pytest.assertion import util
from _pytest.config import Config
from _pytest.nodes import Item


DEFAULT_MAX_LINES = 8
DEFAULT_MAX_CHARS = 8 * 80
USAGE_MSG = "use '-vv' to show"


def truncate_if_required(
    explanation: list[str], item: Item, max_length: int | None = None
) -> list[str]:
    """Truncate this assertion explanation if the given test item is eligible."""
    if _should_truncate_item(item):
        return _truncate_explanation(explanation)
    return explanation


def _should_truncate_item(item: Item) -> bool:
    """Whether or not this test item is eligible for truncation."""
    verbose = item.config.get_verbosity(Config.VERBOSITY_ASSERTIONS)
    return verbose < 2 and not util.running_on_ci()


def _truncate_explanation(
    input_lines: list[str],
    max_lines: int | None = None,
    max_chars: int | None = None,
) -> list[str]:
    """Truncate given list of strings that makes up the assertion explanation.

    Truncates to either 8 lines, or 640 characters - whichever the input reaches
    first, taking the truncation explanation into account. The remaining lines
    will be replaced by a usage message.
    """
    if max_lines is None:
        max_lines = DEFAULT_MAX_LINES
    if max_chars is None:
        max_chars = DEFAULT_MAX_CHARS

    # Check if truncation required
    input_char_count = len("".join(input_lines))
    # The length of the truncation explanation depends on the number of lines
    # removed but is at least 68 characters:
    # The real value is
    # 64 (for the base message:
    # '...\n...Full output truncated (1 line hidden), use '-vv' to show")'
    # )
    # + 1 (for plural)
    # + int(math.log10(len(input_lines) - max_lines)) (number of hidden line, at least 1)
    # + 3 for the '...' added to the truncated line
    # But if there's more than 100 lines it's very likely that we're going to
    # truncate, so we don't need the exact value using log10.
    tolerable_max_chars = (
        max_chars + 70  # 64 + 1 (for plural) + 2 (for '99') + 3 for '...'
    )
    # The truncation explanation add two lines to the output
    tolerable_max_lines = max_lines + 2
    if (
        len(input_lines) <= tolerable_max_lines
        and input_char_count <= tolerable_max_chars
    ):
        return input_lines
    # Truncate first to max_lines, and then truncate to max_chars if necessary
    truncated_explanation = input_lines[:max_lines]
    truncated_char = True
    # We reevaluate the need to truncate chars following removal of some lines
    if len("".join(truncated_explanation)) > tolerable_max_chars:
        truncated_explanation = _truncate_by_char_count(
            truncated_explanation, max_chars
        )
    else:
        truncated_char = False

    truncated_line_count = len(input_lines) - len(truncated_explanation)
    if truncated_explanation[-1]:
        # Add ellipsis and take into account part-truncated final line
        truncated_explanation[-1] = truncated_explanation[-1] + "..."
        if truncated_char:
            # It's possible that we did not remove any char from this line
            truncated_line_count += 1
    else:
        # Add proper ellipsis when we were able to fit a full line exactly
        truncated_explanation[-1] = "..."
    return [
        *truncated_explanation,
        "",
        f"...Full output truncated ({truncated_line_count} line"
        f"{'' if truncated_line_count == 1 else 's'} hidden), {USAGE_MSG}",
    ]


def _truncate_by_char_count(input_lines: list[str], max_chars: int) -> list[str]:
    # Find point at which input length exceeds total allowed length
    iterated_char_count = 0
    for iterated_index, input_line in enumerate(input_lines):
        if iterated_char_count + len(input_line) > max_chars:
            break
        iterated_char_count += len(input_line)

    # Create truncated explanation with modified final line
    truncated_result = input_lines[:iterated_index]
    final_line = input_lines[iterated_index]
    if final_line:
        final_line_truncate_point = max_chars - iterated_char_count
        final_line = final_line[:final_line_truncate_point]
    truncated_result.append(final_line)
    return truncated_result
