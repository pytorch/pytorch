# Owner(s): ["module: docs"]

"""Tests for the MyST Markdown to RST docstring converter in docs/source/conf.py."""

import re
import sys
from pathlib import Path

from torch.testing._internal.common_utils import run_tests, TestCase


def _load_converter_functions():
    """Extract and compile the converter functions from conf.py."""
    conf_path = Path(__file__).resolve().parent.parent / "docs" / "source" / "conf.py"
    source = conf_path.read_text()

    start = source.index("def _is_markdown_docstring")
    end = source.index("def process_docstring")
    func_source = source[start:end]

    namespace = {"re": re}
    exec(compile(func_source, str(conf_path), "exec"), namespace)
    return namespace["_is_markdown_docstring"], namespace["_myst_to_rst"]


_is_markdown_docstring, _myst_to_rst = _load_converter_functions()


class TestMystDetection(TestCase):
    def test_rst_docstring_not_detected(self):
        lines = [
            "Apply a 1D convolution.",
            "",
            ":math:`y = x * w`",
            ":class:`torch.Tensor`",
        ]
        self.assertFalse(_is_markdown_docstring(lines))

    def test_plain_napoleon_not_detected(self):
        lines = [
            "This function does something.",
            "",
            "Args:",
            "    x (int): The input.",
            "",
            "Returns:",
            "    The output.",
        ]
        self.assertFalse(_is_markdown_docstring(lines))

    def test_myst_backtick_directive_detected(self):
        lines = [
            "Description.",
            "",
            "```{note}",
            "Important.",
            "```",
        ]
        self.assertTrue(_is_markdown_docstring(lines))

    def test_myst_colon_fence_detected(self):
        lines = [
            "Description.",
            "",
            ":::{warning}",
            "Careful.",
            ":::",
        ]
        self.assertTrue(_is_markdown_docstring(lines))

    def test_myst_roles_detected(self):
        lines = ["See {py:class}`torch.Tensor` for details."]
        self.assertTrue(_is_markdown_docstring(lines))

    def test_myst_section_label_detected(self):
        lines = ["(my-label)=", "## Section"]
        self.assertTrue(_is_markdown_docstring(lines))

    def test_markdown_links_detected(self):
        lines = ["See [the docs](https://pytorch.org)."]
        self.assertTrue(_is_markdown_docstring(lines))

    def test_inline_math_detected(self):
        lines = ["The result is $x + y$."]
        self.assertTrue(_is_markdown_docstring(lines))

    def test_latex_inline_math_detected(self):
        lines = [r"The result is \(x + y\)."]
        self.assertTrue(_is_markdown_docstring(lines))


class TestMystBacktickDirective(TestCase):
    def test_basic_directive(self):
        lines = ["```{note}", "Content.", "```"]
        rst = _myst_to_rst(lines)
        self.assertIn(".. note::", rst)
        self.assertIn("   Content.", rst)

    def test_directive_with_argument(self):
        lines = [
            "```{admonition} My Title",
            ":class: warning",
            "",
            "Body text.",
            "```",
        ]
        rst = _myst_to_rst(lines)
        self.assertIn(".. admonition:: My Title", rst)
        self.assertIn("   :class: warning", rst)
        self.assertIn("   Body text.", rst)

    def test_directive_with_yaml_options(self):
        lines = [
            "```{figure} image.png",
            "---",
            "width: 80%",
            "alt: A figure",
            "---",
            "Caption.",
            "```",
        ]
        rst = _myst_to_rst(lines)
        self.assertIn(".. figure:: image.png", rst)
        self.assertIn("   :width: 80%", rst)
        self.assertIn("   :alt: A figure", rst)
        self.assertIn("   Caption.", rst)

    def test_eval_rst_passthrough(self):
        lines = [
            "```{eval-rst}",
            ".. autoclass:: MyClass",
            "    :members:",
            "```",
        ]
        rst = _myst_to_rst(lines)
        self.assertIn(".. autoclass:: MyClass", rst)
        self.assertIn("    :members:", rst)
        # eval-rst should NOT produce a ".. eval-rst::" directive
        self.assertNotIn(".. eval-rst::", rst)


class TestMystColonFenceDirective(TestCase):
    def test_basic_colon_fence(self):
        lines = [":::{note}", "A note.", ":::"]
        rst = _myst_to_rst(lines)
        self.assertIn(".. note::", rst)
        self.assertIn("   A note.", rst)

    def test_colon_fence_with_argument(self):
        lines = [
            ":::{admonition} Warning!",
            ":class: danger",
            "",
            "Be careful.",
            ":::",
        ]
        rst = _myst_to_rst(lines)
        self.assertIn(".. admonition:: Warning!", rst)
        self.assertIn("   :class: danger", rst)
        self.assertIn("   Be careful.", rst)

    def test_colon_fence_eval_rst(self):
        lines = [
            ":::{eval-rst}",
            ".. include:: snippet.rst",
            ":::",
        ]
        rst = _myst_to_rst(lines)
        self.assertIn(".. include:: snippet.rst", rst)
        self.assertNotIn(".. eval-rst::", rst)


class TestMystCodeBlocks(TestCase):
    def test_fenced_code_block_with_language(self):
        lines = ["```python", "x = 1", "```"]
        rst = _myst_to_rst(lines)
        self.assertIn(".. code-block:: python", rst)
        self.assertIn("   x = 1", rst)

    def test_fenced_code_block_no_language(self):
        lines = ["```", "x = 1", "```"]
        rst = _myst_to_rst(lines)
        self.assertIn(".. code-block:: python", rst)

    def test_code_block_content_not_transformed(self):
        lines = ["```python", "$not_math$", "{not}`a role`", "```"]
        rst = _myst_to_rst(lines)
        self.assertIn("   $not_math$", rst)
        self.assertIn("   {not}`a role`", rst)


class TestMystInlineConversions(TestCase):
    def test_myst_role(self):
        lines = ["See {func}`torch.add`."]
        rst = _myst_to_rst(lines)
        self.assertIn(":func:`torch.add`", rst[0])

    def test_domain_role(self):
        lines = ["A {py:class}`torch.Tensor`."]
        rst = _myst_to_rst(lines)
        self.assertIn(":py:class:`torch.Tensor`", rst[0])

    def test_markdown_link_anonymous(self):
        lines = ["See [docs](https://pytorch.org)."]
        rst = _myst_to_rst(lines)
        self.assertIn("`docs <https://pytorch.org>`__", rst[0])

    def test_markdown_image(self):
        lines = ["![alt text](images/fig.png)"]
        rst = _myst_to_rst(lines)
        self.assertIn(".. image:: images/fig.png", rst)
        self.assertIn("   :alt: alt text", rst)


class TestMystMath(TestCase):
    def test_inline_dollar_math(self):
        lines = ["The loss is $L = -\\log p$."]
        rst = _myst_to_rst(lines)
        self.assertIn(":math:`L = -\\log p`", rst[0])

    def test_inline_latex_math(self):
        lines = [r"Gradient is \(\nabla f\)."]
        rst = _myst_to_rst(lines)
        self.assertIn(r":math:`\nabla f`", rst[0])

    def test_escaped_dollar_not_converted(self):
        lines = [r"The price is \$5."]
        rst = _myst_to_rst(lines)
        self.assertNotIn(":math:", rst[0])

    def test_display_math_dollar(self):
        lines = ["$$", "E = mc^2", "$$"]
        rst = _myst_to_rst(lines)
        self.assertIn(".. math::", rst)
        self.assertIn("   E = mc^2", rst)

    def test_display_math_latex_brackets(self):
        lines = [r"\[", "E = mc^2", r"\]"]
        rst = _myst_to_rst(lines)
        self.assertIn(".. math::", rst)
        self.assertIn("   E = mc^2", rst)


class TestMystMiscSyntax(TestCase):
    def test_section_label(self):
        lines = ["(my-label)=", "## My Section"]
        rst = _myst_to_rst(lines)
        self.assertIn(".. _my-label:", rst)

    def test_comment(self):
        lines = ["% This is hidden"]
        rst = _myst_to_rst(lines)
        self.assertIn(".. This is hidden", rst)

    def test_heading_levels(self):
        lines = ["# H1", "## H2", "### H3"]
        rst = _myst_to_rst(lines)
        # Check that headings produce text + underline pairs
        h1_idx = rst.index("H1")
        self.assertTrue(all(c == "=" for c in rst[h1_idx + 1]))
        h2_idx = rst.index("H2")
        self.assertTrue(all(c == "-" for c in rst[h2_idx + 1]))
        h3_idx = rst.index("H3")
        self.assertTrue(all(c == "~" for c in rst[h3_idx + 1]))

    def test_yaml_front_matter_stripped(self):
        lines = ["---", "orphan: true", "---", "", "Real content."]
        rst = _myst_to_rst(lines)
        joined = " ".join(rst)
        self.assertNotIn("orphan", joined)
        self.assertIn("Real content.", rst)


class TestMystEndToEnd(TestCase):
    """Test realistic docstrings that mix multiple MyST features."""

    def test_full_myst_docstring(self):
        lines = [
            "Compute the forward pass.",
            "",
            "```{note}",
            "This wraps {func}`torch.add`.",
            "```",
            "",
            "The output satisfies $z = x + y$.",
            "",
            "Args:",
            "    x: First input.",
            "    y: Second input.",
            "",
            "Returns:",
            "    Sum tensor.",
        ]
        self.assertTrue(_is_markdown_docstring(lines))
        rst = _myst_to_rst(lines)
        joined = "\n".join(rst)
        self.assertIn(".. note::", joined)
        self.assertIn(":func:`torch.add`", joined)
        self.assertIn(":math:`z = x + y`", joined)
        # Napoleon sections pass through unchanged
        self.assertIn("Args:", joined)
        self.assertIn("Returns:", joined)

    def test_existing_rst_docstring_unchanged(self):
        lines = [
            "Apply a linear transformation.",
            "",
            ".. math::",
            "",
            "   y = xA^T + b",
            "",
            "Args:",
            "    input: input tensor of shape :math:`(*, H_{in})`",
            "",
            "See :class:`~torch.nn.Linear` for details.",
        ]
        self.assertFalse(_is_markdown_docstring(lines))


if __name__ == "__main__":
    run_tests()
