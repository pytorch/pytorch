# Owner(s): ["module: inductor"]

import os
import tempfile
from unittest import mock

import torch
from functorch.compile import draw_graph
from torch._functorch import partitioners
from torch._inductor import config as inductor_config, debug as inductor_debug
from torch._inductor.debug import DebugFormatter
from torch._inductor.test_case import run_tests, TestCase


def _make_graph_module() -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    y = graph.call_function(torch.neg, (x,))
    graph.output(y)
    return torch.fx.GraphModule({}, graph)


class _FakeDebugHandler:
    def __init__(self, root: str) -> None:
        self.root = root

    def filename(self, name: str) -> str:
        return os.path.join(self.root, name)

    def fopen(self, name: str) -> None:
        raise AssertionError(f"unexpected fopen({name})")

    def fopen_context(self) -> None:
        raise AssertionError("unexpected fopen_context()")


class TestDebugGraphDump(TestCase):
    def test_legacy_svg_flags_default_to_svg_with_global_dot_format(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "INDUCTOR_ORIG_FX_SVG": "1",
                "INDUCTOR_POST_FUSION_SVG": "1",
                "TORCH_COMPILE_GRAPH_FORMAT": "dot",
            },
            clear=True,
        ):
            self.assertEqual(
                inductor_config._get_debug_graph_format(
                    "INDUCTOR_ORIG_FX_GRAPH_FORMAT", "INDUCTOR_ORIG_FX_SVG"
                ),
                "svg",
            )
            self.assertEqual(
                inductor_config._get_debug_graph_format(
                    "INDUCTOR_POST_FUSION_GRAPH_FORMAT",
                    "INDUCTOR_POST_FUSION_SVG",
                ),
                "svg",
            )

    def test_new_graph_flags_use_global_graph_format(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "INDUCTOR_ORIG_FX_GRAPH": "1",
                "INDUCTOR_POST_FUSION_GRAPH": "1",
                "TORCH_COMPILE_GRAPH_FORMAT": "dot",
            },
            clear=True,
        ):
            self.assertEqual(
                inductor_config._get_debug_graph_format(
                    "INDUCTOR_ORIG_FX_GRAPH_FORMAT", "INDUCTOR_ORIG_FX_SVG"
                ),
                "dot",
            )
            self.assertEqual(
                inductor_config._get_debug_graph_format(
                    "INDUCTOR_POST_FUSION_GRAPH_FORMAT",
                    "INDUCTOR_POST_FUSION_SVG",
                ),
                "dot",
            )

    def test_explicit_per_artifact_graph_format_wins(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "INDUCTOR_ORIG_FX_SVG": "1",
                "INDUCTOR_ORIG_FX_GRAPH_FORMAT": "raw",
                "INDUCTOR_POST_FUSION_GRAPH": "1",
                "INDUCTOR_POST_FUSION_GRAPH_FORMAT": "svg",
                "TORCH_COMPILE_GRAPH_FORMAT": "dot",
            },
            clear=True,
        ):
            self.assertEqual(
                inductor_config._get_debug_graph_format(
                    "INDUCTOR_ORIG_FX_GRAPH_FORMAT", "INDUCTOR_ORIG_FX_SVG"
                ),
                "raw",
            )
            self.assertEqual(
                inductor_config._get_debug_graph_format(
                    "INDUCTOR_POST_FUSION_GRAPH_FORMAT",
                    "INDUCTOR_POST_FUSION_SVG",
                ),
                "svg",
            )

    def test_draw_graph_dot_uses_raw_writer(self) -> None:
        gm = _make_graph_module()
        fake_dot_graph = mock.Mock()
        fake_dot_graph.write_dot.side_effect = AssertionError(
            "write_dot invokes Graphviz layout"
        )
        fake_dot_graph.write_svg.side_effect = AssertionError(
            "write_svg invokes Graphviz layout"
        )
        fake_drawer = mock.Mock()
        fake_drawer.return_value.get_main_dot_graph.return_value = fake_dot_graph

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "graph.dot")
            with mock.patch.object(
                partitioners.graph_drawer, "FxGraphDrawer", fake_drawer
            ):
                draw_graph(
                    gm,
                    path,
                    clear_meta=False,
                    prog=["dot", "-Gmaxiter=1"],
                )

        fake_dot_graph.write.assert_called_once_with(path)
        fake_dot_graph.write_dot.assert_not_called()
        fake_dot_graph.write_svg.assert_not_called()

    def test_draw_buffers_dot_does_not_require_graphviz(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "graph.dot")
            with (
                mock.patch.object(
                    inductor_debug,
                    "has_dot",
                    side_effect=AssertionError("raw DOT should not require Graphviz"),
                ),
                mock.patch.object(
                    inductor_debug,
                    "create_fx_from_snodes",
                    return_value=_make_graph_module().graph,
                ),
                mock.patch.object(inductor_debug, "draw_graph") as draw_graph_mock,
            ):
                inductor_debug.draw_buffers([], fname=path)

        draw_graph_mock.assert_called_once()
        self.assertEqual(draw_graph_mock.call_args.args[1], path)

    def test_debug_formatter_uses_configured_graph_formats(self) -> None:
        gm = _make_graph_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            formatter = DebugFormatter(_FakeDebugHandler(tmpdir))
            with (
                inductor_config.patch(
                    {
                        "trace.graph_diagram_format": "dot",
                        "trace.orig_fx_graph_diagram_format": "dot",
                    }
                ),
                mock.patch("torch._inductor.debug.draw_buffers") as draw_buffers,
                mock.patch("torch._inductor.debug.annotate_orig_fx_with_snodes"),
                mock.patch("torch._inductor.debug.draw_graph") as inductor_draw_graph,
            ):
                formatter.graph_diagram([])
                formatter.draw_orig_fx_graph(gm, [])

            draw_buffers.assert_called_once()
            self.assertEqual(
                draw_buffers.call_args.kwargs["fname"],
                os.path.join(tmpdir, "graph_diagram.dot"),
            )
            inductor_draw_graph.assert_called_once()
            self.assertEqual(
                inductor_draw_graph.call_args.kwargs["fname"],
                os.path.join(tmpdir, "orig_fx_graph_diagram.dot"),
            )


if __name__ == "__main__":
    run_tests()
