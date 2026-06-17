from __future__ import annotations

import unittest

from tools.build_with_debinfo import (
    debugify,
    entry_command,
    extract_link_command,
    index_compile_commands,
)


class TestDebugify(unittest.TestCase):
    def test_swaps_optimization_for_debug(self) -> None:
        self.assertEqual(debugify("cc -O2 -c a.cpp"), "cc -g -c a.cpp")
        self.assertEqual(debugify("cc -O3 -c a.cpp"), "cc -g -c a.cpp")

    def test_leaves_other_flags_untouched(self) -> None:
        cmd = "cc -DNDEBUG -I/x -fPIC -c a.cpp -o a.o"
        self.assertEqual(debugify(cmd), cmd)

    def test_metal_gets_debug_flags_once(self) -> None:
        out = debugify("xcrun metal -c a.metal")
        self.assertIn("-frecord-sources", out)
        self.assertIn("-gline-tables-only", out)
        # Idempotent: do not append a second time.
        self.assertEqual(out, debugify(out))


class TestEntryCommand(unittest.TestCase):
    def test_command_form(self) -> None:
        self.assertEqual(entry_command({"command": "cc -c a.cpp"}), "cc -c a.cpp")

    def test_arguments_form_is_quoted(self) -> None:
        entry = {"arguments": ["cc", "-c", "a b.cpp"]}
        self.assertEqual(entry_command(entry), "cc -c 'a b.cpp'")


class TestIndexCompileCommands(unittest.TestCase):
    def test_maps_resolved_source_paths(self) -> None:
        entries = [
            {"directory": "/repo/build", "file": "../torch/csrc/Module.cpp"},
            {"directory": "/repo/build", "file": "/repo/torch/csrc/Other.cpp"},
        ]
        index = index_compile_commands(entries)
        self.assertIn("/repo/torch/csrc/Module.cpp", index)
        self.assertIn("/repo/torch/csrc/Other.cpp", index)


class TestExtractLinkCommand(unittest.TestCase):
    def test_strips_ninja_wrapper(self) -> None:
        out = ": && clang++ -shared -o lib/libtorch_python.so a.o b.o && :"
        self.assertEqual(
            extract_link_command(out, "libtorch_python.so"),
            "clang++ -shared -o lib/libtorch_python.so a.o b.o",
        )

    def test_picks_link_among_compiles(self) -> None:
        out = "\n".join(
            [
                "clang++ -c torch_python.dir/Module.cpp.o",
                ": && clang++ -shared -o lib/libtorch_python.so a.o && :",
            ]
        )
        self.assertEqual(
            extract_link_command(out, "libtorch_python.so"),
            "clang++ -shared -o lib/libtorch_python.so a.o",
        )

    def test_raises_when_absent(self) -> None:
        with self.assertRaises(RuntimeError):
            extract_link_command("clang++ -c a.o", "libtorch_python.so")


if __name__ == "__main__":
    unittest.main()
