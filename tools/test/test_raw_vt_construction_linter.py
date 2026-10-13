from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tools.linter.adapters.raw_vt_construction_linter import (
    _discover_variable_tracker_classes,
    check_file,
    main,
)


class TestRawVTConstructionLinter(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.variables_dir = Path(self.temp_dir.name) / "torch/_dynamo/variables"
        self.write(
            "class VariableTracker:\n"
            "    pass\n"
            "class ValueVariable(VariableTracker):\n"
            "    pass\n"
            "class ContainerVariable(VariableTracker):  # noqa: RAW_VT_CONSTRUCTION\n"
            "    pass\n"
            "class ChildVariable(ContainerVariable):\n"
            "    pass\n"
            "# noqa: RAW_VT_CONSTRUCTION\n"
            "class PlainState(VariableTracker):\n"
            "    pass\n",
            "torch/_dynamo/variables/base.py",
        )
        self.write(
            "from .base import ValueVariable as Parent\n"
            "if enabled:\n"
            "    class StateIterator(Parent):\n"
            "        pass\n"
            "class StateContextManager(StateIterator):\n"
            "    pass\n"
            "class FactoryVariable(Parent):\n"
            "    @classmethod\n"
            "    def create(cls, value):\n"
            "        return FactoryVariable(value)\n",
            "torch/_dynamo/variables/more.py",
        )
        self.write(
            "from .base import ChildVariable, ContainerVariable, PlainState, ValueVariable\n"
            "from .more import FactoryVariable, StateContextManager, StateIterator\n",
            "torch/_dynamo/variables/__init__.py",
        )
        self.inventory = _discover_variable_tracker_classes(self.variables_dir)

    def write(
        self, source: str, relative_path: str = "torch/_dynamo/callsite.py"
    ) -> Path:
        path = Path(self.temp_dir.name) / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        return path

    def check(self, source: str, relative_path: str = "torch/_dynamo/callsite.py"):
        return check_file(str(self.write(source, relative_path)), self.inventory)

    def test_inventory_follows_inheritance_without_name_categories(self) -> None:
        self.assertEqual(
            self.inventory.classes,
            frozenset(
                {
                    "torch._dynamo.variables.base.VariableTracker",
                    "torch._dynamo.variables.base.ValueVariable",
                    "torch._dynamo.variables.base.ContainerVariable",
                    "torch._dynamo.variables.base.ChildVariable",
                    "torch._dynamo.variables.base.PlainState",
                    "torch._dynamo.variables.more.StateIterator",
                    "torch._dynamo.variables.more.StateContextManager",
                    "torch._dynamo.variables.more.FactoryVariable",
                }
            ),
        )

    def test_named_constructors(self) -> None:
        for imports, expression in (
            ("from torch._dynamo.variables import ValueVariable", "ValueVariable(1)"),
            ("from torch._dynamo.variables.base import ValueVariable as V", "V(1)"),
            ("from torch._dynamo import variables as vt", "vt.ValueVariable(1)"),
            ("import torch._dynamo.variables as vt", "vt.ValueVariable(1)"),
            ("import torch._dynamo.variables.base as vt", "vt.ValueVariable(1)"),
            ("import torch", "torch._dynamo.variables.ValueVariable(1)"),
            ("from .variables import ValueVariable as V", "V(1)"),
            ("from torch._dynamo.variables import StateIterator", "StateIterator(1)"),
            (
                "from torch._dynamo.variables import StateContextManager",
                "StateContextManager(1)",
            ),
        ):
            with self.subTest(imports=imports, expression=expression):
                messages = self.check(f"{imports}\n{expression}\n")
                self.assertEqual(len(messages), 1)
                self.assertEqual(messages[0].line, 2)
                self.assertEqual(messages[0].char, 1)
                self.assertEqual(messages[0].code, "RAW_VT_CONSTRUCTION")

    def test_class_noqa_is_exact(self) -> None:
        messages = self.check(
            "from torch._dynamo.variables import ChildVariable, ContainerVariable, PlainState\n"
            "ContainerVariable([])\n"
            "PlainState()\n"
            "ChildVariable([])\n"
        )
        self.assertEqual([message.line for message in messages], [4])

    def test_local_imports_do_not_leak_to_sibling_functions(self) -> None:
        messages = self.check(
            "def plain_fp():\n"
            "    from builtins import int as ValueFP\n"
            "    return ValueFP(1)\n"
            "def tracked_fp():\n"
            "    from torch._dynamo.variables import ValueVariable as ValueFP\n"
            "    return ValueFP(2)\n"
            "def tracked_fn():\n"
            "    from torch._dynamo.variables import ValueVariable as ValueFN\n"
            "    return ValueFN(3)\n"
            "async def plain_fn():\n"
            "    from builtins import int as ValueFN\n"
            "    return ValueFN(4)\n"
        )
        self.assertEqual([message.line for message in messages], [6, 9])

    def test_nested_import_and_definition_scopes(self) -> None:
        for source, expected in (
            (
                "from builtins import int as V\n"
                "def outer():\n"
                "    from torch._dynamo.variables import ValueVariable as V\n"
                "    def nested(V=V(1)):\n"
                "        return V(2)\n"
                "    def closure():\n"
                "        return V(3)\n"
                "    def shadow():\n"
                "        from builtins import int as V\n"
                "        return V(4)\n"
                "    callback = lambda V: V(5)\n"
                "    items = [V(6) for V in things]\n"
                "    outer_items = [V(7) for V in V(8)]\n"
                "    return V(9)\n"
                "V(10)\n",
                [4, 7, 13, 14],
            ),
            (
                "from torch._dynamo.variables import ValueVariable as V\n"
                "class Holder:\n"
                "    from builtins import int as V\n"
                "    plain = V(1)\n"
                "    @decorate(V(2))\n"
                "    def method(self, value=V(3)):\n"
                "        return V(4)\n"
                "    class Child(V(5)):\n"
                "        own = V(6)\n"
                "    callback = lambda: V(7)\n"
                "    items = [V(8) for item in V(9)]\n",
                [7, 9, 10, 11],
            ),
            (
                "from torch._dynamo.variables import ValueVariable as V\n"
                "@decorate(V(0))\n"
                "def factory(arg=V(1)) -> V(2):\n"
                "    from builtins import int as V\n"
                "    return V(3)\n"
                "class Holder(V(4)):\n"
                "    from builtins import int as V\n"
                "    own = V(5)\n",
                [2, 3, 3, 6],
            ),
        ):
            with self.subTest(source=source):
                self.assertEqual(
                    [message.line for message in self.check(source)], expected
                )

    def test_inventory_does_not_export_local_imports(self) -> None:
        self.write(
            "if enabled:\n"
            "    from .base import ValueVariable as Parent\n"
            "def unrelated():\n"
            "    from builtins import int as Parent\n"
            "class Exported(Parent):\n"
            "    pass\n",
            "torch/_dynamo/variables/scopes.py",
        )
        inventory = _discover_variable_tracker_classes(self.variables_dir)
        self.assertEqual(
            inventory.aliases["torch._dynamo.variables.scopes.Parent"],
            "torch._dynamo.variables.base.ValueVariable",
        )
        self.assertIn("torch._dynamo.variables.scopes.Exported", inventory.classes)

    def test_callsite_noqa(self) -> None:
        for call in (
            "ValueVariable(1)  # noqa: RAW_VT_CONSTRUCTION\n",
            "wrapper(ValueVariable(1))  # noqa: RAW_VT_CONSTRUCTION\n",
            "ValueVariable(ValueVariable(1))  # noqa: RAW_VT_CONSTRUCTION\n",
            "# noqa: RAW_VT_CONSTRUCTION\nValueVariable(1)\n",
            "ValueVariable(  # noqa: RAW_VT_CONSTRUCTION\n    1,\n)\n",
            "ValueVariable(  # noqa: RAW_VT_CONSTRUCTION\n    1,\n).call_function()\n",
            "ValueVariable(\n    1,\n)  # noqa: RAW_VT_CONSTRUCTION\n",
            "ValueVariable(1)  # noqa: E123, RAW_VT_CONSTRUCTION\n",
        ):
            with self.subTest(call=call):
                self.assertEqual(
                    self.check(
                        "from torch._dynamo.variables import ValueVariable\n" + call
                    ),
                    [],
                )

    def test_noqa_does_not_cover_other_lines(self) -> None:
        for call in (
            "ValueVariable(  # noqa: RAW_VT_CONSTRUCTION\n    ValueVariable(1),\n)\n",
            "ValueVariable(\n    ValueVariable(1),\n)  # noqa: RAW_VT_CONSTRUCTION\n",
            "wrapper(\n    ValueVariable(1),\n)  # noqa: RAW_VT_CONSTRUCTION\n",
        ):
            with self.subTest(call=call):
                messages = self.check(
                    "from torch._dynamo.variables import ValueVariable\n" + call
                )
                self.assertEqual(len(messages), 1)

    def test_only_explicit_comment_directives_suppress(self) -> None:
        for call in (
            'ValueVariable("# noqa: RAW_VT_CONSTRUCTION")\n',
            'text = "# noqa: RAW_VT_CONSTRUCTION"\nValueVariable(1)\n',
        ):
            with self.subTest(call=call):
                self.assertEqual(
                    len(
                        self.check(
                            "from torch._dynamo.variables import ValueVariable\n" + call
                        )
                    ),
                    1,
                )

    def test_function_noqa_exempts_only_its_body(self) -> None:
        for definition, indent in (
            ("def factory(value):  # noqa: RAW_VT_CONSTRUCTION\n", "    "),
            ("# noqa: RAW_VT_CONSTRUCTION\ndef factory(value):\n", "    "),
            ("async def factory(value):  # noqa: RAW_VT_CONSTRUCTION\n", "    "),
            ("def factory(  # noqa: RAW_VT_CONSTRUCTION\n    value,\n):\n", "    "),
            (
                "class Factory:\n"
                "    def wrap(self, value):  # noqa: RAW_VT_CONSTRUCTION\n",
                "        ",
            ),
        ):
            with self.subTest(definition=definition):
                source = (
                    "from torch._dynamo.variables import StateIterator, ValueVariable\n"
                    + definition
                    + f"{indent}direct = ValueVariable(value)\n"
                    + f"{indent}if value:\n"
                    + f"{indent}    return StateIterator(direct)\n"
                    + f"{indent}return direct\n"
                    + "ValueVariable(1)\n"
                )
                self.assertEqual(
                    [message.line for message in self.check(source)],
                    [len(source.splitlines())],
                )

    def test_function_noqa_excludes_signature_and_nested_scopes(self) -> None:
        messages = self.check(
            "from torch._dynamo.variables import ValueVariable\n"
            "@decorate(ValueVariable(0))\n"
            "def factory(value=ValueVariable(1)) -> ValueVariable(2):  # noqa: RAW_VT_CONSTRUCTION\n"
            "    direct = ValueVariable(value)\n"
            "    def nested(value=ValueVariable(3)):\n"
            "        return ValueVariable(value)\n"
            "    callback = lambda: ValueVariable(value)\n"
            "    class Nested:\n"
            "        own = ValueVariable(value)\n"
            "    def inner_factory(value):  # noqa: RAW_VT_CONSTRUCTION\n"
            "        return ValueVariable(value)\n"
            "    return direct\n"
        )
        self.assertEqual([message.line for message in messages], [2, 3, 3, 5, 6, 7, 9])

    def test_own_create_body_is_exempt(self) -> None:
        messages = check_file(str(self.variables_dir / "more.py"), self.inventory)
        self.assertEqual(messages, [])

    def test_create_only_exempts_its_exact_class(self) -> None:
        for decorator in ("staticmethod", "classmethod"):
            with self.subTest(decorator=decorator):
                messages = self.check(
                    "from .base import ValueVariable\n"
                    "from .more import StateIterator as Alias, StateContextManager\n"
                    "class StateIterator(ValueVariable):\n"
                    f"    @{decorator}\n"
                    "    def create(value):\n"
                    "        own = StateIterator(value)\n"
                    "        alias = Alias(value)\n"
                    "        base = ValueVariable(value)\n"
                    "        child = StateContextManager(value)\n"
                    "        return own\n",
                    "torch/_dynamo/variables/more.py",
                )
                self.assertEqual([message.line for message in messages], [8, 9])

    def test_nonfactory_methods_are_not_exempt(self) -> None:
        for decorator, method in (
            ("", "create"),
            ("staticmethod", "handle"),
            ("classmethod", "create_with_source"),
        ):
            with self.subTest(decorator=decorator, method=method):
                messages = self.check(
                    "from .base import ValueVariable\n"
                    "class FactoryVariable(ValueVariable):\n"
                    + (f"    @{decorator}\n" if decorator else "")
                    + f"    def {method}(value):\n"
                    "        return FactoryVariable(value)\n",
                    "torch/_dynamo/variables/more.py",
                )
                self.assertEqual(len(messages), 1)

    def test_create_exemption_does_not_cover_nested_scopes_or_signature(self) -> None:
        messages = self.check(
            "from .base import ValueVariable\n"
            "class FactoryVariable(ValueVariable):\n"
            "    @staticmethod\n"
            "    @decorate(FactoryVariable(1))\n"
            "    def create(value=FactoryVariable(2)) -> FactoryVariable(3):\n"
            "        direct = FactoryVariable(value)\n"
            "        def nested(value=FactoryVariable(4)):\n"
            "            return FactoryVariable(value)\n"
            "        callback = lambda: FactoryVariable(value)\n"
            "        class Nested:\n"
            "            own = FactoryVariable(value)\n"
            "            @staticmethod\n"
            "            def create():\n"
            "                return FactoryVariable(value)\n"
            "        return direct\n",
            "torch/_dynamo/variables/more.py",
        )
        self.assertEqual(
            [message.line for message in messages], [4, 5, 5, 7, 8, 9, 11, 14]
        )

    def test_methods_and_dynamic_callees_are_outside_rule(self) -> None:
        self.assertEqual(
            self.check(
                "from torch._dynamo.variables import ValueVariable\n"
                "ValueVariable.create(1)\n"
                "ValueVariable.create_with_source(1)\n"
                "ValueVariable.build(tx, 1)\n"
                "cls(1)\n"
                "getattr(vt, 'ValueVariable')(1)\n"
                "alias = ValueVariable\n"
                "alias(1)\n"
            ),
            [],
        )

    def test_unrelated_classes_are_not_vts(self) -> None:
        for source in (
            "from external import ValueVariable\nValueVariable(1)\n",
            "import external as vt\nvt.ValueVariable(1)\n",
            "class ValueVariable:\n    pass\nValueVariable(1)\n",
        ):
            with self.subTest(source=source):
                self.assertEqual(self.check(source), [])

    def test_only_main_builder_is_exempt(self) -> None:
        source = "from torch._dynamo.variables import ValueVariable\nValueVariable(1)\n"
        self.assertEqual(self.check(source, "torch/_dynamo/variables/builder.py"), [])
        self.assertEqual(len(self.check(source, "torch/_dynamo/builder.py")), 1)

    def test_syntax_error_is_a_lint_message(self) -> None:
        messages = self.check("if:\n")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].name, "syntax-error")
        self.assertEqual(messages[0].line, 1)
        self.assertEqual(messages[0].severity, "error")

    def test_unrelated_files_are_not_parsed(self) -> None:
        self.assertEqual(self.check("if:\n", "torch/unrelated.py"), [])

    def test_cli_reads_only_supplied_paths(self) -> None:
        checked = self.write(
            "from torch._dynamo.variables import ConstantVariable\nConstantVariable(1)\n"
        )
        self.write(
            "from torch._dynamo.variables import ConstantVariable\nConstantVariable(2)\n",
            "other.py",
        )
        filenames = self.write(f"{checked}\n", "filenames.txt")
        output = io.StringIO()
        with redirect_stdout(output):
            main([f"@{filenames}"])
        messages = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["path"], str(checked))


if __name__ == "__main__":
    unittest.main()
