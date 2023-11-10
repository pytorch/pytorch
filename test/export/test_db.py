# Owner(s): ["module: dynamo"]

import unittest

import torch._dynamo as torchdynamo
from torch.export import export
from torch._export.db.case import ExportCase, normalize_inputs, SupportLevel
from torch._export.db.examples import (
    filter_examples_by_support_level,
    get_rewrite_cases,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class ExampleTests(TestCase):
    # TODO Maybe we should make this tests actually show up in a file?
    @parametrize(
        "name,case",
        filter_examples_by_support_level(SupportLevel.SUPPORTED).items(),
        name_fn=lambda name, case: f"case_{name}",
    )
    def test_exportdb_supported(self, name: str, case: ExportCase) -> None:
        model = case.model

        inputs = normalize_inputs(case.example_inputs)
        exported_program = export(
            model,
            inputs.args,
            inputs.kwargs,
            dynamic_shapes=case.dynamic_shapes,
        )
        exported_program.graph_module.print_readable()

        self.assertEqual(
            exported_program(*inputs.args, **inputs.kwargs),
            model(*inputs.args, **inputs.kwargs),
        )

        if case.extra_inputs is not None:
            inputs = normalize_inputs(case.extra_inputs)
            self.assertEqual(
                exported_program(*inputs.args, **inputs.kwargs),
                model(*inputs.args, **inputs.kwargs),
            )

    @parametrize(
        "name,case",
        filter_examples_by_support_level(SupportLevel.NOT_SUPPORTED_YET).items(),
        name_fn=lambda name, case: f"case_{name}",
    )
    def test_exportdb_not_supported(self, name: str, case: ExportCase) -> None:
        model = case.model
        # pyre-ignore
        with self.assertRaises((torchdynamo.exc.Unsupported, AssertionError, RuntimeError)):
            inputs = normalize_inputs(case.example_inputs)
            exported_model = export(
                model,
                inputs.args,
                inputs.kwargs,
                dynamic_shapes=case.dynamic_shapes,
            )


instantiate_parametrized_tests(ExampleTests)


if __name__ == "__main__":
    run_tests()
