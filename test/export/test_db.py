# Owner(s): ["oncall: export"]

import copy
import unittest

import torch._dynamo as torchdynamo
from torch._export.db.case import ExportCase, normalize_inputs, SupportLevel
from torch._export.db.examples import (
    filter_examples_by_support_level,
    get_rewrite_cases,
)
from torch.export import export
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_WINDOWS,
    parametrize,
    run_tests,
    TestCase,
)


@unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
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

        inputs_export = normalize_inputs(case.example_inputs)
        inputs_model = copy.deepcopy(inputs_export)
        exported_program = export(
            model,
            inputs_export.args,
            inputs_export.kwargs,
            dynamic_shapes=case.dynamic_shapes,
        )
        exported_program.graph_module.print_readable()

        self.assertEqual(
            exported_program.module()(*inputs_export.args, **inputs_export.kwargs),
            model(*inputs_model.args, **inputs_model.kwargs),
        )

        if case.extra_inputs is not None:
            inputs = normalize_inputs(case.extra_inputs)
            self.assertEqual(
                exported_program.module()(*inputs.args, **inputs.kwargs),
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
        with self.assertRaises(
            (torchdynamo.exc.Unsupported, AssertionError, RuntimeError)
        ):
            inputs = normalize_inputs(case.example_inputs)
            exported_model = export(
                model,
                inputs.args,
                inputs.kwargs,
                dynamic_shapes=case.dynamic_shapes,
            )

    exportdb_not_supported_rewrite_cases = [
        (name, rewrite_case)
        for name, case in filter_examples_by_support_level(
            SupportLevel.NOT_SUPPORTED_YET
        ).items()
        for rewrite_case in get_rewrite_cases(case)
    ]
    if exportdb_not_supported_rewrite_cases:

        @parametrize(
            "name,rewrite_case",
            exportdb_not_supported_rewrite_cases,
            name_fn=lambda name, case: f"case_{name}_{case.name}",
        )
        def test_exportdb_not_supported_rewrite(
            self, name: str, rewrite_case: ExportCase
        ) -> None:
            # pyre-ignore
            inputs = normalize_inputs(rewrite_case.example_inputs)
            exported_model = export(
                rewrite_case.model,
                inputs.args,
                inputs.kwargs,
                dynamic_shapes=rewrite_case.dynamic_shapes,
            )


instantiate_parametrized_tests(ExampleTests)


if __name__ == "__main__":
    run_tests()
