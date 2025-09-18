# Owner(s): ["oncall: export"]

import copy
import unittest

import torch._dynamo as torchdynamo
from torch._export import config
from torch._export.db.case import ExportCase, SupportLevel
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

        args_export = case.example_args
        kwargs_export = case.example_kwargs
        args_model = copy.deepcopy(args_export)
        kwargs_model = copy.deepcopy(kwargs_export)
        with config.patch(use_new_tracer_experimental=True):
            exported_program = export(
                model,
                case.example_args,
                case.example_kwargs,
                dynamic_shapes=case.dynamic_shapes,
                strict=True,
            )
        exported_program.graph_module.print_readable()

        self.assertEqual(
            exported_program.module()(*args_export, **kwargs_export),
            model(*args_model, **kwargs_model),
        )

        if case.extra_args is not None:
            args = case.extra_args
            args_model = copy.deepcopy(args)
            self.assertEqual(
                exported_program.module()(*args),
                model(*args_model),
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
            with config.patch(use_new_tracer_experimental=True):
                _ = export(
                    model,
                    case.example_args,
                    case.example_kwargs,
                    dynamic_shapes=case.dynamic_shapes,
                    strict=True,
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
            export(
                rewrite_case.model,
                rewrite_case.example_args,
                rewrite_case.example_kwargs,
                dynamic_shapes=rewrite_case.dynamic_shapes,
                strict=True,
            )


instantiate_parametrized_tests(ExampleTests)


if __name__ == "__main__":
    run_tests()
