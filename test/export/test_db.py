# Owner(s): ["module: dynamo"]

import unittest
from dataclasses import asdict, dataclass

import torch._dynamo as torchdynamo
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


@dataclass
class _DynamoConfig:
    capture_scalar_outputs: bool = True
    capture_dynamic_output_shape_ops: bool = True
    guard_nn_modules: bool = True
    dynamic_shapes: bool = True
    specialize_int: bool = True
    allow_rnn: bool = True
    verbose: bool = True
    assume_static_by_default: bool = False


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class ExampleTests(TestCase):
    # TODO Maybe we should make this tests actually show up in a file?
    @parametrize(
        "name,case",
        filter_examples_by_support_level(SupportLevel.SUPPORTED).items(),
        name_fn=lambda name, case: "case_{}".format(name),
    )
    def test_exportdb_supported(self, name: str, case: ExportCase) -> None:
        model = case.model
        with torchdynamo.config.patch(asdict(_DynamoConfig())):
            inputs = normalize_inputs(case.example_inputs)
            exported_model, _ = torchdynamo.export(
                model,
                *inputs.args,
                aten_graph=True,
                tracing_mode="symbolic",
                **inputs.kwargs
            )
            exported_model.print_readable()

        self.assertEqual(
            exported_model(*inputs.args, **inputs.kwargs),
            model(*inputs.args, **inputs.kwargs),
        )

        if case.extra_inputs is not None:
            inputs = normalize_inputs(case.extra_inputs)
            self.assertEqual(
                exported_model(*inputs.args, **inputs.kwargs),
                model(*inputs.args, **inputs.kwargs),
            )

    @parametrize(
        "name,case",
        filter_examples_by_support_level(SupportLevel.NOT_SUPPORTED_YET).items(),
        name_fn=lambda name, case: "case_{}".format(name),
    )
    def test_exportdb_not_supported(self, name: str, case: ExportCase) -> None:
        self.skipTest("TODO(zhxchen17) Enable non supported tests.")
        model = case.model
        # pyre-ignore
        with torchdynamo.config.patch(asdict(_DynamoConfig())), self.assertRaises(
            torchdynamo.exc.Unsupported
        ):
            inputs = normalize_inputs(case.example_inputs)
            exported_model, _ = torchdynamo.export(
                model,
                *inputs.args,
                aten_graph=True,
                tracing_mode="symbolic",
                **inputs.kwargs
            )

    @parametrize(
        "name,rewrite_case",
        [
            (name, rewrite_case)
            for name, case in filter_examples_by_support_level(
                SupportLevel.NOT_SUPPORTED_YET
            ).items()
            for rewrite_case in get_rewrite_cases(case)
        ],
        name_fn=lambda name, case: "case_{}_{}".format(name, case.name),
    )
    def test_exportdb_not_supported_rewrite(
        self, name: str, rewrite_case: ExportCase
    ) -> None:
        # pyre-ignore
        with torchdynamo.config.patch(asdict(_DynamoConfig())):
            inputs = normalize_inputs(rewrite_case.example_inputs)
            exported_model, _ = torchdynamo.export(
                rewrite_case.model,
                *inputs.args,
                aten_graph=True,
                tracing_mode="symbolic",
                **inputs.kwargs
            )


instantiate_parametrized_tests(ExampleTests)

if __name__ == "__main__":
    run_tests()
