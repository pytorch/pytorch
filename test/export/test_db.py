# Owner(s): ["oncall: export"]

import copy
import unittest

import torch
import torch._dynamo as torchdynamo
import torch.utils._pytree as pytree
from torch._export import config
from torch._export.db.case import ExportCase, SupportLevel
from torch._export.db.examples import (
    filter_examples_by_support_level,
    get_rewrite_cases,
)
from torch.export import export
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    IS_WINDOWS,
    parametrize,
    run_tests,
    TestCase,
)


def _to_device(obj, device):
    return pytree.tree_map_only(torch.Tensor, lambda x: x.to(device), obj)


@unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class ExampleTests(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    # TODO Maybe we should make this tests actually show up in a file?
    @parametrize(
        "name,case",
        filter_examples_by_support_level(SupportLevel.SUPPORTED).items(),
        name_fn=lambda name, case: f"case_{name}",
    )
    def test_exportdb_supported(self, device, name: str, case: ExportCase) -> None:
        if name in ("optional_input", "static_if"):
            if torch.device(device).type != "cpu":
                self.skipTest(f"{name} constructs CPU tensors in the example")
        model = copy.deepcopy(case.model)
        if isinstance(model, torch.nn.Module):
            model = model.to(device)

        args_export = _to_device(copy.deepcopy(case.example_args), device)
        kwargs_export = _to_device(copy.deepcopy(case.example_kwargs), device)
        args_model = copy.deepcopy(args_export)
        kwargs_model = copy.deepcopy(kwargs_export)
        with config.patch(use_new_tracer_experimental=True):
            exported_program = export(
                model,
                args_export,
                kwargs_export,
                dynamic_shapes=case.dynamic_shapes,
                strict=True,
            )
        exported_program.graph_module.print_readable()

        self.assertEqual(
            exported_program.module()(*args_export, **kwargs_export),
            model(*args_model, **kwargs_model),
        )

        if case.extra_args is not None:
            args = _to_device(copy.deepcopy(case.extra_args), device)
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
    def test_exportdb_not_supported(self, device, name: str, case: ExportCase) -> None:
        model = copy.deepcopy(case.model)
        if isinstance(model, torch.nn.Module):
            model = model.to(device)
        args = _to_device(copy.deepcopy(case.example_args), device)
        kwargs = _to_device(copy.deepcopy(case.example_kwargs), device)
        # pyre-ignore
        with self.assertRaises(
            (torchdynamo.exc.Unsupported, AssertionError, RuntimeError)
        ):
            with config.patch(use_new_tracer_experimental=True):
                _ = export(
                    model,
                    args,
                    kwargs,
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
            self, device, name: str, rewrite_case: ExportCase
        ) -> None:
            model = copy.deepcopy(rewrite_case.model)
            if isinstance(model, torch.nn.Module):
                model = model.to(device)
            args = _to_device(copy.deepcopy(rewrite_case.example_args), device)
            kwargs = _to_device(copy.deepcopy(rewrite_case.example_kwargs), device)
            # pyre-ignore
            export(
                model,
                args,
                kwargs,
                dynamic_shapes=rewrite_case.dynamic_shapes,
                strict=True,
            )


instantiate_device_type_tests(ExampleTests, globals())


if __name__ == "__main__":
    run_tests()
