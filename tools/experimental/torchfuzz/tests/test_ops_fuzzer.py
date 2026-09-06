import unittest

from torchfuzz.ops_fuzzer import _get_template_filtered_operators


class SupportedOpsFilteringTest(unittest.TestCase):
    def _torch_op_names(self, supported_op: str) -> set[str | None]:
        operators = _get_template_filtered_operators(
            template="default", supported_ops=[supported_op]
        )
        return {operator.torch_op_name for operator in operators.values()}

    def test_expand_does_not_match_exp(self) -> None:
        names = self._torch_op_names("torch.expand")

        self.assertIn("torch.expand", names)
        self.assertNotIn("torch.exp", names)

    def test_exp_does_not_match_expand(self) -> None:
        names = self._torch_op_names("torch.exp")

        self.assertIn("torch.exp", names)
        self.assertNotIn("torch.expand", names)

    def test_unsqueeze_does_not_match_squeeze(self) -> None:
        names = self._torch_op_names("torch.unsqueeze")

        self.assertIn("torch.unsqueeze", names)
        self.assertNotIn("torch.squeeze", names)

    def test_squeeze_does_not_match_unsqueeze(self) -> None:
        names = self._torch_op_names("torch.squeeze")

        self.assertIn("torch.squeeze", names)
        self.assertNotIn("torch.unsqueeze", names)

    def test_registry_key_requires_exact_match(self) -> None:
        names = self._torch_op_names("expand")

        self.assertIn("torch.expand", names)
        self.assertNotIn("torch.exp", names)

    def test_partial_registry_key_does_not_match(self) -> None:
        names = self._torch_op_names("expan")

        self.assertNotIn("torch.expand", names)
        self.assertNotIn("torch.exp", names)
