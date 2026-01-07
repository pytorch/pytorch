import torch
import torch._dynamo.test_case
import torch.utils._pytree as pytree


class PytreeRegisteringTestCase(torch._dynamo.test_case.TestCase):
    """TestCase that prunes all temporary pytree registrations and resets Dynamo."""

    def setUp(self) -> None:
        super().setUp()
        self._registered_pytree_nodes: list[type] = []
        self._registered_constant_nodes: list[type] = []

    def tearDown(self) -> None:
        for cls in reversed(self._registered_pytree_nodes):
            pytree._deregister_pytree_node(cls)
        for cls in reversed(self._registered_constant_nodes):
            pytree._deregister_pytree_node(cls)
        torch._dynamo.reset()
        super().tearDown()

    def register_pytree_node(self, cls, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        pytree.register_pytree_node(cls, *args, **kwargs)
        self._registered_pytree_nodes.append(cls)

    def register_constant(self, cls: type) -> None:
        pytree.register_constant(cls)
        self._registered_constant_nodes.append(cls)
