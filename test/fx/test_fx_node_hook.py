# Owner(s): ["module: fx"]
import torch
from torch.fx import symbolic_trace
from torch.testing._internal.common_utils import TestCase


class TestFXNodeHook(TestCase):
    def test_hooks_for_node_update(self):
        global create_node_hook1_called
        global create_node_hook2_called
        global erase_node_hook1_called
        global erase_node_hook2_called
        global replace_node_hook1_called
        global replace_node_hook2_called
        create_node_hook1_called = False
        create_node_hook2_called = False
        erase_node_hook1_called = False
        erase_node_hook2_called = False
        replace_node_hook1_called = False
        replace_node_hook2_called = False

        def fn(a, b, c):
            x = torch.nn.functional.linear(a, b)
            x = x + c
            return x.cos()

        def create_node_hook1(node):
            global create_node_hook1_called
            create_node_hook1_called = True

        def create_node_hook2(node):
            global create_node_hook2_called
            create_node_hook2_called = True

        def erase_node_hook1(node):
            global erase_node_hook1_called
            erase_node_hook1_called = True

        def erase_node_hook2(node):
            global erase_node_hook2_called
            erase_node_hook2_called = True

        def replace_node_hook1(old, new, user):
            global replace_node_hook1_called
            self.assertEqual(old.name, "a")
            self.assertEqual(new, "a_1")
            self.assertEqual(user.name, "linear")
            replace_node_hook1_called = True

        def replace_node_hook2(old, new, user):
            global replace_node_hook2_called
            replace_node_hook2_called = True

        gm = symbolic_trace(fn)
        gm._register_create_node_hook(create_node_hook1)
        gm._register_create_node_hook(create_node_hook2)
        gm._register_erase_node_hook(erase_node_hook1)
        gm._register_erase_node_hook(erase_node_hook2)
        gm._register_replace_node_hook(replace_node_hook1)
        gm._register_replace_node_hook(replace_node_hook2)

        graph = gm.graph
        node_a = None
        for node in graph.find_nodes(op="placeholder"):
            node_a = node
            break
        if node_a is None:
            raise AssertionError("Expected to find a placeholder node")
        # This will create a new node
        node_a_copy = graph.node_copy(node_a)
        node_a.replace_all_uses_with(node_a_copy)
        graph.erase_node(node_a)

        if not (
            create_node_hook1_called
            and create_node_hook2_called
            and erase_node_hook1_called
            and erase_node_hook2_called
            and replace_node_hook1_called
            and replace_node_hook2_called
        ):
            raise AssertionError("Expected all node hooks to be called")

        gm._unregister_create_node_hook(create_node_hook1)
        gm._unregister_create_node_hook(create_node_hook2)
        gm._unregister_erase_node_hook(erase_node_hook1)
        gm._unregister_erase_node_hook(erase_node_hook2)
        gm._unregister_replace_node_hook(replace_node_hook1)
        gm._unregister_replace_node_hook(replace_node_hook2)

        if gm._create_node_hooks != []:
            raise AssertionError("Expected gm._create_node_hooks to be empty")
        if gm._erase_node_hooks != []:
            raise AssertionError("Expected gm._erase_node_hooks to be empty")
        if gm._replace_hooks != []:
            raise AssertionError("Expected gm._replace_hooks to be empty")

    def test_replace_hook_keyword_only_signature(self):
        class M(torch.nn.Module):
            def forward(self, x):
                a = torch.neg(x)
                return a + 1

        gm = symbolic_trace(M())
        calls: list[tuple[str, str, str | None]] = []

        def hook(*, old, new, user):
            calls.append((old.name, new, getattr(user, "name", None)))

        gm._register_replace_node_hook(hook)
        try:
            target_node = next(n for n in gm.graph.nodes if n.op == "call_function")
            target_node.name = target_node.name + "_patched"
        finally:
            gm._unregister_replace_node_hook(hook)

        self.assertTrue(calls)
        self.assertTrue(calls[0][1].endswith("_patched"))

    def test_replace_hook_runs_for_rename(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x + 1

        gm = symbolic_trace(M())
        placeholder = next(n for n in gm.graph.nodes if n.op == "placeholder")
        users = {user.name for user in placeholder.users}
        original_name = placeholder.name
        rename_calls: list[tuple[str, str, str | None]] = []

        def hook(*, old, new, user):
            rename_calls.append((old.name, new, getattr(user, "name", None)))

        gm._register_replace_node_hook(hook)
        try:
            placeholder._rename("renamed_placeholder")
        finally:
            gm._unregister_replace_node_hook(hook)

        self.assertTrue(
            rename_calls,
            "_rename should notify registered replace hooks",
        )
        self.assertTrue(all(entry[0] == original_name for entry in rename_calls))
        self.assertTrue(all(entry[1] == placeholder.name for entry in rename_calls))
        self.assertEqual({entry[2] for entry in rename_calls}, users)


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
