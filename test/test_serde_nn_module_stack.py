import json

import pytest

from torch._export.serde.serialize import (
    GraphModuleSerializer,
    GraphModuleDeserializer,
    ST_DELIMITER,
)


class DummyNode:
    def __init__(self, meta):
        self.meta = meta


@pytest.mark.parametrize(
    "entries",
    [
        {
            "L__self__": ("", "torch.nn.modules.container.Sequential"),
            "fn": ("L['fn']", "torch.nn.modules.container.Sequential"),
            "fn_0": ("getattr(L['fn'], '0')", "torch.nn.modules.container.Sequential"),
            "fn_0_0": (
                "getattr(getattr(L['fn'], '0'), '0')",
                "on_device_ai.application_profile_runs.sweeper_run.Conv2dBNReLU",
            ),
            "getattr_getattr_L__fn_____0_____0___core": (
                "getattr(getattr(L['fn'], '0'), '0').core",
                "torch.nn.modules.container.Sequential",
            ),
            "getattr_getattr_L__fn_____0_____0___core_0": (
                "getattr(getattr(getattr(L['fn'], '0'), '0').core, '0')",
                "torch.nn.modules.conv.Conv2d",
            ),
        }
    ],
)
def test_nn_module_stack_json_roundtrip(entries):
    """Serialize node metadata and then deserialize it back. The nn_module_stack
    entries contain commas/parentheses to exercise the previous bug."""

    node = DummyNode({"nn_module_stack": entries})
    serializer = GraphModuleSerializer(None, [])
    serialized_meta = serializer.serialize_metadata(node)

    assert "nn_module_stack" in serialized_meta
    # The serialized form is a string joined by ST_DELIMITER
    assert isinstance(serialized_meta["nn_module_stack"], str)

    deserializer = GraphModuleDeserializer()
    parsed_meta = deserializer.deserialize_metadata({"nn_module_stack": serialized_meta["nn_module_stack"]})

    assert "nn_module_stack" in parsed_meta
    assert parsed_meta["nn_module_stack"] == entries


def test_nn_module_stack_legacy_parser():
    """Ensure the deserializer still understands the legacy comma-split format."""
    entries = {
        "a": ("getattr(mod, '0')", "torch.nn.Module"),
        "b": ("L['fn']", "torch.nn.Sequential"),
    }
    # Build legacy-style string: key,path,type joined by commas, entries separated by ST_DELIMITER
    legacy_items = [f"{k},{v[0]},{v[1]}" for k, v in entries.items()]
    legacy_str = ST_DELIMITER.join(legacy_items)

    deserializer = GraphModuleDeserializer()
    parsed_meta = deserializer.deserialize_metadata({"nn_module_stack": legacy_str})

    assert "nn_module_stack" in parsed_meta
    assert parsed_meta["nn_module_stack"] == entries
