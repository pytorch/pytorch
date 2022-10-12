import unittest
from typing import Dict

from torchgen.dest.register_dispatch_key import RegisterDispatchKey
from torchgen.model import (
    BackendIndex,
    BackendMetadata,
    DispatchKey,
    Location,
    NativeFunction,
    OperatorName,
)
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import Target

# Test dest.register_dispatch_key.py
class TestRegisterDispatchKey(unittest.TestCase):
    def setUp(self) -> None:
        self.none_selector = SelectiveBuilder.from_yaml_dict(
            {"include_all_operators": False}
        )
        self.include_all_selector = SelectiveBuilder.from_yaml_dict(
            {"include_all_operators": True}
        )
        self.op_1_native_function, op_1_backend_index = NativeFunction.from_yaml(
            {"func": "op_1() -> bool", "dispatch": {"CPU": "kernel_1"}},
            loc=Location(__file__, 1),
            valid_tags=set(),
        )
        backend_indices: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]] = {
            DispatchKey.CPU: {},
            DispatchKey.QuantizedCPU: {},
        }
        BackendIndex.grow_index(backend_indices, op_1_backend_index)
        self.backend_indices = {
            k: BackendIndex(
                dispatch_key=k,
                use_out_as_primary=True,
                external=False,
                device_guard=False,
                index=backend_indices[k],
            )
            for k in backend_indices
        }

    def test_register_dispatch_key_not_selected_returns_none(self) -> None:
        register = RegisterDispatchKey(
            backend_index=self.backend_indices[DispatchKey.CPU],
            target=Target.NAMESPACED_DEFINITION,
            selector=self.none_selector,
            rocm=False,
            symint=False,
            class_method_name=None,
            skip_dispatcher_op_registration=False,
        )
        res = register(self.op_1_native_function)
        self.assertEquals(res, [])

    def test_register_dispatch_key_selected_returns_code(self) -> None:
        register = RegisterDispatchKey(
            backend_index=self.backend_indices[DispatchKey.CPU],
            target=Target.NAMESPACED_DEFINITION,
            selector=self.include_all_selector,
            rocm=False,
            symint=False,
            class_method_name=None,
            skip_dispatcher_op_registration=False,
        )
        res = register(self.op_1_native_function)
        self.assertTrue(len(res) > 0)
