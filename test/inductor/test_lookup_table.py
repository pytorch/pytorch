# Owner(s): ["module: inductor"]
import json
import os
import tempfile
import unittest
from typing import Optional, Union
from unittest.mock import patch

import torch
from torch._inductor import config as inductor_config
from torch._inductor.choices import InductorChoices
from torch._inductor.kernel_inputs import MMKernelInputs
from torch._inductor.lookup_table import recorder
from torch._inductor.lookup_table.choices import LookupTableChoices
from torch._inductor.lookup_table.core import lookup_key_suffix, lookup_template_configs
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON


class MockTensorNode:
    """Mock input node that wraps a real tensor for testing"""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def get_device(self) -> torch.device:
        return self.tensor.device

    def get_dtype(self) -> torch.dtype:
        return self.tensor.dtype

    def get_size(self) -> tuple[int, ...]:
        return tuple(self.tensor.shape)

    def get_stride(self) -> tuple[int, ...]:
        return tuple(self.tensor.stride())


class MockMMKernelInputs(MMKernelInputs):
    """Mock MMKernelInputs that subclasses the real class and uses real tensors"""

    def __init__(
        self,
        tensors: list[torch.Tensor],
        scalars: Optional[dict[str, Union[float, int]]] = None,
        mat1_idx: int = -2,
        mat2_idx: int = -1,
    ):
        """Initialize with real tensors, creating mock nodes for the base class"""
        mock_nodes = [MockTensorNode(t) for t in tensors]
        super().__init__(mock_nodes, scalars, mat1_idx=mat1_idx, mat2_idx=mat2_idx)
        self.tensors = tensors  # Keep reference to original tensors

    def shapes_hinted(self) -> tuple[tuple[int, ...], ...]:
        """Delegate to symbolic since real tensors already have int shapes"""
        return self.shapes_symbolic()

    def strides_hinted(self) -> tuple[tuple[int, ...], ...]:
        """Delegate to symbolic since real tensors already have int strides"""
        return self.strides_symbolic()

    def mnk_hinted(self) -> tuple[int, int, int]:
        """Delegate to symbolic since real tensors already have int dimensions"""
        return self.mnk_symbolic()

    @property
    def device_type(self) -> Optional[str]:
        return self.tensors[0].device.type


class BaseLookupTableTest(TestCase):
    """Base class for lookup table tests with common setup and utilities"""

    def setUp(self):
        super().setUp()
        self.original_table = torch._inductor.config.template_config_lookup_table.table
        self.original_max_autotune = getattr(inductor_config, "max_autotune", False)
        inductor_config.max_autotune = True
        # Set the lookup table choices handler
        V.set_choices_handler(LookupTableChoices())

    def tearDown(self):
        torch._inductor.config.template_config_lookup_table.table = self.original_table
        inductor_config.max_autotune = self.original_max_autotune
        # Restore original choices handler
        V.set_choices_handler(InductorChoices())
        super().tearDown()

    def create_mock_mm_kernel_inputs(
        self,
        shapes: Optional[list[tuple[int, ...]]] = None,
        device: torch.device = torch.device("cuda:0"),
        dtype: torch.dtype = torch.float32,
        scalars: Optional[dict[str, Union[float, int]]] = None,
    ) -> MockMMKernelInputs:
        """Create MockMMKernelInputs with real tensors"""
        if shapes is None:
            shapes = [(128, 128), (128, 128)]  # Default MM shapes

        tensors = []
        for shape in shapes:
            # Create a real tensor with the specified shape, device, and dtype
            tensor = torch.randn(shape, device=device, dtype=dtype)
            tensors.append(tensor)

        return MockMMKernelInputs(tensors, scalars)

    def create_lookup_key(self, method, kernel_inputs):
        """Create a lookup key that matches core.py's make_lookup_key"""
        # This matches exactly what make_lookup_key does in core.py
        flat_key = f"{kernel_inputs.key}+{method}+{lookup_key_suffix()}"
        return flat_key

    def create_config(self, template_id, **kwargs):
        """Create a backend configuration with template_id field"""
        config = {"template_id": template_id}

        # Add minimal defaults based on template type
        if template_id == "triton":
            config.update(
                {
                    "BLOCK_M": 128,
                    "BLOCK_N": 128,
                    "BLOCK_K": 64,
                    "num_stages": 2,
                    "num_warps": 2,
                    "EVEN_K": True,
                    "ALLOW_TF32": False,
                    "USE_FAST_ACCUM": False,
                    "ACC_TYPE": "tl.float32",
                    "GROUP_M": 8,
                }
            )
        elif template_id == "tma":
            config.update(
                {
                    "BLOCK_M": 256,
                    "BLOCK_N": 128,
                    "BLOCK_K": 64,
                    "num_stages": 4,
                    "num_warps": 8,
                    "EVEN_K": True,
                    "ALLOW_TF32": False,
                    "USE_FAST_ACCUM": False,
                    "ACC_TYPE": "tl.float32",
                    "GROUP_M": 8,
                }
            )
        elif template_id == "decompose_k":
            config.update({"k": 4})

        config.update(kwargs)
        return config


@unittest.skipIf(not HAS_CUDA_AND_TRITON, "CUDA not available")
@instantiate_parametrized_tests
class TestLookupTable(BaseLookupTableTest):
    """Consolidated tests for lookup table functionality"""

    def test_lookup_mismatch(self):
        """Test mismatch scenario in lookup table"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        # Mock a different device to create mismatch
        lookup_table_data = {
            self.create_lookup_key("mm", kernel_inputs): [self.create_config("triton")]
        }

        with patch.object(
            inductor_config.template_config_lookup_table, "table", lookup_table_data
        ):
            # looking for addmm but created the entry with mm - should mismatch the key and return
            # an empty result
            result = lookup_template_configs(kernel_inputs, "addmm", ["triton"])
            self.assertEqual(result, {})

    def test_successful_lookup_with_template_filtering(self):
        """Test successful lookup that filters configs by template_id"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        config_list = [
            self.create_config("triton", BLOCK_M=128, BLOCK_N=128),
            self.create_config("triton", BLOCK_M=64, BLOCK_N=64),
            self.create_config("tma", BLOCK_M=256, BLOCK_N=128),
            self.create_config("decompose_k", k_split=4),
        ]

        lookup_table_data = {self.create_lookup_key("mm", kernel_inputs): config_list}

        with patch.object(
            inductor_config.template_config_lookup_table, "table", lookup_table_data
        ):
            # Test triton template filtering
            result = lookup_template_configs(kernel_inputs, "mm", ["triton"])
            assert result is not None, "Result should not be None"
            self.assertEqual(len(result["triton"]), 2)
            for config in result["triton"]:
                self.assertNotIn("template_id", config)
                self.assertIn("BLOCK_M", config)

            # Test tma template filtering
            result = lookup_template_configs(kernel_inputs, "mm", ["tma"])
            assert result is not None, "Result should not be None"
            self.assertEqual(len(result["tma"]), 1)
            self.assertNotIn("template_id", result["tma"][0])
            self.assertEqual(result["tma"][0]["BLOCK_M"], 256)

            # Test decompose_k template filtering
            result = lookup_template_configs(kernel_inputs, "mm", ["decompose_k"])
            assert result is not None, "Result should not be None"
            self.assertEqual(len(result["decompose_k"]), 1)
            self.assertNotIn("template_id", result["decompose_k"][0])
            self.assertEqual(result["decompose_k"][0]["k_split"], 4)

    def test_empty_table(self):
        """Test when template lookup table is empty"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()
        with patch.object(inductor_config.template_config_lookup_table, "table", {}):
            result = lookup_template_configs(kernel_inputs, "mm", ["triton"])
            self.assertEqual(result, {})

    def test_validation_error(self):
        """Test validation error for invalid config"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()
        invalid_config = {"BLOCK_M": 128}  # missing template_id
        lookup_table_data = {
            self.create_lookup_key("mm", kernel_inputs): [invalid_config]
        }

        with patch.object(
            inductor_config.template_config_lookup_table, "table", lookup_table_data
        ):
            with self.assertRaises(ValueError) as cm:
                lookup_template_configs(kernel_inputs, "mm", ["triton"])
            self.assertIn("missing required 'template_id' field", str(cm.exception))

    def test_cpu_input_returns_empty(self):
        """Test that CPU tensor input returns empty dict"""
        # Create kernel inputs with CPU tensors
        kernel_inputs = self.create_mock_mm_kernel_inputs(device=torch.device("cpu"))

        lookup_table_data = {
            self.create_lookup_key("mm", kernel_inputs): [self.create_config("triton")]
        }

        with patch.object(
            inductor_config.template_config_lookup_table, "table", lookup_table_data
        ):
            result = lookup_template_configs(kernel_inputs, "mm", ["triton"])
            self.assertEqual(result, {})  # Should return empty dict for CPU

    @parametrize(
        "allow_tf32,tf32_configs,expected_count",
        [
            (True, [True, False], 2),  # No filtering when allowed
            (False, [True, False], 1),  # Filter TF32=True when not allowed
            (False, [True, True], 0),  # Filter all when all TF32=True
        ],
    )
    def test_tf32_filtering(self, allow_tf32, tf32_configs, expected_count):
        """Test TF32 filtering scenarios"""
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        configs = [
            self.create_config("triton", BLOCK_M=128, ALLOW_TF32=tf32_val)
            for tf32_val in tf32_configs
        ]

        lookup_table_data = {self.create_lookup_key("mm", kernel_inputs): configs}

        with patch.object(
            inductor_config.template_config_lookup_table, "table", lookup_table_data
        ):
            result = lookup_template_configs(kernel_inputs, "mm", ["triton"])
            if expected_count > 0:
                assert result is not None, "Result should not be None"
                self.assertEqual(len(result["triton"]), expected_count)
            else:
                self.assertEqual(result, {})

    def test_multiple_calls_work(self):
        """Test that calling lookup functions multiple times works correctly"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        config_list = [
            self.create_config("triton", BLOCK_M=128),
            self.create_config("tma", BLOCK_M=256),
        ]
        lookup_table_data = {self.create_lookup_key("mm", kernel_inputs): config_list}

        with patch.object(
            inductor_config.template_config_lookup_table, "table", lookup_table_data
        ):
            # First calls
            result1 = lookup_template_configs(kernel_inputs, "mm", ["triton"])
            result2 = lookup_template_configs(kernel_inputs, "mm", ["tma"])
            assert result1 is not None, "Result1 should not be None"
            assert result2 is not None, "Result2 should not be None"
            self.assertEqual(len(result1["triton"]), 1)
            self.assertEqual(len(result2["tma"]), 1)

            # Second calls should work the same
            result3 = lookup_template_configs(kernel_inputs, "mm", ["triton"])
            result4 = lookup_template_configs(kernel_inputs, "mm", ["tma"])
            assert result3 is not None, "Result3 should not be None"
            assert result4 is not None, "Result4 should not be None"
            self.assertEqual(len(result3["triton"]), 1)
            self.assertEqual(len(result4["tma"]), 1)

    def test_batch_lookup_mixed_entries(self):
        """Test batch lookup where some templates have entries and others don't"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        config_list = [
            self.create_config("triton", BLOCK_M=128),
            self.create_config("tma", BLOCK_M=256),
            # No decompose_k config in lookup table
        ]
        lookup_table_data = {self.create_lookup_key("mm", kernel_inputs): config_list}

        with patch.object(
            inductor_config.template_config_lookup_table, "table", lookup_table_data
        ):
            # Test batch lookup with mixed results
            result = lookup_template_configs(
                kernel_inputs, "mm", ["triton", "tma", "decompose_k"]
            )
            assert result is not None, "Result should not be None"

            # Should have entries for triton and tma, but not decompose_k
            self.assertIn("triton", result)
            self.assertIn("tma", result)
            self.assertNotIn("decompose_k", result)

            self.assertEqual(len(result["triton"]), 1)
            self.assertEqual(len(result["tma"]), 1)
            self.assertEqual(result["triton"][0]["BLOCK_M"], 128)
            self.assertEqual(result["tma"][0]["BLOCK_M"], 256)

    @parametrize(
        "config_hash,template_hash,expected_kept",
        [
            # Hash matching (config kept)
            ("hash123", "hash123", True),
            # Hash mismatch (config filtered)
            ("hash123", "hash456", False),
            # Config without hash (config kept)
            (None, "hash123", True),
            # Template without hash (config kept)
            ("hash123", None, True),
            # Both None (config kept)
            (None, None, True),
        ],
    )
    def test_template_hash_checking(self, config_hash, template_hash, expected_kept):
        """Test template hash validation behavior"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        config = self.create_config("triton", BLOCK_M=128, BLOCK_N=64)
        if config_hash is not None:
            config["template_hash"] = config_hash

        lookup_table_data = {self.create_lookup_key("mm", kernel_inputs): [config]}

        template_hash_map = (
            {"triton": template_hash} if template_hash is not None else {}
        )

        with (
            patch.object(
                inductor_config.template_config_lookup_table, "table", lookup_table_data
            ),
            patch.object(
                inductor_config.template_config_lookup_table, "check_src_hash", True
            ),
        ):
            result = lookup_template_configs(
                kernel_inputs, "mm", ["triton"], template_hash_map
            )

            if expected_kept:
                assert result is not None, "Result should not be None"
                self.assertIn("triton", result)
                self.assertEqual(len(result["triton"]), 1)
                # template_hash should be removed from returned config
                self.assertNotIn("template_hash", result["triton"][0])
            else:
                # Config was filtered out due to hash mismatch
                self.assertEqual(result, {})

    def test_template_hash_checking_disabled(self):
        """Test that hash checking is skipped when config flag is disabled"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        # Create config with mismatching hash
        config = self.create_config("triton", BLOCK_M=128, template_hash="hash123")

        lookup_table_data = {self.create_lookup_key("mm", kernel_inputs): [config]}

        # Provide different template hash that would normally cause filtering
        template_hash_map = {"triton": "hash456"}

        with (
            patch.object(
                inductor_config.template_config_lookup_table, "table", lookup_table_data
            ),
            patch.object(
                inductor_config.template_config_lookup_table,
                "check_src_hash",
                False,
            ),
        ):
            result = lookup_template_configs(
                kernel_inputs, "mm", ["triton"], template_hash_map
            )

            # Should keep config even with mismatching hash since checking is disabled
            assert result is not None, "Result should not be None"
            self.assertIn("triton", result)
            self.assertEqual(len(result["triton"]), 1)
            # template_hash should still be removed from returned config
            self.assertNotIn("template_hash", result["triton"][0])

    def test_template_hash_mixed_scenarios(self):
        """Test mixed hash scenarios with multiple configs"""
        kernel_inputs = self.create_mock_mm_kernel_inputs()

        config_list = [
            self.create_config(
                "triton", BLOCK_M=128, template_hash="correct_hash"
            ),  # Should be kept
            self.create_config(
                "triton", BLOCK_M=64, template_hash="wrong_hash"
            ),  # Should be filtered
            self.create_config("triton", BLOCK_M=32),  # No hash, should be kept
        ]

        lookup_table_data = {self.create_lookup_key("mm", kernel_inputs): config_list}

        template_hash_map = {"triton": "correct_hash"}

        with (
            patch.object(
                inductor_config.template_config_lookup_table, "table", lookup_table_data
            ),
            patch.object(
                inductor_config.template_config_lookup_table, "check_src_hash", True
            ),
        ):
            result = lookup_template_configs(
                kernel_inputs, "mm", ["triton"], template_hash_map
            )

            assert result is not None, "Result should not be None"
            self.assertIn("triton", result)
            # Should keep 2 configs: the one with correct hash and the one without hash
            self.assertEqual(len(result["triton"]), 2)

            # Check that kept configs have expected BLOCK_M values
            kept_block_ms = [config["BLOCK_M"] for config in result["triton"]]
            self.assertIn(128, kept_block_ms)  # Config with correct hash
            self.assertIn(32, kept_block_ms)  # Config without hash
            self.assertNotIn(
                64, kept_block_ms
            )  # Config with wrong hash should be filtered

            # template_hash should be removed from returned configs
            for config in result["triton"]:
                self.assertNotIn("template_hash", config)

    def test_recorder_topk_config(self):
        """Test that topk configuration works correctly for recorder"""

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with topk=0 (should record nothing)
            config_patch = {
                "max_autotune_gemm_backends": "TRITON",
                "max_autotune_gemm": True,
                "template_config_lookup_table.recorder_record_dir": temp_dir,
                "template_config_lookup_table.recorder_topk": 0,
            }

            with inductor_config.patch(config_patch):
                # Create mock choices
                choices = []
                timings = {}

                for i in range(3):
                    choice = type("MockChoice", (), {})()
                    choice._ktc = type("MockKTC", (), {})()
                    choice._ktc.template = type(
                        "MockTemplate", (), {"uid": f"template_{i}"}
                    )()
                    choice._ktc.kwargs = {"BLOCK_M": 32 + i * 16}
                    choice._ktc.inputs = type(
                        "MockInputs",
                        (),
                        {
                            "device": lambda: torch.device("cuda"),
                            "key": f"test_key_{i}",
                        },
                    )()

                    timing = 1.0 + i * 0.1  # Increasing timings
                    timings[choice] = timing
                    choices.append(choice)

                # Mock make_lookup_key to return a valid key
                original_make_key = recorder.make_lookup_key
                recorder.make_lookup_key = lambda inputs, op_name: "test_key"

                try:
                    # Clear recorder and test
                    recorder.clear()

                    # Record with topk=0
                    recorder.record_topk_choices(
                        timings=timings,
                        op_name="mm",
                        input_nodes=[],
                        choices=choices,
                        profiled_time_fn=dict,
                        topk=0,
                    )

                    # Should record nothing
                    rec = recorder.get_lookup_table_recorder()
                    self.assertEqual(len(rec.data), 0)

                finally:
                    # Restore original function
                    recorder.make_lookup_key = original_make_key

    def test_recorder_backend_registration(self):
        """Test that recorder backends are properly registered"""

        # Test emit backend registration with different configs
        with inductor_config.patch(
            {"template_config_lookup_table.recorder_emit": True}
        ):
            # Clear and recreate recorder to test registration
            recorder.clear()
            rec = recorder.get_lookup_table_recorder()

            # Should have at least one emit backend registered
            self.assertGreater(
                len(rec.emit_backends), 0, "Should have emit backends registered"
            )

            # Verify it's the LogEmitBackend
            log_backends = [
                b for b in rec.emit_backends if isinstance(b, recorder.LogEmitBackend)
            ]
            self.assertGreater(
                len(log_backends), 0, "Should have LogEmitBackend registered"
            )

        # Test record backend registration with directory config

        with tempfile.TemporaryDirectory() as temp_dir:
            with inductor_config.patch(
                {"template_config_lookup_table.recorder_record_dir": temp_dir}
            ):
                # Clear and recreate recorder
                recorder.clear()
                rec = recorder.get_lookup_table_recorder()

                # Should have at least one record backend registered
                self.assertGreater(
                    len(rec.record_backends),
                    0,
                    "Should have record backends registered",
                )

                # Verify it's the DirectoryRecordBackend
                dir_backends = [
                    b
                    for b in rec.record_backends
                    if isinstance(b, recorder.DirectoryRecordBackend)
                ]
                self.assertGreater(
                    len(dir_backends),
                    0,
                    "Should have DirectoryRecordBackend registered",
                )

    def test_recorder_topk_logic_detailed(self):
        """Test detailed topk logic in the feedback function"""

        # Create mock choices with KTC
        choices = []
        timings = {}

        for i in range(5):
            choice = type("MockChoice", (), {})()
            choice._ktc = type("MockKTC", (), {})()
            choice._ktc.template = type("MockTemplate", (), {"uid": f"template_{i}"})()
            choice._ktc.kwargs = {"BLOCK_M": 32 + i * 16}
            choice._ktc.inputs = type(
                "MockInputs",
                (),
                {"device": lambda: torch.device("cuda"), "key": f"test_key_{i}"},
            )()

            timing = 1.0 + i * 0.1  # Increasing timings
            timings[choice] = timing
            choices.append(choice)

        # Test different topk values
        for topk_val, expected_count in [(1, 1), (3, 3), (10, 5), (-1, 5), (None, 5)]:
            with self.subTest(topk=topk_val):
                # Mock make_lookup_key to return a valid key
                original_make_key = recorder.make_lookup_key
                recorder.make_lookup_key = (
                    lambda inputs, op_name: f"test_key_{topk_val}"
                )

                try:
                    # Clear recorder
                    recorder.clear()
                    rec = recorder.get_lookup_table_recorder()

                    # Add mock backend to capture recordings
                    class MockBackend(recorder.EmitBackend):
                        def __init__(self):
                            self.entries = []

                        def emit(self, entry):
                            self.entries.append(entry)

                    backend = MockBackend()
                    rec.add_backend(backend)

                    # Record with specified topk
                    recorder.record_topk_choices(
                        timings=timings,
                        op_name="mm",
                        input_nodes=[],
                        choices=choices,
                        profiled_time_fn=dict,
                        topk=topk_val,
                    )

                    # Verify expected number of entries
                    if topk_val == 0:
                        self.assertEqual(len(backend.entries), 0)
                    else:
                        self.assertEqual(len(backend.entries), expected_count)

                        # Verify entries are sorted by timing (best first)
                        if len(backend.entries) > 1:
                            prev_timing = None
                            for entry in backend.entries:
                                timing = entry.metadata["timing"]
                                if prev_timing is not None:
                                    self.assertGreater(
                                        timing,
                                        prev_timing,
                                        "Entries should be sorted by timing",
                                    )
                                prev_timing = timing

                finally:
                    # Restore original function
                    recorder.make_lookup_key = original_make_key

    def test_recorder_backend_functionality(self):
        """Test that recorder backends work correctly"""

        # Test emit backend registration
        rec = recorder.get_lookup_table_recorder()

        # Should have at least the default log emit backend when recorder_emit=True
        with inductor_config.patch(
            {"template_config_lookup_table.recorder_emit": True}
        ):
            # Clear and recreate to test registration
            recorder.clear()
            rec = recorder.get_lookup_table_recorder()
            self.assertGreater(
                len(rec.emit_backends), 0, "Should have emit backends registered"
            )

        # Test adding custom backend
        class TestEmitBackend(recorder.EmitBackend):
            def __init__(self):
                self.entries = []

            def emit(self, entry):
                self.entries.append(entry)

        custom_backend = TestEmitBackend()
        recorder.add_backend(custom_backend)

        # Test directory record backend

        with tempfile.TemporaryDirectory() as temp_dir:
            backend = recorder.DirectoryRecordBackend(temp_dir)

            # Test data to dump
            test_data = {
                "key1": [{"template_id": "mm", "BLOCK_M": 32}],
                "key2": [{"template_id": "tma", "BLOCK_N": 64}],
            }

            backend.dump(test_data)

            # Verify file was created
            files = [f for f in os.listdir(temp_dir) if f.startswith("inductor_lut_")]
            self.assertEqual(len(files), 1)

            # Verify content
            filepath = os.path.join(temp_dir, files[0])
            with open(filepath) as f:
                loaded_data = json.load(f)

            self.assertEqual(loaded_data, test_data)

    def test_recorder_emit_and_record(self):
        """Test recorder emit and record functionality using real templates"""

        # Create test backends
        class TestEmitBackend(recorder.EmitBackend):
            def __init__(self):
                self.entries = []

            def emit(self, entry):
                self.entries.append(entry)

        class TestRecordBackend(recorder.RecordBackend):
            def __init__(self):
                self.data = None

            def dump(self, data):
                self.data = data

        # Set up recorder
        rec = recorder.LookupTableRecorder()
        emit_backend = TestEmitBackend()
        record_backend = TestRecordBackend()
        rec.add_backend(emit_backend)
        rec.add_backend(record_backend)

        # Create test entry using real template data - avoid direct imports
        # Get template UID via attribute access like in lookup_table_e2e
        try:
            import torch._inductor.kernel.mm as mm_module

            template_uid = mm_module.mm_template.uid
        except (ImportError, AttributeError):
            # Fallback for test environments
            template_uid = "triton_mm_template"

        entry = recorder.LookupTableEntry(
            key="test_key",
            value={"template_id": template_uid, "BLOCK_M": 32},
            metadata={"timing": 1.0, "rank": 0},
        )

        # Test recording
        rec.record(entry)

        # Verify emission
        self.assertEqual(len(emit_backend.entries), 1)
        self.assertEqual(emit_backend.entries[0], entry)

        # Verify internal data (value stored without metadata)
        expected_data = {"test_key": [{"template_id": template_uid, "BLOCK_M": 32}]}
        self.assertEqual(rec.data, expected_data)

        # Test dumping
        rec.dump()
        self.assertEqual(record_backend.data, expected_data)

    def test_src_hash_from_template(self):
        """Test that src_hash is extracted from template correctly"""

        # Get the actual template, avoiding direct imports like in lookup_table_e2e
        try:
            import torch._inductor.kernel.mm as mm_module

            template = mm_module.mm_template
        except ImportError:
            self.skipTest("MM module not available")

        # Verify template has src_hash
        self.assertTrue(hasattr(template, "src_hash"))
        self.assertIsInstance(template.src_hash, str)
        self.assertGreater(len(template.src_hash), 0)

        # Test entry creation uses src_hash
        class MockKTC:
            def __init__(self):
                self.template = template
                self.kwargs = {"BLOCK_M": 32, "BLOCK_N": 64}
                # Mock inputs
                self.inputs = type(
                    "MockInputs", (), {"device": lambda: torch.device("cuda")}
                )()

        # Mock make_lookup_key to return a valid key
        with inductor_config.patch(
            {"template_config_lookup_table.record_template_hash": True}
        ):
            # Patch make_lookup_key to return a test key
            original_make_key = recorder.make_lookup_key
            recorder.make_lookup_key = lambda inputs, op_name: "test_key"

            try:
                ktc = MockKTC()
                entry = recorder.LookupTableEntry.from_ktc_and_timing(
                    ktc=ktc, timing=1.23, rank=0, op_name="mm"
                )

                self.assertIsNotNone(entry)
                self.assertIn("template_hash", entry.value)
                self.assertEqual(entry.value["template_hash"], template.src_hash)

            finally:
                # Restore original function
                recorder.make_lookup_key = original_make_key


if __name__ == "__main__":
    if HAS_CUDA_AND_TRITON:
        run_tests()
