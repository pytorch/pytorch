# Owner(s): ["module: inductor"]
import json
import os
import shutil
import tempfile
import unittest
from typing import Any

import torch
import torch.nn as nn
from torch._inductor import config as inductor_config
from torch._inductor.lookup_table_recorder import (
    clear,
    DirectoryRecordBackend,
    emit_backend,
    EmitBackend,
    get_lookup_table_recorder,
    LookupTableEntry,
    record_backend,
    RecordBackend,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON
from torch.utils._triton import has_triton_tma_device


class TestEmitBackend(EmitBackend):
    """Test emit backend that captures emitted entries"""

    def __init__(self):
        self.emitted_entries: list[LookupTableEntry] = []

    def emit(self, entry: LookupTableEntry):
        self.emitted_entries.append(entry)


class TestRecordBackend(RecordBackend):
    """Test record backend that captures dumped data"""

    def __init__(self):
        self.dumped_data: dict[str, list[dict[str, Any]]] = {}

    def dump(self, data: dict[str, list[dict[str, Any]]]):
        self.dumped_data = data.copy()


class SimpleMMModel(nn.Module):
    """Simple model that performs matrix multiplication"""

    def forward(self, a, b):
        return torch.mm(a, b)


def force_recorder_reset():
    """Force the global recorder to be recreated on next access"""
    import torch._inductor.lookup_table_recorder as recorder_module

    recorder_module._lookup_table_recorder = None


@unittest.skipIf(not HAS_CUDA_AND_TRITON, "CUDA not available")
class TestLookupTableRecorder(TestCase):
    """Test suite for lookup table recorder functionality"""

    def setUp(self):
        torch._dynamo.reset()
        self.device = torch.device("cuda")
        self.temp_dir = tempfile.mkdtemp()

        # Store original config values
        self.original_recorder_emit = (
            inductor_config.template_lookup_table_config.recorder_emit
        )
        self.original_recorder_record_dir = (
            inductor_config.template_lookup_table_config.recorder_record_dir
        )
        self.original_lookup_table = inductor_config.template_lookup_table

        # Clear any existing recorder
        clear()
        force_recorder_reset()

    def tearDown(self):
        # Restore original config
        inductor_config.template_lookup_table_config.recorder_emit = (
            self.original_recorder_emit
        )
        inductor_config.template_lookup_table_config.recorder_record_dir = (
            self.original_recorder_record_dir
        )
        inductor_config.template_lookup_table = self.original_lookup_table

        # Clean up temp files
        shutil.rmtree(self.temp_dir, ignore_errors=True)

        # Clear recorder
        clear()
        force_recorder_reset()

    def create_simple_mm_tensors(self):
        """Create small test tensors for torch.mm"""
        return [
            torch.randn(64, 32, device=self.device, dtype=torch.float16),
            torch.randn(32, 64, device=self.device, dtype=torch.float16),
        ]

    def compile_and_run_mm(self, config_patches=None):
        """Compile and execute a simple mm operation"""
        default_config = {"max_autotune_gemm": True}
        if config_patches:
            default_config.update(config_patches)
        torch._dynamo.reset()
        with inductor_config.patch(default_config):
            model = SimpleMMModel().to(self.device)
            tensors = self.create_simple_mm_tensors()
            compiled_model = torch.compile(model, mode="max-autotune")
            return compiled_model(*tensors)

    @fresh_cache()
    def test_emit_works(self):
        """Test that emit functionality works correctly"""
        # Force recorder reset and create test backend
        force_recorder_reset()
        test_emit_backend = TestEmitBackend()

        # Get fresh recorder and add our test backend
        recorder = get_lookup_table_recorder()
        recorder.add_backend(test_emit_backend)

        # Trigger compilation with autotuning
        self.compile_and_run_mm()

        # Verify entries were emitted
        self.assertGreater(
            len(test_emit_backend.emitted_entries),
            0,
            "Expected at least one entry to be emitted",
        )

        # Check structure of emitted entries
        for entry in test_emit_backend.emitted_entries:
            self.assertIsInstance(entry, LookupTableEntry)
            self.assertIsInstance(entry.key, str)
            self.assertIsInstance(entry.value, dict)
            self.assertIn("template_id", entry.value)

    @fresh_cache()
    def test_directory_record_backend(self):
        """Test that DirectoryRecordBackend creates timestamped files correctly"""
        # Setup directory for dumping
        dump_dir = os.path.join(self.temp_dir, "test_dump_dir")

        # Force recorder reset and add DirectoryRecordBackend
        force_recorder_reset()
        recorder = get_lookup_table_recorder()
        dir_backend = DirectoryRecordBackend(dump_dir)
        recorder.add_backend(dir_backend)

        # Trigger compilation with autotuning
        self.compile_and_run_mm()

        # Trigger dump
        recorder.dump()

        # Verify directory was created
        self.assertTrue(os.path.exists(dump_dir), "Dump directory should be created")
        self.assertTrue(os.path.isdir(dump_dir), "Dump path should be a directory")

        # Find the generated file
        files = os.listdir(dump_dir)
        json_files = [
            f for f in files if f.endswith(".json") and f.startswith("inductor_lut_")
        ]
        self.assertEqual(len(json_files), 1, "Should have exactly one JSON file")

        # Verify filename format (inductor_lut_YYYYMMDD_HHMMSS_mmm.json)
        filename = json_files[0]
        self.assertTrue(
            filename.startswith("inductor_lut_"),
            "Filename should start with 'inductor_lut_'",
        )
        self.assertTrue(filename.endswith(".json"), "Filename should end with '.json'")

        # Extract timestamp part and verify format
        timestamp_part = filename[len("inductor_lut_") : -len(".json")]
        parts = timestamp_part.split("_")
        self.assertEqual(
            len(parts), 3, "Timestamp should have 3 parts: date, time, milliseconds"
        )

        date_part, time_part, ms_part = parts
        self.assertEqual(len(date_part), 8, "Date part should be 8 digits (YYYYMMDD)")
        self.assertEqual(len(time_part), 6, "Time part should be 6 digits (HHMMSS)")
        self.assertEqual(len(ms_part), 3, "Millisecond part should be 3 digits")

        # Verify file contains valid JSON
        filepath = os.path.join(dump_dir, filename)
        with open(filepath) as f:
            data = json.load(f)

        self.assertIsInstance(data, dict)
        self.assertGreater(len(data), 0, "Expected at least one entry in dump")

        # Check structure of dumped data
        for key, configs in data.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(configs, list)
            self.assertGreater(len(configs), 0)
            for config in configs:
                self.assertIsInstance(config, dict)
                self.assertIn("template_id", config)

    @fresh_cache()
    def test_end_to_end_workflow(self):
        """Test complete workflow from recording to reading and feeding back to inductor"""
        # Step 1: Record a lookup table during first compilation
        dump_dir = os.path.join(self.temp_dir, "e2e_test_dir")

        # Force recorder reset and add DirectoryRecordBackend
        force_recorder_reset()
        recorder = get_lookup_table_recorder()
        dir_backend = DirectoryRecordBackend(dump_dir)
        recorder.add_backend(dir_backend)

        # First compilation - this should record entries
        _ = self.compile_and_run_mm()

        # Dump the recorded table
        recorder.dump()

        # Verify directory and file were created
        self.assertTrue(os.path.exists(dump_dir), "Dump directory should be created")
        files = os.listdir(dump_dir)
        json_files = [
            f for f in files if f.endswith(".json") and f.startswith("inductor_lut_")
        ]
        self.assertEqual(len(json_files), 1, "Should have exactly one JSON file")

        # Step 2: Read the table from the file
        dump_file = os.path.join(dump_dir, json_files[0])
        with open(dump_file) as f:
            recorded_table = json.load(f)

        self.assertGreater(len(recorded_table), 0, "Should have recorded some entries")

        # Step 3: Configure inductor to use the recorded table
        inductor_config.template_lookup_table = recorded_table

        # Clear the recorder to start fresh
        clear()
        force_recorder_reset()

        # Step 4: Compile the same operation again
        # This should work and throw no errors
        _ = self.compile_and_run_mm()

    @fresh_cache()
    def test_recorder_clear_functionality(self):
        """Test that clear functionality works correctly"""
        # Setup recording
        force_recorder_reset()
        recorder = get_lookup_table_recorder()
        test_record_backend = TestRecordBackend()
        recorder.add_backend(test_record_backend)

        # Trigger compilation to populate data
        self.compile_and_run_mm()

        # Verify data exists
        self.assertGreater(len(recorder.data), 0)

        # Clear and verify data is gone
        recorder.clear()
        self.assertEqual(len(recorder.data), 0)

    @fresh_cache()
    def test_decorator_registration_conditional(self):
        """Test that decorator registration respects should_register parameter"""

        # Create two backend classes with decorators - one should register, one shouldn't
        @emit_backend(should_register=True)
        class ShouldRegisterEmitBackend(EmitBackend):
            def emit(self, entry: LookupTableEntry):
                pass

        @emit_backend(should_register=False)
        class ShouldNotRegisterEmitBackend(EmitBackend):
            def emit(self, entry: LookupTableEntry):
                pass

        @record_backend(should_register=True)
        class ShouldRegisterRecordBackend(RecordBackend):
            def dump(self, data: dict[str, list[dict[str, Any]]]):
                pass

        @record_backend(should_register=False)
        class ShouldNotRegisterRecordBackend(RecordBackend):
            def dump(self, data: dict[str, list[dict[str, Any]]]):
                pass

        # Force recorder reset to trigger re-registration with new backends
        force_recorder_reset()
        recorder = get_lookup_table_recorder()

        # Check that only the backends with should_register=True were registered
        registered_emit_types = [type(backend) for backend in recorder.emit_backends]
        registered_record_types = [
            type(backend) for backend in recorder.record_backends
        ]

        # Should have registered the "should register" backends
        self.assertIn(ShouldRegisterEmitBackend, registered_emit_types)
        self.assertIn(ShouldRegisterRecordBackend, registered_record_types)

        # Should NOT have registered the "should not register" backends
        self.assertNotIn(ShouldNotRegisterEmitBackend, registered_emit_types)
        self.assertNotIn(ShouldNotRegisterRecordBackend, registered_record_types)

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @fresh_cache()
    def test_tma_entries_recorded(self):
        """Test that TMA-specific entries are recorded when TMA is enabled"""
        # Create custom record backend to capture data
        test_record_backend = TestRecordBackend()

        # Enable TMA and recording
        inductor_config.template_lookup_table_config.recorder_emit = True

        # Get recorder and add our test backend
        recorder = get_lookup_table_recorder()
        self.assertIsNotNone(recorder)
        assert recorder is not None  # Type narrowing for mypy
        recorder.add_backend(test_record_backend)

        # Trigger compilation with TMA enabled
        self.compile_and_run_mm({"triton.enable_persistent_tma_matmul": True})

        # Trigger dump to populate our test backend
        recorder.dump()

        # Check if any entries contain TMA-related values
        tma_entries_found = False
        for configs in test_record_backend.dumped_data.values():
            for config in configs:
                # Check for TMA in template_id or TMA-specific parameters
                if (
                    "tma" in config.get("template_id", "").lower()
                    or "TMA_SIZE" in config
                    or "NUM_SMS" in config
                    or "TMA_EXPERIMENTAL_API" in config
                ):
                    tma_entries_found = True
                    break
            if tma_entries_found:
                break

        self.assertTrue(
            tma_entries_found,
            "Expected to find at least one entry with TMA-related values",
        )

    @fresh_cache()
    def test_limited_choices_feedback(self):
        """Test that feeding back a limited subset of choices works correctly"""
        # Step 1: Record initial lookup table
        test_record_backend = TestRecordBackend()
        recorder = get_lookup_table_recorder()
        recorder.add_backend(test_record_backend)

        # First compilation - record all choices
        self.compile_and_run_mm()
        recorder.dump()

        # Get the recorded table
        recorded_table = test_record_backend.dumped_data
        self.assertGreater(len(recorded_table), 0, "Should have recorded some entries")

        # Step 2: Modify the table to keep only the first 2 entries for each key
        limited_table = {}
        for key, configs in recorded_table.items():
            limited_table[key] = configs[:2]  # Keep only first 2 entries

        # Step 3: Create a custom backend to capture what gets considered in the second run
        class ChoiceCapturingBackend(RecordBackend):
            def __init__(self):
                self.captured_choices = []

            def dump(self, data):
                self.captured_choices = []
                for configs in data.values():
                    self.captured_choices.extend(configs)

        # Clear recorder and set up for second run
        clear()
        force_recorder_reset()
        choice_capturing_backend = ChoiceCapturingBackend()
        recorder = get_lookup_table_recorder()
        recorder.add_backend(choice_capturing_backend)

        # Step 4: Configure inductor to use the limited table
        inductor_config.template_lookup_table = limited_table

        # Step 5: Run compilation again
        self.compile_and_run_mm()
        recorder.dump()

        # Step 6: Verify that only 2 choices were considered
        self.assertEqual(
            len(choice_capturing_backend.captured_choices),
            2,
            f"Expected exactly 2 choices, got {len(choice_capturing_backend.captured_choices)}",
        )

        # Step 7: Verify the choices match what we fed in
        original_choices = []
        for configs in limited_table.values():
            original_choices.extend(configs)

        # Sort both lists by key parameters to ensure consistent comparison
        def sort_key(config):
            return (
                config["BLOCK_M"],
                config["BLOCK_N"],
                config["BLOCK_K"],
                config["num_stages"],
                config["num_warps"],
            )

        original_sorted = sorted(original_choices, key=sort_key)
        captured_sorted = sorted(
            choice_capturing_backend.captured_choices, key=sort_key
        )

        # Compare the choices
        for i in range(2):
            self.assertEqual(
                captured_sorted[i].get("template_id"),
                original_sorted[i].get("template_id"),
                f"Choice {i} template_id mismatch",
            )
            # Compare key parameters that should match
            for param in ["BLOCK_M", "BLOCK_N", "BLOCK_K", "num_stages", "num_warps"]:
                self.assertEqual(
                    captured_sorted[i].get(param),
                    original_sorted[i].get(param),
                    f"Choice {i} parameter {param} mismatch",
                )


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu
    from torch.testing._internal.inductor_utils import HAS_CPU, HAS_GPU

    # Set env to make it work in CI
    if HAS_GPU and HAS_CPU and is_big_gpu():
        run_tests()
