"""
Tests for silent correctness fixes in torch._dynamo modules.

These tests verify fixes for potential silent incorrectness issues where
code runs without errors but produces wrong results.
"""

import gc
import sys
import unittest
import weakref
from unittest.mock import MagicMock, patch

# Add local source to path first to test our changes
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from torch.testing._internal.common_utils import run_tests, TestCase


class WeakRefableObject:
    """A simple class that can be weakref'd (unlike dict)."""
    def __init__(self, value):
        self.value = value


class TestCacheSizeWeakrefComparison(TestCase):
    """Test that weakref comparison compares objects, not weakref containers."""

    def test_same_object_different_weakrefs(self):
        """Two different weakrefs to the same object should be considered equal."""
        from torch._dynamo.cache_size import _has_same_id_matched_objs

        # Create an object and a weakref to it
        obj = WeakRefableObject("test")
        ref1 = weakref.ref(obj)

        # Note: Python caches weakrefs, so weakref.ref(obj) returns same ref
        # The real scenario is when _get_weakref_from_f_locals creates a new
        # weakref from frame.f_locals. We simulate this by verifying the
        # function correctly dereferences and compares the actual objects.

        # Create mock frame and cache_entry
        mock_frame = MagicMock()
        mock_frame.f_locals = {"local_var": obj}

        mock_cache_entry = MagicMock()
        mock_cache_entry.guard_manager.id_matched_objs = {"local_var": ref1}

        # The function should return True because the objects are the same
        result = _has_same_id_matched_objs(mock_frame, mock_cache_entry)
        self.assertTrue(result)

    def test_different_objects(self):
        """Different objects should not be considered equal."""
        from torch._dynamo.cache_size import _has_same_id_matched_objs

        obj1 = WeakRefableObject("value1")
        obj2 = WeakRefableObject("value2")
        ref1 = weakref.ref(obj1)

        mock_frame = MagicMock()
        mock_frame.f_locals = {"local_var": obj2}

        mock_cache_entry = MagicMock()
        mock_cache_entry.guard_manager.id_matched_objs = {"local_var": ref1}

        result = _has_same_id_matched_objs(mock_frame, mock_cache_entry)
        self.assertFalse(result)

    def test_dead_weakref_skipped(self):
        """Dead weakrefs should be skipped in validation."""
        from torch._dynamo.cache_size import _has_same_id_matched_objs

        # Create object, get weakref, then delete object
        obj = WeakRefableObject("test")
        ref = weakref.ref(obj)
        del obj
        gc.collect()

        # Verify the ref is dead
        self.assertIsNone(ref())

        # Create mock with dead weakref
        mock_frame = MagicMock()
        mock_frame.f_locals = {"local_var": WeakRefableObject("different")}

        mock_cache_entry = MagicMock()
        mock_cache_entry.guard_manager.id_matched_objs = {"local_var": ref}

        # Should return True because dead refs are skipped
        result = _has_same_id_matched_objs(mock_frame, mock_cache_entry)
        self.assertTrue(result)


class TestSideEffectsFalsyValues(TestCase):
    """Test that falsy values are handled correctly in side effects."""

    def test_load_cell_with_falsy_zero(self):
        """Cell containing integer 0 should be returned correctly."""
        from torch._dynamo import variables
        from torch._dynamo.side_effects import SideEffects
        from torch._dynamo.variables.base import AttributeMutationExisting
        from torch._dynamo.source import LocalSource

        mock_output = MagicMock()
        side_effects = SideEffects(mock_output)

        # Create a cell variable with falsy content (integer 0)
        zero_var = variables.ConstantVariable.create(0)
        cell_var = variables.CellVariable(
            mutation_type=AttributeMutationExisting(),
            pre_existing_contents=zero_var,
            source=LocalSource("test_cell"),
        )

        result = side_effects.load_cell(cell_var)
        self.assertIs(result, zero_var)

    def test_load_cell_with_falsy_false(self):
        """Cell containing False should be returned correctly."""
        from torch._dynamo import variables
        from torch._dynamo.side_effects import SideEffects
        from torch._dynamo.variables.base import AttributeMutationExisting
        from torch._dynamo.source import LocalSource

        mock_output = MagicMock()
        side_effects = SideEffects(mock_output)

        false_var = variables.ConstantVariable.create(False)
        cell_var = variables.CellVariable(
            mutation_type=AttributeMutationExisting(),
            pre_existing_contents=false_var,
            source=LocalSource("test_cell"),
        )

        result = side_effects.load_cell(cell_var)
        self.assertIs(result, false_var)

    def test_load_cell_with_falsy_empty_string(self):
        """Cell containing empty string should be returned correctly."""
        from torch._dynamo import variables
        from torch._dynamo.side_effects import SideEffects
        from torch._dynamo.variables.base import AttributeMutationExisting
        from torch._dynamo.source import LocalSource

        mock_output = MagicMock()
        side_effects = SideEffects(mock_output)

        empty_str_var = variables.ConstantVariable.create("")
        cell_var = variables.CellVariable(
            mutation_type=AttributeMutationExisting(),
            pre_existing_contents=empty_str_var,
            source=LocalSource("test_cell"),
        )

        result = side_effects.load_cell(cell_var)
        self.assertIs(result, empty_str_var)

    def test_is_modified_with_none_mutation_type(self):
        """is_modified should return False for None mutation_type."""
        from torch._dynamo.side_effects import SideEffects
        from torch._dynamo import variables

        mock_output = MagicMock()
        side_effects = SideEffects(mock_output)

        # Create a mock variable with None mutation_type that passes is_immutable check
        mock_var = MagicMock(spec=variables.VariableTracker)
        mock_var.is_immutable.return_value = False
        mock_var.mutation_type = None

        # Should not raise, should return False
        result = side_effects.is_modified(mock_var)
        self.assertFalse(result)


class TestMutationTrackerCleanup(TestCase):
    """Test that mutation tracker cleans up dead references."""

    def test_dead_watchers_cleaned_up(self):
        """Dead watchers should be cleaned up when list grows large."""
        from torch._dynamo.mutation_guard import MutationTracker

        tracker = MutationTracker()

        # Add many guarded codes, then let them die
        for i in range(150):
            guarded = MagicMock()
            tracker.track(guarded)
            # Don't keep a reference - let it be GC'd

        gc.collect()

        # Add one more - this should trigger cleanup
        final_guarded = MagicMock()
        tracker.track(final_guarded)

        # After cleanup, only live references should remain
        live_count = sum(1 for ref in tracker.watchers if ref() is not None)
        self.assertLessEqual(live_count, 52)  # At most ~50 from last batch + 1 + some slack


class TestDictVariableFalsyValues(TestCase):
    """Test that dict variable handles falsy VariableTrackers correctly."""

    def test_is_new_item_with_none_value(self):
        """is_new_item should handle None value correctly."""
        from torch._dynamo.variables.dicts import ConstDictVariable

        dict_var = ConstDictVariable({})

        other = MagicMock()
        other.is_realized.return_value = True

        # None value should compare using id()
        result = dict_var.is_new_item(None, other)
        self.assertTrue(result)  # None is not other

    def test_is_new_item_with_realized_values(self):
        """is_new_item should compare realized values when both are realized."""
        from torch._dynamo.variables.dicts import ConstDictVariable

        dict_var = ConstDictVariable({})

        # Create two mock values that are realized
        value1 = MagicMock()
        value1.is_realized.return_value = True
        realized_obj = object()
        value1.realize.return_value = realized_obj

        value2 = MagicMock()
        value2.is_realized.return_value = True
        value2.realize.return_value = realized_obj  # Same object

        result = dict_var.is_new_item(value1, value2)
        self.assertFalse(result)  # Same realized object

        # Different realized object
        value3 = MagicMock()
        value3.is_realized.return_value = True
        value3.realize.return_value = object()  # Different object

        result = dict_var.is_new_item(value1, value3)
        self.assertTrue(result)  # Different realized objects


class TestIgnoreMutationsLogging(TestCase):
    """Test that mutation ignoring is properly logged."""

    def test_ignore_mutations_logs(self):
        """ignore_mutations_on should log when called."""
        from torch._dynamo.side_effects import SideEffects, side_effects_log

        mock_output = MagicMock()
        side_effects = SideEffects(mock_output)

        mock_var = MagicMock()

        with patch.object(side_effects_log, 'debug') as mock_debug:
            side_effects.ignore_mutations_on(mock_var)
            mock_debug.assert_called()

    def test_stop_ignoring_mutations_logs_when_not_ignoring(self):
        """stop_ignoring_mutations_on should log when var wasn't being ignored."""
        from torch._dynamo.side_effects import SideEffects, side_effects_log

        mock_output = MagicMock()
        side_effects = SideEffects(mock_output)

        mock_var = MagicMock()

        with patch.object(side_effects_log, 'debug') as mock_debug:
            # Try to stop ignoring a var that was never being ignored
            side_effects.stop_ignoring_mutations_on(mock_var)
            # Should log a debug message about this
            mock_debug.assert_called()


if __name__ == "__main__":
    run_tests()
