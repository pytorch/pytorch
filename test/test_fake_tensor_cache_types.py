# Owner(s): ["oncall: pt2"]

import copy
import pickle

from torch._subclasses.fake_tensor import (
    _DispatchCacheBypassEntry,
    _DispatchCacheEntryOutputInfo,
    _DispatchCacheKey,
    _DispatchCacheValidEntry,
    SingletonConstant,
)
from torch.testing._internal.common_utils import run_tests, TestCase


# Unit tests for FakeTensor dispatch cache data structures.
# These tests verify the basic properties of the cache types before
# they are ported to C++.
class FakeTensorCacheTypesTest(TestCase):
    def test_cache_key_equality(self):
        """Test that _DispatchCacheKey equality works correctly."""
        key1 = _DispatchCacheKey((1, 2, 3))
        key2 = _DispatchCacheKey((1, 2, 3))
        key3 = _DispatchCacheKey((1, 2, 4))

        self.assertEqual(key1, key2)
        self.assertNotEqual(key1, key3)

    def test_cache_key_hash(self):
        """Test that _DispatchCacheKey hashing is consistent."""
        key1 = _DispatchCacheKey((1, 2, 3))
        key2 = _DispatchCacheKey((1, 2, 3))
        key3 = _DispatchCacheKey((1, 2, 4))

        # Equal keys should have equal hashes
        self.assertEqual(hash(key1), hash(key2))
        # Different keys should (usually) have different hashes
        self.assertNotEqual(hash(key1), hash(key3))

    def test_cache_key_as_dict_key(self):
        """Test that _DispatchCacheKey can be used as a dict key."""
        key1 = _DispatchCacheKey((1, 2, 3))
        key2 = _DispatchCacheKey((1, 2, 3))
        key3 = _DispatchCacheKey((1, 2, 4))

        d = {key1: "value1"}
        # key2 should match key1
        self.assertEqual(d[key2], "value1")
        # key3 should not be in dict
        self.assertNotIn(key3, d)

        d[key3] = "value3"
        self.assertEqual(len(d), 2)

    def test_cache_key_pickle(self):
        """Test that _DispatchCacheKey can be pickled and unpickled."""
        key = _DispatchCacheKey((1, 2, "test", 3.14))
        pickled = pickle.dumps(key)
        unpickled = pickle.loads(pickled)

        self.assertEqual(key, unpickled)
        self.assertEqual(hash(key), hash(unpickled))

    def test_cache_key_deepcopy(self):
        """Test that _DispatchCacheKey can be deepcopied."""
        key = _DispatchCacheKey((1, 2, (3, 4)))
        copied = copy.deepcopy(key)

        self.assertEqual(key, copied)
        self.assertEqual(hash(key), hash(copied))

    def test_output_info_equality(self):
        """Test _DispatchCacheEntryOutputInfo equality."""
        # Inplace op case
        info1 = _DispatchCacheEntryOutputInfo(
            inplace_idx=0, metadata=None, view_idx=None
        )
        info2 = _DispatchCacheEntryOutputInfo(
            inplace_idx=0, metadata=None, view_idx=None
        )
        info3 = _DispatchCacheEntryOutputInfo(
            inplace_idx=1, metadata=None, view_idx=None
        )

        self.assertEqual(info1, info2)
        self.assertNotEqual(info1, info3)

        # Constant value case
        info4 = _DispatchCacheEntryOutputInfo(
            inplace_idx=None, metadata=None, view_idx=None, constant_value=42
        )
        info5 = _DispatchCacheEntryOutputInfo(
            inplace_idx=None, metadata=None, view_idx=None, constant_value=42
        )
        info6 = _DispatchCacheEntryOutputInfo(
            inplace_idx=None, metadata=None, view_idx=None, constant_value=43
        )

        self.assertEqual(info4, info5)
        self.assertNotEqual(info4, info6)

    def test_output_info_hash(self):
        """Test _DispatchCacheEntryOutputInfo hashing."""
        info1 = _DispatchCacheEntryOutputInfo(
            inplace_idx=0, metadata=None, view_idx=None
        )
        info2 = _DispatchCacheEntryOutputInfo(
            inplace_idx=0, metadata=None, view_idx=None
        )

        # Should be hashable (frozen dataclass)
        self.assertEqual(hash(info1), hash(info2))

    def test_output_info_pickle(self):
        """Test _DispatchCacheEntryOutputInfo pickling."""
        info = _DispatchCacheEntryOutputInfo(
            inplace_idx=0, metadata=None, view_idx=None, constant_value=None
        )
        pickled = pickle.dumps(info)
        unpickled = pickle.loads(pickled)

        self.assertEqual(info, unpickled)

    def test_output_info_deepcopy(self):
        """Test _DispatchCacheEntryOutputInfo deepcopy."""
        info = _DispatchCacheEntryOutputInfo(
            inplace_idx=0, metadata=None, view_idx=None
        )
        copied = copy.deepcopy(info)

        self.assertEqual(info, copied)

    def test_valid_entry_equality(self):
        """Test _DispatchCacheValidEntry equality."""
        info1 = _DispatchCacheEntryOutputInfo(
            inplace_idx=0, metadata=None, view_idx=None
        )
        info2 = _DispatchCacheEntryOutputInfo(
            inplace_idx=1, metadata=None, view_idx=None
        )

        entry1 = _DispatchCacheValidEntry(output_infos=(info1,), is_output_tuple=False)
        entry2 = _DispatchCacheValidEntry(output_infos=(info1,), is_output_tuple=False)
        entry3 = _DispatchCacheValidEntry(output_infos=(info2,), is_output_tuple=False)

        self.assertEqual(entry1, entry2)
        self.assertNotEqual(entry1, entry3)

        # Test tuple output
        entry4 = _DispatchCacheValidEntry(
            output_infos=(info1, info2), is_output_tuple=True
        )
        entry5 = _DispatchCacheValidEntry(
            output_infos=(info1, info2), is_output_tuple=True
        )
        self.assertEqual(entry4, entry5)

    def test_valid_entry_hash(self):
        """Test _DispatchCacheValidEntry hashing."""
        info = _DispatchCacheEntryOutputInfo(
            inplace_idx=0, metadata=None, view_idx=None
        )
        entry1 = _DispatchCacheValidEntry(output_infos=(info,), is_output_tuple=False)
        entry2 = _DispatchCacheValidEntry(output_infos=(info,), is_output_tuple=False)

        self.assertEqual(hash(entry1), hash(entry2))

    def test_valid_entry_pickle(self):
        """Test _DispatchCacheValidEntry pickling."""
        info = _DispatchCacheEntryOutputInfo(
            inplace_idx=0, metadata=None, view_idx=None
        )
        entry = _DispatchCacheValidEntry(output_infos=(info,), is_output_tuple=False)

        pickled = pickle.dumps(entry)
        unpickled = pickle.loads(pickled)

        self.assertEqual(entry, unpickled)

    def test_valid_entry_deepcopy(self):
        """Test _DispatchCacheValidEntry deepcopy."""
        info = _DispatchCacheEntryOutputInfo(
            inplace_idx=0, metadata=None, view_idx=None
        )
        entry = _DispatchCacheValidEntry(output_infos=(info,), is_output_tuple=False)

        copied = copy.deepcopy(entry)
        self.assertEqual(entry, copied)

    def test_bypass_entry_equality(self):
        """Test _DispatchCacheBypassEntry equality."""
        entry1 = _DispatchCacheBypassEntry(reason="test reason")
        entry2 = _DispatchCacheBypassEntry(reason="test reason")
        entry3 = _DispatchCacheBypassEntry(reason="different reason")

        self.assertEqual(entry1, entry2)
        self.assertNotEqual(entry1, entry3)

    def test_bypass_entry_hash(self):
        """Test _DispatchCacheBypassEntry hashing."""
        entry1 = _DispatchCacheBypassEntry(reason="test reason")
        entry2 = _DispatchCacheBypassEntry(reason="test reason")

        self.assertEqual(hash(entry1), hash(entry2))

    def test_bypass_entry_pickle(self):
        """Test _DispatchCacheBypassEntry pickling."""
        entry = _DispatchCacheBypassEntry(reason="test reason")

        pickled = pickle.dumps(entry)
        unpickled = pickle.loads(pickled)

        self.assertEqual(entry, unpickled)

    def test_bypass_entry_deepcopy(self):
        """Test _DispatchCacheBypassEntry deepcopy."""
        entry = _DispatchCacheBypassEntry(reason="test reason")
        copied = copy.deepcopy(entry)

        self.assertEqual(entry, copied)

    def test_singleton_constant_identity(self):
        """Test that SingletonConstant is a singleton."""
        # SingletonConstant should be a class, not an instance
        self.assertTrue(isinstance(SingletonConstant, type))

        # Default constant_value should be SingletonConstant (the class itself)
        info = _DispatchCacheEntryOutputInfo(
            inplace_idx=None, metadata=None, view_idx=None
        )
        self.assertIs(info.constant_value, SingletonConstant)

    def test_cache_key_nested_structures(self):
        """Test _DispatchCacheKey with nested structures."""
        # Test with nested tuples (must be hashable)
        key1 = _DispatchCacheKey((1, (2, 3), (4, 5)))
        key2 = _DispatchCacheKey((1, (2, 3), (4, 5)))
        key3 = _DispatchCacheKey((1, (2, 3), (4, 6)))

        self.assertEqual(key1, key2)
        self.assertNotEqual(key1, key3)

    def test_cache_key_with_none(self):
        """Test _DispatchCacheKey with None values."""
        key1 = _DispatchCacheKey((None, 1, None))
        key2 = _DispatchCacheKey((None, 1, None))
        key3 = _DispatchCacheKey((None, 2, None))

        self.assertEqual(key1, key2)
        self.assertNotEqual(key1, key3)

    def test_entry_tuple_vs_single_output(self):
        """Test that is_output_tuple flag correctly distinguishes entry types."""
        info = _DispatchCacheEntryOutputInfo(
            inplace_idx=0, metadata=None, view_idx=None
        )

        # Single output
        entry_single = _DispatchCacheValidEntry(
            output_infos=(info,), is_output_tuple=False
        )
        # Tuple output with one element
        entry_tuple = _DispatchCacheValidEntry(
            output_infos=(info,), is_output_tuple=True
        )

        # These should be different
        self.assertNotEqual(entry_single, entry_tuple)


if __name__ == "__main__":
    run_tests()
