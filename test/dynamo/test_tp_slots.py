# Owner(s): ["module: dynamo"]

"""Tests for CPython type slot detection in Dynamo.

Tests that get_type_slots correctly identifies which protocol methods
(sequence, mapping, number, type) are implemented by various types.
"""

import collections
import collections.abc
import dataclasses
import enum

from torch._C._dynamo import (
    get_type_slots,
    has_slot,
    PyMappingSlots,
    PyNumberSlots,
    PySequenceSlots,
    PyTypeSlots,
)
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.dicts import ConstDictVariable, OrderedItemsDictVariable
from torch._dynamo.variables.lists import ListVariable, SizeVariable, TupleVariable
from torch._dynamo.variables.sets import (
    DictKeySetVariable,
    FrozensetVariable,
    OrderedSetVariable,
    SetVariable,
)
from torch._dynamo.variables.user_defined import UserDefinedObjectVariable
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


class TestTypeSlots(TestCase):
    """Test suite for type slot detection."""

    def _get_slot_info(self, obj_type):
        """Helper to get and unpack slot information."""
        seq_slots, map_slots, num_slots, type_slots = get_type_slots(obj_type)
        return seq_slots, map_slots, num_slots, type_slots

    def test_dict_slots(self):
        """Test that dict has mapping protocol but not sequence protocol."""
        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(dict)

        # dict should NOT have sq_length (sequence protocol)
        self.assertFalse(has_slot(seq_slots, PySequenceSlots.SQ_LENGTH))

        # dict SHOULD have mapping protocol
        self.assertTrue(has_slot(map_slots, PyMappingSlots.MP_LENGTH))
        self.assertTrue(has_slot(map_slots, PyMappingSlots.MP_SUBSCRIPT))
        self.assertTrue(has_slot(map_slots, PyMappingSlots.MP_ASS_SUBSCRIPT))

    def test_list_slots(self):
        """Test that list has sequence protocol."""
        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(list)

        # list SHOULD have sequence protocol
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_LENGTH))
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_ITEM))
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_CONTAINS))

        # list also has mapping protocol for compatibility (mp_length, mp_subscript, mp_ass_subscript)
        self.assertTrue(has_slot(map_slots, PyMappingSlots.MP_LENGTH))
        self.assertTrue(has_slot(map_slots, PyMappingSlots.MP_SUBSCRIPT))

    def test_tuple_slots(self):
        """Test that tuple has sequence protocol."""
        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(tuple)

        # tuple SHOULD have sequence protocol
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_LENGTH))
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_ITEM))
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_CONTAINS))

    def test_set_slots(self):
        """Test that set has sequence protocol for contains."""
        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(set)

        # set SHOULD have sq_contains
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_CONTAINS))

        # set should NOT have mapping protocol
        self.assertFalse(has_slot(map_slots, PyMappingSlots.MP_LENGTH))

    def test_dict_subclass_slots(self):
        """Test that dict subclasses have both mapping and sequence protocol."""

        class DictSubclass(dict):
            pass

        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(DictSubclass)

        # Dict subclasses expose both protocols for compatibility
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_LENGTH))
        self.assertTrue(has_slot(map_slots, PyMappingSlots.MP_LENGTH))

    def test_list_subclass_slots(self):
        """Test that list subclasses have sequence protocol."""

        class ListSubclass(list):
            pass

        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(ListSubclass)

        # List subclasses should have sequence protocol
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_LENGTH))
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_ITEM))

    def test_int_slots(self):
        """Test that int has number protocol."""
        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(int)

        # int SHOULD have number protocol operations
        self.assertTrue(has_slot(num_slots, PyNumberSlots.NB_ADD))
        self.assertTrue(has_slot(num_slots, PyNumberSlots.NB_SUBTRACT))
        self.assertTrue(has_slot(num_slots, PyNumberSlots.NB_MULTIPLY))

    def test_str_slots(self):
        """Test that str has sequence protocol."""
        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(str)

        # str SHOULD have sequence protocol
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_LENGTH))
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_ITEM))
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_CONTAINS))

    def test_type_has_call_slot(self):
        """Test that type objects have tp_call slot."""
        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(type)

        # type SHOULD have tp_call (for calling classes/types)
        self.assertTrue(has_slot(type_slots, PyTypeSlots.TP_CALL))

    def test_type_has_hash_slot(self):
        """Test that most types have tp_hash slot."""
        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(int)

        # int SHOULD have tp_hash
        self.assertTrue(has_slot(type_slots, PyTypeSlots.TP_HASH))

    def test_custom_class_slots(self):
        """Test that custom user-defined classes have minimal slots."""

        class CustomClass:
            pass

        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(CustomClass)

        # Custom classes should not have sequence/mapping/number protocol by default
        self.assertFalse(has_slot(seq_slots, PySequenceSlots.SQ_LENGTH))
        self.assertFalse(has_slot(map_slots, PyMappingSlots.MP_LENGTH))

    def test_set_subclass_slots(self):
        """Test that set subclasses inherit set protocol."""

        class SetSubclass(set):
            pass

        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(SetSubclass)

        # Set subclasses should have sq_contains
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_CONTAINS))

    def test_tuple_subclass_slots(self):
        """Test that tuple subclasses inherit sequence protocol."""

        class TupleSubclass(tuple):
            __slots__ = ()

        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(TupleSubclass)

        # Tuple subclasses should have sequence protocol
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_LENGTH))
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_ITEM))

    def test_mutable_mapping_slots(self):
        """Test that MutableMapping ABC has mapping protocol."""

        class MyMapping(collections.abc.MutableMapping):
            def __init__(self):
                self._data = {}

            def __getitem__(self, key):
                return self._data[key]

            def __setitem__(self, key, value):
                self._data[key] = value

            def __delitem__(self, key):
                del self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(MyMapping)

        # MutableMapping subclasses should have mapping protocol
        self.assertTrue(has_slot(map_slots, PyMappingSlots.MP_LENGTH))
        self.assertTrue(has_slot(map_slots, PyMappingSlots.MP_SUBSCRIPT))
        self.assertTrue(has_slot(map_slots, PyMappingSlots.MP_ASS_SUBSCRIPT))

    def test_frozen_dataclass_slots(self):
        """Test that frozen dataclasses have standard object slots."""

        @dataclasses.dataclass(frozen=True)
        class FrozenData:
            x: int
            y: str

        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(FrozenData)

        # Frozen dataclasses should have basic type slots like hash
        self.assertTrue(has_slot(type_slots, PyTypeSlots.TP_HASH))

    def test_enum_slots(self):
        """Test that Enum types have expected slots."""

        class Color(enum.Enum):
            RED = 1
            GREEN = 2
            BLUE = 3

        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(Color)

        # Enums should have type protocol slots
        self.assertTrue(has_slot(type_slots, PyTypeSlots.TP_HASH))

    def test_enum_member_slots(self):
        """Test that individual enum members work correctly."""

        class Status(enum.Enum):
            PENDING = "pending"
            ACTIVE = "active"

        # Enum members are instances, test the enum class
        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(Status)
        self.assertTrue(has_slot(type_slots, PyTypeSlots.TP_HASH))

    def test_metaclass_slots(self):
        """Test that metaclass types have tp_call for instantiation."""

        class MyMeta(type):
            pass

        class MyClass(metaclass=MyMeta):
            pass

        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(MyMeta)

        # Metaclasses should have tp_call for creating instances
        self.assertTrue(has_slot(type_slots, PyTypeSlots.TP_CALL))

    def test_dict_subclass_with_custom_len(self):
        """Test dict subclass with custom __len__ (the original bug case)."""

        class DictWithCustomLen(dict):
            def __len__(self):
                return super().__len__() - 1

        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(
            DictWithCustomLen
        )

        # Should have both protocols (CPython compatibility)
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_LENGTH))
        self.assertTrue(has_slot(map_slots, PyMappingSlots.MP_LENGTH))

    def test_frozenset_slots(self):
        """Test that frozenset has sequence protocol for contains."""
        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(frozenset)

        # frozenset SHOULD have sq_contains
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_CONTAINS))

    def test_float_slots(self):
        """Test that float has number protocol."""
        seq_slots, map_slots, num_slots, type_slots = self._get_slot_info(float)

        # float SHOULD have number protocol operations
        self.assertTrue(has_slot(num_slots, PyNumberSlots.NB_ADD))
        self.assertTrue(has_slot(num_slots, PyNumberSlots.NB_SUBTRACT))
        self.assertTrue(has_slot(num_slots, PyNumberSlots.NB_MULTIPLY))

    def test_deque_sq_item_only(self):
        """deque has sq_item but NOT mp_subscript — vt_getitem dispatches to Branch 2."""
        seq_slots, map_slots, _, _ = self._get_slot_info(collections.deque)
        self.assertFalse(has_slot(map_slots, PyMappingSlots.MP_SUBSCRIPT))
        self.assertTrue(has_slot(seq_slots, PySequenceSlots.SQ_ITEM))

    def test_nb_index_int_bool(self):
        """int and bool have nb_index (pass _PyIndex_Check)."""
        for t in [int, bool]:
            _, _, num_slots, _ = self._get_slot_info(t)
            self.assertTrue(
                has_slot(num_slots, PyNumberSlots.NB_INDEX),
                f"{t.__name__} should have nb_index",
            )

    def test_nb_index_absent(self):
        """float, str, complex do NOT have nb_index (fail _PyIndex_Check)."""
        for t in [float, str, complex]:
            _, _, num_slots, _ = self._get_slot_info(t)
            self.assertFalse(
                has_slot(num_slots, PyNumberSlots.NB_INDEX),
                f"{t.__name__} should NOT have nb_index",
            )

    def test_mp_subscript_on_subscriptable_types(self):
        """All types with mp_subscript_impl overrides must have the C-level slot."""
        import torch

        for t in [
            list,
            tuple,
            dict,
            str,
            bytes,
            range,
            collections.OrderedDict,
            collections.defaultdict,
            torch.Tensor,
            torch.Size,
        ]:
            _, map_slots, _, _ = self._get_slot_info(t)
            self.assertTrue(
                has_slot(map_slots, PyMappingSlots.MP_SUBSCRIPT),
                f"{t.__name__} should have mp_subscript",
            )


# Builtin VTs whose type is fixed by the VT class, so tp_name is a static slot
# and must equal what python_type().__name__ returns.
_TP_NAME_CASES = {
    "list": lambda: ListVariable([]),
    "tuple": lambda: TupleVariable([]),
    "Size": lambda: SizeVariable([]),
    "set": lambda: SetVariable([]),
    "frozenset": lambda: FrozensetVariable([]),
    "OrderedSet": lambda: OrderedSetVariable([]),
    "dict_keys": lambda: DictKeySetVariable([]),
    "dict": lambda: ConstDictVariable({}),
    "OrderedDict": lambda: OrderedItemsDictVariable({}),
}


class VariableTrackerTpNameDocTests(TestCase):
    """tp_name / tp_doc slots on VariableTracker (analogous to CPython's
    PyTypeObject tp_name / tp_doc): describe the type a VT represents, not a
    per-instance attribute. tp_name backs python_type_name(); tp_doc backs
    __doc__ reads. Both default to None (derive from python_type())."""

    @parametrize("name", list(_TP_NAME_CASES))
    def test_tp_name_matches_python_type_name(self, name):
        vt = _TP_NAME_CASES[name]()
        self.assertEqual(vt.tp_name, name)
        self.assertEqual(vt.python_type_name(), name)

    def test_base_slots_default_to_none(self):
        self.assertIsNone(VariableTracker.tp_name)
        self.assertIsNone(VariableTracker.tp_doc)

    def test_dynamic_vt_derives_name_from_python_type(self):
        # UserDefinedObjectVariable represents arbitrary classes, so its type is
        # per instance: tp_name stays None and python_type_name() derives the
        # name from python_type() (value_type).
        class Foo:
            pass

        self.assertIsNone(UserDefinedObjectVariable.tp_name)
        self.assertEqual(UserDefinedObjectVariable(Foo()).python_type_name(), "Foo")
        self.assertEqual(
            UserDefinedObjectVariable(object()).python_type_name(), "object"
        )

    def test_tp_doc_absent_is_not_a_constant(self):
        # No tp_doc declared: const_getattr does not treat __doc__ as a constant
        # and falls through to NotImplementedError.
        vt = ConstDictVariable({})
        self.assertIsNone(vt.tp_doc)
        with self.assertRaises(NotImplementedError):
            vt.const_getattr(None, "__doc__")

    def test_tp_doc_backs_dunder_doc(self):
        # A VT that declares tp_doc answers __doc__ from it (CPython: an
        # instance's __doc__ resolves to its type's tp_doc).
        class DocDictVariable(ConstDictVariable):
            tp_doc = "my docstring"

        self.assertEqual(
            DocDictVariable({}).const_getattr(None, "__doc__"), "my docstring"
        )


instantiate_parametrized_tests(VariableTrackerTpNameDocTests)


if __name__ == "__main__":
    run_tests()
