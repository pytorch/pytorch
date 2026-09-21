# Owner(s): ["module: dynamo"]

"""
Comprehensive tests for len() builtin and __len__() method protocol in PyTorch Dynamo.

Tests cover:
- len(obj) builtin calls
- obj.__len__() method calls
- type(obj).__len__(obj) unbound method calls (marked as expectedFailure - Dynamo limitation)
- Various container types: list, tuple, dict, set, frozenset, range, str, Tensor, nn.Module
- Dict views: keys(), values(), items()
- User-defined classes with __len__
"""

import collections
import dataclasses
import types

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


class _BaseSequenceLen:
    """Base class for testing len() on sequence types (list, tuple)"""

    thetype = None  # Override in subclass

    def setUp(self):
        if self.thetype is None:
            self.skipTest("Base class - not meant to be run directly")
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_basic(self):
        seq = self.thetype([1, 2, 3])
        self.assertEqual(len(seq), 3)
        self.assertEqual(seq.__len__(), 3)

    @make_dynamo_test
    def test_len_empty(self):
        seq = self.thetype([])
        self.assertEqual(len(seq), 0)
        self.assertEqual(seq.__len__(), 0)

    @make_dynamo_test
    def test_len_single_element(self):
        seq = self.thetype([42])
        self.assertEqual(len(seq), 1)
        self.assertEqual(seq.__len__(), 1)

    @make_dynamo_test
    def test_len_nested(self):
        inner = self.thetype([1, 2])
        seq = self.thetype([inner, inner, inner])
        self.assertEqual(len(seq), 3)
        self.assertEqual(seq.__len__(), 3)

    @make_dynamo_test
    def test_len_with_tensors(self):
        seq = self.thetype([torch.tensor(1), torch.tensor(2), torch.tensor(3)])
        self.assertEqual(len(seq), 3)
        self.assertEqual(seq.__len__(), 3)

    @make_dynamo_test
    def test_len_with_mixed_types(self):
        seq = self.thetype([1, "hello", 3.14, torch.tensor(4)])
        self.assertEqual(len(seq), 4)
        self.assertEqual(seq.__len__(), 4)

    @make_dynamo_test
    def test_len_large(self):
        seq = self.thetype(range(100))
        self.assertEqual(len(seq), 100)
        self.assertEqual(seq.__len__(), 100)


class TestListLen(_BaseSequenceLen, torch._dynamo.test_case.TestCase):
    """Tests for len() on list objects"""

    thetype = list


class TestTupleLen(_BaseSequenceLen, torch._dynamo.test_case.TestCase):
    """Tests for len() on tuple objects"""

    thetype = tuple


class _BaseMappingLen:
    """Base class for testing len() on mapping types (dict, OrderedDict)"""

    def get_mapping(self, items):
        """Override in subclass to return appropriate mapping type"""
        return dict(items)

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_basic(self):
        d = self.get_mapping({1: "a", 2: "b", 3: "c"}.items())
        self.assertEqual(len(d), 3)
        self.assertEqual(d.__len__(), 3)

    @make_dynamo_test
    def test_len_empty(self):
        d = self.get_mapping({}.items())
        self.assertEqual(len(d), 0)
        self.assertEqual(d.__len__(), 0)

    @make_dynamo_test
    def test_len_single_entry(self):
        d = self.get_mapping({"key": "value"}.items())
        self.assertEqual(len(d), 1)
        self.assertEqual(d.__len__(), 1)

    @make_dynamo_test
    def test_len_string_keys(self):
        d = self.get_mapping({"one": 1, "two": 2, "three": 3}.items())
        self.assertEqual(len(d), 3)
        self.assertEqual(d.__len__(), 3)

    @make_dynamo_test
    def test_len_int_keys(self):
        d = self.get_mapping({1: "one", 2: "two", 3: "three", 4: "four"}.items())
        self.assertEqual(len(d), 4)
        self.assertEqual(d.__len__(), 4)

    @make_dynamo_test
    def test_len_with_tensor_values(self):
        d = self.get_mapping(
            {"a": torch.tensor(1), "b": torch.tensor(2), "c": torch.tensor(3)}.items()
        )
        self.assertEqual(len(d), 3)
        self.assertEqual(d.__len__(), 3)

    @make_dynamo_test
    def test_len_large(self):
        d = self.get_mapping({i: i * 2 for i in range(20)}.items())
        self.assertEqual(len(d), 20)
        self.assertEqual(d.__len__(), 20)


class TestDictLen(_BaseMappingLen, torch._dynamo.test_case.TestCase):
    """Tests for len() on dict objects"""


class TestOrderedDictLen(_BaseMappingLen, torch._dynamo.test_case.TestCase):
    """Tests for len() on OrderedDict objects"""

    def get_mapping(self, items):
        return collections.OrderedDict(items)


class TestDefaultDictLen(_BaseMappingLen, torch._dynamo.test_case.TestCase):
    """Tests for len() on defaultdict objects"""

    def get_mapping(self, items):
        d = collections.defaultdict(int)
        for k, v in items:
            d[k] = v
        return d


class _BaseSetLen:
    """Base class for testing len() on set types"""

    __test__ = False  # Prevent pytest from collecting this as a test class

    def get_set(self, items):
        """Override in subclass to return appropriate set type"""
        return set(items)

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_basic(self):
        s = self.get_set([1, 2, 3])
        self.assertEqual(len(s), 3)
        self.assertEqual(s.__len__(), 3)

    @make_dynamo_test
    def test_len_empty(self):
        s = self.get_set([])
        self.assertEqual(len(s), 0)
        self.assertEqual(s.__len__(), 0)

    @make_dynamo_test
    def test_len_single_element(self):
        s = self.get_set([42])
        self.assertEqual(len(s), 1)
        self.assertEqual(s.__len__(), 1)

    @make_dynamo_test
    def test_len_with_strings(self):
        s = self.get_set(["a", "b", "c", "d"])
        self.assertEqual(len(s), 4)
        self.assertEqual(s.__len__(), 4)

    @make_dynamo_test
    def test_len_with_duplicates(self):
        # Set constructor deduplicates
        s = self.get_set([1, 2, 2, 3, 3, 3])
        self.assertEqual(len(s), 3)
        self.assertEqual(s.__len__(), 3)

    @make_dynamo_test
    def test_len_large(self):
        s = self.get_set(range(50))
        self.assertEqual(len(s), 50)
        self.assertEqual(s.__len__(), 50)


class TestSetLen(_BaseSetLen, torch._dynamo.test_case.TestCase):
    """Tests for len() on set objects"""

    def get_set(self, items):
        return set(items)


class TestFrozenSetLen(_BaseSetLen, torch._dynamo.test_case.TestCase):
    """Tests for len() on frozenset objects"""

    def get_set(self, items):
        return frozenset(items)


class TestRangeLen(torch._dynamo.test_case.TestCase):
    """Tests for len() on range objects"""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_basic(self):
        r = range(5)
        self.assertEqual(len(r), 5)
        self.assertEqual(r.__len__(), 5)
        self.assertEqual(range.__len__(r), 5)

    @make_dynamo_test
    def test_len_with_start_stop(self):
        r = range(5, 15)
        self.assertEqual(len(r), 10)
        self.assertEqual(r.__len__(), 10)
        self.assertEqual(range.__len__(r), 10)

    @make_dynamo_test
    def test_len_with_step(self):
        r = range(0, 10, 2)
        self.assertEqual(len(r), 5)
        self.assertEqual(r.__len__(), 5)
        self.assertEqual(range.__len__(r), 5)

    @make_dynamo_test
    def test_len_negative_step(self):
        r = range(10, 0, -1)
        self.assertEqual(len(r), 10)
        self.assertEqual(r.__len__(), 10)
        self.assertEqual(range.__len__(r), 10)

    @make_dynamo_test
    def test_len_empty(self):
        r = range(5, 5)
        self.assertEqual(len(r), 0)
        self.assertEqual(r.__len__(), 0)
        self.assertEqual(range.__len__(r), 0)


class TestStringLen(torch._dynamo.test_case.TestCase):
    """Tests for len() on string objects"""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_empty(self):
        s = ""
        self.assertEqual(len(s), 0)
        self.assertEqual(s.__len__(), 0)
        self.assertEqual(str.__len__(s), 0)

    @make_dynamo_test
    def test_len_single_char(self):
        s = "a"
        self.assertEqual(len(s), 1)
        self.assertEqual(s.__len__(), 1)
        self.assertEqual(str.__len__(s), 1)

    @make_dynamo_test
    def test_len_multiple_chars(self):
        s = "hello"
        self.assertEqual(len(s), 5)
        self.assertEqual(s.__len__(), 5)
        self.assertEqual(str.__len__(s), 5)

    @make_dynamo_test
    def test_len_with_spaces(self):
        s = "hello world"
        self.assertEqual(len(s), 11)
        self.assertEqual(s.__len__(), 11)
        self.assertEqual(str.__len__(s), 11)


class TestTensorLen(torch._dynamo.test_case.TestCase):
    """Tests for len() on torch.Tensor objects"""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_1d(self):
        t = torch.tensor([1, 2, 3, 4, 5])
        self.assertEqual(len(t), 5)
        self.assertEqual(t.__len__(), 5)
        self.assertEqual(torch.Tensor.__len__(t), 5)

    @make_dynamo_test
    def test_len_2d(self):
        t = torch.tensor([[1, 2], [3, 4], [5, 6]])
        self.assertEqual(len(t), 3)
        self.assertEqual(t.__len__(), 3)
        self.assertEqual(torch.Tensor.__len__(t), 3)

    @make_dynamo_test
    def test_len_3d(self):
        t = torch.randn(4, 5, 6)
        self.assertEqual(len(t), 4)
        self.assertEqual(t.__len__(), 4)
        self.assertEqual(torch.Tensor.__len__(t), 4)

    @make_dynamo_test
    def test_len_empty(self):
        t = torch.tensor([])
        self.assertEqual(len(t), 0)
        self.assertEqual(t.__len__(), 0)
        self.assertEqual(torch.Tensor.__len__(t), 0)

    @make_dynamo_test
    def test_len_large_batch(self):
        t = torch.randn(100, 5, 5)
        self.assertEqual(len(t), 100)
        self.assertEqual(t.__len__(), 100)
        self.assertEqual(torch.Tensor.__len__(t), 100)

    @make_dynamo_test
    def test_len_different_dtypes(self):
        t = torch.tensor([1, 2, 3], dtype=torch.float32)
        self.assertEqual(len(t), 3)
        self.assertEqual(t.__len__(), 3)
        self.assertEqual(torch.Tensor.__len__(t), 3)


class TestNNModuleLen(torch._dynamo.test_case.TestCase):
    """Tests for len() on torch.nn module containers"""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

        # Pre-construct nn.Module instances outside compiled regions
        self.seq_3layers = torch.nn.Sequential(
            torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 5)
        )
        self.empty_seq = torch.nn.Sequential()
        self.ml_2modules = torch.nn.ModuleList(
            [torch.nn.Linear(10, 20), torch.nn.Linear(20, 30)]
        )
        self.empty_ml = torch.nn.ModuleList()
        self.md_2modules = torch.nn.ModuleDict(
            {"layer1": torch.nn.Linear(10, 20), "layer2": torch.nn.Linear(20, 5)}
        )
        self.empty_md = torch.nn.ModuleDict()
        self.seq_5layers = torch.nn.Sequential(
            *[torch.nn.Linear(10, 10) for _ in range(5)]
        )

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_sequential(self):
        seq = self.seq_3layers
        self.assertEqual(len(seq), 3)
        self.assertEqual(seq.__len__(), 3)

    @make_dynamo_test
    def test_len_empty_sequential(self):
        seq = self.empty_seq
        self.assertEqual(len(seq), 0)
        self.assertEqual(seq.__len__(), 0)

    @make_dynamo_test
    def test_len_module_list(self):
        ml = self.ml_2modules
        self.assertEqual(len(ml), 2)
        self.assertEqual(ml.__len__(), 2)

    @make_dynamo_test
    def test_len_empty_module_list(self):
        ml = self.empty_ml
        self.assertEqual(len(ml), 0)
        self.assertEqual(ml.__len__(), 0)

    @make_dynamo_test
    def test_len_module_dict(self):
        md = self.md_2modules
        self.assertEqual(len(md), 2)
        self.assertEqual(md.__len__(), 2)

    @make_dynamo_test
    def test_len_empty_module_dict(self):
        md = self.empty_md
        self.assertEqual(len(md), 0)
        self.assertEqual(md.__len__(), 0)

    @make_dynamo_test
    def test_len_sequential_with_multiple_layers(self):
        seq = self.seq_5layers
        self.assertEqual(len(seq), 5)
        self.assertEqual(seq.__len__(), 5)


class TestDictViewLen(torch._dynamo.test_case.TestCase):
    """Tests for len() on dict view objects (keys, values, items)"""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_dict_keys_view(self):
        d = {"a": 1, "b": 2, "c": 3}
        keys = d.keys()
        self.assertEqual(len(keys), 3)
        self.assertEqual(keys.__len__(), 3)

    @make_dynamo_test
    def test_len_dict_values_view(self):
        d = {"a": 1, "b": 2, "c": 3}
        values = d.values()
        self.assertEqual(len(values), 3)
        self.assertEqual(values.__len__(), 3)

    @make_dynamo_test
    def test_len_dict_items_view(self):
        d = {"x": 10, "y": 20, "z": 30}
        items = d.items()
        self.assertEqual(len(items), 3)
        self.assertEqual(items.__len__(), 3)

    @make_dynamo_test
    def test_len_dict_keys_empty(self):
        d = {}
        keys = d.keys()
        self.assertEqual(len(keys), 0)
        self.assertEqual(keys.__len__(), 0)

    @make_dynamo_test
    def test_len_dict_values_empty(self):
        d = {}
        values = d.values()
        self.assertEqual(len(values), 0)
        self.assertEqual(values.__len__(), 0)

    @make_dynamo_test
    def test_len_dict_items_empty(self):
        d = {}
        items = d.items()
        self.assertEqual(len(items), 0)
        self.assertEqual(items.__len__(), 0)

    @make_dynamo_test
    def test_len_dict_keys_single_entry(self):
        """Test len() on dict.keys() with single entry"""
        d = {"key": "value"}
        keys = d.keys()
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys.__len__(), 1)

    @make_dynamo_test
    def test_len_dict_values_single_entry(self):
        """Test len() on dict.values() with single entry"""
        d = {"key": "value"}
        values = d.values()
        self.assertEqual(len(values), 1)
        self.assertEqual(values.__len__(), 1)

    @make_dynamo_test
    def test_len_dict_items_single_entry(self):
        """Test len() on dict.items() with single entry"""
        d = {"key": "value"}
        items = d.items()
        self.assertEqual(len(items), 1)
        self.assertEqual(items.__len__(), 1)

    @make_dynamo_test
    def test_len_dict_keys_int_keys(self):
        """Test len() on dict.keys() with integer keys"""
        d = {1: "one", 2: "two", 3: "three", 4: "four"}
        keys = d.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys.__len__(), 4)

    @make_dynamo_test
    def test_len_dict_values_tensor_values(self):
        """Test len() on dict.values() with tensor values"""
        d = {"a": torch.tensor(1), "b": torch.tensor(2), "c": torch.tensor(3)}
        values = d.values()
        self.assertEqual(len(values), 3)
        self.assertEqual(values.__len__(), 3)

    @make_dynamo_test
    def test_len_dict_items_mixed_types(self):
        """Test len() on dict.items() with mixed key/value types"""
        d = {"str": "value", 42: torch.tensor(1), (1, 2): "tuple_key"}
        items = d.items()
        self.assertEqual(len(items), 3)
        self.assertEqual(items.__len__(), 3)

    @make_dynamo_test
    def test_len_dict_keys_large(self):
        """Test len() on dict.keys() with large number of entries"""
        d = {i: i * 2 for i in range(50)}
        keys = d.keys()
        self.assertEqual(len(keys), 50)
        self.assertEqual(keys.__len__(), 50)

    @make_dynamo_test
    def test_len_dict_values_large(self):
        """Test len() on dict.values() with large number of entries"""
        d = {i: i * 2 for i in range(50)}
        values = d.values()
        self.assertEqual(len(values), 50)
        self.assertEqual(values.__len__(), 50)

    @make_dynamo_test
    def test_len_dict_items_large(self):
        """Test len() on dict.items() with large number of entries"""
        d = {i: i * 2 for i in range(50)}
        items = d.items()
        self.assertEqual(len(items), 50)
        self.assertEqual(items.__len__(), 50)


# User-defined classes for TestUserDefinedLen
class CustomList:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)


class CustomContainer:
    def __len__(self):
        return 0


class Container:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


class FixedSize:
    def __len__(self):
        return 10


class ListWrapper:
    def __init__(self):
        self.data = [1, 2, 3, 4, 5]

    def __len__(self):
        return len(self.data)


class ListSubclassCustomLen(list):
    def __len__(self):
        return super().__len__() * 2


class CustomMapping:
    """A user-defined mapping (dict-like) class"""

    def __init__(self, data):
        self._data = dict(data) if not isinstance(data, dict) else data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        return self._data[key]

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()


class CustomMappingSubclass(CustomMapping):
    """A subclass of CustomMapping"""

    def __len__(self):
        # Custom len implementation (e.g., filtered length)
        return super().__len__() + 1


class SimpleDictLike:
    """Minimal dict-like class with just __len__"""

    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


class TupleSubclassCustomLen(tuple):
    __slots__ = ()

    def __len__(self):
        return super().__len__() + 1


class DictSubclassCustomLen(dict):
    def __len__(self):
        return super().__len__() - 1


class SetSubclassCustomLen(set):
    def __len__(self):
        return 0


class TestUserDefinedLen(torch._dynamo.test_case.TestCase):
    """Tests for len() on user-defined classes with __len__"""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_simple_custom_class(self):
        obj = CustomList([1, 2, 3])
        self.assertEqual(len(obj), 3)
        self.assertEqual(obj.__len__(), 3)
        self.assertEqual(CustomList.__len__(obj), 3)

    @make_dynamo_test
    def test_len_custom_class_empty(self):
        obj = CustomContainer()
        self.assertEqual(len(obj), 0)
        self.assertEqual(obj.__len__(), 0)
        self.assertEqual(CustomContainer.__len__(obj), 0)

    @make_dynamo_test
    def test_len_custom_class_with_properties(self):
        obj = Container(42)
        self.assertEqual(len(obj), 42)
        self.assertEqual(obj.__len__(), 42)
        self.assertEqual(Container.__len__(obj), 42)

    @make_dynamo_test
    def test_len_custom_class_constant_return(self):
        obj = FixedSize()
        self.assertEqual(len(obj), 10)
        self.assertEqual(obj.__len__(), 10)
        self.assertEqual(FixedSize.__len__(obj), 10)

    @make_dynamo_test
    def test_len_custom_class_with_list_attr(self):
        obj = ListWrapper()
        self.assertEqual(len(obj), 5)
        self.assertEqual(obj.__len__(), 5)
        self.assertEqual(ListWrapper.__len__(obj), 5)


class TestSubclassOverloadedLen(torch._dynamo.test_case.TestCase):
    """Tests for custom classes that inherit from builtins and overload __len__"""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_list_subclass_custom_len(self):
        obj = ListSubclassCustomLen([1, 2, 3])
        self.assertEqual(len(obj), 6)
        self.assertEqual(obj.__len__(), 6)
        self.assertEqual(ListSubclassCustomLen.__len__(obj), 6)

    @make_dynamo_test
    def test_tuple_subclass_custom_len(self):
        obj = TupleSubclassCustomLen([1, 2, 3])
        self.assertEqual(len(obj), 4)
        self.assertEqual(obj.__len__(), 4)
        self.assertEqual(TupleSubclassCustomLen.__len__(obj), 4)

    @make_dynamo_test
    def test_dict_subclass_custom_len(self):
        obj = DictSubclassCustomLen({"a": 1, "b": 2, "c": 3})
        self.assertEqual(len(obj), 2)
        self.assertEqual(obj.__len__(), 2)
        self.assertEqual(DictSubclassCustomLen.__len__(obj), 2)

    @make_dynamo_test
    def test_set_subclass_custom_len(self):
        obj = SetSubclassCustomLen([1, 2, 3])
        self.assertEqual(len(obj), 0)
        self.assertEqual(obj.__len__(), 0)
        self.assertEqual(SetSubclassCustomLen.__len__(obj), 0)


class DescriptorLenClass:
    """Test class with __len__ as a regular instance method"""

    def __len__(self):
        """Regular instance method __len__ - should return 40"""
        return 40


class PartialLenClass:
    """Class where __len__ is a lambda/callable object"""

    def __init__(self):
        self._items = [1, 2, 3, 4, 5]

    def __len__(self):
        return len(self._items)


class StaticMethodLenClass:
    """Test class where __len__ is a staticmethod (unusual, likely to fail)"""

    @staticmethod
    def __len__():
        """Staticmethod __len__ - unusual pattern"""
        return 10


class ClassMethodLenClass:
    """Test class where __len__ is a classmethod (unusual, likely to fail)"""

    @classmethod
    def __len__(cls):
        """Classmethod __len__ - unusual pattern"""
        return 20


class CustomDescriptorLenClass:
    """Test class where __len__ is a custom descriptor"""

    class CustomDescriptorLen:
        """Custom descriptor that implements __get__ method"""

        def __get__(self, obj, objtype=None):
            """Descriptor protocol: return a callable that returns len"""
            if obj is None:
                return self
            return lambda: 50

    __len__ = CustomDescriptorLen()


class TestUserDefinedMappingLen(torch._dynamo.test_case.TestCase):
    """Tests for len() on user-defined mapping (dict-like) classes"""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_custom_mapping_basic(self):
        """Test len() on a basic custom mapping class"""
        m = CustomMapping({"a": 1, "b": 2, "c": 3})
        self.assertEqual(len(m), 3)
        self.assertEqual(m.__len__(), 3)

    @make_dynamo_test
    def test_len_custom_mapping_empty(self):
        """Test len() on an empty custom mapping"""
        m = CustomMapping({})
        self.assertEqual(len(m), 0)
        self.assertEqual(m.__len__(), 0)

    @make_dynamo_test
    def test_len_custom_mapping_single_entry(self):
        """Test len() on a custom mapping with single entry"""
        m = CustomMapping({"key": "value"})
        self.assertEqual(len(m), 1)
        self.assertEqual(m.__len__(), 1)

    @make_dynamo_test
    def test_len_custom_mapping_string_keys(self):
        """Test len() on a custom mapping with string keys"""
        m = CustomMapping({"one": 1, "two": 2, "three": 3})
        self.assertEqual(len(m), 3)
        self.assertEqual(m.__len__(), 3)

    @make_dynamo_test
    def test_len_custom_mapping_int_keys(self):
        """Test len() on a custom mapping with int keys"""
        m = CustomMapping({1: "one", 2: "two", 3: "three", 4: "four"})
        self.assertEqual(len(m), 4)
        self.assertEqual(m.__len__(), 4)

    @make_dynamo_test
    def test_len_custom_mapping_with_tensor_values(self):
        """Test len() on a custom mapping with tensor values"""
        m = CustomMapping(
            {"a": torch.tensor(1), "b": torch.tensor(2), "c": torch.tensor(3)}
        )
        self.assertEqual(len(m), 3)
        self.assertEqual(m.__len__(), 3)

    @make_dynamo_test
    def test_len_custom_mapping_large(self):
        """Test len() on a large custom mapping"""
        m = CustomMapping({i: i * 2 for i in range(20)})
        self.assertEqual(len(m), 20)
        self.assertEqual(m.__len__(), 20)

    @make_dynamo_test
    def test_len_custom_mapping_subclass(self):
        """Test len() on a subclass of CustomMapping"""
        m = CustomMappingSubclass({"a": 1, "b": 2})
        # CustomMappingSubclass.__len__ returns len + 1
        self.assertEqual(len(m), 3)
        self.assertEqual(m.__len__(), 3)

    @make_dynamo_test
    def test_len_simple_dict_like(self):
        """Test len() on a minimal dict-like class"""
        m = SimpleDictLike(42)
        self.assertEqual(len(m), 42)
        self.assertEqual(m.__len__(), 42)


class TestDescriptorLenImpl(torch._dynamo.test_case.TestCase):
    """Test that len_impl handles descriptor-based __len__ correctly"""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_regular_instance_len(self):
        """Test regular instance method __len__"""
        obj = DescriptorLenClass()

        # Regular instance methods should work fine
        self.assertEqual(len(obj), 40)
        self.assertEqual(obj.__len__(), 40)

    @make_dynamo_test
    def test_partial_callable_len(self):
        """Test lambda/callable as __len__"""
        obj = PartialLenClass()

        # Callable __len__ should work
        self.assertEqual(len(obj), 5)
        self.assertEqual(obj.__len__(), 5)

    @make_dynamo_test
    def test_staticmethod_len_works(self):
        """Test that staticmethod as __len__ actually works (unusual but supported)

        Staticmethods are descriptors that don't bind. CPython resolves the
        descriptor and calls the underlying function without passing self.
        """
        obj = StaticMethodLenClass()

        # Surprisingly, CPython's descriptor protocol makes this work
        # staticmethod.__get__ returns the unwrapped function
        self.assertEqual(len(obj), 10)

    @make_dynamo_test
    def test_classmethod_len_works(self):
        """Test that classmethod as __len__ actually works (unusual but supported)

        Classmethods are descriptors that bind to the class. CPython's
        descriptor protocol handles this and passes the class instead of instance.
        """
        obj = ClassMethodLenClass()

        # Surprisingly, CPython's descriptor protocol makes this work
        # classmethod.__get__ returns a bound method with the class
        self.assertEqual(len(obj), 20)

    @make_dynamo_test
    def test_custom_descriptor_len(self):
        """Test custom descriptor with __get__ method as __len__

        Custom descriptors implement the descriptor protocol via __get__.
        The descriptor is resolved and the returned callable is used as __len__.
        """
        obj = CustomDescriptorLenClass()

        # Custom descriptor's __get__ returns a callable that returns 50
        self.assertEqual(len(obj), 50)


class TestRaisesTypeError(torch._dynamo.test_case.TestCase):
    """Tests for types that don't support len() - should raise TypeError like Python"""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_slice_raises_type_error(self):
        """slice objects do not support len() - should raise TypeError"""
        s = slice(1, 5)
        with self.assertRaises(TypeError):
            len(s)

    @make_dynamo_test
    def test_len_slice_with_step_raises_type_error(self):
        """slice with step also raises TypeError"""
        s = slice(0, 10, 2)
        with self.assertRaises(TypeError):
            len(s)

    @make_dynamo_test
    def test_len_list_iterator_raises_type_error(self):
        """list iterator does not support len() - should raise TypeError"""
        it = iter([1, 2, 3])
        with self.assertRaises(TypeError):
            len(it)

    @make_dynamo_test
    def test_len_empty_list_iterator_raises_type_error(self):
        """empty list iterator also raises TypeError"""
        it = iter([])
        with self.assertRaises(TypeError):
            len(it)

    @make_dynamo_test
    def test_len_tuple_iterator_raises_type_error(self):
        """tuple iterator does not support len() - should raise TypeError"""
        it = iter((1, 2, 3))
        with self.assertRaises(TypeError):
            len(it)

    @make_dynamo_test
    def test_len_range_iterator_raises_type_error(self):
        """range iterator does not support len() - should raise TypeError"""
        it = iter(range(5))
        with self.assertRaises(TypeError):
            len(it)

    @make_dynamo_test
    def test_len_dict_iterator_raises_type_error(self):
        """dict iterator (keys) does not support len() - should raise TypeError"""
        d = {"a": 1, "b": 2}
        it = iter(d)
        with self.assertRaises(TypeError):
            len(it)


class TestDequeLen(torch._dynamo.test_case.TestCase):
    """Tests for len() on collections.deque objects"""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_basic(self):
        d = collections.deque([1, 2, 3])
        self.assertEqual(len(d), 3)
        self.assertEqual(d.__len__(), 3)

    @make_dynamo_test
    def test_len_empty(self):
        d = collections.deque([])
        self.assertEqual(len(d), 0)
        self.assertEqual(d.__len__(), 0)

    @make_dynamo_test
    def test_len_single_element(self):
        d = collections.deque([42])
        self.assertEqual(len(d), 1)
        self.assertEqual(d.__len__(), 1)

    @make_dynamo_test
    def test_len_with_maxlen(self):
        d = collections.deque([1, 2, 3, 4, 5], maxlen=3)
        self.assertEqual(len(d), 3)
        self.assertEqual(d.__len__(), 3)

    @make_dynamo_test
    def test_len_with_strings(self):
        d = collections.deque(["a", "b", "c", "d"])
        self.assertEqual(len(d), 4)
        self.assertEqual(d.__len__(), 4)

    @make_dynamo_test
    def test_len_large(self):
        d = collections.deque(range(50))
        self.assertEqual(len(d), 50)
        self.assertEqual(d.__len__(), 50)


class TestMappingProxyLen(torch._dynamo.test_case.TestCase):
    """Tests for len() on types.MappingProxyType objects"""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_basic(self):
        """Test len() on a basic MappingProxyType"""
        d = types.MappingProxyType({"a": 1, "b": 2, "c": 3})
        self.assertEqual(len(d), 3)
        self.assertEqual(d.__len__(), 3)

    @make_dynamo_test
    def test_len_empty(self):
        """Test len() on an empty MappingProxyType"""
        d = types.MappingProxyType({})
        self.assertEqual(len(d), 0)
        self.assertEqual(d.__len__(), 0)

    @make_dynamo_test
    def test_len_single_entry(self):
        """Test len() on a MappingProxyType with single entry"""
        d = types.MappingProxyType({"key": "value"})
        self.assertEqual(len(d), 1)
        self.assertEqual(d.__len__(), 1)

    @make_dynamo_test
    def test_len_string_keys(self):
        """Test len() on a MappingProxyType with string keys"""
        d = types.MappingProxyType({"one": 1, "two": 2, "three": 3})
        self.assertEqual(len(d), 3)
        self.assertEqual(d.__len__(), 3)

    @make_dynamo_test
    def test_len_int_keys(self):
        """Test len() on a MappingProxyType with int keys"""
        d = types.MappingProxyType({1: "one", 2: "two", 3: "three", 4: "four"})
        self.assertEqual(len(d), 4)
        self.assertEqual(d.__len__(), 4)

    @make_dynamo_test
    def test_len_with_tensor_values(self):
        """Test len() on a MappingProxyType with tensor values"""
        d = types.MappingProxyType(
            {"a": torch.tensor(1), "b": torch.tensor(2), "c": torch.tensor(3)}
        )
        self.assertEqual(len(d), 3)
        self.assertEqual(d.__len__(), 3)

    @make_dynamo_test
    def test_len_large(self):
        """Test len() on a large MappingProxyType"""
        d = types.MappingProxyType({i: i * 2 for i in range(20)})
        self.assertEqual(len(d), 20)
        self.assertEqual(d.__len__(), 20)


class MetaclassWithLen(type):
    """A metaclass that defines __len__ on the class itself"""

    def __len__(cls):
        """Return the number of items defined in the metaclass"""
        return 5


class SimpleMetaclassClass(metaclass=MetaclassWithLen):
    """A class using the MetaclassWithLen metaclass"""


class TestMetaclassLen(torch._dynamo.test_case.TestCase):
    """Tests for len() on metaclasses, classmethods, staticmethods, and properties"""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_metaclass_len_basic(self):
        """Test len() on a class with __len__ defined in metaclass"""
        self.assertEqual(len(SimpleMetaclassClass), 5)

    @make_dynamo_test
    def test_metaclass_len_direct_call(self):
        """Test direct call to __len__() on a class with metaclass-defined __len__"""
        self.assertEqual(SimpleMetaclassClass.__len__(), 5)


class CustomMutableMapping(collections.abc.MutableMapping):
    """Custom mutable mapping implementation with __len__."""

    def __init__(self, data=None):
        self._data = data if data is not None else {}

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)


@dataclasses.dataclass(frozen=True)
class FrozenPoint:
    """Frozen dataclass with __len__ method."""

    x: float
    y: float
    z: float

    def __len__(self):
        return 3


@dataclasses.dataclass(frozen=True)
class FrozenData:
    """Frozen dataclass with __len__ based on items."""

    items: tuple

    def __len__(self):
        return len(self.items)


class TestMutableMappingLen(torch._dynamo.test_case.TestCase):
    """Tests for len() on mutable mapping types."""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_custom_mutable_mapping(self):
        """Test len() on custom mutable mapping."""
        m = CustomMutableMapping({"x": 10, "y": 20, "z": 30})
        self.assertEqual(len(m), 3)
        self.assertEqual(m.__len__(), 3)

    @make_dynamo_test
    def test_len_custom_mutable_mapping_empty(self):
        """Test len() on empty custom mutable mapping."""
        m = CustomMutableMapping()
        self.assertEqual(len(m), 0)


class TestFrozenDataclassLen(torch._dynamo.test_case.TestCase):
    """Tests for len() on frozen dataclasses."""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_len_frozen_dataclass_with_len(self):
        """Test len() on frozen dataclass with custom __len__."""
        p = FrozenPoint(1.0, 2.0, 3.0)
        self.assertEqual(len(p), 3)
        self.assertEqual(p.__len__(), 3)

    @make_dynamo_test
    def test_len_frozen_dataclass_via_tuple(self):
        """Test len() on frozen dataclass with __len__ based on contained data."""
        obj = FrozenData((1, 2, 3, 4))
        self.assertEqual(len(obj), 4)
        self.assertEqual(obj.__len__(), 4)

    @make_dynamo_test
    def test_len_frozen_dataclass_consistency(self):
        """Test that len() on frozen dataclass is consistent across multiple calls."""
        obj = FrozenData(("a", "b", "c"))
        # Call len() twice to ensure consistency
        len1 = len(obj)
        len2 = len(obj)
        self.assertEqual(len1, 3)
        self.assertEqual(len2, 3)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
