# Owner(s): ["module: dynamo"]

import enum
import types
import unittest

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


class CustomIterable:
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)


class CustomIterator:
    def __init__(self, max_val):
        self.max_val = max_val
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.max_val:
            raise StopIteration
        self.current += 1
        return self.current


class SequenceClass:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class CustomList(list):
    def __iter__(self):
        # Return elements in reverse order
        return iter(reversed(self))


class CustomDict(dict):
    def __iter__(self):
        # Return keys in reverse order
        return iter(sorted(self.keys(), reverse=True))


class CustomSet(set):
    def __iter__(self):
        # Return elements multiplied by 2
        for item in set.__iter__(self):
            yield item * 2


class CustomSetDefaultIter(set):
    def __init__(self, iterable) -> None:
        super().__init__([10, 20, 30])


class TestIterators(torch._dynamo.test_case.TestCase):
    """Test iterator support in Dynamo"""

    def setUp(self):
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev

    @make_dynamo_test
    def test_list_iteration(self):
        """Test iteration over list"""
        lst = [1, 2, 3, 4, 5]
        result = []
        for item in lst.__iter__():
            result.append(item * 2)
        self.assertEqual(result, [2, 4, 6, 8, 10])

    @make_dynamo_test
    def test_tuple_iteration(self):
        """Test iteration over tuple"""
        tup = (1, 2, 3, 4, 5)
        result = []
        for item in tup.__iter__():
            result.append(item + 10)
        self.assertEqual(result, [11, 12, 13, 14, 15])

    @make_dynamo_test
    def test_set_iteration(self):
        """Test iteration over set"""
        s = {1, 2, 3, 4, 5}
        result = []
        for item in s.__iter__():
            result.append(item * 2)
        self.assertEqual(sorted(result), [2, 4, 6, 8, 10])

    @make_dynamo_test
    def test_mappingproxy_iteration(self):
        m = types.MappingProxyType({"a": 1, "b": 2, "c": 3})
        result = []
        for key in m.__iter__():
            result.append(key)  # noqa: PERF402
        self.assertEqual(sorted(result), ["a", "b", "c"])

        result = []
        for key in iter(m):
            result.append(key)
        self.assertEqual(sorted(result), ["a", "b", "c"])

    @make_dynamo_test
    def test_dict_keys_iteration(self):
        """Test iteration over dict keys"""
        d = {"a": 1, "b": 2, "c": 3}
        result = []
        for key in d.__iter__():
            result.append(key)  # noqa: PERF402
        self.assertEqual(sorted(result), ["a", "b", "c"])

    @make_dynamo_test
    def test_dict_container_types_iteration(self):
        """Test iteration over dict and dict views"""
        d = {"a": 1, "b": 2, "c": 3}

        # Test dict keys
        result = []
        for key in d.__iter__():
            result.append(key)  # noqa: PERF402
        self.assertEqual(sorted(result), ["a", "b", "c"])

        # Test dict values
        result = []
        for val in d.values().__iter__():
            result.append(val * 2)
        self.assertEqual(sorted(result), [2, 4, 6])

        # Test dict items
        result = []
        for key, val in d.items().__iter__():
            result.append((key, val * 2))
        self.assertEqual(sorted(result), [("a", 2), ("b", 4), ("c", 6)])

    @make_dynamo_test
    def test_range_iteration(self):
        """Test iteration over range"""
        result = []
        for i in range(5):
            result.append(i * 3)
        self.assertEqual(result, [0, 3, 6, 9, 12])

    @make_dynamo_test
    def test_enumerate_iteration(self):
        """Test enumerate iterator, then with mutation"""
        lst = [10, 20, 30, 40]
        result = []
        # Explicitly call __iter__() on enumerate
        for idx, val in enumerate(lst).__iter__():
            result.append((idx, val * 2))
        self.assertEqual(result, [(0, 20), (1, 40), (2, 60), (3, 80)])

        # Test with mutation
        lst2 = [10, 20, 30]
        result2 = []
        count = 0
        for idx, val in enumerate(lst2).__iter__():
            result2.append((idx, val))
            count += 1  # noqa: SIM113
            if count == 1:
                lst2.append(40)
        self.assertTrue(any(val == 40 for _, val in result2))

    @make_dynamo_test
    def test_enumerate_with_start(self):
        """Test enumerate with start parameter"""
        lst = ["a", "b", "c"]
        result = []
        # Explicitly call __iter__() on enumerate with start
        for idx, val in enumerate(lst, start=1).__iter__():
            result.append((idx, val))
        self.assertEqual(result, [(1, "a"), (2, "b"), (3, "c")])

    @make_dynamo_test
    def test_zip_iteration(self):
        """Test zip iterator with two lists, then with mutation"""
        lst1 = [1, 2, 3]
        lst2 = [10, 20, 30]
        result = []
        # Explicitly call __iter__() on zip
        for a, b in zip(lst1, lst2).__iter__():
            result.append((a, b * 2))
        self.assertEqual(result, [(1, 20), (2, 40), (3, 60)])

        # Test with mutation
        lst1_mut = [1, 2, 3]
        lst2_mut = [10, 20, 30]
        result2 = []
        count = 0
        for a, b in zip(lst1_mut, lst2_mut).__iter__():
            result2.append((a, b))
            count += 1  # noqa: SIM113
            if count == 1:
                lst1_mut.append(4)
        self.assertTrue(len(result2) >= 3)

    @make_dynamo_test
    def test_zip_three_iterables(self):
        """Test zip with three iterables"""
        lst1 = [1, 2]
        lst2 = [10, 20]
        lst3 = [100, 200]
        result = []
        # Explicitly call __iter__() on zip
        for a, b, c in zip(lst1, lst2, lst3).__iter__():
            result.append(a + b + c)
        self.assertEqual(result, [111, 222])

    @make_dynamo_test
    def test_nested_iteration(self):
        """Test nested for loops with iterators"""
        lst1 = [1, 2, 3]
        lst2 = [10, 20]
        result = []
        # Explicitly call __iter__() on both lists
        for a in lst1.__iter__():
            for b in lst2.__iter__():
                result.append(a * b)
        self.assertEqual(result, [10, 20, 20, 40, 30, 60])

    @make_dynamo_test
    def test_break_in_iteration(self):
        """Test break statement in iteration"""
        lst = [1, 2, 3, 4, 5]
        result = []
        # Explicitly call __iter__()
        for item in lst.__iter__():
            if item > 3:
                break
            result.append(item)
        self.assertEqual(result, [1, 2, 3])

    @make_dynamo_test
    def test_continue_in_iteration(self):
        """Test continue statement in iteration"""
        lst = [1, 2, 3, 4, 5]
        result = []
        # Explicitly call iter()
        for item in lst.__iter__():
            if item % 2 == 0:
                continue
            result.append(item)
        self.assertEqual(result, [1, 3, 5])

    @make_dynamo_test
    def test_iterator_with_default_value(self):
        """Test iter() with callable and default value"""
        counter = {"i": 0}

        def get_next():
            counter["i"] += 1
            return counter["i"]

        result = []
        for val in iter(get_next, 3):  # iter(callable, sentinel) form requires iter()
            result.append(val)  # noqa: PERF402
        self.assertEqual(result, [1, 2])

    @make_dynamo_test
    def test_iter_idempotency(self):
        """Test that iter(iter(x)) is the same as iter(x)"""
        lst = [1, 2, 3]
        it1 = lst.__iter__()
        it2 = it1.__iter__()
        self.assertIs(it1, it2)

    @make_dynamo_test
    def test_multiple_independent_iterators(self):
        """Test multiple independent iterators on same container, then with mutation"""
        lst = [10, 20, 30, 40]
        it1 = lst.__iter__()
        it2 = lst.__iter__()

        result = []
        val1 = next(it1)
        val2 = next(it2)
        result.append((val1, val2))

        val1 = next(it1)
        val2 = next(it2)
        result.append((val1, val2))

        self.assertEqual(result, [(10, 10), (20, 20)])

        # Test with mutation
        lst_mut = [1, 2, 3]
        it1_mut = lst_mut.__iter__()
        it2_mut = lst_mut.__iter__()

        val1 = next(it1_mut)
        self.assertEqual(val1, 1)
        lst_mut.append(4)

        val1_next = next(it1_mut)
        val2_first = next(it2_mut)
        self.assertEqual(val1_next, 2)
        self.assertEqual(val2_first, 1)

        rest1 = list(it1_mut)
        rest2 = list(it2_mut)
        self.assertIn(4, rest1)
        self.assertIn(4, rest2)

    @make_dynamo_test
    def test_string_iteration(self):
        """Test iteration over string"""
        s = "hello"
        result = []
        for char in s.__iter__():
            result.append(char.upper())
        self.assertEqual("".join(result), "HELLO")

    @unittest.expectedFailure
    @make_dynamo_test
    def test_bytes_iteration(self):
        """Test iteration over bytes"""
        b = b"hello"
        result = []
        for byte_val in b.__iter__():
            result.append(byte_val * 2)
        # Each byte value is multiplied by 2
        expected = [
            ord("h") * 2,
            ord("e") * 2,
            ord("l") * 2,
            ord("l") * 2,
            ord("o") * 2,
        ]
        self.assertEqual(result, expected)

    @make_dynamo_test
    def test_comprehensions_with_iterator(self):
        """Test different comprehension types with iterators"""
        lst = [1, 2, 3, 4, 5]

        # List comprehension
        result = [x * 2 for x in lst.__iter__()]
        self.assertEqual(result, [2, 4, 6, 8, 10])

        # Dict comprehension
        result = {x: x * 2 for x in lst.__iter__()}
        self.assertEqual(result, {1: 2, 2: 4, 3: 6, 4: 8, 5: 10})

        # Set comprehension
        result = {x % 2 for x in lst.__iter__()}
        self.assertEqual(result, {0, 1})

    @make_dynamo_test
    def test_generator_expression(self):
        """Test generator expression (iterator)"""
        lst = [1, 2, 3, 4]
        # Generator expression implicitly calls iter()
        gen = (x * 2 for x in lst.__iter__())
        result = list(gen)
        self.assertEqual(result, [2, 4, 6, 8])

    @make_dynamo_test
    def test_filter_iterator(self):
        """Test filter built-in iterator"""
        lst = [1, 2, 3, 4, 5]
        # Explicitly call iter() on filter result
        filtered = filter(lambda x: x > 2, lst.__iter__())
        result = list(filtered)
        self.assertEqual(result, [3, 4, 5])

    @make_dynamo_test
    def test_map_iterator(self):
        """Test map built-in iterator"""
        lst = [1, 2, 3, 4]
        # Explicitly call iter() on map result
        mapped = map(lambda x: x * 3, lst.__iter__())  # noqa: C417
        result = list(mapped)
        self.assertEqual(result, [3, 6, 9, 12])

    @make_dynamo_test
    def test_reversed_iterator(self):
        """Test reversed iterator"""
        lst = [1, 2, 3, 4, 5]
        # Explicitly call __iter__() on reversed result
        rev = reversed(lst).__iter__()
        result = list(rev)
        self.assertEqual(result, [5, 4, 3, 2, 1])

    @make_dynamo_test
    def test_next_with_default(self):
        """Test next() with default value"""
        lst = [1, 2, 3]
        it = lst.__iter__()
        # Exhaust the iterator
        for _ in lst:
            next(it)
        # Try to get next with default
        result = next(it, "EMPTY")
        self.assertEqual(result, "EMPTY")

    # Corner case tests

    @make_dynamo_test
    def test_iterator_exhaustion(self):
        """Test that exhausted iterator raises StopIteration"""
        lst = [1, 2]
        it = lst.__iter__()
        next(it)
        next(it)
        # Next call on exhausted iterator should raise StopIteration
        with self.assertRaises(StopIteration):
            next(it)

    @make_dynamo_test
    def test_empty_and_single_item_containers(self):
        """Test iteration over empty and single-item containers"""
        # Empty containers
        self.assertEqual(list([].__iter__()), [])
        self.assertEqual(list({}.__iter__()), [])
        self.assertEqual(list(set().__iter__()), [])
        self.assertEqual(list(().__iter__()), [])

        # Single item containers
        self.assertEqual(list([42].__iter__()), [42])
        self.assertEqual(list((42,).__iter__()), [42])
        self.assertEqual(list({42}.__iter__()), [42])

    @make_dynamo_test
    def test_custom_iterator_class(self):
        """Test custom class implementing iterator protocol"""

        it = CustomIterator(3)
        result = []
        for val in iter(it):
            result.append(val)  # noqa: PERF402
        self.assertEqual(result, [1, 2, 3])

    @make_dynamo_test
    def test_custom_iterable_class(self):
        """Test custom class with __iter__ returning different object"""
        obj = CustomIterable([10, 20, 30])
        result = list(obj.__iter__())
        self.assertEqual(result, [10, 20, 30])

    @make_dynamo_test
    def test_iter_idempotency_callable(self):
        """Test iter(iter(x)) returns same iterator"""
        lst = [1, 2, 3]
        it = lst.__iter__()
        it2 = it.__iter__()
        # Both should be the same object
        self.assertIs(it, it2)

    @make_dynamo_test
    def test_zip_unequal_lengths(self):
        """Test zip stops at shortest iterable"""
        lst1 = [1, 2, 3, 4, 5]
        lst2 = [10, 20]
        result = list(zip(lst1, lst2.__iter__()))
        # Should stop at shortest (lst2)
        self.assertEqual(result, [(1, 10), (2, 20)])

    @make_dynamo_test
    def test_enumerate_with_large_start(self):
        """Test enumerate starting at a large index"""
        lst = ["a", "b"]
        result = list(enumerate(lst, start=100).__iter__())
        self.assertEqual(result, [(100, "a"), (101, "b")])

    @make_dynamo_test
    def test_for_else_with_break(self):
        """Test for/else: else does not run with break"""
        lst = [1, 2, 3, 4, 5]
        result = []
        else_executed = False
        for item in lst.__iter__():
            if item == 3:
                break
            result.append(item)
        else:
            else_executed = True
        self.assertEqual(result, [1, 2])
        self.assertFalse(else_executed)

    @make_dynamo_test
    def test_for_else_without_break(self):
        """Test for/else: else runs when loop completes naturally"""
        lst = [1, 2, 3]
        result = []
        else_executed = False
        for item in lst.__iter__():
            result.append(item)  # noqa: PERF402
        else:
            else_executed = True
        self.assertEqual(result, [1, 2, 3])
        self.assertTrue(else_executed)

    @make_dynamo_test
    def test_generator_function(self):
        """Test generator function as iterator"""

        def gen():
            yield 1
            yield 2
            yield 3

        result = list(gen().__iter__())
        self.assertEqual(result, [1, 2, 3])

    @make_dynamo_test
    def test_iterator_unpacking(self):
        """Test unpacking values from an iterator"""
        lst = [1, 2, 3]
        it = lst.__iter__()
        a, b, c = it
        self.assertEqual(a, 1)
        self.assertEqual(b, 2)
        self.assertEqual(c, 3)

    @make_dynamo_test
    def test_dict_value_mutation_allowed(self):
        """Test that modifying dict values during iteration is allowed"""
        d = {"a": 1, "b": 2, "c": 3}
        it = d.__iter__()
        first_key = next(it)
        # Modify a value (not adding/removing keys)
        d[first_key] = 99
        # Should be able to continue iteration
        rest = list(it)
        self.assertEqual(len(rest), 2)
        self.assertNotIn(first_key, rest)

    @make_dynamo_test
    def test_list_clear_during_iteration(self):
        """Test that clearing a list during iteration works"""
        lst = [1, 2, 3, 4, 5]
        it = lst.__iter__()
        first = next(it)
        self.assertEqual(first, 1)
        # Clear the list
        lst.clear()
        # Iteration should continue but see the cleared state
        rest = list(it)
        # After clear, remaining elements might be empty or implementation-dependent
        self.assertIsInstance(rest, list)

    @make_dynamo_test
    def test_range_iterator(self):
        """Test range object as iterator"""
        r = range(3, 8, 2)
        result = list(r.__iter__())
        self.assertEqual(result, [3, 5, 7])

    @make_dynamo_test
    def test_reversed_on_list(self):
        """Test reversed() returns iterator"""
        lst = [1, 2, 3]
        result = list(reversed(lst).__iter__())
        self.assertEqual(result, [3, 2, 1])

    @make_dynamo_test
    def test_iter_sequence_protocol(self):
        """Test iteration using sequence protocol (__getitem__)"""

        seq = SequenceClass([10, 20, 30])
        # Even without __iter__, should be iterable via __getitem__
        result = []
        for val in seq:
            result.append(val)  # noqa: PERF402
        self.assertEqual(result, [10, 20, 30])

    @make_dynamo_test
    def test_filter_empty_result(self):
        """Test filter that produces empty result"""
        lst = [1, 2, 3, 4, 5]
        filtered = filter(lambda x: x > 10, lst.__iter__())
        result = list(filtered)
        self.assertEqual(result, [])

    @make_dynamo_test
    def test_map_empty_input(self):
        """Test map on empty input"""
        lst = []
        mapped = map(lambda x: x * 2, lst.__iter__())  # noqa: C417
        result = list(mapped)
        self.assertEqual(result, [])

    @make_dynamo_test
    def test_nested_generators(self):
        """Test nested generator expressions"""
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = [x for row in matrix.__iter__() for x in row.__iter__()]
        self.assertEqual(result, [1, 2, 3, 4, 5, 6, 7, 8, 9])

    @make_dynamo_test
    def test_iter_two_arg_callable_sentinel(self):
        """Test iter(callable, sentinel) form"""
        counter = {"val": 0}

        def get_next():
            counter["val"] += 1
            return counter["val"]

        # Stop when we get 4
        result = list(iter(get_next, 4))  # iter(callable, sentinel) requires iter()
        self.assertEqual(result, [1, 2, 3])

    @make_dynamo_test
    def test_zip_with_empty(self):
        """Test zip with one empty iterable"""
        lst1 = [1, 2, 3]
        lst2 = []
        result = list(zip(lst1, lst2.__iter__()))
        self.assertEqual(result, [])

    @make_dynamo_test
    def test_dict_keys_view_iteration(self):
        """Test iteration over dict.keys() view"""
        d = {"a": 1, "b": 2, "c": 3}
        # Explicitly call __iter__() on dict.keys()
        keys = d.keys().__iter__()
        result = sorted(keys)
        self.assertEqual(result, ["a", "b", "c"])

    @make_dynamo_test
    def test_dict_values_view_iteration(self):
        """Test iteration over dict.values() view"""
        d = {"a": 1, "b": 2, "c": 3}
        # Explicitly call __iter__() on dict.values()
        values = d.values().__iter__()
        result = sorted(values)
        self.assertEqual(result, [1, 2, 3])

    @make_dynamo_test
    def test_dict_items_view_iteration(self):
        """Test iteration over dict.items() view"""
        d = {"a": 1, "b": 2, "c": 3}
        # Explicitly call __iter__() on dict.items()
        items = d.items().__iter__()
        result = sorted(items)
        self.assertEqual(result, [("a", 1), ("b", 2), ("c", 3)])

    @unittest.expectedFailure
    @make_dynamo_test
    def test_custom_list_subclass_with_custom_iter(self):
        """Test custom list subclass that overloads __iter__"""

        cl = CustomList([1, 2, 3])
        result = list(cl.__iter__())
        self.assertEqual(result, [3, 2, 1])

    @make_dynamo_test
    def test_custom_dict_subclass_with_custom_iter(self):
        """Test custom dict subclass that overloads __iter__"""

        cd = CustomDict({"a": 1, "b": 2, "c": 3})
        result = list(cd.__iter__())
        self.assertEqual(result, ["c", "b", "a"])

    @make_dynamo_test
    def test_custom_set_subclass_with_custom_iter(self):
        """Test custom set subclass that overloads __iter__"""

        cs = CustomSet([1, 2, 3])
        result = sorted(cs.__iter__())
        self.assertEqual(result, [2, 4, 6])

    @make_dynamo_test
    def test_custom_set_subclass_with_default_iter(self):
        """Test custom set subclass with default __iter__"""

        cs = CustomSetDefaultIter([1, 2, 3])
        result = sorted(iter(cs))
        self.assertEqual(result, [10, 20, 30])

    @make_dynamo_test
    def test_dict_keys_view_direct_iteration(self):
        """Test direct for loop over dict.keys() view"""
        d = {"x": 10, "y": 20, "z": 30}
        result = list(d)
        self.assertEqual(sorted(result), ["x", "y", "z"])

    @make_dynamo_test
    def test_dict_values_view_in_comprehension(self):
        """Test dict.values() in list comprehension"""
        d = {"a": 1, "b": 2, "c": 3}
        result = [v * 2 for v in d.values()]
        self.assertEqual(sorted(result), [2, 4, 6])

    @make_dynamo_test
    def test_dict_items_view_in_comprehension(self):
        """Test dict.items() in list comprehension"""
        d = {"a": 1, "b": 2, "c": 3}
        result = [(k, v * 2) for k, v in d.items()]
        self.assertEqual(sorted(result), [("a", 2), ("b", 4), ("c", 6)])


# Define custom subclasses outside of test functions
class CustomListWithReverseIter(list):
    """Custom list that iterates in reverse order"""

    def __iter__(self):
        # return iter(reversed(self))
        yield 1
        yield 2
        yield 3


class CustomTupleWithDoubleIter(tuple):
    """Custom tuple that yields each element twice"""

    __slots__ = ()

    def __iter__(self):
        for item in tuple.__iter__(self):
            yield item
            yield item


class CustomSetWithFilteredIter(set):
    """Custom set that only iterates even numbers"""

    def __iter__(self):
        for item in set.__iter__(self):
            if item % 2 == 0:
                yield item


class CustomSetWith123Iter(set):
    def __iter__(self):
        yield 1
        yield 2
        yield 3


class CustomDictWithKeyLengthIter(dict):
    """Custom dict that only iterates keys with length > 1"""

    def __iter__(self):
        for key in dict.__iter__(self):
            if len(str(key)) > 1:
                yield key


class CustomObjectWithCustomIter:
    def __iter__(self):
        yield 1
        yield 2
        yield 3


class GeneratorIterIterable:
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        yield from self.data


class TestCustomIteratorMethods(torch._dynamo.test_case.TestCase):
    """Test custom __iter__ implementations on user-defined subclasses"""

    def setUp(self):
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev

    @make_dynamo_test
    def test_custom_object_with_custom_iter(self):
        obj = CustomListWithReverseIter()
        result = list(obj.__iter__())
        self.assertEqual(result, [1, 2, 3])

    @make_dynamo_test
    def test_custom_list_with_reverse_iter(self):
        lst = CustomListWithReverseIter([1, 2, 3, 4, 5])
        result = list(lst.__iter__())
        self.assertEqual(result, [1, 2, 3])

    @make_dynamo_test
    def test_custom_tuple_with_double_iter(self):
        """Test custom tuple that yields each element twice"""
        tup = CustomTupleWithDoubleIter([10, 20, 30])
        result = list(tup.__iter__())
        self.assertEqual(result, [10, 10, 20, 20, 30, 30])

    @make_dynamo_test
    def test_custom_set_with_filtered_iter(self):
        """Test custom set that only iterates even numbers"""
        s = CustomSetWithFilteredIter([1, 2, 3, 4, 5, 6])
        result = sorted(s.__iter__())
        self.assertEqual(result, [2, 4, 6])

    @make_dynamo_test
    def test_custom_set_with_123_iter(self):
        """Test custom set that only iterates even numbers"""
        s = CustomSetWith123Iter([1, 2, 3, 4, 5, 6])
        result = sorted(s.__iter__())
        self.assertEqual(result, [1, 2, 3])

    @make_dynamo_test
    def test_custom_dict_with_key_length_iter(self):
        """Test custom dict that filters keys by length"""
        d = CustomDictWithKeyLengthIter({"a": 1, "ab": 2, "abc": 3, "b": 4})
        result = sorted(d.__iter__())
        self.assertEqual(result, ["ab", "abc"])

    @make_dynamo_test
    def test_custom_object_yield_from(self):
        """Test custom object whose __iter__ uses yield from self.data"""
        obj = GeneratorIterIterable([10, 20, 30])
        result = list(iter(obj))
        self.assertEqual(result, [10, 20, 30])


class TestIteratorMutationSemantics(torch._dynamo.test_case.TestCase):
    """Test iterator behavior when containers are mutated during iteration.

    These tests explore whether Dynamo preserves CPython's iterator semantics
    when the underlying container is modified.
    """

    def setUp(self):
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev

    @make_dynamo_test
    def test_dict_mutation_during_iteration(self):
        """Dict mutation during iteration

        In CPython, mutating a dict during iteration raises RuntimeError.
        Dynamo currently allows mutations (known limitation).
        """
        d = {"a": 1, "b": 2}
        it = d.__iter__()

        first = next(it)
        self.assertIn(first, ["a", "b"])

        d["c"] = 3

        try:
            rest = list(it)
            return rest
        except RuntimeError:
            pass

    @make_dynamo_test
    def test_set_mutation_during_iteration(self):
        """Set mutation during iteration

        In CPython, mutating a set during iteration raises RuntimeError.
        Dynamo currently allows mutations (known limitation).
        """
        s = {1, 2, 3}
        it = s.__iter__()

        first = next(it)
        self.assertIn(first, {1, 2, 3})

        s.add(4)

        try:
            rest = list(it)
            return rest
        except RuntimeError:
            pass

    @make_dynamo_test
    def test_list_insert_before_iterator_index(self):
        """List iterator uses absolute indices (CPython behavior)"""
        lst = [1, 2, 3]
        it = lst.__iter__()

        val1 = next(it)
        self.assertEqual(val1, 1)

        lst.insert(0, 0)

        val2 = next(it)
        self.assertEqual(val2, 1)

    @make_dynamo_test
    def test_list_remove_before_iterator_index(self):
        """List iterator uses absolute indices with removals"""
        lst = [1, 2, 3, 4, 5]
        it = lst.__iter__()

        val1 = next(it)
        val2 = next(it)
        self.assertEqual(val1, 1)
        self.assertEqual(val2, 2)

        lst.pop(0)

        val3 = next(it)
        self.assertEqual(val3, 4)

    @make_dynamo_test
    def test_multiple_iterators_shared_mutation(self):
        """Multiple iterators see shared mutations"""
        lst = [1, 2, 3]
        it1 = lst.__iter__()
        it2 = lst.__iter__()

        self.assertEqual(next(it1), 1)
        self.assertEqual(next(it2), 1)

        lst.append(4)

        self.assertEqual(next(it1), 2)
        self.assertEqual(next(it2), 2)

        self.assertEqual(next(it1), 3)
        self.assertEqual(next(it2), 3)

        self.assertEqual(next(it1), 4)
        self.assertEqual(next(it2), 4)


class Color(enum.Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


class Direction(enum.Enum):
    NORTH = "N"
    SOUTH = "S"
    EAST = "E"
    WEST = "W"


class Priority(enum.IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class TestEnumIteration(torch._dynamo.test_case.TestCase):
    """Test iteration over enum classes"""

    def setUp(self):
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev

    @make_dynamo_test
    def test_enum_iter(self):
        result = list(iter(Color))
        self.assertEqual(result, [Color.RED, Color.GREEN, Color.BLUE])

    @make_dynamo_test
    def test_enum_for_loop(self):
        result = list(Color)
        self.assertEqual(result, [Color.RED, Color.GREEN, Color.BLUE])

    @make_dynamo_test
    def test_enum_iter_values(self):
        result = [m.value for m in Color]
        self.assertEqual(result, [1, 2, 3])

    @make_dynamo_test
    def test_enum_iter_names(self):
        result = [m.name for m in Color]
        self.assertEqual(result, ["RED", "GREEN", "BLUE"])

    @make_dynamo_test
    def test_enum_string_values_iter(self):
        result = list(iter(Direction))
        self.assertEqual(
            result, [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]
        )

    @make_dynamo_test
    def test_int_enum_iter(self):
        result = list(iter(Priority))
        self.assertEqual(result, [Priority.LOW, Priority.MEDIUM, Priority.HIGH])

    @make_dynamo_test
    def test_enum_iter_unpacking(self):
        a, b, c = Color
        self.assertEqual(a, Color.RED)
        self.assertEqual(b, Color.GREEN)
        self.assertEqual(c, Color.BLUE)


class TestIterWithBuiltins(torch._dynamo.test_case.TestCase):
    """Test iter() with builtin iterators like zip, map, filter, reversed"""

    def setUp(self):
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev

    @make_dynamo_test
    def test_iter_of_zip(self):
        """Test iter(zip(a, b)) returns an iterator"""
        lst1 = [1, 2, 3]
        lst2 = [10, 20, 30]
        zip_obj = zip(lst1, lst2)
        it = iter(zip_obj)
        result = []
        for a, b in it:
            result.append((a, b))
        self.assertEqual(result, [(1, 10), (2, 20), (3, 30)])

    @make_dynamo_test
    def test_iter_of_map(self):
        """Test iter(map(...)) returns an iterator"""
        lst = [1, 2, 3, 4]
        map_obj = map(lambda x: x * 2, lst)  # noqa: C417
        it = iter(map_obj)
        result = list(it)
        self.assertEqual(result, [2, 4, 6, 8])

    @make_dynamo_test
    def test_iter_of_filter(self):
        """Test iter(filter(...)) returns an iterator"""
        lst = [1, 2, 3, 4, 5, 6]
        filter_obj = filter(lambda x: x > 3, lst)
        it = iter(filter_obj)
        result = list(it)
        self.assertEqual(result, [4, 5, 6])

    @make_dynamo_test
    def test_iter_of_reversed(self):
        """Test iter(reversed(list)) returns an iterator"""
        lst = [1, 2, 3, 4, 5]
        rev_obj = reversed(lst)
        it = iter(rev_obj)
        result = list(it)
        self.assertEqual(result, [5, 4, 3, 2, 1])

    @make_dynamo_test
    def test_iter_of_enumerate(self):
        """Test iter(enumerate(...)) returns an iterator"""
        lst = ["a", "b", "c"]
        enum_obj = enumerate(lst)
        it = iter(enum_obj)
        result = list(it)
        self.assertEqual(result, [(0, "a"), (1, "b"), (2, "c")])

    @make_dynamo_test
    def test_iter_of_zip_multiple(self):
        """Test iter(zip(...)) with multiple iterables"""
        a = [1, 2, 3]
        b = [10, 20, 30]
        c = [100, 200, 300]
        zip_obj = zip(a, b, c)
        it = iter(zip_obj)
        result = list(it)
        self.assertEqual(result, [(1, 10, 100), (2, 20, 200), (3, 30, 300)])

    @make_dynamo_test
    def test_iter_of_map_with_filter(self):
        """Test iter(map(...)) followed by filter"""
        lst = [1, 2, 3, 4, 5]
        mapped = map(lambda x: x * 10, lst)  # noqa: C417
        filtered = filter(lambda x: x > 20, mapped)
        it = iter(filtered)
        result = list(it)
        self.assertEqual(result, [30, 40, 50])

    @make_dynamo_test
    def test_iter_preserves_iterator_state(self):
        """Test that iter() on iterator returns the same iterator"""
        lst = [1, 2, 3, 4]
        map_obj = map(lambda x: x * 2, lst)  # noqa: C417
        it1 = iter(map_obj)
        it2 = iter(it1)
        # They should be the same object
        self.assertIs(it1, it2)
        # Advancing one should affect the other
        val1 = next(it1)
        self.assertEqual(val1, 2)
        val2 = next(it2)
        self.assertEqual(val2, 4)

    @make_dynamo_test
    def test_iter_idempotent_on_iterators(self):
        """Test that iter() called multiple times on iterator works"""
        lst = [1, 2, 3]
        it = iter(lst)
        it_again = iter(it)
        it_again_2 = iter(it_again)
        val1 = next(it_again_2)
        self.assertEqual(val1, 1)
        val2 = next(it)
        self.assertEqual(val2, 2)

    @make_dynamo_test
    def test_iter_of_zip_empty(self):
        """Test iter(zip(...)) with empty iterables"""
        lst1 = []
        lst2 = []
        zip_obj = zip(lst1, lst2)
        it = iter(zip_obj)
        result = list(it)
        self.assertEqual(result, [])

    @make_dynamo_test
    def test_iter_of_filter_empty_result(self):
        """Test iter(filter(...)) when filter returns empty"""
        lst = [1, 2, 3]
        filter_obj = filter(lambda x: x > 10, lst)
        it = iter(filter_obj)
        result = list(it)
        self.assertEqual(result, [])

    @make_dynamo_test
    def test_iter_of_map_with_zip(self):
        """Test iter(map(...)) with zip"""
        a = [1, 2, 3]
        b = [10, 20, 30]
        zip_obj = zip(a, b)
        mapped = map(lambda pair: pair[0] + pair[1], zip_obj)  # noqa: C417
        it = iter(mapped)
        result = list(it)
        self.assertEqual(result, [11, 22, 33])

    @make_dynamo_test
    def test_iter_exhaustion_with_builtin_iterators(self):
        """Test exhausting iter() of builtin iterators"""
        lst = [1, 2]
        map_obj = map(lambda x: x * 2, lst)  # noqa: C417
        it = iter(map_obj)
        next(it)
        next(it)
        with self.assertRaises(StopIteration):
            next(it)

    @make_dynamo_test
    def test_iter_next_with_default_on_builtin(self):
        """Test next() with default on iter() of builtin iterator"""
        lst = [1, 2]
        filter_obj = filter(lambda x: x > 1, lst)
        it = iter(filter_obj)
        val = next(it)
        self.assertEqual(val, 2)
        result = next(it, "EMPTY")
        self.assertEqual(result, "EMPTY")

    @make_dynamo_test
    def test_iter_of_zip_uneven_lengths(self):
        """Test iter(zip(...)) with uneven length iterables"""
        a = [1, 2, 3]
        b = [10, 20]
        zip_obj = zip(a, b)
        it = iter(zip_obj)
        result = list(it)
        self.assertEqual(result, [(1, 10), (2, 20)])

    @make_dynamo_test
    def test_iter_chaining_builtin_iterators(self):
        """Test chaining multiple iter() calls on builtin iterators"""
        lst = [1, 2, 3, 4, 5]
        # Create a chain of iterators: map -> filter -> iter
        mapped = map(lambda x: x * 2, lst)  # noqa: C417
        filtered = filter(lambda x: x > 4, mapped)
        it = iter(filtered)
        result = list(it)
        self.assertEqual(result, [6, 8, 10])

    @make_dynamo_test
    def test_iter_enumerate_with_start(self):
        """Test iter(enumerate(...)) with start parameter"""
        lst = ["a", "b", "c"]
        enum_obj = enumerate(lst, start=10)
        it = iter(enum_obj)
        result = list(it)
        self.assertEqual(result, [(10, "a"), (11, "b"), (12, "c")])

    @make_dynamo_test
    def test_iter_zip_with_filter_and_map(self):
        """Test iter() with nested builtin iterators"""
        a = [1, 2, 3, 4, 5]
        b = [10, 20, 30, 40, 50]
        zipped = zip(a, b)
        # Flatten and just test that iteration works
        it = iter(zipped)
        result = list(it)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0], (1, 10))


class NotIterable:
    pass


class BadIterReturnsNonIterator:
    def __iter__(self):
        return 42  # not an iterator


class IterWithoutNext:
    def __iter__(self):
        return self  # no __next__


class NoIterNoGetitem:
    pass


class TestIterErrors(torch._dynamo.test_case.TestCase):
    """Test that iter() raises the correct exceptions"""

    def setUp(self):
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev

    @make_dynamo_test
    def test_iter_non_iterable_raises_type_error(self):
        with self.assertRaises(TypeError):
            iter(42)

    @make_dynamo_test
    def test_iter_custom_non_iterable_raises_type_error(self):
        with self.assertRaises(TypeError):
            iter(NotIterable())

    @make_dynamo_test
    def test_iter_two_arg_non_callable_raises_type_error(self):
        with self.assertRaises(TypeError):
            iter(42, None)

    @unittest.expectedFailure
    @make_dynamo_test
    def test_iter_bad_iter_return_raises_type_error(self):
        with self.assertRaises(TypeError):
            iter(BadIterReturnsNonIterator())

    @unittest.expectedFailure
    @make_dynamo_test
    def test_iter_returns_self_without_next_raises_type_error(self):
        with self.assertRaises(TypeError):
            iter(IterWithoutNext())

    @make_dynamo_test
    def test_iter_no_iter_no_getitem_raises_type_error(self):
        with self.assertRaises(TypeError):
            iter(NoIterNoGetitem())

    @make_dynamo_test
    def test_next_on_exhausted_raises_stop_iteration(self):
        it = iter([])
        with self.assertRaises(StopIteration):
            next(it)

    @make_dynamo_test
    def test_next_default_on_exhausted(self):
        it = iter([])
        result = next(it, "default")
        self.assertEqual(result, "default")

    @make_dynamo_test
    def test_error_on_dict_keys_mutation_during_iteration(self):
        """Test that mutating a dict during iteration raises RuntimeError"""
        d = {"a": 1, "b": 2, "c": 3}
        it = iter(d.keys())
        next(it)  # Get first key
        # Mutate dict during iteration - should raise RuntimeError
        with self.assertRaises(RuntimeError):
            d["d"] = 4
            next(it)

    @make_dynamo_test
    def test_error_on_dict_values_mutation_during_iteration(self):
        """Test that mutating a dict during values() iteration raises RuntimeError"""
        d = {"a": 1, "b": 2, "c": 3}
        it = iter(d.values())
        next(it)  # Get first value
        # Mutate dict during iteration - should raise RuntimeError
        with self.assertRaises(RuntimeError):
            d["d"] = 4
            next(it)

    @make_dynamo_test
    def test_error_on_dict_items_mutation_during_iteration(self):
        """Test that mutating a dict during items() iteration raises RuntimeError"""
        d = {"a": 1, "b": 2, "c": 3}
        it = iter(d.items())
        next(it)  # Get first item
        # Mutate dict during iteration - should raise RuntimeError
        with self.assertRaises(RuntimeError):
            d["d"] = 4
            next(it)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
