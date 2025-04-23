# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import (
    TEST_WITH_TORCHDYNAMO,
    run_tests,
    xfailIfTorchDynamo,
)

if TEST_WITH_TORCHDYNAMO:
    __TestCase = CPythonTestCase
else:
    __TestCase = unittest.TestCase

# redirect import statements
import sys
import importlib.abc

redirect_imports = (
    "test.mapping_tests",
    "test.typinganndata",
    "test.test_grammar",
    "test.test_math",
    "test.test_iter",
    "test.typinganndata.ann_module",
)

class RedirectImportFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        # Check if the import is the problematic one
        if fullname in redirect_imports:
            try:
                # Attempt to import the standalone module
                name = fullname.removeprefix("test.")
                r = importlib.import_module(name)
                # Redirect the module in sys.modules
                sys.modules[fullname] = r
                # Return a module spec from the found module
                return importlib.util.find_spec(name)
            except ImportError:
                return None
        return None

# Add the custom finder to sys.meta_path
sys.meta_path.insert(0, RedirectImportFinder())


# ======= END DYNAMO PATCH =======

import collections.abc
import types
import unittest
from test.support import get_c_recursion_limit

class TestExceptionGroupTypeHierarchy(__TestCase):
    def test_exception_group_types(self):
        self.assertTrue(issubclass(ExceptionGroup, Exception))
        self.assertTrue(issubclass(ExceptionGroup, BaseExceptionGroup))
        self.assertTrue(issubclass(BaseExceptionGroup, BaseException))

    def test_exception_is_not_generic_type(self):
        with self.assertRaisesRegex(TypeError, 'Exception'):
            Exception[OSError]

    def test_exception_group_is_generic_type(self):
        E = OSError
        self.assertIsInstance(ExceptionGroup[E], types.GenericAlias)
        self.assertIsInstance(BaseExceptionGroup[E], types.GenericAlias)


class BadConstructorArgs(__TestCase):
    def test_bad_EG_construction__too_many_args(self):
        MSG = r'BaseExceptionGroup.__new__\(\) takes exactly 2 arguments'
        with self.assertRaisesRegex(TypeError, MSG):
            ExceptionGroup('no errors')
        with self.assertRaisesRegex(TypeError, MSG):
            ExceptionGroup([ValueError('no msg')])
        with self.assertRaisesRegex(TypeError, MSG):
            ExceptionGroup('eg', [ValueError('too')], [TypeError('many')])

    def test_bad_EG_construction__bad_message(self):
        MSG = 'argument 1 must be str, not '
        with self.assertRaisesRegex(TypeError, MSG):
            ExceptionGroup(ValueError(12), SyntaxError('bad syntax'))
        with self.assertRaisesRegex(TypeError, MSG):
            ExceptionGroup(None, [ValueError(12)])

    def test_bad_EG_construction__bad_excs_sequence(self):
        MSG = r'second argument \(exceptions\) must be a sequence'
        with self.assertRaisesRegex(TypeError, MSG):
            ExceptionGroup('errors not sequence', {ValueError(42)})
        with self.assertRaisesRegex(TypeError, MSG):
            ExceptionGroup("eg", None)

        MSG = r'second argument \(exceptions\) must be a non-empty sequence'
        with self.assertRaisesRegex(ValueError, MSG):
            ExceptionGroup("eg", [])

    def test_bad_EG_construction__nested_non_exceptions(self):
        MSG = (r'Item [0-9]+ of second argument \(exceptions\)'
              ' is not an exception')
        with self.assertRaisesRegex(ValueError, MSG):
            ExceptionGroup('expect instance, not type', [OSError]);
        with self.assertRaisesRegex(ValueError, MSG):
            ExceptionGroup('bad error', ["not an exception"])


class InstanceCreation(__TestCase):
    def test_EG_wraps_Exceptions__creates_EG(self):
        excs = [ValueError(1), TypeError(2)]
        self.assertIs(
            type(ExceptionGroup("eg", excs)),
            ExceptionGroup)

    def test_BEG_wraps_Exceptions__creates_EG(self):
        excs = [ValueError(1), TypeError(2)]
        self.assertIs(
            type(BaseExceptionGroup("beg", excs)),
            ExceptionGroup)

    def test_EG_wraps_BaseException__raises_TypeError(self):
        MSG= "Cannot nest BaseExceptions in an ExceptionGroup"
        with self.assertRaisesRegex(TypeError, MSG):
            eg = ExceptionGroup("eg", [ValueError(1), KeyboardInterrupt(2)])

    def test_BEG_wraps_BaseException__creates_BEG(self):
        beg = BaseExceptionGroup("beg", [ValueError(1), KeyboardInterrupt(2)])
        self.assertIs(type(beg), BaseExceptionGroup)

    def test_EG_subclass_wraps_non_base_exceptions(self):
        class MyEG(ExceptionGroup):
            pass

        self.assertIs(
            type(MyEG("eg", [ValueError(12), TypeError(42)])),
            MyEG)

    def test_EG_subclass_does_not_wrap_base_exceptions(self):
        class MyEG(ExceptionGroup):
            pass

        msg = "Cannot nest BaseExceptions in 'MyEG'"
        with self.assertRaisesRegex(TypeError, msg):
            MyEG("eg", [ValueError(12), KeyboardInterrupt(42)])

    def test_BEG_and_E_subclass_does_not_wrap_base_exceptions(self):
        class MyEG(BaseExceptionGroup, ValueError):
            pass

        msg = "Cannot nest BaseExceptions in 'MyEG'"
        with self.assertRaisesRegex(TypeError, msg):
            MyEG("eg", [ValueError(12), KeyboardInterrupt(42)])

    def test_EG_and_specific_subclass_can_wrap_any_nonbase_exception(self):
        class MyEG(ExceptionGroup, ValueError):
            pass

        # The restriction is specific to Exception, not "the other base class"
        MyEG("eg", [ValueError(12), Exception()])

    def test_BEG_and_specific_subclass_can_wrap_any_nonbase_exception(self):
        class MyEG(BaseExceptionGroup, ValueError):
            pass

        # The restriction is specific to Exception, not "the other base class"
        MyEG("eg", [ValueError(12), Exception()])


    def test_BEG_subclass_wraps_anything(self):
        class MyBEG(BaseExceptionGroup):
            pass

        self.assertIs(
            type(MyBEG("eg", [ValueError(12), TypeError(42)])),
            MyBEG)
        self.assertIs(
            type(MyBEG("eg", [ValueError(12), KeyboardInterrupt(42)])),
            MyBEG)


class StrAndReprTests(__TestCase):
    def test_ExceptionGroup(self):
        eg = BaseExceptionGroup(
            'flat', [ValueError(1), TypeError(2)])

        self.assertEqual(str(eg), "flat (2 sub-exceptions)")
        self.assertEqual(repr(eg),
            "ExceptionGroup('flat', [ValueError(1), TypeError(2)])")

        eg = BaseExceptionGroup(
            'nested', [eg, ValueError(1), eg, TypeError(2)])

        self.assertEqual(str(eg), "nested (4 sub-exceptions)")
        self.assertEqual(repr(eg),
            "ExceptionGroup('nested', "
                "[ExceptionGroup('flat', "
                    "[ValueError(1), TypeError(2)]), "
                 "ValueError(1), "
                 "ExceptionGroup('flat', "
                    "[ValueError(1), TypeError(2)]), TypeError(2)])")

    def test_BaseExceptionGroup(self):
        eg = BaseExceptionGroup(
            'flat', [ValueError(1), KeyboardInterrupt(2)])

        self.assertEqual(str(eg), "flat (2 sub-exceptions)")
        self.assertEqual(repr(eg),
            "BaseExceptionGroup("
                "'flat', "
                "[ValueError(1), KeyboardInterrupt(2)])")

        eg = BaseExceptionGroup(
            'nested', [eg, ValueError(1), eg])

        self.assertEqual(str(eg), "nested (3 sub-exceptions)")
        self.assertEqual(repr(eg),
            "BaseExceptionGroup('nested', "
                "[BaseExceptionGroup('flat', "
                    "[ValueError(1), KeyboardInterrupt(2)]), "
                "ValueError(1), "
                "BaseExceptionGroup('flat', "
                    "[ValueError(1), KeyboardInterrupt(2)])])")

    def test_custom_exception(self):
        class MyEG(ExceptionGroup):
            pass

        eg = MyEG(
            'flat', [ValueError(1), TypeError(2)])

        self.assertEqual(str(eg), "flat (2 sub-exceptions)")
        self.assertEqual(repr(eg), "MyEG('flat', [ValueError(1), TypeError(2)])")

        eg = MyEG(
            'nested', [eg, ValueError(1), eg, TypeError(2)])

        self.assertEqual(str(eg), "nested (4 sub-exceptions)")
        self.assertEqual(repr(eg), (
                 "MyEG('nested', "
                     "[MyEG('flat', [ValueError(1), TypeError(2)]), "
                      "ValueError(1), "
                      "MyEG('flat', [ValueError(1), TypeError(2)]), "
                      "TypeError(2)])"))


def create_simple_eg():
    excs = []
    try:
        try:
            raise MemoryError("context and cause for ValueError(1)")
        except MemoryError as e:
            raise ValueError(1) from e
    except ValueError as e:
        excs.append(e)

    try:
        try:
            raise OSError("context for TypeError")
        except OSError as e:
            raise TypeError(int)
    except TypeError as e:
        excs.append(e)

    try:
        try:
            raise ImportError("context for ValueError(2)")
        except ImportError as e:
            raise ValueError(2)
    except ValueError as e:
        excs.append(e)

    try:
        raise ExceptionGroup('simple eg', excs)
    except ExceptionGroup as e:
        return e


class ExceptionGroupFields(__TestCase):
    def test_basics_ExceptionGroup_fields(self):
        eg = create_simple_eg()

        # check msg
        self.assertEqual(eg.message, 'simple eg')
        self.assertEqual(eg.args[0], 'simple eg')

        # check cause and context
        self.assertIsInstance(eg.exceptions[0], ValueError)
        self.assertIsInstance(eg.exceptions[0].__cause__, MemoryError)
        self.assertIsInstance(eg.exceptions[0].__context__, MemoryError)
        self.assertIsInstance(eg.exceptions[1], TypeError)
        self.assertIsNone(eg.exceptions[1].__cause__)
        self.assertIsInstance(eg.exceptions[1].__context__, OSError)
        self.assertIsInstance(eg.exceptions[2], ValueError)
        self.assertIsNone(eg.exceptions[2].__cause__)
        self.assertIsInstance(eg.exceptions[2].__context__, ImportError)

        # check tracebacks
        line0 = create_simple_eg.__code__.co_firstlineno
        tb_linenos = [line0 + 27,
                      [line0 + 6, line0 + 14, line0 + 22]]
        self.assertEqual(eg.__traceback__.tb_lineno, tb_linenos[0])
        self.assertIsNone(eg.__traceback__.tb_next)
        for i in range(3):
            tb = eg.exceptions[i].__traceback__
            self.assertIsNone(tb.tb_next)
            self.assertEqual(tb.tb_lineno, tb_linenos[1][i])

    def test_fields_are_readonly(self):
        eg = ExceptionGroup('eg', [TypeError(1), OSError(2)])

        self.assertEqual(type(eg.exceptions), tuple)

        eg.message
        with self.assertRaises(AttributeError):
            eg.message = "new msg"

        eg.exceptions
        with self.assertRaises(AttributeError):
            eg.exceptions = [OSError('xyz')]


class ExceptionGroupTestBase(__TestCase):
    def assertMatchesTemplate(self, exc, exc_type, template):
        """ Assert that the exception matches the template

            A template describes the shape of exc. If exc is a
            leaf exception (i.e., not an exception group) then
            template is an exception instance that has the
            expected type and args value of exc. If exc is an
            exception group, then template is a list of the
            templates of its nested exceptions.
        """
        if exc_type is not None:
            self.assertIs(type(exc), exc_type)

        if isinstance(exc, BaseExceptionGroup):
            self.assertIsInstance(template, collections.abc.Sequence)
            self.assertEqual(len(exc.exceptions), len(template))
            for e, t in zip(exc.exceptions, template):
                self.assertMatchesTemplate(e, None, t)
        else:
            self.assertIsInstance(template, BaseException)
            self.assertEqual(type(exc), type(template))
            self.assertEqual(exc.args, template.args)

class Predicate:
    def __init__(self, func):
        self.func = func

    def __call__(self, e):
        return self.func(e)

    def method(self, e):
        return self.func(e)

class ExceptionGroupSubgroupTests(ExceptionGroupTestBase):
    def setUp(self):
        self.eg = create_simple_eg()
        self.eg_template = [ValueError(1), TypeError(int), ValueError(2)]
        super().setUp()

    def test_basics_subgroup_split__bad_arg_type(self):
        class C:
            pass

        bad_args = ["bad arg",
                    C,
                    OSError('instance not type'),
                    [OSError, TypeError],
                    (OSError, 42),
                   ]
        for arg in bad_args:
            with self.assertRaises(TypeError):
                self.eg.subgroup(arg)
            with self.assertRaises(TypeError):
                self.eg.split(arg)

    def test_basics_subgroup_by_type__passthrough(self):
        eg = self.eg
        self.assertIs(eg, eg.subgroup(BaseException))
        self.assertIs(eg, eg.subgroup(Exception))
        self.assertIs(eg, eg.subgroup(BaseExceptionGroup))
        self.assertIs(eg, eg.subgroup(ExceptionGroup))

    def test_basics_subgroup_by_type__no_match(self):
        self.assertIsNone(self.eg.subgroup(OSError))

    def test_basics_subgroup_by_type__match(self):
        eg = self.eg
        testcases = [
            # (match_type, result_template)
            (ValueError, [ValueError(1), ValueError(2)]),
            (TypeError, [TypeError(int)]),
            ((ValueError, TypeError), self.eg_template)]

        for match_type, template in testcases:
            with self.subTest(match=match_type):
                subeg = eg.subgroup(match_type)
                self.assertEqual(subeg.message, eg.message)
                self.assertMatchesTemplate(subeg, ExceptionGroup, template)

    def test_basics_subgroup_by_predicate__passthrough(self):
        f = lambda e: True
        for callable in [f, Predicate(f), Predicate(f).method]:
            self.assertIs(self.eg, self.eg.subgroup(callable))

    def test_basics_subgroup_by_predicate__no_match(self):
        f = lambda e: False
        for callable in [f, Predicate(f), Predicate(f).method]:
            self.assertIsNone(self.eg.subgroup(callable))

    def test_basics_subgroup_by_predicate__match(self):
        eg = self.eg
        testcases = [
            # (match_type, result_template)
            (ValueError, [ValueError(1), ValueError(2)]),
            (TypeError, [TypeError(int)]),
            ((ValueError, TypeError), self.eg_template)]

        for match_type, template in testcases:
            f = lambda e: isinstance(e, match_type)
            for callable in [f, Predicate(f), Predicate(f).method]:
                with self.subTest(callable=callable):
                    subeg = eg.subgroup(f)
                    self.assertEqual(subeg.message, eg.message)
                    self.assertMatchesTemplate(subeg, ExceptionGroup, template)


class ExceptionGroupSplitTests(ExceptionGroupTestBase):
    def setUp(self):
        self.eg = create_simple_eg()
        self.eg_template = [ValueError(1), TypeError(int), ValueError(2)]
        super().setUp()

    def test_basics_split_by_type__passthrough(self):
        for E in [BaseException, Exception,
                  BaseExceptionGroup, ExceptionGroup]:
            match, rest = self.eg.split(E)
            self.assertMatchesTemplate(
                match, ExceptionGroup, self.eg_template)
            self.assertIsNone(rest)

    def test_basics_split_by_type__no_match(self):
        match, rest = self.eg.split(OSError)
        self.assertIsNone(match)
        self.assertMatchesTemplate(
            rest, ExceptionGroup, self.eg_template)

    def test_basics_split_by_type__match(self):
        eg = self.eg
        VE = ValueError
        TE = TypeError
        testcases = [
            # (matcher, match_template, rest_template)
            (VE, [VE(1), VE(2)], [TE(int)]),
            (TE, [TE(int)], [VE(1), VE(2)]),
            ((VE, TE), self.eg_template, None),
            ((OSError, VE), [VE(1), VE(2)], [TE(int)]),
        ]

        for match_type, match_template, rest_template in testcases:
            match, rest = eg.split(match_type)
            self.assertEqual(match.message, eg.message)
            self.assertMatchesTemplate(
                match, ExceptionGroup, match_template)
            if rest_template is not None:
                self.assertEqual(rest.message, eg.message)
                self.assertMatchesTemplate(
                    rest, ExceptionGroup, rest_template)
            else:
                self.assertIsNone(rest)

    def test_basics_split_by_predicate__passthrough(self):
        f = lambda e: True
        for callable in [f, Predicate(f), Predicate(f).method]:
            match, rest = self.eg.split(callable)
            self.assertMatchesTemplate(match, ExceptionGroup, self.eg_template)
            self.assertIsNone(rest)

    def test_basics_split_by_predicate__no_match(self):
        f = lambda e: False
        for callable in [f, Predicate(f), Predicate(f).method]:
            match, rest = self.eg.split(callable)
            self.assertIsNone(match)
            self.assertMatchesTemplate(rest, ExceptionGroup, self.eg_template)

    def test_basics_split_by_predicate__match(self):
        eg = self.eg
        VE = ValueError
        TE = TypeError
        testcases = [
            # (matcher, match_template, rest_template)
            (VE, [VE(1), VE(2)], [TE(int)]),
            (TE, [TE(int)], [VE(1), VE(2)]),
            ((VE, TE), self.eg_template, None),
        ]

        for match_type, match_template, rest_template in testcases:
            f = lambda e: isinstance(e, match_type)
            for callable in [f, Predicate(f), Predicate(f).method]:
                match, rest = eg.split(callable)
                self.assertEqual(match.message, eg.message)
                self.assertMatchesTemplate(
                    match, ExceptionGroup, match_template)
                if rest_template is not None:
                    self.assertEqual(rest.message, eg.message)
                    self.assertMatchesTemplate(
                        rest, ExceptionGroup, rest_template)


class DeepRecursionInSplitAndSubgroup(__TestCase):
    def make_deep_eg(self):
        e = TypeError(1)
        for i in range(get_c_recursion_limit() + 1):
            e = ExceptionGroup('eg', [e])
        return e

    def test_deep_split(self):
        e = self.make_deep_eg()
        with self.assertRaises(RecursionError):
            e.split(TypeError)

    def test_deep_subgroup(self):
        e = self.make_deep_eg()
        with self.assertRaises(RecursionError):
            e.subgroup(TypeError)


def leaf_generator(exc, tbs=None):
    if tbs is None:
        tbs = []
    tbs.append(exc.__traceback__)
    if isinstance(exc, BaseExceptionGroup):
        for e in exc.exceptions:
            yield from leaf_generator(e, tbs)
    else:
        # exc is a leaf exception and its traceback
        # is the concatenation of the traceback
        # segments in tbs
        yield exc, tbs
    tbs.pop()


class LeafGeneratorTest(__TestCase):
    # The leaf_generator is mentioned in PEP 654 as a suggestion
    # on how to iterate over leaf nodes of an EG. Is is also
    # used below as a test utility. So we test it here.

    def test_leaf_generator(self):
        eg = create_simple_eg()

        self.assertSequenceEqual(
            [e for e, _ in leaf_generator(eg)],
            eg.exceptions)

        for e, tbs in leaf_generator(eg):
            self.assertSequenceEqual(
                tbs, [eg.__traceback__, e.__traceback__])


def create_nested_eg():
    excs = []
    try:
        try:
            raise TypeError(bytes)
        except TypeError as e:
            raise ExceptionGroup("nested", [e])
    except ExceptionGroup as e:
        excs.append(e)

    try:
        try:
            raise MemoryError('out of memory')
        except MemoryError as e:
            raise ValueError(1) from e
    except ValueError as e:
        excs.append(e)

    try:
        raise ExceptionGroup("root", excs)
    except ExceptionGroup as eg:
        return eg


class NestedExceptionGroupBasicsTest(ExceptionGroupTestBase):
    def test_nested_group_matches_template(self):
        eg = create_nested_eg()
        self.assertMatchesTemplate(
            eg,
            ExceptionGroup,
            [[TypeError(bytes)], ValueError(1)])

    def test_nested_group_chaining(self):
        eg = create_nested_eg()
        self.assertIsInstance(eg.exceptions[1].__context__, MemoryError)
        self.assertIsInstance(eg.exceptions[1].__cause__, MemoryError)
        self.assertIsInstance(eg.exceptions[0].__context__, TypeError)

    def test_nested_exception_group_tracebacks(self):
        eg = create_nested_eg()

        line0 = create_nested_eg.__code__.co_firstlineno
        for (tb, expected) in [
            (eg.__traceback__, line0 + 19),
            (eg.exceptions[0].__traceback__, line0 + 6),
            (eg.exceptions[1].__traceback__, line0 + 14),
            (eg.exceptions[0].exceptions[0].__traceback__, line0 + 4),
        ]:
            self.assertEqual(tb.tb_lineno, expected)
            self.assertIsNone(tb.tb_next)

    def test_iteration_full_tracebacks(self):
        eg = create_nested_eg()
        # check that iteration over leaves
        # produces the expected tracebacks
        self.assertEqual(len(list(leaf_generator(eg))), 2)

        line0 = create_nested_eg.__code__.co_firstlineno
        expected_tbs = [ [line0 + 19, line0 + 6, line0 + 4],
                         [line0 + 19, line0 + 14]]

        for (i, (_, tbs)) in enumerate(leaf_generator(eg)):
            self.assertSequenceEqual(
                [tb.tb_lineno for tb in tbs],
                expected_tbs[i])


class ExceptionGroupSplitTestBase(ExceptionGroupTestBase):

    def split_exception_group(self, eg, types):
        """ Split an EG and do some sanity checks on the result """
        self.assertIsInstance(eg, BaseExceptionGroup)

        match, rest = eg.split(types)
        sg = eg.subgroup(types)

        if match is not None:
            self.assertIsInstance(match, BaseExceptionGroup)
            for e,_ in leaf_generator(match):
                self.assertIsInstance(e, types)

            self.assertIsNotNone(sg)
            self.assertIsInstance(sg, BaseExceptionGroup)
            for e,_ in leaf_generator(sg):
                self.assertIsInstance(e, types)

        if rest is not None:
            self.assertIsInstance(rest, BaseExceptionGroup)

        def leaves(exc):
            return [] if exc is None else [e for e,_ in leaf_generator(exc)]

        # match and subgroup have the same leaves
        self.assertSequenceEqual(leaves(match), leaves(sg))

        match_leaves = leaves(match)
        rest_leaves = leaves(rest)
        # each leaf exception of eg is in exactly one of match and rest
        self.assertEqual(
            len(leaves(eg)),
            len(leaves(match)) + len(leaves(rest)))

        for e in leaves(eg):
            self.assertNotEqual(
                match and e in match_leaves,
                rest and e in rest_leaves)

        # message, cause and context, traceback and note equal to eg
        for part in [match, rest, sg]:
            if part is not None:
                self.assertEqual(eg.message, part.message)
                self.assertIs(eg.__cause__, part.__cause__)
                self.assertIs(eg.__context__, part.__context__)
                self.assertIs(eg.__traceback__, part.__traceback__)
                self.assertEqual(
                    getattr(eg, '__notes__', None),
                    getattr(part, '__notes__', None))

        def tbs_for_leaf(leaf, eg):
            for e, tbs in leaf_generator(eg):
                if e is leaf:
                    return tbs

        def tb_linenos(tbs):
            return [tb.tb_lineno for tb in tbs if tb]

        # full tracebacks match
        for part in [match, rest, sg]:
            for e in leaves(part):
                self.assertSequenceEqual(
                    tb_linenos(tbs_for_leaf(e, eg)),
                    tb_linenos(tbs_for_leaf(e, part)))

        return match, rest


class NestedExceptionGroupSplitTest(ExceptionGroupSplitTestBase):

    def test_split_by_type(self):
        class MyExceptionGroup(ExceptionGroup):
            pass

        def raiseVE(v):
            raise ValueError(v)

        def raiseTE(t):
            raise TypeError(t)

        def nested_group():
            def level1(i):
                excs = []
                for f, arg in [(raiseVE, i), (raiseTE, int), (raiseVE, i+1)]:
                    try:
                        f(arg)
                    except Exception as e:
                        excs.append(e)
                raise ExceptionGroup('msg1', excs)

            def level2(i):
                excs = []
                for f, arg in [(level1, i), (level1, i+1), (raiseVE, i+2)]:
                    try:
                        f(arg)
                    except Exception as e:
                        excs.append(e)
                raise MyExceptionGroup('msg2', excs)

            def level3(i):
                excs = []
                for f, arg in [(level2, i+1), (raiseVE, i+2)]:
                    try:
                        f(arg)
                    except Exception as e:
                        excs.append(e)
                raise ExceptionGroup('msg3', excs)

            level3(5)

        try:
            nested_group()
        except ExceptionGroup as e:
            e.add_note(f"the note: {id(e)}")
            eg = e

        eg_template = [
            [
                [ValueError(6), TypeError(int), ValueError(7)],
                [ValueError(7), TypeError(int), ValueError(8)],
                ValueError(8),
            ],
            ValueError(7)]

        valueErrors_template = [
            [
                [ValueError(6), ValueError(7)],
                [ValueError(7), ValueError(8)],
                ValueError(8),
            ],
            ValueError(7)]

        typeErrors_template = [[[TypeError(int)], [TypeError(int)]]]

        self.assertMatchesTemplate(eg, ExceptionGroup, eg_template)

        # Match Nothing
        match, rest = self.split_exception_group(eg, SyntaxError)
        self.assertIsNone(match)
        self.assertMatchesTemplate(rest, ExceptionGroup, eg_template)

        # Match Everything
        match, rest = self.split_exception_group(eg, BaseException)
        self.assertMatchesTemplate(match, ExceptionGroup, eg_template)
        self.assertIsNone(rest)
        match, rest = self.split_exception_group(eg, (ValueError, TypeError))
        self.assertMatchesTemplate(match, ExceptionGroup, eg_template)
        self.assertIsNone(rest)

        # Match ValueErrors
        match, rest = self.split_exception_group(eg, ValueError)
        self.assertMatchesTemplate(match, ExceptionGroup, valueErrors_template)
        self.assertMatchesTemplate(rest, ExceptionGroup, typeErrors_template)

        # Match TypeErrors
        match, rest = self.split_exception_group(eg, (TypeError, SyntaxError))
        self.assertMatchesTemplate(match, ExceptionGroup, typeErrors_template)
        self.assertMatchesTemplate(rest, ExceptionGroup, valueErrors_template)

        # Match ExceptionGroup
        match, rest = eg.split(ExceptionGroup)
        self.assertIs(match, eg)
        self.assertIsNone(rest)

        # Match MyExceptionGroup (ExceptionGroup subclass)
        match, rest = eg.split(MyExceptionGroup)
        self.assertMatchesTemplate(match, ExceptionGroup, [eg_template[0]])
        self.assertMatchesTemplate(rest, ExceptionGroup, [eg_template[1]])

    def test_split_BaseExceptionGroup(self):
        def exc(ex):
            try:
                raise ex
            except BaseException as e:
                return e

        try:
            raise BaseExceptionGroup(
                "beg", [exc(ValueError(1)), exc(KeyboardInterrupt(2))])
        except BaseExceptionGroup as e:
            beg = e

        # Match Nothing
        match, rest = self.split_exception_group(beg, TypeError)
        self.assertIsNone(match)
        self.assertMatchesTemplate(
            rest, BaseExceptionGroup, [ValueError(1), KeyboardInterrupt(2)])

        # Match Everything
        match, rest = self.split_exception_group(
            beg, (ValueError, KeyboardInterrupt))
        self.assertMatchesTemplate(
            match, BaseExceptionGroup, [ValueError(1), KeyboardInterrupt(2)])
        self.assertIsNone(rest)

        # Match ValueErrors
        match, rest = self.split_exception_group(beg, ValueError)
        self.assertMatchesTemplate(
            match, ExceptionGroup, [ValueError(1)])
        self.assertMatchesTemplate(
            rest, BaseExceptionGroup, [KeyboardInterrupt(2)])

        # Match KeyboardInterrupts
        match, rest = self.split_exception_group(beg, KeyboardInterrupt)
        self.assertMatchesTemplate(
            match, BaseExceptionGroup, [KeyboardInterrupt(2)])
        self.assertMatchesTemplate(
            rest, ExceptionGroup, [ValueError(1)])

    def test_split_copies_notes(self):
        # make sure each exception group after a split has its own __notes__ list
        eg = ExceptionGroup("eg", [ValueError(1), TypeError(2)])
        eg.add_note("note1")
        eg.add_note("note2")
        orig_notes = list(eg.__notes__)
        match, rest = eg.split(TypeError)
        self.assertEqual(eg.__notes__, orig_notes)
        self.assertEqual(match.__notes__, orig_notes)
        self.assertEqual(rest.__notes__, orig_notes)
        self.assertIsNot(eg.__notes__, match.__notes__)
        self.assertIsNot(eg.__notes__, rest.__notes__)
        self.assertIsNot(match.__notes__, rest.__notes__)
        eg.add_note("eg")
        match.add_note("match")
        rest.add_note("rest")
        self.assertEqual(eg.__notes__, orig_notes + ["eg"])
        self.assertEqual(match.__notes__, orig_notes + ["match"])
        self.assertEqual(rest.__notes__, orig_notes + ["rest"])

    def test_split_does_not_copy_non_sequence_notes(self):
        # __notes__ should be a sequence, which is shallow copied.
        # If it is not a sequence, the split parts don't get any notes.
        eg = ExceptionGroup("eg", [ValueError(1), TypeError(2)])
        eg.__notes__ = 123
        match, rest = eg.split(TypeError)
        self.assertFalse(hasattr(match, '__notes__'))
        self.assertFalse(hasattr(rest, '__notes__'))

    def test_drive_invalid_return_value(self):
        class MyEg(ExceptionGroup):
            def derive(self, excs):
                return 42

        eg = MyEg('eg', [TypeError(1), ValueError(2)])
        msg = "derive must return an instance of BaseExceptionGroup"
        with self.assertRaisesRegex(TypeError, msg):
            eg.split(TypeError)
        with self.assertRaisesRegex(TypeError, msg):
            eg.subgroup(TypeError)


class NestedExceptionGroupSubclassSplitTest(ExceptionGroupSplitTestBase):

    def test_split_ExceptionGroup_subclass_no_derive_no_new_override(self):
        class EG(ExceptionGroup):
            pass

        try:
            try:
                try:
                    raise TypeError(2)
                except TypeError as te:
                    raise EG("nested", [te])
            except EG as nested:
                try:
                    raise ValueError(1)
                except ValueError as ve:
                    raise EG("eg", [ve, nested])
        except EG as e:
            eg = e

        self.assertMatchesTemplate(eg, EG, [ValueError(1), [TypeError(2)]])

        # Match Nothing
        match, rest = self.split_exception_group(eg, OSError)
        self.assertIsNone(match)
        self.assertMatchesTemplate(
            rest, ExceptionGroup, [ValueError(1), [TypeError(2)]])

        # Match Everything
        match, rest = self.split_exception_group(eg, (ValueError, TypeError))
        self.assertMatchesTemplate(
            match, ExceptionGroup, [ValueError(1), [TypeError(2)]])
        self.assertIsNone(rest)

        # Match ValueErrors
        match, rest = self.split_exception_group(eg, ValueError)
        self.assertMatchesTemplate(match, ExceptionGroup, [ValueError(1)])
        self.assertMatchesTemplate(rest, ExceptionGroup, [[TypeError(2)]])

        # Match TypeErrors
        match, rest = self.split_exception_group(eg, TypeError)
        self.assertMatchesTemplate(match, ExceptionGroup, [[TypeError(2)]])
        self.assertMatchesTemplate(rest, ExceptionGroup, [ValueError(1)])

    def test_split_BaseExceptionGroup_subclass_no_derive_new_override(self):
        class EG(BaseExceptionGroup):
            def __new__(cls, message, excs, unused):
                # The "unused" arg is here to show that split() doesn't call
                # the actual class constructor from the default derive()
                # implementation (it would fail on unused arg if so because
                # it assumes the BaseExceptionGroup.__new__ signature).
                return super().__new__(cls, message, excs)

        try:
            raise EG("eg", [ValueError(1), KeyboardInterrupt(2)], "unused")
        except EG as e:
            eg = e

        self.assertMatchesTemplate(
            eg, EG, [ValueError(1), KeyboardInterrupt(2)])

        # Match Nothing
        match, rest = self.split_exception_group(eg, OSError)
        self.assertIsNone(match)
        self.assertMatchesTemplate(
            rest, BaseExceptionGroup, [ValueError(1), KeyboardInterrupt(2)])

        # Match Everything
        match, rest = self.split_exception_group(
            eg, (ValueError, KeyboardInterrupt))
        self.assertMatchesTemplate(
            match, BaseExceptionGroup, [ValueError(1), KeyboardInterrupt(2)])
        self.assertIsNone(rest)

        # Match ValueErrors
        match, rest = self.split_exception_group(eg, ValueError)
        self.assertMatchesTemplate(match, ExceptionGroup, [ValueError(1)])
        self.assertMatchesTemplate(
            rest, BaseExceptionGroup, [KeyboardInterrupt(2)])

        # Match KeyboardInterrupt
        match, rest = self.split_exception_group(eg, KeyboardInterrupt)
        self.assertMatchesTemplate(
            match, BaseExceptionGroup, [KeyboardInterrupt(2)])
        self.assertMatchesTemplate(rest, ExceptionGroup, [ValueError(1)])

    def test_split_ExceptionGroup_subclass_derive_and_new_overrides(self):
        class EG(ExceptionGroup):
            def __new__(cls, message, excs, code):
                obj = super().__new__(cls, message, excs)
                obj.code = code
                return obj

            def derive(self, excs):
                return EG(self.message, excs, self.code)

        try:
            try:
                try:
                    raise TypeError(2)
                except TypeError as te:
                    raise EG("nested", [te], 101)
            except EG as nested:
                try:
                    raise ValueError(1)
                except ValueError as ve:
                    raise EG("eg", [ve, nested], 42)
        except EG as e:
            eg = e

        self.assertMatchesTemplate(eg, EG, [ValueError(1), [TypeError(2)]])

        # Match Nothing
        match, rest = self.split_exception_group(eg, OSError)
        self.assertIsNone(match)
        self.assertMatchesTemplate(rest, EG, [ValueError(1), [TypeError(2)]])
        self.assertEqual(rest.code, 42)
        self.assertEqual(rest.exceptions[1].code, 101)

        # Match Everything
        match, rest = self.split_exception_group(eg, (ValueError, TypeError))
        self.assertMatchesTemplate(match, EG, [ValueError(1), [TypeError(2)]])
        self.assertEqual(match.code, 42)
        self.assertEqual(match.exceptions[1].code, 101)
        self.assertIsNone(rest)

        # Match ValueErrors
        match, rest = self.split_exception_group(eg, ValueError)
        self.assertMatchesTemplate(match, EG, [ValueError(1)])
        self.assertEqual(match.code, 42)
        self.assertMatchesTemplate(rest, EG, [[TypeError(2)]])
        self.assertEqual(rest.code, 42)
        self.assertEqual(rest.exceptions[0].code, 101)

        # Match TypeErrors
        match, rest = self.split_exception_group(eg, TypeError)
        self.assertMatchesTemplate(match, EG, [[TypeError(2)]])
        self.assertEqual(match.code, 42)
        self.assertEqual(match.exceptions[0].code, 101)
        self.assertMatchesTemplate(rest, EG, [ValueError(1)])
        self.assertEqual(rest.code, 42)


if __name__ == '__main__':
    if TEST_WITH_TORCHDYNAMO:
        run_tests()
    else:
        unittest.main()
