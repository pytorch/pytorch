# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_reprlib.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

"""
  Test cases for the repr module
  Nick Mathewson
"""

import sys
import os
import shutil
import importlib
import importlib.util
import unittest
import textwrap

from test.support import verbose
from test.support.os_helper import create_empty_file
from reprlib import repr as r # Don't shadow builtin repr
from reprlib import Repr
from reprlib import recursive_repr


def nestedTuple(nesting):
    t = ()
    for i in range(nesting):
        t = (t,)
    return t

class ReprTests(CPythonTestCase):

    def test_init_kwargs(self):
        example_kwargs = {
            "maxlevel": 101,
            "maxtuple": 102,
            "maxlist": 103,
            "maxarray": 104,
            "maxdict": 105,
            "maxset": 106,
            "maxfrozenset": 107,
            "maxdeque": 108,
            "maxstring": 109,
            "maxlong": 110,
            "maxother": 111,
            "fillvalue": "x" * 112,
            "indent": "x" * 113,
        }
        r1 = Repr()
        for attr, val in example_kwargs.items():
            setattr(r1, attr, val)
        r2 = Repr(**example_kwargs)
        for attr in example_kwargs:
            self.assertEqual(getattr(r1, attr), getattr(r2, attr), msg=attr)

    def test_string(self):
        eq = self.assertEqual
        eq(r("abc"), "'abc'")
        eq(r("abcdefghijklmnop"),"'abcdefghijklmnop'")

        s = "a"*30+"b"*30
        expected = repr(s)[:13] + "..." + repr(s)[-14:]
        eq(r(s), expected)

        eq(r("\"'"), repr("\"'"))
        s = "\""*30+"'"*100
        expected = repr(s)[:13] + "..." + repr(s)[-14:]
        eq(r(s), expected)

    def test_tuple(self):
        eq = self.assertEqual
        eq(r((1,)), "(1,)")

        t3 = (1, 2, 3)
        eq(r(t3), "(1, 2, 3)")

        r2 = Repr()
        r2.maxtuple = 2
        expected = repr(t3)[:-2] + "...)"
        eq(r2.repr(t3), expected)

        # modified fillvalue:
        r3 = Repr()
        r3.fillvalue = '+++'
        r3.maxtuple = 2
        expected = repr(t3)[:-2] + "+++)"
        eq(r3.repr(t3), expected)

    def test_container(self):
        from array import array
        from collections import deque

        eq = self.assertEqual
        # Tuples give up after 6 elements
        eq(r(()), "()")
        eq(r((1,)), "(1,)")
        eq(r((1, 2, 3)), "(1, 2, 3)")
        eq(r((1, 2, 3, 4, 5, 6)), "(1, 2, 3, 4, 5, 6)")
        eq(r((1, 2, 3, 4, 5, 6, 7)), "(1, 2, 3, 4, 5, 6, ...)")

        # Lists give up after 6 as well
        eq(r([]), "[]")
        eq(r([1]), "[1]")
        eq(r([1, 2, 3]), "[1, 2, 3]")
        eq(r([1, 2, 3, 4, 5, 6]), "[1, 2, 3, 4, 5, 6]")
        eq(r([1, 2, 3, 4, 5, 6, 7]), "[1, 2, 3, 4, 5, 6, ...]")

        # Sets give up after 6 as well
        eq(r(set([])), "set()")
        eq(r(set([1])), "{1}")
        eq(r(set([1, 2, 3])), "{1, 2, 3}")
        eq(r(set([1, 2, 3, 4, 5, 6])), "{1, 2, 3, 4, 5, 6}")
        eq(r(set([1, 2, 3, 4, 5, 6, 7])), "{1, 2, 3, 4, 5, 6, ...}")

        # Frozensets give up after 6 as well
        eq(r(frozenset([])), "frozenset()")
        eq(r(frozenset([1])), "frozenset({1})")
        eq(r(frozenset([1, 2, 3])), "frozenset({1, 2, 3})")
        eq(r(frozenset([1, 2, 3, 4, 5, 6])), "frozenset({1, 2, 3, 4, 5, 6})")
        eq(r(frozenset([1, 2, 3, 4, 5, 6, 7])), "frozenset({1, 2, 3, 4, 5, 6, ...})")

        # collections.deque after 6
        eq(r(deque([1, 2, 3, 4, 5, 6, 7])), "deque([1, 2, 3, 4, 5, 6, ...])")

        # Dictionaries give up after 4.
        eq(r({}), "{}")
        d = {'alice': 1, 'bob': 2, 'charles': 3, 'dave': 4}
        eq(r(d), "{'alice': 1, 'bob': 2, 'charles': 3, 'dave': 4}")
        d['arthur'] = 1
        eq(r(d), "{'alice': 1, 'arthur': 1, 'bob': 2, 'charles': 3, ...}")

        # array.array after 5.
        eq(r(array('i')), "array('i')")
        eq(r(array('i', [1])), "array('i', [1])")
        eq(r(array('i', [1, 2])), "array('i', [1, 2])")
        eq(r(array('i', [1, 2, 3])), "array('i', [1, 2, 3])")
        eq(r(array('i', [1, 2, 3, 4])), "array('i', [1, 2, 3, 4])")
        eq(r(array('i', [1, 2, 3, 4, 5])), "array('i', [1, 2, 3, 4, 5])")
        eq(r(array('i', [1, 2, 3, 4, 5, 6])),
                   "array('i', [1, 2, 3, 4, 5, ...])")

    def test_set_literal(self):
        eq = self.assertEqual
        eq(r({1}), "{1}")
        eq(r({1, 2, 3}), "{1, 2, 3}")
        eq(r({1, 2, 3, 4, 5, 6}), "{1, 2, 3, 4, 5, 6}")
        eq(r({1, 2, 3, 4, 5, 6, 7}), "{1, 2, 3, 4, 5, 6, ...}")

    def test_frozenset(self):
        eq = self.assertEqual
        eq(r(frozenset({1})), "frozenset({1})")
        eq(r(frozenset({1, 2, 3})), "frozenset({1, 2, 3})")
        eq(r(frozenset({1, 2, 3, 4, 5, 6})), "frozenset({1, 2, 3, 4, 5, 6})")
        eq(r(frozenset({1, 2, 3, 4, 5, 6, 7})), "frozenset({1, 2, 3, 4, 5, 6, ...})")

    def test_numbers(self):
        eq = self.assertEqual
        eq(r(123), repr(123))
        eq(r(123), repr(123))
        eq(r(1.0/3), repr(1.0/3))

        n = 10**100
        expected = repr(n)[:18] + "..." + repr(n)[-19:]
        eq(r(n), expected)

    def test_instance(self):
        eq = self.assertEqual
        i1 = ClassWithRepr("a")
        eq(r(i1), repr(i1))

        i2 = ClassWithRepr("x"*1000)
        expected = repr(i2)[:13] + "..." + repr(i2)[-14:]
        eq(r(i2), expected)

        i3 = ClassWithFailingRepr()
        eq(r(i3), ("<ClassWithFailingRepr instance at %#x>"%id(i3)))

        s = r(ClassWithFailingRepr)
        self.assertTrue(s.startswith("<class "))
        self.assertTrue(s.endswith(">"))
        self.assertIn(s.find("..."), [12, 13])

    def test_lambda(self):
        r = repr(lambda x: x)
        self.assertTrue(r.startswith("<function ReprTests.test_lambda.<locals>.<lambda"), r)
        # XXX anonymous functions?  see func_repr

    def test_builtin_function(self):
        eq = self.assertEqual
        # Functions
        eq(repr(hash), '<built-in function hash>')
        # Methods
        self.assertTrue(repr(''.split).startswith(
            '<built-in method split of str object at 0x'))

    def test_range(self):
        eq = self.assertEqual
        eq(repr(range(1)), 'range(0, 1)')
        eq(repr(range(1, 2)), 'range(1, 2)')
        eq(repr(range(1, 4, 3)), 'range(1, 4, 3)')

    def test_nesting(self):
        eq = self.assertEqual
        # everything is meant to give up after 6 levels.
        eq(r([[[[[[[]]]]]]]), "[[[[[[[]]]]]]]")
        eq(r([[[[[[[[]]]]]]]]), "[[[[[[[...]]]]]]]")

        eq(r(nestedTuple(6)), "(((((((),),),),),),)")
        eq(r(nestedTuple(7)), "(((((((...),),),),),),)")

        eq(r({ nestedTuple(5) : nestedTuple(5) }),
           "{((((((),),),),),): ((((((),),),),),)}")
        eq(r({ nestedTuple(6) : nestedTuple(6) }),
           "{((((((...),),),),),): ((((((...),),),),),)}")

        eq(r([[[[[[{}]]]]]]), "[[[[[[{}]]]]]]")
        eq(r([[[[[[[{}]]]]]]]), "[[[[[[[...]]]]]]]")

    def test_cell(self):
        def get_cell():
            x = 42
            def inner():
                return x
            return inner
        x = get_cell().__closure__[0]
        self.assertRegex(repr(x), r'<cell at 0x[0-9A-Fa-f]+: '
                                  r'int object at 0x[0-9A-Fa-f]+>')
        self.assertRegex(r(x), r'<cell at 0x.*\.\.\..*>')

    def test_descriptors(self):
        eq = self.assertEqual
        # method descriptors
        eq(repr(dict.items), "<method 'items' of 'dict' objects>")
        # XXX member descriptors
        # XXX attribute descriptors
        # XXX slot descriptors
        # static and class methods
        class C:
            def foo(cls): pass
        x = staticmethod(C.foo)
        self.assertEqual(repr(x), f'<staticmethod({C.foo!r})>')
        x = classmethod(C.foo)
        self.assertEqual(repr(x), f'<classmethod({C.foo!r})>')

    def test_unsortable(self):
        # Repr.repr() used to call sorted() on sets, frozensets and dicts
        # without taking into account that not all objects are comparable
        x = set([1j, 2j, 3j])
        y = frozenset(x)
        z = {1j: 1, 2j: 2}
        r(x)
        r(y)
        r(z)

    def test_valid_indent(self):
        test_cases = [
            {
                'object': (),
                'tests': (
                    (dict(indent=None), '()'),
                    (dict(indent=False), '()'),
                    (dict(indent=True), '()'),
                    (dict(indent=0), '()'),
                    (dict(indent=1), '()'),
                    (dict(indent=4), '()'),
                    (dict(indent=4, maxlevel=2), '()'),
                    (dict(indent=''), '()'),
                    (dict(indent='-->'), '()'),
                    (dict(indent='....'), '()'),
                ),
            },
            {
                'object': '',
                'tests': (
                    (dict(indent=None), "''"),
                    (dict(indent=False), "''"),
                    (dict(indent=True), "''"),
                    (dict(indent=0), "''"),
                    (dict(indent=1), "''"),
                    (dict(indent=4), "''"),
                    (dict(indent=4, maxlevel=2), "''"),
                    (dict(indent=''), "''"),
                    (dict(indent='-->'), "''"),
                    (dict(indent='....'), "''"),
                ),
            },
            {
                'object': [1, 'spam', {'eggs': True, 'ham': []}],
                'tests': (
                    (dict(indent=None), '''\
                        [1, 'spam', {'eggs': True, 'ham': []}]'''),
                    (dict(indent=False), '''\
                        [
                        1,
                        'spam',
                        {
                        'eggs': True,
                        'ham': [],
                        },
                        ]'''),
                    (dict(indent=True), '''\
                        [
                         1,
                         'spam',
                         {
                          'eggs': True,
                          'ham': [],
                         },
                        ]'''),
                    (dict(indent=0), '''\
                        [
                        1,
                        'spam',
                        {
                        'eggs': True,
                        'ham': [],
                        },
                        ]'''),
                    (dict(indent=1), '''\
                        [
                         1,
                         'spam',
                         {
                          'eggs': True,
                          'ham': [],
                         },
                        ]'''),
                    (dict(indent=4), '''\
                        [
                            1,
                            'spam',
                            {
                                'eggs': True,
                                'ham': [],
                            },
                        ]'''),
                    (dict(indent=4, maxlevel=2), '''\
                        [
                            1,
                            'spam',
                            {
                                'eggs': True,
                                'ham': [],
                            },
                        ]'''),
                    (dict(indent=''), '''\
                        [
                        1,
                        'spam',
                        {
                        'eggs': True,
                        'ham': [],
                        },
                        ]'''),
                    (dict(indent='-->'), '''\
                        [
                        -->1,
                        -->'spam',
                        -->{
                        -->-->'eggs': True,
                        -->-->'ham': [],
                        -->},
                        ]'''),
                    (dict(indent='....'), '''\
                        [
                        ....1,
                        ....'spam',
                        ....{
                        ........'eggs': True,
                        ........'ham': [],
                        ....},
                        ]'''),
                ),
            },
            {
                'object': {
                    1: 'two',
                    b'three': [
                        (4.5, 6.7),
                        [set((8, 9)), frozenset((10, 11))],
                    ],
                },
                'tests': (
                    (dict(indent=None), '''\
                        {1: 'two', b'three': [(4.5, 6.7), [{8, 9}, frozenset({10, 11})]]}'''),
                    (dict(indent=False), '''\
                        {
                        1: 'two',
                        b'three': [
                        (
                        4.5,
                        6.7,
                        ),
                        [
                        {
                        8,
                        9,
                        },
                        frozenset({
                        10,
                        11,
                        }),
                        ],
                        ],
                        }'''),
                    (dict(indent=True), '''\
                        {
                         1: 'two',
                         b'three': [
                          (
                           4.5,
                           6.7,
                          ),
                          [
                           {
                            8,
                            9,
                           },
                           frozenset({
                            10,
                            11,
                           }),
                          ],
                         ],
                        }'''),
                    (dict(indent=0), '''\
                        {
                        1: 'two',
                        b'three': [
                        (
                        4.5,
                        6.7,
                        ),
                        [
                        {
                        8,
                        9,
                        },
                        frozenset({
                        10,
                        11,
                        }),
                        ],
                        ],
                        }'''),
                    (dict(indent=1), '''\
                        {
                         1: 'two',
                         b'three': [
                          (
                           4.5,
                           6.7,
                          ),
                          [
                           {
                            8,
                            9,
                           },
                           frozenset({
                            10,
                            11,
                           }),
                          ],
                         ],
                        }'''),
                    (dict(indent=4), '''\
                        {
                            1: 'two',
                            b'three': [
                                (
                                    4.5,
                                    6.7,
                                ),
                                [
                                    {
                                        8,
                                        9,
                                    },
                                    frozenset({
                                        10,
                                        11,
                                    }),
                                ],
                            ],
                        }'''),
                    (dict(indent=4, maxlevel=2), '''\
                        {
                            1: 'two',
                            b'three': [
                                (...),
                                [...],
                            ],
                        }'''),
                    (dict(indent=''), '''\
                        {
                        1: 'two',
                        b'three': [
                        (
                        4.5,
                        6.7,
                        ),
                        [
                        {
                        8,
                        9,
                        },
                        frozenset({
                        10,
                        11,
                        }),
                        ],
                        ],
                        }'''),
                    (dict(indent='-->'), '''\
                        {
                        -->1: 'two',
                        -->b'three': [
                        -->-->(
                        -->-->-->4.5,
                        -->-->-->6.7,
                        -->-->),
                        -->-->[
                        -->-->-->{
                        -->-->-->-->8,
                        -->-->-->-->9,
                        -->-->-->},
                        -->-->-->frozenset({
                        -->-->-->-->10,
                        -->-->-->-->11,
                        -->-->-->}),
                        -->-->],
                        -->],
                        }'''),
                    (dict(indent='....'), '''\
                        {
                        ....1: 'two',
                        ....b'three': [
                        ........(
                        ............4.5,
                        ............6.7,
                        ........),
                        ........[
                        ............{
                        ................8,
                        ................9,
                        ............},
                        ............frozenset({
                        ................10,
                        ................11,
                        ............}),
                        ........],
                        ....],
                        }'''),
                ),
            },
        ]
        for test_case in test_cases:
            with self.subTest(test_object=test_case['object']):
                for repr_settings, expected_repr in test_case['tests']:
                    with self.subTest(repr_settings=repr_settings):
                        r = Repr()
                        for attribute, value in repr_settings.items():
                            setattr(r, attribute, value)
                        resulting_repr = r.repr(test_case['object'])
                        expected_repr = textwrap.dedent(expected_repr)
                        self.assertEqual(resulting_repr, expected_repr)

    def test_invalid_indent(self):
        test_object = [1, 'spam', {'eggs': True, 'ham': []}]
        test_cases = [
            (-1, (ValueError, '[Nn]egative|[Pp]ositive')),
            (-4, (ValueError, '[Nn]egative|[Pp]ositive')),
            ((), (TypeError, None)),
            ([], (TypeError, None)),
            ((4,), (TypeError, None)),
            ([4,], (TypeError, None)),
            (object(), (TypeError, None)),
        ]
        for indent, (expected_error, expected_msg) in test_cases:
            with self.subTest(indent=indent):
                r = Repr()
                r.indent = indent
                expected_msg = expected_msg or f'{type(indent)}'
                with self.assertRaisesRegex(expected_error, expected_msg):
                    r.repr(test_object)

    def test_shadowed_stdlib_array(self):
        # Issue #113570: repr() should not be fooled by an array
        class array:
            def __repr__(self):
                return "not array.array"

        self.assertEqual(r(array()), "not array.array")

    def test_shadowed_builtin(self):
        # Issue #113570: repr() should not be fooled
        # by a shadowed builtin function
        class list:
            def __repr__(self):
                return "not builtins.list"

        self.assertEqual(r(list()), "not builtins.list")

    def test_custom_repr(self):
        class MyRepr(Repr):

            def repr_TextIOWrapper(self, obj, level):
                if obj.name in {'<stdin>', '<stdout>', '<stderr>'}:
                    return obj.name
                return repr(obj)

        aRepr = MyRepr()
        self.assertEqual(aRepr.repr(sys.stdin), "<stdin>")

    def test_custom_repr_class_with_spaces(self):
        class TypeWithSpaces:
            pass

        t = TypeWithSpaces()
        type(t).__name__ = "type with spaces"
        self.assertEqual(type(t).__name__, "type with spaces")

        class MyRepr(Repr):
            def repr_type_with_spaces(self, obj, level):
                return "Type With Spaces"


        aRepr = MyRepr()
        self.assertEqual(aRepr.repr(t), "Type With Spaces")

def write_file(path, text):
    with open(path, 'w', encoding='ASCII') as fp:
        fp.write(text)

class LongReprTest(CPythonTestCase):
    longname = 'areallylongpackageandmodulenametotestreprtruncation'

    def setUp(self):
        self.pkgname = os.path.join(self.longname)
        self.subpkgname = os.path.join(self.longname, self.longname)
        # Make the package and subpackage
        shutil.rmtree(self.pkgname, ignore_errors=True)
        os.mkdir(self.pkgname)
        create_empty_file(os.path.join(self.pkgname, '__init__.py'))
        shutil.rmtree(self.subpkgname, ignore_errors=True)
        os.mkdir(self.subpkgname)
        create_empty_file(os.path.join(self.subpkgname, '__init__.py'))
        # Remember where we are
        self.here = os.getcwd()
        sys.path.insert(0, self.here)
        # When regrtest is run with its -j option, this command alone is not
        # enough.
        importlib.invalidate_caches()

    def tearDown(self):
        actions = []
        for dirpath, dirnames, filenames in os.walk(self.pkgname):
            for name in dirnames + filenames:
                actions.append(os.path.join(dirpath, name))
        actions.append(self.pkgname)
        actions.sort()
        actions.reverse()
        for p in actions:
            if os.path.isdir(p):
                os.rmdir(p)
            else:
                os.remove(p)
        del sys.path[0]

    def _check_path_limitations(self, module_name):
        # base directory
        source_path_len = len(self.here)
        # a path separator + `longname` (twice)
        source_path_len += 2 * (len(self.longname) + 1)
        # a path separator + `module_name` + ".py"
        source_path_len += len(module_name) + 1 + len(".py")
        cached_path_len = (source_path_len +
            len(importlib.util.cache_from_source("x.py")) - len("x.py"))
        if os.name == 'nt' and cached_path_len >= 258:
            # Under Windows, the max path len is 260 including C's terminating
            # NUL character.
            # (see http://msdn.microsoft.com/en-us/library/windows/desktop/aa365247%28v=vs.85%29.aspx#maxpath)
            self.skipTest("test paths too long (%d characters) for Windows' 260 character limit"
                          % cached_path_len)
        elif os.name == 'nt' and verbose:
            print("cached_path_len =", cached_path_len)

    def test_module(self):
        self.maxDiff = None
        self._check_path_limitations(self.pkgname)
        create_empty_file(os.path.join(self.subpkgname, self.pkgname + '.py'))
        importlib.invalidate_caches()
        from areallylongpackageandmodulenametotestreprtruncation.areallylongpackageandmodulenametotestreprtruncation import areallylongpackageandmodulenametotestreprtruncation
        module = areallylongpackageandmodulenametotestreprtruncation
        self.assertEqual(repr(module), "<module %r from %r>" % (module.__name__, module.__file__))
        self.assertEqual(repr(sys), "<module 'sys' (built-in)>")

    def test_type(self):
        self._check_path_limitations('foo')
        eq = self.assertEqual
        write_file(os.path.join(self.subpkgname, 'foo.py'), '''\
class foo(object):
    pass
''')
        importlib.invalidate_caches()
        from areallylongpackageandmodulenametotestreprtruncation.areallylongpackageandmodulenametotestreprtruncation import foo
        eq(repr(foo.foo),
               "<class '%s.foo'>" % foo.__name__)

    @unittest.skip('need a suitable object')
    def test_object(self):
        # XXX Test the repr of a type with a really long tp_name but with no
        # tp_repr.  WIBNI we had ::Inline? :)
        pass

    def test_class(self):
        self._check_path_limitations('bar')
        write_file(os.path.join(self.subpkgname, 'bar.py'), '''\
class bar:
    pass
''')
        importlib.invalidate_caches()
        from areallylongpackageandmodulenametotestreprtruncation.areallylongpackageandmodulenametotestreprtruncation import bar
        # Module name may be prefixed with "test.", depending on how run.
        self.assertEqual(repr(bar.bar), "<class '%s.bar'>" % bar.__name__)

    def test_instance(self):
        self._check_path_limitations('baz')
        write_file(os.path.join(self.subpkgname, 'baz.py'), '''\
class baz:
    pass
''')
        importlib.invalidate_caches()
        from areallylongpackageandmodulenametotestreprtruncation.areallylongpackageandmodulenametotestreprtruncation import baz
        ibaz = baz.baz()
        self.assertTrue(repr(ibaz).startswith(
            "<%s.baz object at 0x" % baz.__name__))

    def test_method(self):
        self._check_path_limitations('qux')
        eq = self.assertEqual
        write_file(os.path.join(self.subpkgname, 'qux.py'), '''\
class aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:
    def amethod(self): pass
''')
        importlib.invalidate_caches()
        from areallylongpackageandmodulenametotestreprtruncation.areallylongpackageandmodulenametotestreprtruncation import qux
        # Unbound methods first
        r = repr(qux.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.amethod)
        self.assertTrue(r.startswith('<function aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.amethod'), r)
        # Bound method next
        iqux = qux.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa()
        r = repr(iqux.amethod)
        self.assertTrue(r.startswith(
            '<bound method aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.amethod of <%s.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa object at 0x' \
            % (qux.__name__,) ), r)

    @unittest.skip('needs a built-in function with a really long name')
    def test_builtin_function(self):
        # XXX test built-in functions and methods with really long names
        pass

class ClassWithRepr:
    def __init__(self, s):
        self.s = s
    def __repr__(self):
        return "ClassWithRepr(%r)" % self.s


class ClassWithFailingRepr:
    def __repr__(self):
        raise Exception("This should be caught by Repr.repr_instance")

class MyContainer:
    'Helper class for TestRecursiveRepr'
    def __init__(self, values):
        self.values = list(values)
    def append(self, value):
        self.values.append(value)
    @recursive_repr()
    def __repr__(self):
        return '<' + ', '.join(map(str, self.values)) + '>'

class MyContainer2(MyContainer):
    @recursive_repr('+++')
    def __repr__(self):
        return '<' + ', '.join(map(str, self.values)) + '>'

class MyContainer3:
    def __repr__(self):
        'Test document content'
        pass
    wrapped = __repr__
    wrapper = recursive_repr()(wrapped)

class TestRecursiveRepr(CPythonTestCase):
    def test_recursive_repr(self):
        m = MyContainer(list('abcde'))
        m.append(m)
        m.append('x')
        m.append(m)
        self.assertEqual(repr(m), '<a, b, c, d, e, ..., x, ...>')
        m = MyContainer2(list('abcde'))
        m.append(m)
        m.append('x')
        m.append(m)
        self.assertEqual(repr(m), '<a, b, c, d, e, +++, x, +++>')

    def test_assigned_attributes(self):
        from functools import WRAPPER_ASSIGNMENTS as assigned
        wrapped = MyContainer3.wrapped
        wrapper = MyContainer3.wrapper
        for name in assigned:
            self.assertIs(getattr(wrapper, name), getattr(wrapped, name))

    def test__wrapped__(self):
        class X:
            def __repr__(self):
                return 'X()'
            f = __repr__ # save reference to check it later
            __repr__ = recursive_repr()(__repr__)

        self.assertIs(X.f, X.__repr__.__wrapped__)

    def test__type_params__(self):
        class My:
            @recursive_repr()
            def __repr__[T: str](self, default: T = '') -> str:
                return default

        type_params = My().__repr__.__type_params__
        self.assertEqual(len(type_params), 1)
        self.assertEqual(type_params[0].__name__, 'T')
        self.assertEqual(type_params[0].__bound__, str)

if __name__ == "__main__":
    run_tests()
