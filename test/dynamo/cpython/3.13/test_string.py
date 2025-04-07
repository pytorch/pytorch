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

import unittest
import string
from string import Template


class ModuleTest(__TestCase):

    def test_attrs(self):
        # While the exact order of the items in these attributes is not
        # technically part of the "language spec", in practice there is almost
        # certainly user code that depends on the order, so de-facto it *is*
        # part of the spec.
        self.assertEqual(string.whitespace, ' \t\n\r\x0b\x0c')
        self.assertEqual(string.ascii_lowercase, 'abcdefghijklmnopqrstuvwxyz')
        self.assertEqual(string.ascii_uppercase, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.assertEqual(string.ascii_letters, string.ascii_lowercase + string.ascii_uppercase)
        self.assertEqual(string.digits, '0123456789')
        self.assertEqual(string.hexdigits, string.digits + 'abcdefABCDEF')
        self.assertEqual(string.octdigits, '01234567')
        self.assertEqual(string.punctuation, '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        self.assertEqual(string.printable, string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation + string.whitespace)

    def test_capwords(self):
        self.assertEqual(string.capwords('abc def ghi'), 'Abc Def Ghi')
        self.assertEqual(string.capwords('abc\tdef\nghi'), 'Abc Def Ghi')
        self.assertEqual(string.capwords('abc\t   def  \nghi'), 'Abc Def Ghi')
        self.assertEqual(string.capwords('ABC DEF GHI'), 'Abc Def Ghi')
        self.assertEqual(string.capwords('ABC-DEF-GHI', '-'), 'Abc-Def-Ghi')
        self.assertEqual(string.capwords('ABC-def DEF-ghi GHI'), 'Abc-def Def-ghi Ghi')
        self.assertEqual(string.capwords('   aBc  DeF   '), 'Abc Def')
        self.assertEqual(string.capwords('\taBc\tDeF\t'), 'Abc Def')
        self.assertEqual(string.capwords('\taBc\tDeF\t', '\t'), '\tAbc\tDef\t')

    def test_basic_formatter(self):
        fmt = string.Formatter()
        self.assertEqual(fmt.format("foo"), "foo")
        self.assertEqual(fmt.format("foo{0}", "bar"), "foobar")
        self.assertEqual(fmt.format("foo{1}{0}-{1}", "bar", 6), "foo6bar-6")
        self.assertRaises(TypeError, fmt.format)
        self.assertRaises(TypeError, string.Formatter.format)

    def test_format_keyword_arguments(self):
        fmt = string.Formatter()
        self.assertEqual(fmt.format("-{arg}-", arg='test'), '-test-')
        self.assertRaises(KeyError, fmt.format, "-{arg}-")
        self.assertEqual(fmt.format("-{self}-", self='test'), '-test-')
        self.assertRaises(KeyError, fmt.format, "-{self}-")
        self.assertEqual(fmt.format("-{format_string}-", format_string='test'),
                         '-test-')
        self.assertRaises(KeyError, fmt.format, "-{format_string}-")
        with self.assertRaisesRegex(TypeError, "format_string"):
            fmt.format(format_string="-{arg}-", arg='test')

    def test_auto_numbering(self):
        fmt = string.Formatter()
        self.assertEqual(fmt.format('foo{}{}', 'bar', 6),
                         'foo{}{}'.format('bar', 6))
        self.assertEqual(fmt.format('foo{1}{num}{1}', None, 'bar', num=6),
                         'foo{1}{num}{1}'.format(None, 'bar', num=6))
        self.assertEqual(fmt.format('{:^{}}', 'bar', 6),
                         '{:^{}}'.format('bar', 6))
        self.assertEqual(fmt.format('{:^{}} {}', 'bar', 6, 'X'),
                         '{:^{}} {}'.format('bar', 6, 'X'))
        self.assertEqual(fmt.format('{:^{pad}}{}', 'foo', 'bar', pad=6),
                         '{:^{pad}}{}'.format('foo', 'bar', pad=6))

        with self.assertRaises(ValueError):
            fmt.format('foo{1}{}', 'bar', 6)

        with self.assertRaises(ValueError):
            fmt.format('foo{}{1}', 'bar', 6)

    def test_conversion_specifiers(self):
        fmt = string.Formatter()
        self.assertEqual(fmt.format("-{arg!r}-", arg='test'), "-'test'-")
        self.assertEqual(fmt.format("{0!s}", 'test'), 'test')
        self.assertRaises(ValueError, fmt.format, "{0!h}", 'test')
        # issue13579
        self.assertEqual(fmt.format("{0!a}", 42), '42')
        self.assertEqual(fmt.format("{0!a}",  string.ascii_letters),
            "'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'")
        self.assertEqual(fmt.format("{0!a}",  chr(255)), "'\\xff'")
        self.assertEqual(fmt.format("{0!a}",  chr(256)), "'\\u0100'")

    def test_name_lookup(self):
        fmt = string.Formatter()
        class AnyAttr:
            def __getattr__(self, attr):
                return attr
        x = AnyAttr()
        self.assertEqual(fmt.format("{0.lumber}{0.jack}", x), 'lumberjack')
        with self.assertRaises(AttributeError):
            fmt.format("{0.lumber}{0.jack}", '')

    def test_index_lookup(self):
        fmt = string.Formatter()
        lookup = ["eggs", "and", "spam"]
        self.assertEqual(fmt.format("{0[2]}{0[0]}", lookup), 'spameggs')
        with self.assertRaises(IndexError):
            fmt.format("{0[2]}{0[0]}", [])
        with self.assertRaises(KeyError):
            fmt.format("{0[2]}{0[0]}", {})

    def test_override_get_value(self):
        class NamespaceFormatter(string.Formatter):
            def __init__(self, namespace={}):
                string.Formatter.__init__(self)
                self.namespace = namespace

            def get_value(self, key, args, kwds):
                if isinstance(key, str):
                    try:
                        # Check explicitly passed arguments first
                        return kwds[key]
                    except KeyError:
                        return self.namespace[key]
                else:
                    string.Formatter.get_value(key, args, kwds)

        fmt = NamespaceFormatter({'greeting':'hello'})
        self.assertEqual(fmt.format("{greeting}, world!"), 'hello, world!')


    def test_override_format_field(self):
        class CallFormatter(string.Formatter):
            def format_field(self, value, format_spec):
                return format(value(), format_spec)

        fmt = CallFormatter()
        self.assertEqual(fmt.format('*{0}*', lambda : 'result'), '*result*')


    def test_override_convert_field(self):
        class XFormatter(string.Formatter):
            def convert_field(self, value, conversion):
                if conversion == 'x':
                    return None
                return super().convert_field(value, conversion)

        fmt = XFormatter()
        self.assertEqual(fmt.format("{0!r}:{0!x}", 'foo', 'foo'), "'foo':None")


    def test_override_parse(self):
        class BarFormatter(string.Formatter):
            # returns an iterable that contains tuples of the form:
            # (literal_text, field_name, format_spec, conversion)
            def parse(self, format_string):
                for field in format_string.split('|'):
                    if field[0] == '+':
                        # it's markup
                        field_name, _, format_spec = field[1:].partition(':')
                        yield '', field_name, format_spec, None
                    else:
                        yield field, None, None, None

        fmt = BarFormatter()
        self.assertEqual(fmt.format('*|+0:^10s|*', 'foo'), '*   foo    *')

    def test_check_unused_args(self):
        class CheckAllUsedFormatter(string.Formatter):
            def check_unused_args(self, used_args, args, kwargs):
                # Track which arguments actually got used
                unused_args = set(kwargs.keys())
                unused_args.update(range(0, len(args)))

                for arg in used_args:
                    unused_args.remove(arg)

                if unused_args:
                    raise ValueError("unused arguments")

        fmt = CheckAllUsedFormatter()
        self.assertEqual(fmt.format("{0}", 10), "10")
        self.assertEqual(fmt.format("{0}{i}", 10, i=100), "10100")
        self.assertEqual(fmt.format("{0}{i}{1}", 10, 20, i=100), "1010020")
        self.assertRaises(ValueError, fmt.format, "{0}{i}{1}", 10, 20, i=100, j=0)
        self.assertRaises(ValueError, fmt.format, "{0}", 10, 20)
        self.assertRaises(ValueError, fmt.format, "{0}", 10, 20, i=100)
        self.assertRaises(ValueError, fmt.format, "{i}", 10, 20, i=100)

    def test_vformat_recursion_limit(self):
        fmt = string.Formatter()
        args = ()
        kwargs = dict(i=100)
        with self.assertRaises(ValueError) as err:
            fmt._vformat("{i}", args, kwargs, set(), -1)
        self.assertIn("recursion", str(err.exception))


# Template tests (formerly housed in test_pep292.py)

class Bag:
    pass

class Mapping:
    def __getitem__(self, name):
        obj = self
        for part in name.split('.'):
            try:
                obj = getattr(obj, part)
            except AttributeError:
                raise KeyError(name)
        return obj


class TestTemplate(__TestCase):
    def test_regular_templates(self):
        s = Template('$who likes to eat a bag of $what worth $$100')
        self.assertEqual(s.substitute(dict(who='tim', what='ham')),
                         'tim likes to eat a bag of ham worth $100')
        self.assertRaises(KeyError, s.substitute, dict(who='tim'))
        self.assertRaises(TypeError, Template.substitute)

    def test_regular_templates_with_braces(self):
        s = Template('$who likes ${what} for ${meal}')
        d = dict(who='tim', what='ham', meal='dinner')
        self.assertEqual(s.substitute(d), 'tim likes ham for dinner')
        self.assertRaises(KeyError, s.substitute,
                          dict(who='tim', what='ham'))

    def test_regular_templates_with_upper_case(self):
        s = Template('$WHO likes ${WHAT} for ${MEAL}')
        d = dict(WHO='tim', WHAT='ham', MEAL='dinner')
        self.assertEqual(s.substitute(d), 'tim likes ham for dinner')

    def test_regular_templates_with_non_letters(self):
        s = Template('$_wh0_ likes ${_w_h_a_t_} for ${mea1}')
        d = dict(_wh0_='tim', _w_h_a_t_='ham', mea1='dinner')
        self.assertEqual(s.substitute(d), 'tim likes ham for dinner')

    def test_escapes(self):
        eq = self.assertEqual
        s = Template('$who likes to eat a bag of $$what worth $$100')
        eq(s.substitute(dict(who='tim', what='ham')),
           'tim likes to eat a bag of $what worth $100')
        s = Template('$who likes $$')
        eq(s.substitute(dict(who='tim', what='ham')), 'tim likes $')

    def test_percents(self):
        eq = self.assertEqual
        s = Template('%(foo)s $foo ${foo}')
        d = dict(foo='baz')
        eq(s.substitute(d), '%(foo)s baz baz')
        eq(s.safe_substitute(d), '%(foo)s baz baz')

    def test_stringification(self):
        eq = self.assertEqual
        s = Template('tim has eaten $count bags of ham today')
        d = dict(count=7)
        eq(s.substitute(d), 'tim has eaten 7 bags of ham today')
        eq(s.safe_substitute(d), 'tim has eaten 7 bags of ham today')
        s = Template('tim has eaten ${count} bags of ham today')
        eq(s.substitute(d), 'tim has eaten 7 bags of ham today')

    def test_tupleargs(self):
        eq = self.assertEqual
        s = Template('$who ate ${meal}')
        d = dict(who=('tim', 'fred'), meal=('ham', 'kung pao'))
        eq(s.substitute(d), "('tim', 'fred') ate ('ham', 'kung pao')")
        eq(s.safe_substitute(d), "('tim', 'fred') ate ('ham', 'kung pao')")

    def test_SafeTemplate(self):
        eq = self.assertEqual
        s = Template('$who likes ${what} for ${meal}')
        eq(s.safe_substitute(dict(who='tim')), 'tim likes ${what} for ${meal}')
        eq(s.safe_substitute(dict(what='ham')), '$who likes ham for ${meal}')
        eq(s.safe_substitute(dict(what='ham', meal='dinner')),
           '$who likes ham for dinner')
        eq(s.safe_substitute(dict(who='tim', what='ham')),
           'tim likes ham for ${meal}')
        eq(s.safe_substitute(dict(who='tim', what='ham', meal='dinner')),
           'tim likes ham for dinner')

    def test_invalid_placeholders(self):
        raises = self.assertRaises
        s = Template('$who likes $')
        raises(ValueError, s.substitute, dict(who='tim'))
        s = Template('$who likes ${what)')
        raises(ValueError, s.substitute, dict(who='tim'))
        s = Template('$who likes $100')
        raises(ValueError, s.substitute, dict(who='tim'))
        # Template.idpattern should match to only ASCII characters.
        # https://bugs.python.org/issue31672
        s = Template("$who likes $\u0131")  # (DOTLESS I)
        raises(ValueError, s.substitute, dict(who='tim'))
        s = Template("$who likes $\u0130")  # (LATIN CAPITAL LETTER I WITH DOT ABOVE)
        raises(ValueError, s.substitute, dict(who='tim'))

    def test_idpattern_override(self):
        class PathPattern(Template):
            idpattern = r'[_a-z][._a-z0-9]*'
        m = Mapping()
        m.bag = Bag()
        m.bag.foo = Bag()
        m.bag.foo.who = 'tim'
        m.bag.what = 'ham'
        s = PathPattern('$bag.foo.who likes to eat a bag of $bag.what')
        self.assertEqual(s.substitute(m), 'tim likes to eat a bag of ham')

    def test_flags_override(self):
        class MyPattern(Template):
            flags = 0
        s = MyPattern('$wHO likes ${WHAT} for ${meal}')
        d = dict(wHO='tim', WHAT='ham', meal='dinner', w='fred')
        self.assertRaises(ValueError, s.substitute, d)
        self.assertEqual(s.safe_substitute(d), 'fredHO likes ${WHAT} for dinner')

    def test_idpattern_override_inside_outside(self):
        # bpo-1198569: Allow the regexp inside and outside braces to be
        # different when deriving from Template.
        class MyPattern(Template):
            idpattern = r'[a-z]+'
            braceidpattern = r'[A-Z]+'
            flags = 0
        m = dict(foo='foo', BAR='BAR')
        s = MyPattern('$foo ${BAR}')
        self.assertEqual(s.substitute(m), 'foo BAR')

    def test_idpattern_override_inside_outside_invalid_unbraced(self):
        # bpo-1198569: Allow the regexp inside and outside braces to be
        # different when deriving from Template.
        class MyPattern(Template):
            idpattern = r'[a-z]+'
            braceidpattern = r'[A-Z]+'
            flags = 0
        m = dict(foo='foo', BAR='BAR')
        s = MyPattern('$FOO')
        self.assertRaises(ValueError, s.substitute, m)
        s = MyPattern('${bar}')
        self.assertRaises(ValueError, s.substitute, m)

    def test_pattern_override(self):
        class MyPattern(Template):
            pattern = r"""
            (?P<escaped>@{2})                   |
            @(?P<named>[_a-z][._a-z0-9]*)       |
            @{(?P<braced>[_a-z][._a-z0-9]*)}    |
            (?P<invalid>@)
            """
        m = Mapping()
        m.bag = Bag()
        m.bag.foo = Bag()
        m.bag.foo.who = 'tim'
        m.bag.what = 'ham'
        s = MyPattern('@bag.foo.who likes to eat a bag of @bag.what')
        self.assertEqual(s.substitute(m), 'tim likes to eat a bag of ham')

        class BadPattern(Template):
            pattern = r"""
            (?P<badname>.*)                     |
            (?P<escaped>@{2})                   |
            @(?P<named>[_a-z][._a-z0-9]*)       |
            @{(?P<braced>[_a-z][._a-z0-9]*)}    |
            (?P<invalid>@)                      |
            """
        s = BadPattern('@bag.foo.who likes to eat a bag of @bag.what')
        self.assertRaises(ValueError, s.substitute, {})
        self.assertRaises(ValueError, s.safe_substitute, {})

    def test_braced_override(self):
        class MyTemplate(Template):
            pattern = r"""
            \$(?:
              (?P<escaped>$)                     |
              (?P<named>[_a-z][_a-z0-9]*)        |
              @@(?P<braced>[_a-z][_a-z0-9]*)@@   |
              (?P<invalid>)                      |
           )
           """

        tmpl = 'PyCon in $@@location@@'
        t = MyTemplate(tmpl)
        self.assertRaises(KeyError, t.substitute, {})
        val = t.substitute({'location': 'Cleveland'})
        self.assertEqual(val, 'PyCon in Cleveland')

    def test_braced_override_safe(self):
        class MyTemplate(Template):
            pattern = r"""
            \$(?:
              (?P<escaped>$)                     |
              (?P<named>[_a-z][_a-z0-9]*)        |
              @@(?P<braced>[_a-z][_a-z0-9]*)@@   |
              (?P<invalid>)                      |
           )
           """

        tmpl = 'PyCon in $@@location@@'
        t = MyTemplate(tmpl)
        self.assertEqual(t.safe_substitute(), tmpl)
        val = t.safe_substitute({'location': 'Cleveland'})
        self.assertEqual(val, 'PyCon in Cleveland')

    def test_invalid_with_no_lines(self):
        # The error formatting for invalid templates
        # has a special case for no data that the default
        # pattern can't trigger (always has at least '$')
        # So we craft a pattern that is always invalid
        # with no leading data.
        class MyTemplate(Template):
            pattern = r"""
              (?P<invalid>) |
              unreachable(
                (?P<named>)   |
                (?P<braced>)  |
                (?P<escaped>)
              )
            """
        s = MyTemplate('')
        with self.assertRaises(ValueError) as err:
            s.substitute({})
        self.assertIn('line 1, col 1', str(err.exception))

    def test_unicode_values(self):
        s = Template('$who likes $what')
        d = dict(who='t\xffm', what='f\xfe\fed')
        self.assertEqual(s.substitute(d), 't\xffm likes f\xfe\x0ced')

    def test_keyword_arguments(self):
        eq = self.assertEqual
        s = Template('$who likes $what')
        eq(s.substitute(who='tim', what='ham'), 'tim likes ham')
        eq(s.substitute(dict(who='tim'), what='ham'), 'tim likes ham')
        eq(s.substitute(dict(who='fred', what='kung pao'),
                        who='tim', what='ham'),
           'tim likes ham')
        s = Template('the mapping is $mapping')
        eq(s.substitute(dict(foo='none'), mapping='bozo'),
           'the mapping is bozo')
        eq(s.substitute(dict(mapping='one'), mapping='two'),
           'the mapping is two')

        s = Template('the self is $self')
        eq(s.substitute(self='bozo'), 'the self is bozo')

    def test_keyword_arguments_safe(self):
        eq = self.assertEqual
        raises = self.assertRaises
        s = Template('$who likes $what')
        eq(s.safe_substitute(who='tim', what='ham'), 'tim likes ham')
        eq(s.safe_substitute(dict(who='tim'), what='ham'), 'tim likes ham')
        eq(s.safe_substitute(dict(who='fred', what='kung pao'),
                        who='tim', what='ham'),
           'tim likes ham')
        s = Template('the mapping is $mapping')
        eq(s.safe_substitute(dict(foo='none'), mapping='bozo'),
           'the mapping is bozo')
        eq(s.safe_substitute(dict(mapping='one'), mapping='two'),
           'the mapping is two')
        d = dict(mapping='one')
        raises(TypeError, s.substitute, d, {})
        raises(TypeError, s.safe_substitute, d, {})

        s = Template('the self is $self')
        eq(s.safe_substitute(self='bozo'), 'the self is bozo')

    def test_delimiter_override(self):
        eq = self.assertEqual
        raises = self.assertRaises
        class AmpersandTemplate(Template):
            delimiter = '&'
        s = AmpersandTemplate('this &gift is for &{who} &&')
        eq(s.substitute(gift='bud', who='you'), 'this bud is for you &')
        raises(KeyError, s.substitute)
        eq(s.safe_substitute(gift='bud', who='you'), 'this bud is for you &')
        eq(s.safe_substitute(), 'this &gift is for &{who} &')
        s = AmpersandTemplate('this &gift is for &{who} &')
        raises(ValueError, s.substitute, dict(gift='bud', who='you'))
        eq(s.safe_substitute(), 'this &gift is for &{who} &')

        class PieDelims(Template):
            delimiter = '@'
        s = PieDelims('@who likes to eat a bag of @{what} worth $100')
        self.assertEqual(s.substitute(dict(who='tim', what='ham')),
                         'tim likes to eat a bag of ham worth $100')

    def test_is_valid(self):
        eq = self.assertEqual
        s = Template('$who likes to eat a bag of ${what} worth $$100')
        self.assertTrue(s.is_valid())

        s = Template('$who likes to eat a bag of ${what} worth $100')
        self.assertFalse(s.is_valid())

        # if the pattern has an unrecognized capture group,
        # it should raise ValueError like substitute and safe_substitute do
        class BadPattern(Template):
            pattern = r"""
            (?P<badname>.*)                  |
            (?P<escaped>@{2})                   |
            @(?P<named>[_a-z][._a-z0-9]*)       |
            @{(?P<braced>[_a-z][._a-z0-9]*)}    |
            (?P<invalid>@)                      |
            """
        s = BadPattern('@bag.foo.who likes to eat a bag of @bag.what')
        self.assertRaises(ValueError, s.is_valid)

    def test_get_identifiers(self):
        eq = self.assertEqual
        raises = self.assertRaises
        s = Template('$who likes to eat a bag of ${what} worth $$100')
        ids = s.get_identifiers()
        eq(ids, ['who', 'what'])

        # repeated identifiers only included once
        s = Template('$who likes to eat a bag of ${what} worth $$100; ${who} likes to eat a bag of $what worth $$100')
        ids = s.get_identifiers()
        eq(ids, ['who', 'what'])

        # invalid identifiers are ignored
        s = Template('$who likes to eat a bag of ${what} worth $100')
        ids = s.get_identifiers()
        eq(ids, ['who', 'what'])

        # if the pattern has an unrecognized capture group,
        # it should raise ValueError like substitute and safe_substitute do
        class BadPattern(Template):
            pattern = r"""
            (?P<badname>.*)                  |
            (?P<escaped>@{2})                   |
            @(?P<named>[_a-z][._a-z0-9]*)       |
            @{(?P<braced>[_a-z][._a-z0-9]*)}    |
            (?P<invalid>@)                      |
            """
        s = BadPattern('@bag.foo.who likes to eat a bag of @bag.what')
        self.assertRaises(ValueError, s.get_identifiers)


if __name__ == "__main__":
    if TEST_WITH_TORCHDYNAMO:
        run_tests()
    else:
        unittest.main()
