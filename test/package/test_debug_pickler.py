from pickle import PicklingError
from textwrap import dedent

from torch.package import sys_importer
from torch.package._package_pickler import debug_dumps
from torch.testing._internal.common_utils import run_tests

try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase


class TestDebugPickler(PackageTestCase):
    def test_tuple(self):
        from package_a.bad_pickle import BadPickle, GoodPickle
        obj = [
            GoodPickle(),
            (GoodPickle(), GoodPickle(), BadPickle()),
        ]
        debug_dumps(sys_importer, obj)

    def test_basic_msg(self):
        from package_a.bad_pickle import BadPickle, GoodPickle

        a = GoodPickle()
        a.b = GoodPickle()
        a.b.c = GoodPickle()
        a.b.c.d = BadPickle()
        with self.assertRaises(PicklingError) as e:
            debug_dumps(sys_importer, a)

        self.assertEqual(
            str(e.exception),
            dedent(
                """\
                I can't be pickled!.

                We think the problematic object is found at:
                <pickled object> (<class 'package_a.bad_pickle.GoodPickle'>)
                  .b (<class 'package_a.bad_pickle.GoodPickle'>)
                  .c (<class 'package_a.bad_pickle.GoodPickle'>)
                  .d (<class 'package_a.bad_pickle.BadPickle'>)
                """
            ),
        )

    def test_nested_lists(self):
        from package_a.bad_pickle import BadPickle, GoodPickle

        obj = [
            GoodPickle(),
            [GoodPickle(), GoodPickle(), [GoodPickle(), [BadPickle()]]],
        ]
        self.assertTrue(isinstance(obj[1][2][1][0], BadPickle))
        with self.assertRaises(PicklingError) as e:
            debug_dumps(sys_importer, obj)

        self.assertEqual(
            str(e.exception),
            dedent(
                """\
                I can't be pickled!.

                We think the problematic object is found at:
                <pickled object> (<class 'list'>)
                  <object @ idx 1> (<class 'list'>)
                  <object @ idx 2> (<class 'list'>)
                  <object @ idx 1> (<class 'list'>)
                  <object @ idx 0> (<class 'package_a.bad_pickle.BadPickle'>)
                """
            ),
        )

    def test_nested_list_obj(self):
        from package_a.bad_pickle import BadPickle, GoodPickle

        a = GoodPickle()
        a.b = GoodPickle()
        a.b.c = GoodPickle()
        a.b.c.d = BadPickle()
        obj = [GoodPickle(), [GoodPickle(), GoodPickle(), [GoodPickle(), [a]]]]
        with self.assertRaises(PicklingError) as e:
            debug_dumps(sys_importer, obj)

        self.assertEqual(
            str(e.exception),
            dedent(
                """\
                I can't be pickled!.

                We think the problematic object is found at:
                <pickled object> (<class 'list'>)
                  <object @ idx 1> (<class 'list'>)
                  <object @ idx 2> (<class 'list'>)
                  <object @ idx 1> (<class 'list'>)
                  <object @ idx 0> (<class 'package_a.bad_pickle.GoodPickle'>)
                  .b (<class 'package_a.bad_pickle.GoodPickle'>)
                  .c (<class 'package_a.bad_pickle.GoodPickle'>)
                  .d (<class 'package_a.bad_pickle.BadPickle'>)
                """
            ),
        )

    def test_in_list_msg(self):
        from package_a.bad_pickle import BadPickle, GoodPickle

        a = GoodPickle()
        a.bad_list = [GoodPickle(), GoodPickle(), BadPickle()]
        with self.assertRaises(PicklingError) as e:
            debug_dumps(sys_importer, a)

        self.assertEqual(
            str(e.exception),
            dedent(
                """\
                I can't be pickled!.

                We think the problematic object is found at:
                <pickled object> (<class 'package_a.bad_pickle.GoodPickle'>)
                  .bad_list (<class 'list'>)
                  <object @ idx 2> (<class 'package_a.bad_pickle.BadPickle'>)
                """
            ),
        )

    def test_dict(self):
        from package_a.bad_pickle import BadPickle, GoodPickle

        obj = {"foo": GoodPickle(), "bar": BadPickle()}
        with self.assertRaises(PicklingError) as e:
            debug_dumps(sys_importer, obj)

        self.assertEqual(
            str(e.exception),
            dedent(
                """\
                I can't be pickled!.

                We think the problematic object is found at:
                <pickled object> (<class 'dict'>)
                  <object @ key bar> (<class 'package_a.bad_pickle.BadPickle'>)
                """
            ),
        )

    def test_nested_list_dict(self):
        from package_a.bad_pickle import BadPickle, GoodPickle

        a = GoodPickle()
        a.b = GoodPickle()
        a.b.c = GoodPickle()
        a.b.c.d = BadPickle()
        obj = {
            "foo": GoodPickle(),
            "bar": [GoodPickle(), [GoodPickle(), GoodPickle(), [GoodPickle(), [a]]]],
        }

        with self.assertRaises(PicklingError) as e:
            debug_dumps(sys_importer, obj)

        self.assertEqual(
            str(e.exception),
            dedent(
                """\
                I can't be pickled!.

                We think the problematic object is found at:
                <pickled object> (<class 'dict'>)
                  <object @ key bar> (<class 'list'>)
                  <object @ idx 1> (<class 'list'>)
                  <object @ idx 2> (<class 'list'>)
                  <object @ idx 1> (<class 'list'>)
                  <object @ idx 0> (<class 'package_a.bad_pickle.GoodPickle'>)
                  .b (<class 'package_a.bad_pickle.GoodPickle'>)
                  .c (<class 'package_a.bad_pickle.GoodPickle'>)
                  .d (<class 'package_a.bad_pickle.BadPickle'>)
                """
            ),
        )

    def test_good_pickle(self):
        """Passing an object that actually pickles should raise a ValueError."""
        from package_a.bad_pickle import GoodPickle

        with self.assertRaises(ValueError):
            debug_dumps(sys_importer, GoodPickle())


if __name__ == "__main__":
    run_tests()
