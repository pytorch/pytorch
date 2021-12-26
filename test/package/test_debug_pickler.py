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

    def test_set(self):
        from package_a.bad_pickle import BadPickle, GoodPickle
        lst = [GoodPickle(), GoodPickle() ,BadPickle()]
        obj = [
            GoodPickle(),
            set(lst),
        ]
        with self.assertRaises(PicklingError) as e3:
            debug_dumps(sys_importer, obj, protocol=3)

        with self.assertRaises(PicklingError) as e4:
            debug_dumps(sys_importer, obj, protocol=4)
        error = dedent(
                """\
                I can't be pickled!.

                We think the problematic object is found at:
                <pickled object> (<class 'list'>)
                  <object @ idx 1> (<class 'set'>)
                  <object BadPickle> (<class 'package_a.bad_pickle.BadPickle'>)
                """
            )
        self.assertEqual(str(e3.exception), error)
        self.assertEqual(str(e4.exception), error)

    def test_frozenset(self):
        from package_a.bad_pickle import BadPickle, GoodPickle
        lst = [GoodPickle(), GoodPickle() ,BadPickle()]
        obj = [
            GoodPickle(),
            frozenset(lst),
        ]
        with self.assertRaises(PicklingError) as e3:
            debug_dumps(sys_importer, obj, protocol=3)

        with self.assertRaises(PicklingError) as e4:
            debug_dumps(sys_importer, obj, protocol=4)
        error = dedent(
                """\
                I can't be pickled!.

                We think the problematic object is found at:
                <pickled object> (<class 'list'>)
                  <object @ idx 1> (<class 'frozenset'>)
                  <object BadPickle> (<class 'package_a.bad_pickle.BadPickle'>)
                """
            )
        self.assertEqual(str(e3.exception), error)
        self.assertEqual(str(e4.exception), error)

    def test_tuple(self):
        from package_a.bad_pickle import BadPickle, GoodPickle
        obj1 = [
            GoodPickle(),
            (GoodPickle(), GoodPickle() ,BadPickle()),
        ]
        obj2 = [
            GoodPickle(),
            (GoodPickle(), GoodPickle(), BadPickle(), GoodPickle(), GoodPickle()),
        ]
        with self.assertRaises(PicklingError) as e1_protocol3:
            debug_dumps(sys_importer, obj1, protocol=3)
        with self.assertRaises(PicklingError) as e2_protocol3:
            debug_dumps(sys_importer, obj2, protocol=3)
        with self.assertRaises(PicklingError) as e1_protocol4:
            debug_dumps(sys_importer, obj1, protocol=4)
        with self.assertRaises(PicklingError) as e2_protocol4:
            debug_dumps(sys_importer, obj2, protocol=4)

        error = dedent(
                """\
                I can't be pickled!.

                We think the problematic object is found at:
                <pickled object> (<class 'list'>)
                  <object @ idx 1> (<class 'tuple'>)
                  <object @ idx 2> (<class 'package_a.bad_pickle.BadPickle'>)
                """
            )

        self.assertEqual(str(e1_protocol3.exception), error)
        self.assertEqual(str(e2_protocol3.exception), error)
        self.assertEqual(str(e1_protocol4.exception), error)
        self.assertEqual(str(e2_protocol4.exception), error)

    def test_basic_msg(self):
        from package_a.bad_pickle import BadPickle, GoodPickle

        a = GoodPickle()
        a.b = GoodPickle()
        a.b.c = GoodPickle()
        a.b.c.d = BadPickle()
        with self.assertRaises(PicklingError) as e3:
            debug_dumps(sys_importer, a, protocol=3)
        with self.assertRaises(PicklingError) as e4:
            debug_dumps(sys_importer, a, protocol=4)
        error = dedent(
                """\
                I can't be pickled!.

                We think the problematic object is found at:
                <pickled object> (<class 'package_a.bad_pickle.GoodPickle'>)
                  .b (<class 'package_a.bad_pickle.GoodPickle'>)
                  .c (<class 'package_a.bad_pickle.GoodPickle'>)
                  .d (<class 'package_a.bad_pickle.BadPickle'>)
                """
            )
        self.assertEqual(str(e3.exception), error)
        self.assertEqual(str(e4.exception), error)

    def test_nested_lists(self):
        from package_a.bad_pickle import BadPickle, GoodPickle

        obj = [
            GoodPickle(),
            [GoodPickle(), GoodPickle(), [GoodPickle(), [BadPickle()]]],
        ]
        self.assertTrue(isinstance(obj[1][2][1][0], BadPickle))
        with self.assertRaises(PicklingError) as e3:
            debug_dumps(sys_importer, obj, protocol=3)

        with self.assertRaises(PicklingError) as e4:
            debug_dumps(sys_importer, obj, protocol=4)
        error = dedent(
                """\
                I can't be pickled!.

                We think the problematic object is found at:
                <pickled object> (<class 'list'>)
                  <object @ idx 1> (<class 'list'>)
                  <object @ idx 2> (<class 'list'>)
                  <object @ idx 1> (<class 'list'>)
                  <object @ idx 0> (<class 'package_a.bad_pickle.BadPickle'>)
                """
            )
        self.assertEqual(str(e3.exception), error)
        self.assertEqual(str(e4.exception), error)

    def test_nested_list_obj(self):
        from package_a.bad_pickle import BadPickle, GoodPickle

        a = GoodPickle()
        a.b = GoodPickle()
        a.b.c = GoodPickle()
        a.b.c.d = BadPickle()
        obj = [GoodPickle(), [GoodPickle(), GoodPickle(), [GoodPickle(), [a]]]]
        with self.assertRaises(PicklingError) as e3:
            debug_dumps(sys_importer, obj, protocol=3)

        with self.assertRaises(PicklingError) as e4:
            debug_dumps(sys_importer, obj, protocol=4)
        error = dedent(
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
            )
        self.assertEqual(str(e3.exception), error)
        self.assertEqual(str(e4.exception), error)

    def test_in_list_msg(self):
        from package_a.bad_pickle import BadPickle, GoodPickle

        a = GoodPickle()
        a.bad_list = [GoodPickle(), GoodPickle(), BadPickle()]
        with self.assertRaises(PicklingError) as e3:
            debug_dumps(sys_importer, a, protocol=3)
        with self.assertRaises(PicklingError) as e4:
            debug_dumps(sys_importer, a, protocol=4)
        error = dedent(
                """\
                I can't be pickled!.

                We think the problematic object is found at:
                <pickled object> (<class 'package_a.bad_pickle.GoodPickle'>)
                  .bad_list (<class 'list'>)
                  <object @ idx 2> (<class 'package_a.bad_pickle.BadPickle'>)
                """
            )
        self.assertEqual(str(e3.exception), error)
        self.assertEqual(str(e4.exception), error)

    def test_dict(self):
        from package_a.bad_pickle import BadPickle, GoodPickle

        obj = {"foo": GoodPickle(), "bar": BadPickle()}
        with self.assertRaises(PicklingError) as e3:
            debug_dumps(sys_importer, obj, protocol=3)

        with self.assertRaises(PicklingError) as e4:
            debug_dumps(sys_importer, obj, protocol=4)
        error = dedent(
                """\
                I can't be pickled!.

                We think the problematic object is found at:
                <pickled object> (<class 'dict'>)
                  <object @ key bar> (<class 'package_a.bad_pickle.BadPickle'>)
                """
            )
        self.assertEqual(str(e3.exception), error)
        self.assertEqual(str(e4.exception), error)

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

        with self.assertRaises(PicklingError) as e3:
            debug_dumps(sys_importer, obj, protocol=3)

        with self.assertRaises(PicklingError) as e4:
            debug_dumps(sys_importer, obj, protocol=4)
        error = dedent(
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
            )
        self.assertEqual(str(e3.exception), error)
        self.assertEqual(str(e4.exception), error)

    def test_good_pickle(self):
        """Passing an object that actually pickles should raise a ValueError."""
        from package_a.bad_pickle import GoodPickle

        with self.assertRaises(ValueError):
            debug_dumps(sys_importer, GoodPickle())


if __name__ == "__main__":
    run_tests()
