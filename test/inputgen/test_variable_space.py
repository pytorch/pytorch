# Owner(s): ["module: tests"]

from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inputgen.variable.space import (
    Discrete,
    Interval,
    Intervals,
)


class TestDiscrete(TestCase):
    def test_init(self):
        values = Discrete()
        self.assertEqual(values.values, set())
        self.assertEqual(values.initialized, False)
        values = Discrete([1, 2, 3])
        self.assertEqual(values.values, {1, 2, 3})
        self.assertEqual(values.initialized, True)

    def test_empty(self):
        with self.assertRaises(Exception):
            Discrete().empty()
        self.assertTrue(Discrete([]).empty())
        self.assertFalse(Discrete([1]).empty())

    def test_contains(self):
        with self.assertRaises(Exception):
            Discrete().contains(1)
        values = Discrete([1, "a", False, 0.5])
        self.assertTrue(values.contains(True))
        self.assertTrue(values.contains(1))
        self.assertTrue(values.contains(1.0))
        self.assertTrue(values.contains(False))
        self.assertTrue(values.contains(0))
        self.assertTrue(values.contains(0.0))
        self.assertTrue(values.contains("a"))
        self.assertTrue(values.contains(0.5))
        self.assertFalse(values.contains(2))
        self.assertFalse(values.contains("b"))
        self.assertFalse(values.contains(2.5))
        self.assertFalse(values.contains(float("inf")))
        self.assertFalse(values.contains(float("-inf")))

        values = Discrete([1, float("inf")])
        self.assertTrue(values.contains(1))
        self.assertTrue(values.contains(float("inf")))
        self.assertFalse(values.contains(float("-inf")))

    def test_remove(self):
        with self.assertRaises(Exception):
            Discrete().remove(1)
        values = Discrete([1, "a", False, 0.5])
        self.assertTrue(values.contains("a"))
        values.remove("a")
        self.assertEqual(values.values, {1, False, 0.5})

    def test_filter(self):
        with self.assertRaises(Exception):
            Discrete().filter(lambda v: True)
        values = Discrete([1, "a", False, 0.5])
        values.filter(lambda v: isinstance(v, int))
        self.assertEqual(values.values, {1, False})

    def test_nan_error(self):
        with self.assertRaises(Exception):
            values = Discrete([1, 2, 3, float("nan")])


class TestInterval(TestCase):
    def test_empty(self):
        self.assertFalse(Interval().empty())
        self.assertFalse(Interval(-3, 4).empty())
        self.assertFalse(Interval(5.7, 5.7).empty())
        self.assertTrue(Interval(5.7, 5.7, True, False).empty())
        self.assertTrue(Interval(5.7, 5.7, False, True).empty())
        self.assertTrue(Interval(5.8, 5.7).empty())

    def test_contains(self):
        self.assertTrue(Interval(-3, 4).contains(1))
        self.assertTrue(Interval(-3, 4).contains(-3))
        self.assertTrue(Interval(-3, 4).contains(4))
        self.assertFalse(Interval(-3, 4).contains(5))
        self.assertFalse(Interval(-3, 4).contains(-4))
        self.assertFalse(Interval(-3, 4).contains(float("inf")))
        self.assertFalse(Interval(-3, 4, True, False).contains(-3))
        self.assertTrue(Interval(-3, 4, True, False).contains(-2.999))
        self.assertTrue(Interval(-3, 4, True, False).contains(4))
        self.assertFalse(Interval(-3, 4, False, True).contains(4))
        self.assertTrue(Interval(-3, 4, False, True).contains(-3))
        self.assertTrue(Interval(-3, 4, False, True).contains(3.999))

    def test_contains_int(self):
        self.assertTrue(Interval().contains_int())
        self.assertTrue(Interval(5.3, float("inf")).contains_int())
        self.assertTrue(Interval(float("-inf"), 5.3).contains_int())
        self.assertTrue(Interval(3, 4).contains_int())
        self.assertTrue(Interval(3, 4, True, False).contains_int())
        self.assertTrue(Interval(3, 4, False, True).contains_int())
        self.assertFalse(Interval(3, 4, True, True).contains_int())
        self.assertFalse(Interval(3.001, 3.999).contains_int())
        self.assertTrue(Interval(3.999, 4.001).contains_int())


class TestIntervals(TestCase):
    def test_empty(self):
        self.assertTrue(Intervals([]).empty())
        self.assertTrue(Intervals([Interval(3, 3, True, True)]).empty())
        self.assertTrue(Intervals([Interval(3, 3, True, False)]).empty())
        self.assertTrue(
            Intervals(
                [Interval(3, 3, True, False), Interval(4, 4, False, True)]
            ).empty()
        )
        self.assertFalse(
            Intervals(
                [Interval(3, 3, True, False), Interval(4, 4.001, False, True)]
            ).empty()
        )

    def test_contains(self):
        self.assertFalse(Intervals([]).contains(1))
        intervals = Intervals(
            [Interval(-2, 3, True, False), Interval(4.3, 5.7, False, True)]
        )
        self.assertFalse(intervals.contains(-2.1))
        self.assertFalse(intervals.contains(-2))
        self.assertTrue(intervals.contains(False))
        self.assertTrue(intervals.contains(True))
        self.assertTrue(intervals.contains(3))
        self.assertFalse(intervals.contains(3.5))
        self.assertTrue(intervals.contains(4.3))
        self.assertTrue(intervals.contains(5))
        self.assertFalse(intervals.contains(5.7))
        self.assertFalse(intervals.contains(6))

    def test_remove(self):
        # start with full range interval
        intervals = Intervals()
        self.assertEqual(str(intervals), "[-inf, inf]")

        # remove 1
        self.assertTrue(intervals.contains(1))
        intervals.remove(1)
        self.assertEqual(str(intervals), "[-inf, 1) (1, inf]")
        self.assertFalse(intervals.contains(1))

        # remove inf
        self.assertTrue(intervals.contains(float("inf")))
        intervals.remove(float("inf"))
        self.assertEqual(str(intervals), "[-inf, 1) (1, inf)")
        self.assertFalse(intervals.contains(float("inf")))

        # remove 2
        self.assertTrue(intervals.contains(2))
        intervals.remove(2)
        self.assertEqual(str(intervals), "[-inf, 1) (1, 2) (2, inf)")
        self.assertFalse(intervals.contains(2))

        # start with a separate collection of intervals
        intervals = Intervals([Interval(3, 4), Interval(5, 6)])
        self.assertEqual(str(intervals), "[3, 4] [5, 6]")
        self.assertTrue(intervals.contains(4))
        intervals.remove(4)
        self.assertEqual(str(intervals), "[3, 4) [5, 6]")
        self.assertFalse(intervals.contains(4))

    def test_lower_and_upper(self):
        # start with full range interval
        intervals = Intervals()
        self.assertEqual(str(intervals), "[-inf, inf]")

        intervals.set_lower(3, True)
        self.assertEqual(str(intervals), "(3, inf]")

        intervals.set_upper(float("inf"), True)
        self.assertEqual(str(intervals), "(3, inf)")

        intervals.set_upper(1e300, False)
        self.assertEqual(str(intervals), "(3, 1e+300]")

        intervals.set_lower(2, False)
        self.assertEqual(str(intervals), "(3, 1e+300]")

        intervals.set_lower(3, True)
        self.assertEqual(str(intervals), "(3, 1e+300]")

        # start with a separate collection of intervals
        intervals = Intervals([Interval(3, 4), Interval(5, 6)])
        self.assertEqual(str(intervals), "[3, 4] [5, 6]")

        intervals.set_lower(3, True)
        self.assertEqual(str(intervals), "(3, 4] [5, 6]")

        intervals.set_upper(5, False)
        self.assertEqual(str(intervals), "(3, 4] [5, 5]")

        intervals.set_upper(5, True)
        self.assertEqual(str(intervals), "(3, 4]")

        # start with a separate collection of intervals
        intervals = Intervals([Interval(3, 4), Interval(5, 6, True, False)])
        self.assertEqual(str(intervals), "[3, 4] (5, 6]")

        intervals.set_upper(5, False)
        self.assertEqual(str(intervals), "[3, 4]")

        # start with a separate collection of intervals
        intervals = Intervals(
            [Interval(3, 4, False, True), Interval(5, 6, True, False)]
        )
        self.assertEqual(str(intervals), "[3, 4) (5, 6]")

        intervals.set_lower(4, False)
        self.assertEqual(str(intervals), "(5, 6]")


if __name__ == "__main__":
    run_tests()
