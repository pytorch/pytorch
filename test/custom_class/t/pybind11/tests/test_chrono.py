from pybind11_tests import chrono as m
import datetime


def test_chrono_system_clock():

    # Get the time from both c++ and datetime
    date1 = m.test_chrono1()
    date2 = datetime.datetime.today()

    # The returned value should be a datetime
    assert isinstance(date1, datetime.datetime)

    # The numbers should vary by a very small amount (time it took to execute)
    diff = abs(date1 - date2)

    # There should never be a days/seconds difference
    assert diff.days == 0
    assert diff.seconds == 0

    # We test that no more than about 0.5 seconds passes here
    # This makes sure that the dates created are very close to the same
    # but if the testing system is incredibly overloaded this should still pass
    assert diff.microseconds < 500000


def test_chrono_system_clock_roundtrip():
    date1 = datetime.datetime.today()

    # Roundtrip the time
    date2 = m.test_chrono2(date1)

    # The returned value should be a datetime
    assert isinstance(date2, datetime.datetime)

    # They should be identical (no information lost on roundtrip)
    diff = abs(date1 - date2)
    assert diff.days == 0
    assert diff.seconds == 0
    assert diff.microseconds == 0


def test_chrono_duration_roundtrip():

    # Get the difference between two times (a timedelta)
    date1 = datetime.datetime.today()
    date2 = datetime.datetime.today()
    diff = date2 - date1

    # Make sure this is a timedelta
    assert isinstance(diff, datetime.timedelta)

    cpp_diff = m.test_chrono3(diff)

    assert cpp_diff.days == diff.days
    assert cpp_diff.seconds == diff.seconds
    assert cpp_diff.microseconds == diff.microseconds


def test_chrono_duration_subtraction_equivalence():

    date1 = datetime.datetime.today()
    date2 = datetime.datetime.today()

    diff = date2 - date1
    cpp_diff = m.test_chrono4(date2, date1)

    assert cpp_diff.days == diff.days
    assert cpp_diff.seconds == diff.seconds
    assert cpp_diff.microseconds == diff.microseconds


def test_chrono_steady_clock():
    time1 = m.test_chrono5()
    assert isinstance(time1, datetime.timedelta)


def test_chrono_steady_clock_roundtrip():
    time1 = datetime.timedelta(days=10, seconds=10, microseconds=100)
    time2 = m.test_chrono6(time1)

    assert isinstance(time2, datetime.timedelta)

    # They should be identical (no information lost on roundtrip)
    assert time1.days == time2.days
    assert time1.seconds == time2.seconds
    assert time1.microseconds == time2.microseconds


def test_floating_point_duration():
    # Test using a floating point number in seconds
    time = m.test_chrono7(35.525123)

    assert isinstance(time, datetime.timedelta)

    assert time.seconds == 35
    assert 525122 <= time.microseconds <= 525123

    diff = m.test_chrono_float_diff(43.789012, 1.123456)
    assert diff.seconds == 42
    assert 665556 <= diff.microseconds <= 665557
