"""One passing and one failing test, run with a broken report writer
(TestReportFailureIsolation)."""


def test_pass():
    pass


def test_fail():
    raise RuntimeError("expected failure")
