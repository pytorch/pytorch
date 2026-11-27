"""Every junit outcome the pytest path emits, including the <rerun> elements that
test/conftest.py's LogXMLReruns adds. pytest-rerunfailures reruns in-process,
hence module-level attempt counters."""

import pytest


_ATTEMPTS: dict[str, int] = {"rerun_then_pass": 0, "rerun_then_fail": 0}


@pytest.fixture
def broken_setup():
    raise RuntimeError("setup error")


@pytest.fixture
def broken_teardown():
    yield
    raise RuntimeError("teardown error")


class TestJunitOutcomes:
    def test_pass(self) -> None:
        pass

    def test_assert_failure(self) -> None:
        raise AssertionError("values differ")

    def test_raises_non_assertion(self) -> None:
        # <failure> here, <error> on the unittest path
        raise RuntimeError("runtime error!")

    def test_error_in_setup(self, broken_setup) -> None:
        pass

    def test_error_in_teardown(self, broken_teardown) -> None:
        pass

    @pytest.mark.skip(reason="skipped unconditionally")
    def test_skipped(self) -> None:
        pass

    @pytest.mark.skipif(True, reason="skipped conditionally")
    def test_skipif(self) -> None:
        pass

    @pytest.mark.xfail(reason="known bad", strict=False)
    def test_xfail(self) -> None:
        raise AssertionError("expected to fail")

    @pytest.mark.xfail(reason="unexpectedly fixed", strict=False)
    def test_xpass_non_strict(self) -> None:
        pass

    @pytest.mark.xfail(reason="strictly expected to fail", strict=True)
    def test_xpass_strict(self) -> None:
        pass

    @pytest.mark.flaky(reruns=2)
    def test_rerun_then_pass(self) -> None:
        _ATTEMPTS["rerun_then_pass"] += 1
        if _ATTEMPTS["rerun_then_pass"] < 3:
            raise AssertionError(f"attempt {_ATTEMPTS['rerun_then_pass']} fails")

    @pytest.mark.flaky(reruns=2)
    def test_rerun_then_fail(self) -> None:
        _ATTEMPTS["rerun_then_fail"] += 1
        raise AssertionError(f"attempt {_ATTEMPTS['rerun_then_fail']} fails")

    @pytest.mark.flaky(reruns=2)
    def test_no_rerun_needed(self) -> None:
        pass
