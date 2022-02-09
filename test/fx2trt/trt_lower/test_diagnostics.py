# Owner(s): ["oncall: aiacc"]
from typing import Union
from unittest import TestCase
import functools
import glob
import os
import shutil
import tempfile
import fx2trt_oss.fx.diagnostics as diag


def reset_diag(fn):
    @functools.wraps(fn)
    def reset(*a, **kw):
        try:
            tok1 = diag._CURRENT_COLLECTOR.set(None)
            tok2 = diag._CURRENT_WRITER.set(None)
            tok3 = diag._SUBSEQUENT_COLLECT_SUPPRESSED_BY.set(None)
            return fn(*a, **kw)
        finally:
            diag._CURRENT_COLLECTOR.reset(tok1)
            diag._CURRENT_WRITER.reset(tok2)
            diag._SUBSEQUENT_COLLECT_SUPPRESSED_BY.reset(tok3)
    return reset


class Fx2trtDiagnosticsTest(TestCase):
    @reset_diag
    def test_diagnostics(self):
        collector = diag.ZipDiagnosticsCollector(
            writer=diag.get_current_writer()
        )

        diag.set_current_collector(collector)

        try:
            with diag.collect_when_fail():
                diag.write("aaa", "hello")
                diag.write("bbb", lambda: "world")
                diag.write("ccc", b"123")
                diag.write("ddd", lambda: b"456")

                def boom() -> str:
                    raise AssertionError("Error generating diagnostics.")
                diag.write("eee", boom)

                diag.write("zzz", "done")
                raise _UserDefinedError("Error while lowering")
        except _UserDefinedError:
            pass

        zip_fn = collector._last_zip_path_for_test
        assert os.path.exists(zip_fn)
        with tempfile.TemporaryDirectory() as tempdir:
            print(f"Unpacking into {tempdir}")
            shutil.unpack_archive(zip_fn, tempdir)
            _check_file(tempdir, "aaa", "hello")
            _check_file(tempdir, "bbb", "world")
            _check_file(tempdir, "ccc", b"123")
            _check_file(tempdir, "ddd", b"456")
            _check_file(tempdir, "zzz", "done")
            # file eee should still exist to contain err msg
            _check_file(tempdir, "eee", "")

    @reset_diag
    def test_condition_func_name(self):
        collector = diag.ZipDiagnosticsCollector(
            writer=diag.get_current_writer()
        )
        diag.set_current_collector(collector)

        with diag.collect_when(
            diag.CollectionConditions.when_called_by_function(self.test_condition_func_name.__name__)
        ):
            diag.write("aaa", "hello")

        zip_fn = collector._last_zip_path_for_test
        assert os.path.exists(zip_fn)
        with tempfile.TemporaryDirectory() as tempdir:
            print(f"Unpacking into {tempdir}")
            shutil.unpack_archive(zip_fn, tempdir)
            _check_file(tempdir, "aaa", "hello")

    @reset_diag
    def test_write_without_collect(self):
        collector = diag.ZipDiagnosticsCollector(
            writer=diag.get_current_writer()
        )
        diag.set_current_collector(collector)
        diag.write("aaa", "hello")
        root_dir = diag.get_current_writer().root_dir()
        res = glob.glob(f"{root_dir}/*")
        assert not res  # root dir should be empty

    def test_conditions(self):

        _test_cond(
            diag.CollectionConditions.when_called_by_function(self.test_conditions.__name__),
            should_collect=True,
        )

        _test_cond(
            diag.CollectionConditions.when_called_by_function("moo_baa_la_la_la"),
            should_collect=False,
        )

        _test_cond(
            diag.CollectionConditions.any(
                diag.CollectionConditions.never(),
                diag.CollectionConditions.always(),
            ),
            True,
        )

        _test_cond(
            diag.CollectionConditions.all(
                diag.CollectionConditions.never(),
                diag.CollectionConditions.always(),
            ),
            False,
        )

        # nested
        _test_cond(
            diag.CollectionConditions.any(
                diag.CollectionConditions.never(),
                diag.CollectionConditions.any(
                    diag.CollectionConditions.always(),
                ),
            ),
            True,
        )


@reset_diag
def _test_cond(
    cond: diag.CollectionCondition,
    should_collect: bool,
) -> None:
    collector = diag.ZipDiagnosticsCollector(
        writer=diag.get_current_writer()
    )
    diag.set_current_collector(collector)

    with diag.collect_when(cond):
        diag.write("aaa", "hello")

    zip_fn = collector._last_zip_path_for_test
    if should_collect:
        assert os.path.exists(zip_fn)
        with tempfile.TemporaryDirectory() as tempdir:
            print(f"Unpacking into {tempdir}")
            shutil.unpack_archive(zip_fn, tempdir)
            _check_file(tempdir, "aaa", "hello")
    else:
        assert not zip_fn, "the collection should not have triggered"


def _check_file(dir: str, fn: str, content: Union[str, bytes]):
    fp = os.path.join(dir, fn)
    res = glob.glob(f"{fp}*")
    assert len(res) == 1
    fp = res[0]
    if not os.path.exists(fp):
        raise _CheckFileDoesNotExist(f"{fp} must exist")
    if not content:
        # don't check content then
        return
    if isinstance(content, bytes):
        with open(fp, "rb") as f:
            content_actual = f.read()
            assert content == content_actual
    else:
        content: str
        with open(fp, "r", encoding="utf-8") as f:
            content_actual = f.read()
            assert content == content_actual


class _UserDefinedError(Exception):
    pass

class _CheckFileDoesNotExist(AssertionError):
    pass
