import copy
import functools
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from types import MethodType
from typing import Any, List, Optional, TYPE_CHECKING, Union

import pytest
from _pytest.config import Config, filename_arg
from _pytest.config.argparsing import Parser
from _pytest.junitxml import _NodeReporter, bin_xml_escape, LogXML
from _pytest.python import Module
from _pytest.reports import TestReport
from _pytest.stash import StashKey
from _pytest.terminal import _get_raw_skip_reason
from pytest_shard_custom import pytest_addoptions as shard_addoptions, PytestShardPlugin

if TYPE_CHECKING:
    from _pytest._code.code import ReprFileLocation

# a lot of this file is copied from _pytest.junitxml and modified to get rerun info

xml_key = StashKey["LogXMLReruns"]()
STEPCURRENT_CACHE_DIR = "cache/stepcurrent"


def pytest_addoption(parser: Parser) -> None:
    group = parser.getgroup("general")
    group.addoption(
        "--scs",
        action="store",
        default=None,
        dest="stepcurrent_skip",
    )
    group.addoption(
        "--sc",
        action="store",
        default=None,
        dest="stepcurrent",
    )

    parser.addoption("--use-main-module", action="store_true")
    group = parser.getgroup("terminal reporting")
    group.addoption(
        "--junit-xml-reruns",
        action="store",
        dest="xmlpath_reruns",
        metavar="path",
        type=functools.partial(filename_arg, optname="--junit-xml-reruns"),
        default=None,
        help="create junit-xml style report file at given path.",
    )
    group.addoption(
        "--junit-prefix-reruns",
        action="store",
        metavar="str",
        default=None,
        help="prepend prefix to classnames in junit-xml output",
    )
    parser.addini(
        "junit_suite_name_reruns", "Test suite name for JUnit report", default="pytest"
    )
    parser.addini(
        "junit_logging_reruns",
        "Write captured log messages to JUnit report: "
        "one of no|log|system-out|system-err|out-err|all",
        default="no",
    )
    parser.addini(
        "junit_log_passing_tests_reruns",
        "Capture log information for passing tests to JUnit report: ",
        type="bool",
        default=True,
    )
    parser.addini(
        "junit_duration_report_reruns",
        "Duration time to report: one of total|call",
        default="total",
    )
    parser.addini(
        "junit_family_reruns",
        "Emit XML for schema: one of legacy|xunit1|xunit2",
        default="xunit2",
    )
    shard_addoptions(parser)


def pytest_configure(config: Config) -> None:
    xmlpath = config.option.xmlpath_reruns
    # Prevent opening xmllog on worker nodes (xdist).
    if xmlpath and not hasattr(config, "workerinput"):
        junit_family = config.getini("junit_family_reruns")
        config.stash[xml_key] = LogXMLReruns(
            xmlpath,
            config.option.junitprefix,
            config.getini("junit_suite_name_reruns"),
            config.getini("junit_logging_reruns"),
            config.getini("junit_duration_report_reruns"),
            junit_family,
            config.getini("junit_log_passing_tests_reruns"),
        )
        config.pluginmanager.register(config.stash[xml_key])
    if config.getoption("stepcurrent_skip"):
        config.option.stepcurrent = config.getoption("stepcurrent_skip")
    if config.getoption("stepcurrent"):
        config.pluginmanager.register(StepcurrentPlugin(config), "stepcurrentplugin")
    if config.getoption("num_shards"):
        config.pluginmanager.register(PytestShardPlugin(config), "pytestshardplugin")


def pytest_unconfigure(config: Config) -> None:
    xml = config.stash.get(xml_key, None)
    if xml:
        del config.stash[xml_key]
        config.pluginmanager.unregister(xml)


class _NodeReporterReruns(_NodeReporter):
    def _prepare_content(self, content: str, header: str) -> str:
        return content

    def _write_content(self, report: TestReport, content: str, jheader: str) -> None:
        if content == "":
            return
        tag = ET.Element(jheader)
        tag.text = bin_xml_escape(content)
        self.append(tag)

    def append_skipped(self, report: TestReport) -> None:
        # Referenced from the below
        # https://github.com/pytest-dev/pytest/blob/2178ee86d7c1ee93748cfb46540a6e40b4761f2d/src/_pytest/junitxml.py#L236C6-L236C6
        # Modified to escape characters not supported by xml in the skip reason.  Everything else should be the same.
        if hasattr(report, "wasxfail"):
            # Super here instead of the actual code so we can reduce possible divergence
            super().append_skipped(report)
        else:
            assert isinstance(report.longrepr, tuple)
            filename, lineno, skipreason = report.longrepr
            if skipreason.startswith("Skipped: "):
                skipreason = skipreason[9:]
            details = f"{filename}:{lineno}: {skipreason}"

            skipped = ET.Element(
                "skipped", type="pytest.skip", message=bin_xml_escape(skipreason)
            )
            skipped.text = bin_xml_escape(details)
            self.append(skipped)
            self.write_captured_output(report)


class LogXMLReruns(LogXML):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def append_rerun(self, reporter: _NodeReporter, report: TestReport) -> None:
        if hasattr(report, "wasxfail"):
            reporter._add_simple("skipped", "xfail-marked test passes unexpectedly")
        else:
            assert report.longrepr is not None
            reprcrash: Optional[ReprFileLocation] = getattr(
                report.longrepr, "reprcrash", None
            )
            if reprcrash is not None:
                message = reprcrash.message
            else:
                message = str(report.longrepr)
            message = bin_xml_escape(message)
            reporter._add_simple("rerun", message, str(report.longrepr))

    def pytest_runtest_logreport(self, report: TestReport) -> None:
        super().pytest_runtest_logreport(report)
        if report.outcome == "rerun":
            reporter = self._opentestcase(report)
            self.append_rerun(reporter, report)
        if report.outcome == "skipped":
            if isinstance(report.longrepr, tuple):
                fspath, lineno, reason = report.longrepr
                reason = f"{report.nodeid}: {_get_raw_skip_reason(report)}"
                report.longrepr = (fspath, lineno, reason)

    def node_reporter(self, report: Union[TestReport, str]) -> _NodeReporterReruns:
        nodeid: Union[str, TestReport] = getattr(report, "nodeid", report)
        # Local hack to handle xdist report order.
        workernode = getattr(report, "node", None)

        key = nodeid, workernode

        if key in self.node_reporters:
            # TODO: breaks for --dist=each
            return self.node_reporters[key]

        reporter = _NodeReporterReruns(nodeid, self)

        self.node_reporters[key] = reporter
        self.node_reporters_ordered.append(reporter)

        return reporter


# imitating summary_failures in pytest's terminal.py
# both hookwrapper and tryfirst to make sure this runs before pytest's
@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # prints stack traces for reruns
    if terminalreporter.config.option.tbstyle != "no":
        reports = terminalreporter.getreports("rerun")
        if reports:
            terminalreporter.write_sep("=", "RERUNS")
            if terminalreporter.config.option.tbstyle == "line":
                for rep in reports:
                    line = terminalreporter._getcrashline(rep)
                    terminalreporter.write_line(line)
            else:
                for rep in reports:
                    msg = terminalreporter._getfailureheadline(rep)
                    terminalreporter.write_sep("_", msg, red=True, bold=True)
                    terminalreporter._outrep_summary(rep)
                    terminalreporter._handle_teardown_sections(rep.nodeid)
    yield


@pytest.hookimpl(tryfirst=True)
def pytest_pycollect_makemodule(module_path, path, parent) -> Module:
    if parent.config.getoption("--use-main-module"):
        mod = Module.from_parent(parent, path=module_path)
        mod._getobj = MethodType(lambda x: sys.modules["__main__"], mod)
        return mod


@pytest.hookimpl(hookwrapper=True)
def pytest_report_teststatus(report, config):
    # Add the test time to the verbose output, unforunately I don't think this
    # includes setup or teardown
    pluggy_result = yield
    if not isinstance(report, pytest.TestReport):
        return
    outcome, letter, verbose = pluggy_result.get_result()
    if verbose:
        pluggy_result.force_result(
            (outcome, letter, f"{verbose} [{report.duration:.4f}s]")
        )


l = "test_decomp.py::TestDecompCUDA::test_comprehensive_H_cuda_int8 PASSED [1.6720s] [  0%]test_decomp.py::TestDecompCUDA::test_comprehensive___radd___cuda_bfloat16 PASSED [0.0467s] [  0%]test_decomp.py::TestDecompCUDA::test_comprehensive___rdiv___cuda_int64 PASSED [0.3445s] [  0%]test_decomp.py::TestDecompCUDA::test_comprehensive___rmod___cuda_float32 PASSED [1.8809s] [  1%]test_decomp.py::TestDecompCUDA::test_comprehensive__chunk_cat_cuda_bool PASSED [0.6603s] [  1%]test_decomp.py::TestDecompCUDA::test_comprehensive__chunk_cat_cuda_float32 PASSED [0.9240s] [  1%]test_decomp.py::TestDecompCUDA::test_comprehensive__upsample_bilinear2d_aa_cuda_float16 SKIPPED [0.0050s] (_upsample_bilinear2d_aa in torch.float16 not supported) [  1%]test_decomp.py::TestDecompCUDA::test_comprehensive_acos_cuda_int8 PASSED [0.0152s] [  2%]test_decomp.py::TestDecompCUDA::test_comprehensive_acosh_cuda_float16 PASSED [0.0225s] [  2%]test_decomp.py::TestDecompCUDA::test_comprehensive_acosh_cuda_float32 PASSED [0.0710s] [  2%]test_decomp.py::TestDecompCUDA::test_comprehensive_acosh_cuda_uint8 PASSED [0.0161s] [  3%]test_decomp.py::TestDecompCUDA::test_comprehensive_addr_cuda_bfloat16 PASSED [1.1096s] [  3%]test_decomp.py::TestDecompCUDA::test_comprehensive_alias_copy_cuda_bool PASSED [0.0146s] [  3%]test_decomp.py::TestDecompCUDA::test_comprehensive_argmax_cuda_float32 PASSED [0.0110s] [  3%]test_decomp.py::TestDecompCUDA::test_comprehensive_as_strided_cuda_complex128 PASSED [0.4074s] [  4%]test_decomp.py::TestDecompCUDA::test_comprehensive_as_strided_partial_views_cuda_complex32 PASSED [0.0103s] [  4%]test_decomp.py::TestDecompCUDA::test_comprehensive_as_strided_partial_views_cuda_complex64 PASSED [0.3027s] [  4%]test_decomp.py::TestDecompCUDA::test_comprehensive_asinh_cuda_complex32 PASSED [0.6897s] [  5%]test_decomp.py::TestDecompCUDA::test_comprehensive_atan2_cuda_int8 PASSED [0.2291s] [  5%]test_decomp.py::TestDecompCUDA::test_comprehensive_atleast_2d_cuda_bool PASSED [0.0159s] [  5%]test_decomp.py::TestDecompCUDA::test_comprehensive_bincount_cuda_int8 PASSED [0.0166s] [  5%]test_decomp.py::TestDecompCUDA::test_comprehensive_bitwise_left_shift_cuda_int64 PASSED [0.1751s] [  6%]test_decomp.py::TestDecompCUDA::test_comprehensive_bitwise_not_cuda_bool PASSED [0.0141s] [  6%]test_decomp.py::TestDecompCUDA::test_comprehensive_broadcast_to_cuda_float16 PASSED [0.0286s] [  6%]test_decomp.py::TestDecompCUDA::test_comprehensive_chalf_cuda_complex64 PASSED [0.1175s] [  7%]test_decomp.py::TestDecompCUDA::test_comprehensive_char_cuda_complex32 PASSED [0.0541s] [  7%]test_decomp.py::TestDecompCUDA::test_comprehensive_clone_cuda_int64 PASSED [0.0389s] [  7%]test_decomp.py::TestDecompCUDA::test_comprehensive_contiguous_cuda_complex32 PASSED [0.0100s] [  7%]test_decomp.py::TestDecompCUDA::test_comprehensive_cov_cuda_float32 PASSED [7.8116s] [  8%]test_decomp.py::TestDecompCUDA::test_comprehensive_cov_cuda_float64 PASSED [8.1576s] [  8%]test_decomp.py::TestDecompCUDA::test_comprehensive_cummax_cuda_uint8 PASSED [0.0112s] [  8%]test_decomp.py::TestDecompCUDA::test_comprehensive_diag_embed_cuda_bool PASSED [0.9846s] [  9%]test_decomp.py::TestDecompCUDA::test_comprehensive_diag_embed_cuda_int16 PASSED [1.4133s] [  9%]test_decomp.py::TestDecompCUDA::test_comprehensive_diagflat_cuda_int16 PASSED [0.1647s] [  9%]test_decomp.py::TestDecompCUDA::test_comprehensive_diagonal_copy_cuda_int32 PASSED [0.1277s] [  9%]test_decomp.py::TestDecompCUDA::test_comprehensive_diagonal_cuda_float16 PASSED [0.1025s] [ 10%]test_decomp.py::TestDecompCUDA::test_comprehensive_diagonal_scatter_cuda_bool PASSED [0.9605s] [ 10%]test_decomp.py::TestDecompCUDA::test_comprehensive_diff_cuda_bool PASSED [3.2323s] [ 10%]test_decomp.py::TestDecompCUDA::test_comprehensive_diff_cuda_int64 PASSED [3.0922s] [ 11%]test_decomp.py::TestDecompCUDA::test_comprehensive_double_cuda_float16 PASSED [0.0240s] [ 11%]test_decomp.py::TestDecompCUDA::test_comprehensive_dsplit_cuda_bfloat16 PASSED [0.0401s] [ 11%]test_decomp.py::TestDecompCUDA::test_comprehensive_dstack_cuda_bool PASSED [0.0677s] [ 11%]test_decomp.py::TestDecompCUDA::test_comprehensive_dstack_cuda_int8 PASSED [0.0677s] [ 12%]test_decomp.py::TestDecompCUDA::test_comprehensive_eq_cuda_uint8 PASSED [0.1751s] [ 12%]test_decomp.py::TestDecompCUDA::test_comprehensive_equal_cuda_complex64 PASSED [0.0115s] [ 12%]test_decomp.py::TestDecompCUDA::test_comprehensive_equal_cuda_uint8 PASSED [0.0120s] [ 12%]test_decomp.py::TestDecompCUDA::test_comprehensive_erfc_cuda_float16 PASSED [0.7577s] [ 13%]test_decomp.py::TestDecompCUDA::test_comprehensive_eye_cuda_complex64 PASSED [4.1691s] [ 13%]test_decomp.py::TestDecompCUDA::test_comprehensive_eye_cuda_int8 PASSED [2.6816s] [ 13%]test_decomp.py::TestDecompCUDA::test_comprehensive_fft_fft2_cuda_float32 PASSED [0.6508s] [ 14%]test_decomp.py::TestDecompCUDA::test_comprehensive_fft_fftn_cuda_float32 PASSED [0.6762s] [ 14%]test_decomp.py::TestDecompCUDA::test_comprehensive_fft_fftn_cuda_int16 PASSED [0.4416s] [ 14%]test_decomp.py::TestDecompCUDA::test_comprehensive_fft_hfft2_cuda_uint8 PASSED [0.6577s] [ 14%]test_decomp.py::TestDecompCUDA::test_comprehensive_fft_hfftn_cuda_int32 PASSED [0.7846s] [ 15%]test_decomp.py::TestDecompCUDA::test_comprehensive_fft_ifftn_cuda_complex32 PASSED [0.9756s] [ 15%]test_decomp.py::TestDecompCUDA::test_comprehensive_fft_ifftshift_cuda_uint8 PASSED [0.2288s] [ 15%]test_decomp.py::TestDecompCUDA::test_comprehensive_fft_irfft_cuda_complex64 PASSED [0.7875s] [ 16%]test_decomp.py::TestDecompCUDA::test_comprehensive_flatten_cuda_complex32 PASSED [0.1081s] [ 16%]test_decomp.py::TestDecompCUDA::test_comprehensive_flip_cuda_uint8 PASSED [0.1525s] [ 16%]test_decomp.py::TestDecompCUDA::test_comprehensive_float_cuda_bfloat16 PASSED [0.0225s] [ 16%]test_decomp.py::TestDecompCUDA::test_comprehensive_float_cuda_float32 PASSED [0.0301s] [ 17%]test_decomp.py::TestDecompCUDA::test_comprehensive_floor_divide_cuda_float64 PASSED [0.5199s] [ 17%]test_decomp.py::TestDecompCUDA::test_comprehensive_fmax_cuda_bool PASSED [0.1709s] [ 17%]test_decomp.py::TestDecompCUDA::test_comprehensive_fmax_cuda_float16 PASSED [0.5650s] [ 18%]test_decomp.py::TestDecompCUDA::test_comprehensive_fmax_cuda_int64 PASSED [0.1814s] [ 18%]test_decomp.py::TestDecompCUDA::test_comprehensive_fmin_cuda_float64 PASSED [3.9607s] [ 18%]test_decomp.py::TestDecompCUDA::test_comprehensive_fmod_cuda_bfloat16 PASSED [0.0817s] [ 18%]test_decomp.py::TestDecompCUDA::test_comprehensive_frexp_cuda_float32 PASSED [0.5405s] [ 19%]test_decomp.py::TestDecompCUDA::test_comprehensive_ge_cuda_int8 PASSED [0.1693s] [ 19%]test_decomp.py::TestDecompCUDA::test_comprehensive_gradient_cuda_float64 PASSED [6.4988s] [ 19%]test_decomp.py::TestDecompCUDA::test_comprehensive_gradient_cuda_int32 PASSED [1.7219s] [ 20%]test_decomp.py::TestDecompCUDA::test_comprehensive_grid_sampler_2d_cuda_float64 PASSED [1250.2746s] [ 20%]test_decomp.py::TestDecompCUDA::test_comprehensive_gt_cuda_int64 PASSED [0.1831s] [ 20%]test_decomp.py::TestDecompCUDA::test_comprehensive_half_cuda_complex64 PASSED [0.1149s] [ 20%]test_decomp.py::TestDecompCUDA::test_comprehensive_hstack_cuda_float32 PASSED [0.1096s] [ 21%]test_decomp.py::TestDecompCUDA::test_comprehensive_index_add_cuda_float64 PASSED [0.3755s] [ 21%]test_decomp.py::TestDecompCUDA::test_comprehensive_index_select_cuda_int16 PASSED [0.0246s] [ 21%]test_decomp.py::TestDecompCUDA::test_comprehensive_inner_cuda_float32 PASSED [0.1381s] [ 22%]test_decomp.py::TestDecompCUDA::test_comprehensive_int_cuda_int32 PASSED [0.0212s] [ 22%]test_decomp.py::TestDecompCUDA::test_comprehensive_isclose_cuda_float32 PASSED [4.1766s] [ 22%]test_decomp.py::TestDecompCUDA::test_comprehensive_isnan_cuda_complex128 PASSED [0.0968s] [ 22%]test_decomp.py::TestDecompCUDA::test_comprehensive_isnan_cuda_float32 PASSED [0.0542s] [ 23%]test_decomp.py::TestDecompCUDA::test_comprehensive_jiterator_4inputs_with_extra_args_cuda_int32 PASSED [0.8157s] [ 23%]test_decomp.py::TestDecompCUDA::test_comprehensive_jiterator_unary_cuda_float16 PASSED [0.2102s] [ 23%]test_decomp.py::TestDecompCUDA::test_comprehensive_kthvalue_cuda_float16 PASSED [0.0504s] [ 24%]test_decomp.py::TestDecompCUDA::test_comprehensive_kthvalue_cuda_float64 PASSED [0.1740s] [ 24%]test_decomp.py::TestDecompCUDA::test_comprehensive_linalg_cross_cuda_bfloat16 PASSED [0.1770s] [ 24%]test_decomp.py::TestDecompCUDA::test_comprehensive_linalg_det_cuda_float32 PASSED [0.7883s] [ 24%]test_decomp.py::TestDecompCUDA::test_comprehensive_linalg_householder_product_cuda_float64 PASSED [15.7259s] [ 25%]test_decomp.py::TestDecompCUDA::test_comprehensive_linalg_ldl_factor_ex_cuda_float64 PASSED [0.0144s] [ 25%]test_decomp.py::TestDecompCUDA::test_comprehensive_linalg_lu_cuda_float64 PASSED [24.2167s] [ 25%]test_decomp.py::TestDecompCUDA::test_comprehensive_linalg_matrix_norm_cuda_float64 PASSED [5.2868s] [ 25%]test_decomp.py::TestDecompCUDA::test_comprehensive_linalg_matrix_rank_hermitian_cuda_float32 PASSED [0.2123s] [ 26%]test_decomp.py::TestDecompCUDA::test_comprehensive_linalg_pinv_hermitian_cuda_float64 PASSED [1.3390s] [ 26%]test_decomp.py::TestDecompCUDA::test_comprehensive_linalg_svdvals_cuda_complex64 PASSED [1.0527s] [ 26%]test_decomp.py::TestDecompCUDA::test_comprehensive_linalg_vector_norm_cuda_float64 PASSED [18.5228s] [ 27%]test_decomp.py::TestDecompCUDA::test_comprehensive_linspace_cuda_int64 PASSED [1.7232s] [ 27%]test_decomp.py::TestDecompCUDA::test_comprehensive_log10_cuda_bfloat16 PASSED [0.0243s] [ 27%]test_decomp.py::TestDecompCUDA::test_comprehensive_log_normal_cuda_bfloat16 PASSED [0.0232s] [ 27%]test_decomp.py::TestDecompCUDA::test_comprehensive_log_softmax_with_dtype_cuda_bfloat16 PASSED [0.0917s] [ 28%]test_decomp.py::TestDecompCUDA::test_comprehensive_log_softmax_with_dtype_cuda_complex64 PASSED [1.2436s] [ 28%]test_decomp.py::TestDecompCUDA::test_comprehensive_logaddexp2_cuda_float64 PASSED [0.4767s] [ 28%]test_decomp.py::TestDecompCUDA::test_comprehensive_logaddexp_cuda_float32 PASSED [8.0786s] [ 29%]test_decomp.py::TestDecompCUDA::test_comprehensive_logical_not_cuda_int16 PASSED [0.0173s] [ 29%]test_decomp.py::TestDecompCUDA::test_comprehensive_logit_cuda_float64 PASSED [1.6762s] [ 29%]test_decomp.py::TestDecompCUDA::test_comprehensive_logspace_cuda_int32 XFAIL [0.8634s] [ 29%]test_decomp.py::TestDecompCUDA::test_comprehensive_long_cuda_bfloat16 PASSED [0.0291s] [ 30%]test_decomp.py::TestDecompCUDA::test_comprehensive_long_cuda_int16 PASSED [0.0301s] [ 30%]test_decomp.py::TestDecompCUDA::test_comprehensive_mT_cuda_bfloat16 PASSED [0.0275s] [ 30%]test_decomp.py::TestDecompCUDA::test_comprehensive_masked_argmax_cuda_bfloat16 PASSED [0.2235s] [ 31%]test_decomp.py::TestDecompCUDA::test_comprehensive_masked_argmax_cuda_int32 PASSED [0.4173s] [ 31%]test_decomp.py::TestDecompCUDA::test_comprehensive_masked_median_cuda_float64 PASSED [1.3666s] [ 31%]test_decomp.py::TestDecompCUDA::test_comprehensive_masked_sum_cuda_bool PASSED [0.7884s] [ 31%]test_decomp.py::TestDecompCUDA::test_comprehensive_masked_sum_cuda_int8 PASSED [0.7982s] [ 32%]test_decomp.py::TestDecompCUDA::test_comprehensive_masked_var_cuda_float32 PASSED [17.8078s] [ 32%]test_decomp.py::TestDecompCUDA::test_comprehensive_masked_var_cuda_int8 PASSED [8.6109s] [ 32%]test_decomp.py::TestDecompCUDA::test_comprehensive_max_reduction_no_dim_cuda_int32 PASSED [0.0118s] [ 33%]test_decomp.py::TestDecompCUDA::test_comprehensive_max_reduction_no_dim_cuda_uint8 PASSED [0.0109s] [ 33%]test_decomp.py::TestDecompCUDA::test_comprehensive_meshgrid_list_of_tensors_cuda_bfloat16 SKIPPED [0.0054s] (meshgrid in torch.bfloat16 not supported) [ 33%]test_decomp.py::TestDecompCUDA::test_comprehensive_mode_cuda_float16 PASSED [0.0371s] [ 33%]test_decomp.py::TestDecompCUDA::test_comprehensive_mode_cuda_float64 PASSED [0.1133s] [ 34%]test_decomp.py::TestDecompCUDA::test_comprehensive_nan_to_num_cuda_int16 PASSED [0.0179s] [ 34%]test_decomp.py::TestDecompCUDA::test_comprehensive_nanmean_cuda_complex32 PASSED [7.7939s] [ 34%]test_decomp.py::TestDecompCUDA::test_comprehensive_narrow_cuda_int16 PASSED [0.1522s] [ 35%]test_decomp.py::TestDecompCUDA::test_comprehensive_new_full_cuda_bool PASSED [0.0433s] [ 35%]test_decomp.py::TestDecompCUDA::test_comprehensive_new_full_cuda_float32 PASSED [0.0715s] [ 35%]test_decomp.py::TestDecompCUDA::test_comprehensive_new_ones_cuda_int16 PASSED [0.0424s] [ 35%]test_decomp.py::TestDecompCUDA::test_comprehensive_nn_functional_adaptive_avg_pool2d_cuda_float16 PASSED [0.6365s] [ 36%]test_decomp.py::TestDecompCUDA::test_comprehensive_nn_functional_avg_pool1d_cuda_bfloat16 PASSED [0.0483s] [ 36%]test_decomp.py::TestDecompCUDA::test_comprehensive_nn_functional_conv_transpose2d_cuda_complex128 PASSED [18.0572s] [ 36%]test_decomp.py::TestDecompCUDA::test_comprehensive_nn_functional_cosine_embedding_loss_cuda_float64 PASSED [1.9207s] [ 37%]test_decomp.py::TestDecompCUDA::test_comprehensive_nn_functional_cross_entropy_cuda_float32 PASSED [4.7998s] [ 37%]test_decomp.py::TestDecompCUDA::test_comprehensive_nn_functional_elu_cuda_float32 PASSED [0.1986s] [ 37%]test_decomp.py::TestDecompCUDA::test_comprehensive_nn_functional_embedding_bag_cuda_float16 PASSED [0.0952s] [ 37%]test_decomp.py::TestDecompCUDA::test_comprehensive_nn_functional_gaussian_nll_loss_cuda_float32 PASSED [125.5811s] [ 38%]test_decomp.py::TestDecompCUDA::test_comprehensive_nn_functional_hardsigmoid_cuda_float64 PASSED [0.2730s] [ 38%]test_decomp.py::TestDecompCUDA::test_comprehensive_nn_functional_hardswish_cuda_float16 PASSED [0.4556s] [ 38%]test_decomp.py::TestDecompCUDA::test_comprehensive_nn_functional_instance_norm_cuda_float16 PASSED [0.5534s] [ 38%]test_decomp.py::TestDecompCUDA::test_comprehensive_nn_functional_interpolate_linear_cuda_float64 PASSED [3.3219s] [ 39%]test_decomp.py::TestDecompCUDA::test_comprehensive_nn_functional_interpolate_nearest_cuda_float16 PASSED [0.4348s] [ 39%]test_decomp.py::TestDecompCUDA::test_comprehensive_nn_functional_kl_div_cuda_float32 PASSED [2.3525s] [ 39%]test_decomp.py::TestDecompCUDA::test_comprehensive_nn_functional_linear_cuda_float16 PASSED [0.2868s] [ 40%]test_decomp.py::TestDecompCUDA::test_comprehensive_nn_functional_max_pool3d_cuda_float64 PASSED [0.4045s] [ 40%]test_decomp.py::TestDecompCUDA::test_comprehensive_nn_functional_pad_circular_cuda_float32 SKIPPED [0.0003s] (Expected: new_empty_strided is not comparable) [ 40%]test_decomp.py::TestDecompCUDA::test_comprehensive_nn_functional_pad_constant_cuda_float32 PASSED [2.6988s] [ 40%]test_decomp.py::TestDecompCUDA::test_comprehensive_nn_functional_pad_constant_cuda_float64 PASSED [2.6402s] [ 41%]test_decomp.py::TestDecompCUDA::test_comprehensive_nn_functional_pad_constant_cuda_int32 PASSED [0.9556s] [ 41%]test_decomp.py::TestDecompCUDA::test_comprehensive_nn_functional_pad_constant_cuda_int8 PASSED [0.9518s] [ 41%]test_decomp.py::TestDecompCUDA::test_comprehensive_nn_functional_silu_cuda_bfloat16 PASSED [0.0548s] [ 42%]test_decomp.py::TestDecompCUDA::test_comprehensive_nn_functional_soft_margin_loss_cuda_bfloat16 PASSED [0.0856s] [ 42%]test_decomp.py::TestDecompCUDA::test_comprehensive_nn_functional_triplet_margin_with_distance_loss_cuda_complex64 PASSED [7.0648s] [ 42%]test_decomp.py::TestDecompCUDA::test_comprehensive_nonzero_static_cuda_int32 SKIPPED [0.0017s] (Only runs on cpu) [ 42%]test_decomp.py::TestDecompCUDA::test_comprehensive_norm_nuc_cuda_float32 SKIPPED [0.0065s] (norm in torch.float32 not supported) [ 43%]test_decomp.py::TestDecompCUDA::test_comprehensive_pca_lowrank_cuda_float32 PASSED [196.7299s] [ 43%]test_decomp.py::TestDecompCUDA::test_comprehensive_permute_cuda_int16 PASSED [0.0221s] [ 43%]test_decomp.py::TestDecompCUDA::test_comprehensive_permute_cuda_int64 PASSED [0.0215s] [ 44%]test_decomp.py::TestDecompCUDA::test_comprehensive_polygamma_polygamma_n_0_cuda_int64 PASSED [1.2790s] [ 44%]test_decomp.py::TestDecompCUDA::test_comprehensive_prod_cuda_complex32 PASSED [0.8143s] [ 44%]test_decomp.py::TestDecompCUDA::test_comprehensive_rand_like_cuda_complex32 PASSED [0.0172s] [ 44%]test_decomp.py::TestDecompCUDA::test_comprehensive_ravel_cuda_complex128 PASSED [0.1914s] [ 45%]test_decomp.py::TestDecompCUDA::test_comprehensive_real_cuda_int32 PASSED [0.0093s] [ 45%]test_decomp.py::TestDecompCUDA::test_comprehensive_renorm_cuda_float32 PASSED [3.0797s] [ 45%]test_decomp.py::TestDecompCUDA::test_comprehensive_repeat_cuda_float16 PASSED [0.1149s] [ 46%]test_decomp.py::TestDecompCUDA::test_comprehensive_repeat_interleave_cuda_float32 PASSED [0.2710s] [ 46%]test_decomp.py::TestDecompCUDA::test_comprehensive_reshape_cuda_float64 PASSED [0.2371s] [ 46%]test_decomp.py::TestDecompCUDA::test_comprehensive_resolve_neg_cuda_bfloat16 PASSED [0.0105s] [ 46%]test_decomp.py::TestDecompCUDA::test_comprehensive_rsqrt_cuda_bool PASSED [0.0150s] [ 47%]test_decomp.py::TestDecompCUDA::test_comprehensive_rsqrt_cuda_uint8 PASSED [0.0144s] [ 47%]test_decomp.py::TestDecompCUDA::test_comprehensive_rsub_cuda_int32 PASSED [0.3165s] [ 47%]test_decomp.py::TestDecompCUDA::test_comprehensive_scalar_tensor_cuda_complex64 PASSED [0.0097s] [ 48%]test_decomp.py::TestDecompCUDA::test_comprehensive_scatter_cuda_bool PASSED [0.0174s] [ 48%]test_decomp.py::TestDecompCUDA::test_comprehensive_scatter_reduce_amin_cuda_bfloat16 PASSED [0.9014s] [ 48%]test_decomp.py::TestDecompCUDA::test_comprehensive_sgn_cuda_complex64 PASSED [2.9046s] [ 48%]test_decomp.py::TestDecompCUDA::test_comprehensive_short_cuda_complex64 PASSED [0.0546s] [ 49%]test_decomp.py::TestDecompCUDA::test_comprehensive_sigmoid_cuda_complex32 PASSED [1.2100s] [ 49%]test_decomp.py::TestDecompCUDA::test_comprehensive_signal_windows_cosine_cuda_float64 PASSED [0.3206s] [ 49%]test_decomp.py::TestDecompCUDA::test_comprehensive_sinc_cuda_float64 PASSED [0.6113s] [ 50%]test_decomp.py::TestDecompCUDA::test_comprehensive_slice_cuda_float32 PASSED [0.5662s] [ 50%]test_decomp.py::TestDecompCUDA::test_comprehensive_slice_cuda_float64 PASSED [0.5934s] [ 50%]test_decomp.py::TestDecompCUDA::test_comprehensive_slice_scatter_cuda_int64 PASSED [0.0118s] [ 50%]test_decomp.py::TestDecompCUDA::test_comprehensive_slice_scatter_cuda_int8 PASSED [0.0116s] [ 51%]test_decomp.py::TestDecompCUDA::test_comprehensive_sort_cuda_float16 PASSED [0.1094s] [ 51%]test_decomp.py::TestDecompCUDA::test_comprehensive_sparse_mm_reduce_cuda_float32 SKIPPED [0.0014s] (Only runs on cpu) [ 51%]test_decomp.py::TestDecompCUDA::test_comprehensive_special_chebyshev_polynomial_t_cuda_int8 PASSED [1.6470s] [ 51%]test_decomp.py::TestDecompCUDA::test_comprehensive_special_chebyshev_polynomial_v_cuda_uint8 SKIPPED [0.0003s] (Skipping - testing takes an unreasonably long time, #79528) [ 52%]test_decomp.py::TestDecompCUDA::test_comprehensive_special_hermite_polynomial_h_cuda_float64 PASSED [0.8373s] [ 52%]test_decomp.py::TestDecompCUDA::test_comprehensive_special_i1_cuda_uint8 PASSED [0.6965s] [ 52%]test_decomp.py::TestDecompCUDA::test_comprehensive_special_i1e_cuda_uint8 PASSED [0.6290s] [ 53%]test_decomp.py::TestDecompCUDA::test_comprehensive_special_laguerre_polynomial_l_cuda_int64 PASSED [1.6045s] [ 53%]test_decomp.py::TestDecompCUDA::test_comprehensive_special_modified_bessel_i0_cuda_int16 PASSED [0.3774s] [ 53%]test_decomp.py::TestDecompCUDA::test_comprehensive_special_modified_bessel_i0_cuda_int8 PASSED [0.0105s] [ 53%]test_decomp.py::TestDecompCUDA::test_comprehensive_special_modified_bessel_i1_cuda_uint8 PASSED [0.3667s] [ 54%]test_decomp.py::TestDecompCUDA::test_comprehensive_special_ndtr_cuda_int32 SKIPPED [0.0049s] (special.ndtr in torch.int32 not supported) [ 54%]test_decomp.py::TestDecompCUDA::test_comprehensive_special_ndtr_cuda_int64 SKIPPED [0.0050s] (special.ndtr in torch.int64 not supported) [ 54%]test_decomp.py::TestDecompCUDA::test_comprehensive_special_ndtri_cuda_int8 PASSED [0.7487s] [ 55%]test_decomp.py::TestDecompCUDA::test_comprehensive_special_scaled_modified_bessel_k0_cuda_float32 PASSED [0.4157s] [ 55%]test_decomp.py::TestDecompCUDA::test_comprehensive_special_shifted_chebyshev_polynomial_t_cuda_int64 SKIPPED [0.0002s] (Skipping - testing takes an unreasonably long time, #79528) [ 55%]test_decomp.py::TestDecompCUDA::test_comprehensive_special_shifted_chebyshev_polynomial_u_cuda_int8 SKIPPED [0.0008s] (Skipping - testing takes an unreasonably long time, #79528) [ 55%]test_decomp.py::TestDecompCUDA::test_comprehensive_special_shifted_chebyshev_polynomial_u_cuda_uint8 SKIPPED [0.0002s] (Skipping - testing takes an unreasonably long time, #79528) [ 56%]test_decomp.py::TestDecompCUDA::test_comprehensive_split_cuda_complex32 PASSED [0.5935s] [ 56%]test_decomp.py::TestDecompCUDA::test_comprehensive_split_list_args_cuda_bfloat16 PASSED [0.0250s] [ 56%]test_decomp.py::TestDecompCUDA::test_comprehensive_squeeze_cuda_int8 PASSED [0.0913s] [ 57%]test_decomp.py::TestDecompCUDA::test_comprehensive_stack_cuda_float16 PASSED [0.0416s] [ 57%]test_decomp.py::TestDecompCUDA::test_comprehensive_std_mean_unbiased_cuda_float32 PASSED [0.2953s] [ 57%]test_decomp.py::TestDecompCUDA::test_comprehensive_std_unbiased_cuda_float32 PASSED [0.2054s] [ 57%]test_decomp.py::TestDecompCUDA::test_comprehensive_take_along_dim_cuda_complex128 PASSED [0.1683s] [ 58%]test_decomp.py::TestDecompCUDA::test_comprehensive_tan_cuda_bool PASSED [0.0439s] [ 58%]test_decomp.py::TestDecompCUDA::test_comprehensive_tan_cuda_complex32 PASSED [1.3777s] [ 58%]test_decomp.py::TestDecompCUDA::test_comprehensive_tanh_cuda_float16 PASSED [0.0215s] [ 59%]test_decomp.py::TestDecompCUDA::test_comprehensive_tensor_split_cuda_int32 PASSED [0.4644s] [ 59%]test_decomp.py::TestDecompCUDA::test_comprehensive_to_cuda_int8 PASSED [5.9423s] [ 59%]test_decomp.py::TestDecompCUDA::test_comprehensive_to_sparse_cuda_complex128 PASSED [0.0156s] [ 59%]test_decomp.py::TestDecompCUDA::test_comprehensive_to_sparse_cuda_float64 PASSED [0.0170s] [ 60%]test_decomp.py::TestDecompCUDA::test_comprehensive_torch__scaled_mm_cuda_float8_e4m3fn SKIPPED [0.0013s] (Requires CUDA SM >= 9.0) [ 60%]test_decomp.py::TestDecompCUDA::test_comprehensive_trace_cuda_float32 PASSED [0.0499s] [ 60%]test_decomp.py::TestDecompCUDA::test_comprehensive_transpose_cuda_int8 PASSED [0.0718s] [ 61%]test_decomp.py::TestDecompCUDA::test_comprehensive_trapz_cuda_bfloat16 PASSED [0.2677s] [ 61%]test_decomp.py::TestDecompCUDA::test_comprehensive_true_divide_cuda_float16 PASSED [0.1029s] [ 61%]test_decomp.py::TestDecompCUDA::test_comprehensive_unfold_cuda_bfloat16 PASSED [0.2272s] [ 61%]test_decomp.py::TestDecompCUDA::test_comprehensive_uniform_cuda_complex64 PASSED [0.0126s] [ 62%]test_decomp.py::TestDecompCUDA::test_comprehensive_unique_cuda_float64 PASSED [0.2505s] [ 62%]test_decomp.py::TestDecompCUDA::test_comprehensive_unsafe_split_cuda_float32 PASSED [0.3300s] [ 62%]test_decomp.py::TestDecompCUDA::test_comprehensive_unsqueeze_cuda_bool PASSED [0.0519s] [ 62%]test_decomp.py::TestDecompCUDA::test_comprehensive_var_cuda_float32 PASSED [1.0512s] [ 63%]test_decomp.py::TestDecompCUDA::test_comprehensive_var_mean_cuda_complex64 PASSED [3.0277s] [ 63%]test_decomp.py::TestDecompCUDA::test_comprehensive_view_as_real_cuda_complex128 PASSED [0.0096s] [ 63%]test_decomp.py::TestDecompCUDA::test_comprehensive_view_copy_cuda_bool PASSED [0.0114s] [ 64%]test_decomp.py::TestDecompCUDA::test_comprehensive_view_cuda_int8 PASSED [0.0832s] [ 64%]test_decomp.py::TestDecompCUDA::test_comprehensive_view_cuda_uint8 PASSED [0.0845s] [ 64%]test_decomp.py::TestDecompCUDA::test_quick__softmax_backward_data_cuda_float32 PASSED [0.1024s] [ 64%]test_decomp.py::TestDecompCUDA::test_quick__unsafe_masked_index_cuda_complex64 PASSED [0.3565s] [ 65%]test_decomp.py::TestDecompCUDA::test_quick__unsafe_masked_index_cuda_float16 PASSED [0.0215s] [ 65%]test_decomp.py::TestDecompCUDA::test_quick__unsafe_masked_index_put_accumulate_cuda_complex128 PASSED [0.8547s] [ 65%]test_decomp.py::TestDecompCUDA::test_quick_addmv_cuda_complex64 PASSED [0.1646s] [ 66%]test_decomp.py::TestDecompCUDA::test_quick_addr_cuda_bool PASSED [0.0312s] [ 66%]test_decomp.py::TestDecompCUDA::test_quick_addr_cuda_int8 PASSED [0.0310s] [ 66%]test_decomp.py::TestDecompCUDA::test_quick_alias_copy_cuda_complex32 PASSED [0.0224s] [ 66%]test_decomp.py::TestDecompCUDA::test_quick_amax_cuda_float64 PASSED [0.0998s] [ 67%]test_decomp.py::TestDecompCUDA::test_quick_amax_cuda_int16 PASSED [0.0547s] [ 67%]test_decomp.py::TestDecompCUDA::test_quick_any_cuda_int16 PASSED [0.0548s] [ 67%]test_decomp.py::TestDecompCUDA::test_quick_as_strided_copy_cuda_complex64 PASSED [0.0444s] [ 68%]test_decomp.py::TestDecompCUDA::test_quick_atan_cuda_int16 PASSED [0.0433s] [ 68%]test_decomp.py::TestDecompCUDA::test_quick_baddbmm_cuda_complex64 PASSED [1.5888s] [ 68%]test_decomp.py::TestDecompCUDA::test_quick_bitwise_right_shift_cuda_int16 PASSED [0.1339s] [ 68%]test_decomp.py::TestDecompCUDA::test_quick_bucketize_cuda_int32 PASSED [0.1529s] [ 69%]test_decomp.py::TestDecompCUDA::test_quick_cat_cuda_complex32 PASSED [0.1435s] [ 69%]test_decomp.py::TestDecompCUDA::test_quick_clamp_min_cuda_bfloat16 PASSED [0.0171s] [ 69%]test_decomp.py::TestDecompCUDA::test_quick_clone_cuda_float32 PASSED [0.0494s] [ 70%]test_decomp.py::TestDecompCUDA::test_quick_constant_pad_nd_cuda_complex64 PASSED [0.8214s] [ 70%]test_decomp.py::TestDecompCUDA::test_quick_constant_pad_nd_cuda_uint8 PASSED [0.3571s] [ 70%]test_decomp.py::TestDecompCUDA::test_quick_core_backward_lerp_cuda_float64 XFAIL [27.8669s] [ 70%]test_decomp.py::TestDecompCUDA::test_quick_core_backward_special_entr_cuda_float64 PASSED [3.7822s] [ 71%]test_decomp.py::TestDecompCUDA::test_quick_core_backward_special_log_ndtr_cuda_float64 PASSED [6.9675s] [ 71%]test_decomp.py::TestDecompCUDA::test_quick_cosh_cuda_float32 PASSED [0.0211s] [ 71%]test_decomp.py::TestDecompCUDA::test_quick_cumprod_cuda_complex64 PASSED [0.2659s] [ 72%]test_decomp.py::TestDecompCUDA::test_quick_cumprod_cuda_float32 PASSED [0.1601s] [ 72%]test_decomp.py::TestDecompCUDA::test_quick_cumprod_cuda_int32 PASSED [0.1158s] [ 72%]test_decomp.py::TestDecompCUDA::test_quick_diag_cuda_complex32 SKIPPED [0.0053s] (diag in torch.complex32 not supported) [ 72%]test_decomp.py::TestDecompCUDA::test_quick_diagonal_scatter_cuda_complex128 PASSED [1.0082s] [ 73%]test_decomp.py::TestDecompCUDA::test_quick_diagonal_scatter_cuda_int8 PASSED [0.3675s] [ 73%]test_decomp.py::TestDecompCUDA::test_quick_digamma_cuda_int32 PASSED [0.8790s] [ 73%]test_decomp.py::TestDecompCUDA::test_quick_div_floor_rounding_cuda_int32 PASSED [0.1481s] [ 74%]test_decomp.py::TestDecompCUDA::test_quick_empty_strided_cuda_uint8 SKIPPED [0.0059s] (empty_strided in torch.uint8 not supported) [ 74%]test_decomp.py::TestDecompCUDA::test_quick_erf_cuda_bool PASSED [0.0471s] [ 74%]test_decomp.py::TestDecompCUDA::test_quick_exp2_cuda_complex128 PASSED [0.5126s] [ 74%]test_decomp.py::TestDecompCUDA::test_quick_eye_cuda_uint8 PASSED [0.0785s] [ 75%]test_decomp.py::TestDecompCUDA::test_quick_fft_hfft2_cuda_float32 PASSED [0.5703s] [ 75%]test_decomp.py::TestDecompCUDA::test_quick_fft_ifft2_cuda_complex64 PASSED [0.1459s] [ 75%]test_decomp.py::TestDecompCUDA::test_quick_fft_ifft_cuda_float64 PASSED [0.0488s] [ 75%]test_decomp.py::TestDecompCUDA::test_quick_fft_ifftn_cuda_complex32 SKIPPED [0.0097s] (only backwards is decomposed, but dtype doesn't support AD) [ 76%]test_decomp.py::TestDecompCUDA::test_quick_fft_irfftn_cuda_complex32 SKIPPED [0.0096s] (only backwards is decomposed, but dtype doesn't support AD) [ 76%]test_decomp.py::TestDecompCUDA::test_quick_fft_irfftn_cuda_complex64 PASSED [0.1493s] [ 76%]test_decomp.py::TestDecompCUDA::test_quick_fmin_cuda_bool PASSED [0.1366s] [ 77%]test_decomp.py::TestDecompCUDA::test_quick_gcd_cuda_int8 PASSED [0.9374s] [ 77%]test_decomp.py::TestDecompCUDA::test_quick_ge_cuda_int16 PASSED [0.1456s] [ 77%]test_decomp.py::TestDecompCUDA::test_quick_grid_sampler_2d_cuda_bfloat16 PASSED [0.1962s] [ 77%]test_decomp.py::TestDecompCUDA::test_quick_gt_cuda_float16 PASSED [0.1077s] [ 78%]test_decomp.py::TestDecompCUDA::test_quick_heaviside_cuda_bfloat16 PASSED [0.0221s] [ 78%]test_decomp.py::TestDecompCUDA::test_quick_heaviside_cuda_float16 PASSED [0.0208s] [ 78%]test_decomp.py::TestDecompCUDA::test_quick_hypot_cuda_float64 PASSED [0.2173s] [ 79%]test_decomp.py::TestDecompCUDA::test_quick_index_add_cuda_float32 PASSED [0.0967s] [ 79%]test_decomp.py::TestDecompCUDA::test_quick_index_select_cuda_complex128 PASSED [0.0345s] [ 79%]test_decomp.py::TestDecompCUDA::test_quick_isneginf_cuda_float64 PASSED [0.0540s] [ 79%]test_decomp.py::TestDecompCUDA::test_quick_isposinf_cuda_float16 PASSED [0.0315s] [ 80%]test_decomp.py::TestDecompCUDA::test_quick_isposinf_cuda_uint8 PASSED [0.0424s] [ 80%]test_decomp.py::TestDecompCUDA::test_quick_item_cuda_complex64 SKIPPED [0.0049s] (item in torch.complex64 not supported) [ 80%]test_decomp.py::TestDecompCUDA::test_quick_linalg_vector_norm_cuda_float16 PASSED [0.1815s] [ 81%]test_decomp.py::TestDecompCUDA::test_quick_linspace_cuda_bfloat16 PASSED [0.0510s] [ 81%]test_decomp.py::TestDecompCUDA::test_quick_log_cuda_complex128 PASSED [0.3733s] [ 81%]test_decomp.py::TestDecompCUDA::test_quick_logaddexp2_cuda_float32 PASSED [0.0222s] [ 81%]test_decomp.py::TestDecompCUDA::test_quick_logical_and_cuda_int8 PASSED [0.1376s] [ 82%]test_decomp.py::TestDecompCUDA::test_quick_logit_cuda_bfloat16 PASSED [0.0201s] [ 82%]test_decomp.py::TestDecompCUDA::test_quick_maximum_cuda_float64 PASSED [0.2000s] [ 82%]test_decomp.py::TestDecompCUDA::test_quick_mean_cuda_bfloat16 PASSED [0.0314s] [ 83%]test_decomp.py::TestDecompCUDA::test_quick_mul_cuda_bfloat16 PASSED [0.0206s] [ 83%]test_decomp.py::TestDecompCUDA::test_quick_mvlgamma_mvlgamma_p_1_cuda_int8 PASSED [0.3813s] [ 83%]test_decomp.py::TestDecompCUDA::test_quick_nan_to_num_cuda_int16 PASSED [0.0135s] [ 83%]test_decomp.py::TestDecompCUDA::test_quick_nan_to_num_cuda_int32 PASSED [0.0136s] [ 84%]test_decomp.py::TestDecompCUDA::test_quick_nan_to_num_cuda_int64 PASSED [0.0144s] [ 84%]test_decomp.py::TestDecompCUDA::test_quick_nansum_cuda_int32 PASSED [0.0635s] [ 84%]test_decomp.py::TestDecompCUDA::test_quick_narrow_copy_cuda_complex128 PASSED [0.1800s] [ 85%]test_decomp.py::TestDecompCUDA::test_quick_narrow_copy_cuda_complex64 PASSED [0.1793s] [ 85%]test_decomp.py::TestDecompCUDA::test_quick_native_dropout_backward_cuda_bfloat16 PASSED [0.0191s] [ 85%]test_decomp.py::TestDecompCUDA::test_quick_nextafter_cuda_bfloat16 PASSED [0.0181s] [ 85%]test_decomp.py::TestDecompCUDA::test_quick_nn_functional_hardsigmoid_cuda_float32 PASSED [0.0375s] [ 86%]test_decomp.py::TestDecompCUDA::test_quick_nn_functional_huber_loss_cuda_float32 PASSED [0.6375s] [ 86%]test_decomp.py::TestDecompCUDA::test_quick_nn_functional_logsigmoid_cuda_float64 PASSED [0.0501s] [ 86%]test_decomp.py::TestDecompCUDA::test_quick_nn_functional_mish_cuda_float16 PASSED [0.0191s] [ 87%]test_decomp.py::TestDecompCUDA::test_quick_nn_functional_mse_loss_cuda_float16 PASSED [0.0325s] [ 87%]test_decomp.py::TestDecompCUDA::test_quick_nn_functional_pad_constant_cuda_int8 PASSED [0.3627s] [ 87%]test_decomp.py::TestDecompCUDA::test_quick_nn_functional_relu6_cuda_int32 SKIPPED [0.0052s] (nn.functional.relu6 in torch.int32 not supported) [ 87%]test_decomp.py::TestDecompCUDA::test_quick_nn_functional_relu6_cuda_int64 SKIPPED [0.0062s] (nn.functional.relu6 in torch.int64 not supported) [ 88%]test_decomp.py::TestDecompCUDA::test_quick_norm_cuda_float64 SKIPPED [0.0051s] (norm in torch.float64 not supported) [ 88%]test_decomp.py::TestDecompCUDA::test_quick_norm_fro_cuda_float16 SKIPPED [0.0047s] (norm in torch.float16 not supported) [ 88%]test_decomp.py::TestDecompCUDA::test_quick_polar_cuda_float64 PASSED [0.1980s] [ 88%]test_decomp.py::TestDecompCUDA::test_quick_rot90_cuda_bfloat16 PASSED [0.0428s] [ 89%]test_decomp.py::TestDecompCUDA::test_quick_rot90_cuda_complex128 PASSED [1.1708s] [ 89%]test_decomp.py::TestDecompCUDA::test_quick_round_cuda_bfloat16 PASSED [0.0112s] [ 89%]test_decomp.py::TestDecompCUDA::test_quick_rsqrt_cuda_bfloat16 PASSED [0.0117s] [ 90%]test_decomp.py::TestDecompCUDA::test_quick_rsqrt_cuda_uint8 PASSED [0.0144s] [ 90%]test_decomp.py::TestDecompCUDA::test_quick_select_cuda_complex32 SKIPPED [0.0089s] (only backwards is decomposed, but dtype doesn't support AD) [ 90%]test_decomp.py::TestDecompCUDA::test_quick_select_scatter_cuda_bfloat16 PASSED [0.0151s] [ 90%]test_decomp.py::TestDecompCUDA::test_quick_sgn_cuda_complex128 PASSED [0.3642s] [ 91%]test_decomp.py::TestDecompCUDA::test_quick_sgn_cuda_uint8 PASSED [0.0409s] [ 91%]test_decomp.py::TestDecompCUDA::test_quick_sigmoid_cuda_uint8 PASSED [0.0170s] [ 91%]test_decomp.py::TestDecompCUDA::test_quick_sign_cuda_int16 PASSED [0.0419s] [ 92%]test_decomp.py::TestDecompCUDA::test_quick_signbit_cuda_float32 PASSED [0.0559s] [ 92%]test_decomp.py::TestDecompCUDA::test_quick_slice_cuda_float16 PASSED [0.0127s] [ 92%]test_decomp.py::TestDecompCUDA::test_quick_special_entr_cuda_bfloat16 PASSED [0.2412s] [ 92%]test_decomp.py::TestDecompCUDA::test_quick_special_i0e_cuda_float64 PASSED [0.3445s] [ 93%]test_decomp.py::TestDecompCUDA::test_quick_special_i0e_cuda_int16 PASSED [0.6931s] [ 93%]test_decomp.py::TestDecompCUDA::test_quick_special_i0e_cuda_int32 PASSED [0.0141s] [ 93%]test_decomp.py::TestDecompCUDA::test_quick_special_zeta_cuda_int8 PASSED [4.2382s] [ 94%]test_decomp.py::TestDecompCUDA::test_quick_split_list_args_cuda_int8 PASSED [0.1340s] [ 94%]test_decomp.py::TestDecompCUDA::test_quick_split_with_sizes_copy_cuda_float64 PASSED [0.2621s] [ 94%]test_decomp.py::TestDecompCUDA::test_quick_split_with_sizes_cuda_bool PASSED [0.1784s] [ 94%]test_decomp.py::TestDecompCUDA::test_quick_split_with_sizes_cuda_int32 PASSED [0.1759s] [ 95%]test_decomp.py::TestDecompCUDA::test_quick_sqrt_cuda_complex32 PASSED [1.2542s] [ 95%]test_decomp.py::TestDecompCUDA::test_quick_tan_cuda_complex128 PASSED [0.6821s] [ 95%]test_decomp.py::TestDecompCUDA::test_quick_tanh_cuda_float64 PASSED [0.1456s] [ 96%]test_decomp.py::TestDecompCUDA::test_quick_triu_indices_cuda_int64 PASSED [0.0167s] [ 96%]test_decomp.py::TestDecompCUDA::test_quick_trunc_cuda_bfloat16 PASSED [0.0109s] [ 96%]test_decomp.py::TestDecompCUDA::test_quick_unbind_cuda_int16 PASSED [0.1704s] [ 96%]test_decomp.py::TestDecompCUDA::test_quick_unsqueeze_cuda_int16 PASSED [0.0498s] [ 97%]test_decomp.py::TestDecompCUDA::test_quick_unsqueeze_cuda_int64 PASSED [0.0501s] [ 97%]test_decomp.py::TestDecompCUDA::test_quick_unsqueeze_cuda_int8 PASSED [0.0500s] [ 97%]test_decomp.py::TestDecompCUDA::test_quick_var_cuda_complex128 PASSED [0.3184s] [ 98%]test_decomp.py::TestDecompCUDA::test_quick_var_mean_cuda_complex64 PASSED [0.6364s] [ 98%]test_decomp.py::TestDecompCUDA::test_quick_var_mean_cuda_float16 PASSED [0.0328s] [ 98%]test_decomp.py::TestDecompCUDA::test_quick_view_cuda_complex32 PASSED [0.1932s] [ 98%]test_decomp.py::TestDecompCUDA::test_quick_where_cuda_uint8 PASSED [0.1325s] [ 99%]test_decomp.py::TestDecompCUDA::test_quick_zeros_like_cuda_float16 PASSED [0.0148s] [ 99%]test_decomp.py::DecompOneOffTestsCUDA::test_contiguous_log_softmax_cuda PASSED [0.0020s] [ 99%]test_decomp.py::DecompOneOffTestsCUDA::test_elu_backward_cuda PASSED [0.0020s] [100%]"
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items: List[Any]) -> None:
    """
    This hook is used when rerunning disabled tests to get rid of all skipped tests
    instead of running and skipping them N times. This avoids flooding the console
    and XML outputs with junk. So we want this to run last when collecting tests.
    """
    yes = []
    for item in items:
        if item.nodeid in l:
            print(item.nodeid)
            yes.append(l)
    items.clear()
    items.extend(yes)
    return

    rerun_disabled_tests = os.getenv("PYTORCH_TEST_RERUN_DISABLED_TESTS", "0") == "1"
    if not rerun_disabled_tests:
        return

    disabled_regex = re.compile(r"(?P<test_name>.+)\s+\([^\.]+\.(?P<test_class>.+)\)")
    disabled_tests = defaultdict(set)

    # This environment has already been set by run_test before it calls pytest
    disabled_tests_file = os.getenv("DISABLED_TESTS_FILE", "")
    if not disabled_tests_file or not os.path.exists(disabled_tests_file):
        return

    with open(disabled_tests_file) as fp:
        for disabled_test in json.load(fp):
            m = disabled_regex.match(disabled_test)
            if m:
                test_name = m["test_name"]
                test_class = m["test_class"]
                disabled_tests[test_class].add(test_name)

    # When rerunning disabled test, ignore all test cases that are not disabled
    filtered_items = []

    for item in items:
        test_name = item.name
        test_class = item.parent.name

        if (
            test_class not in disabled_tests
            or test_name not in disabled_tests[test_class]
        ):
            continue

        cpy = copy.copy(item)
        cpy._initrequest()

        filtered_items.append(cpy)

    items.clear()
    # NB: Need to edit items directly here to have the list reflected back to pytest
    items.extend(filtered_items)


class StepcurrentPlugin:
    # Modified fromo _pytest/stepwise.py in order to save the currently running
    # test instead of the last failed test
    def __init__(self, config: Config) -> None:
        self.config = config
        self.report_status = ""
        assert config.cache is not None
        self.cache: pytest.Cache = config.cache
        self.directory = f"{STEPCURRENT_CACHE_DIR}/{config.getoption('stepcurrent')}"
        self.lastrun: Optional[str] = self.cache.get(self.directory, None)
        self.initial_val = self.lastrun
        self.skip: bool = config.getoption("stepcurrent_skip")

    def pytest_collection_modifyitems(self, config: Config, items: List[Any]) -> None:
        if not self.lastrun:
            self.report_status = "Cannot find last run test, not skipping"
            return

        # check all item nodes until we find a match on last run
        failed_index = None
        for index, item in enumerate(items):
            if item.nodeid == self.lastrun:
                failed_index = index
                if self.skip:
                    failed_index += 1
                break

        # If the previously failed test was not found among the test items,
        # do not skip any tests.
        if failed_index is None:
            self.report_status = "previously run test not found, not skipping."
        else:
            self.report_status = f"skipping {failed_index} already run items."
            deselected = items[:failed_index]
            del items[:failed_index]
            config.hook.pytest_deselected(items=deselected)

    def pytest_report_collectionfinish(self) -> Optional[str]:
        if self.config.getoption("verbose") >= 0 and self.report_status:
            return f"stepcurrent: {self.report_status}"
        return None

    def pytest_runtest_protocol(self, item, nextitem) -> None:
        self.lastrun = item.nodeid
        self.cache.set(self.directory, self.lastrun)

    def pytest_sessionfinish(self, session, exitstatus):
        if exitstatus == 0:
            self.cache.set(self.directory, self.initial_val)
