from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from test_nestedtensor import (
    BACKWARD_SKIPS_AND_XFAILS,
    COMPILE_BACKWARD_SKIPS_AND_XFAILS,
    COMPILE_FORWARD_SKIPS_AND_XFAILS,
    FORWARD_SKIPS_AND_XFAILS,
)

import torch
from torch.nested._internal.nested_tensor import NestedTensor
from torch.testing._internal.opinfo.core import SampleRule, SkipRule, XFailRule
from torch.testing._internal.opinfo.definitions.nested import njt_op_db
from torch.utils._pytree import tree_flatten


# Contains info about a single SampleInput
@dataclass
class SampleInfo:
    # the first rule that was matched; could be a skip or an xfail. None == success
    matched_rule: Optional[SampleRule] = None
    # are all NJT inputs contiguous?
    contiguous: bool = True


# Contains info about all SampleInputs for a single test set (e.g. for forward tests)
@dataclass
class TestResultInfo:
    device_type: str = None
    dtype: torch.dtype = torch.float32
    sample_infos: List[SampleInfo] = None

    def __post_init__(self):
        if self.device_type is None:
            raise ValueError("device_type must be set")
        if self.sample_infos is None:
            raise ValueError("sample_infos must be set")

    def num_xfails(self, contiguous=None):
        if contiguous is None:
            return len(
                [
                    si
                    for si in self.sample_infos
                    if isinstance(si.matched_rule, XFailRule)
                ]
            )
        else:
            return len(
                [
                    si
                    for si in self.sample_infos
                    if isinstance(si.matched_rule, XFailRule)
                    and si.contiguous == contiguous
                ]
            )

    def num_skips(self, contiguous=None):
        if contiguous is None:
            return len(
                [
                    si
                    for si in self.sample_infos
                    if isinstance(si.matched_rule, SkipRule)
                ]
            )
        else:
            return len(
                [
                    si
                    for si in self.sample_infos
                    if isinstance(si.matched_rule, SkipRule)
                    and si.contiguous == contiguous
                ]
            )

    def num_samples(self, contiguous=None):
        if contiguous is None:
            return len(self.sample_infos)
        else:
            return len([si for si in self.sample_infos if si.contiguous == contiguous])

    def success_rate(self, contiguous=None):
        num_samples = self.num_samples(contiguous)
        if num_samples == 0:
            # avoid division by 0
            return None
        return 1 - (
            float(self.num_xfails(contiguous) + self.num_skips(contiguous))
            / float(num_samples)
        )


# Status around op support for NJT OpInfo tests.
# We're only able to get info for those ops that have an NJT-compatible OpInfo entry.
# The op may not have one or it may not support float32, which means the tests won't run (yet).
class OpInfoStatus(Enum):
    VALID_OPINFO = 1
    NO_OPINFO = 2
    NO_FLOAT32_SUPPORT = 3


# Contains info about all test sets for a given op
@dataclass
class OpTestResultInfo:
    status: OpInfoStatus = None
    # mapping from test set name (e.g. "forward") -> test set info.
    # should be None if this isn't able to be calculated.
    test_results: Dict[str, TestResultInfo] = None

    def __post_init__(self):
        if self.status is None:
            raise ValueError("status must be set")


@dataclass
class TestSet:
    name: str = None
    skips_and_xfails: List[SampleRule] = None
    needs_requires_grad: bool = False

    def __post_init__(self):
        if self.name is None:
            raise ValueError("name must be set")
        if self.skips_and_xfails is None:
            raise ValueError("skips_and_xfails must be set")


TEST_SETS = [
    TestSet(
        name="forward",
        skips_and_xfails=FORWARD_SKIPS_AND_XFAILS,
        needs_requires_grad=False,
    ),
    TestSet(
        name="compile_forward",
        skips_and_xfails=COMPILE_FORWARD_SKIPS_AND_XFAILS,
        needs_requires_grad=False,
    ),
    TestSet(
        name="backward",
        skips_and_xfails=BACKWARD_SKIPS_AND_XFAILS,
        needs_requires_grad=True,
    ),
    TestSet(
        name="compile_backward",
        skips_and_xfails=COMPILE_BACKWARD_SKIPS_AND_XFAILS,
        needs_requires_grad=True,
    ),
]


def write_header(f):
    f.write(
        """
<html>
<head>
<title>NJT OpInfo Testing</title>
<style>
table {
  border-collapse: collapse;
  width: 98%;
  color: #333;
  font-family: Arial, sans-serif;
  font-size: 13px;
  text-align: left;
  border-radius: 10px;
  box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
  margin-top: 0px;
}
body: {
  background-color: #fdfdfd;
}
thead, tbody: {
  display: block;
}
table thead {
  position: sticky;
  top: 0;
  background-color: #232a25;
  border-radius: 10px;
}
table th {
  background-color: #232a25;
  color: #ddd;
  font-weight: bold;
  padding: 10px;
  text-transform: uppercase;
  letter-spacing: 1px;
  border-bottom: 1px solid #ccc;
}
table th:first-child {
  border-radius: 10px 0 0 0;
  box-shadow: 0 -2.1rem 0 .6rem #fdfdfd
}
table th:last-child {
  border-radius: 0 10px 0 0;
  box-shadow: 1rem -2.1rem 0 .6rem #fdfdfd
}
table td {
  background-color: #ddd;
  padding: 10px;
  border-bottom: 1px solid #333;
  border-right: 1px solid #333;
  font-weight: bold;
}
table tr td:last-child {
  border-right: 0px solid #333
}
table tr:last-child td {
  border-bottom: 0px solid #333
}
table tr:last-child td:first-child {
  border-radius: 0 0 0 10px;
}
table tr:last-child td:last-child {
  border-radius: 0 0 10px 0;
}
</style>
</head>
<body>
<table>
<thead>
<tr>
<th>Op Name</th>
"""
    )

    for contiguous in [True, False]:
        for test_set in TEST_SETS:
            f.write(
                f"""
<th>{test_set.name}<br/>({"contiguous" if contiguous else "non-contiguous"})</th>
"""
            )

    f.write(
        """
</tr>
</thead>
<tbody>
"""
    )


def write_footer(f):
    f.write(
        """
</tbody>
</table>
</body>
</html>
"""
    )


# returns (fgcolor, bgcolor) for given success rate
def success_colors(success_rate):
    if success_rate is None:
        return ("#333333", "#dddddd")
    elif success_rate == 1.0:
        # green
        return ("#dddddd", "#495b52")
    elif success_rate > 0.5:
        # yellow
        return ("#dddddd", "#a47146")
    else:
        # red
        return ("#dddddd", "#a04c46")


def write_table_row(f, op, result):
    table_row = f"<tr><td>{op.full_name}</td>"
    for contiguous in [True, False]:
        for test_set in TEST_SETS:
            success_rate = None
            if result.status == OpInfoStatus.VALID_OPINFO:
                set_result = result.test_results[test_set.name]
                success_rate = set_result.success_rate(contiguous)
                if success_rate is None:
                    success_rate_text = "N/A"
                elif success_rate == 1.0:
                    success_rate_text = f"{success_rate:.2f}"
                else:
                    success_rate_text = f"{success_rate:.2f}"
                    success_rate_text += (
                        f" (xfails: {set_result.num_xfails(contiguous)}, "
                        f"skips: {set_result.num_skips(contiguous)}, "
                        f"samples: {set_result.num_samples(contiguous)})"
                    )
            elif result.status == OpInfoStatus.NO_OPINFO:
                success_rate_text = "N/A (no OpInfo)"
            elif result.status == OpInfoStatus.NO_FLOAT32_SUPPORT:
                success_rate_text = "N/A (no float32 support)"
            else:
                raise ValueError("invalid OpInfoStatus encountered")

            fgcolor, bgcolor = success_colors(success_rate)
            style = f'style="color: {fgcolor}; background-color: {bgcolor}"'
            table_row += f'<td {style}">{success_rate_text}</td>\n'

    table_row += "</tr>"
    f.write(table_row)


# Returns first matched rule or None if none matched
def match_first_rule(rules, sample, device):
    for rule in rules:
        if rule.sample_match_fn(device, sample):
            return rule
    return None


def get_test_result_info(
    op, rules, device_type, dtype, requires_grad
) -> TestResultInfo:
    device = torch.device(device_type)
    sample_infos = []
    op_rules = [rule for rule in rules if rule.op_match_fn(device_type, op)]
    for sample in op.sample_inputs(
        device=device_type, dtype=dtype, requires_grad=requires_grad
    ):
        all_njts_contiguous = all(
            njt.is_contiguous()
            for njt in tree_flatten((sample.input, sample.args, sample.kwargs))[0]
            if isinstance(njt, NestedTensor)
        )

        sample_infos.append(
            SampleInfo(
                matched_rule=match_first_rule(op_rules, sample, device),
                contiguous=all_njts_contiguous,
            )
        )

    return TestResultInfo(
        device_type=device_type,
        dtype=dtype,
        sample_infos=sample_infos,
    )


with open("njt_op_report.html", "w") as f:
    write_header(f)
    njt_op_db.sort(key=lambda op: op.full_name)
    for op in njt_op_db:
        # TODO: un-hardcode these
        device_type = "cuda"
        supported_dtypes = op.dtypesIfCUDA if device_type == "cuda" else op.dtypes
        dtype = torch.float32

        if not op.supports_njt:
            result = OpTestResultInfo(status=OpInfoStatus.NO_OPINFO)
        elif dtype not in supported_dtypes:
            result = OpTestResultInfo(status=OpInfoStatus.NO_FLOAT32_SUPPORT)
        else:
            test_results = {
                test_set.name: get_test_result_info(
                    op,
                    test_set.skips_and_xfails,
                    device_type,
                    dtype,
                    test_set.needs_requires_grad,
                )
                for test_set in TEST_SETS
            }
            result = OpTestResultInfo(
                status=OpInfoStatus.VALID_OPINFO,
                test_results=test_results,
            )

        write_table_row(f, op, result)

    write_footer(f)
