#!/usr/bin/env python3

import argparse
import csv
import os
import re
import pandas as pd
from enum import Enum
from itertools import chain
from pathlib import Path
from upload_test_stats import (
        parse_xml_report,
        get_pytest_parallel_times,
        summarize_test_cases,
)

# unit test status list
UT_STATUS_LIST = [
    "PASSED",
    "MISSED",
    "SKIPPED",
    "FAILED",
    "XFAILED",
    "ERROR"
]

# excluded test suites for comparison
EXCLUDED_TEST_SUITES = [
    "_nvfuser.test_dynamo",
    "_nvfuser.test_python_frontend",
    "_nvfuser.test_torchscript",
    "test_nvfuser_dynamo",
    "test_nvfuser_frontend",
    # Blocklisted on ROCm upstream, so it never runs there and would otherwise
    # show up as falsely MISSED against the CUDA baseline.
    "test_jit_cuda_fuser",
    "inductor.test_cpu_repro"
]


EXCLUDED_TEST_CLASSES = [
    "nvfuser_tests",
    "TensorPipeCudaDdpComparisonTest",
    "TensorPipeCudaDistAutogradTest",
    "TensorPipeCudaRemoteModuleTest",
    "TensorPipeCudaRpcTest",
    "TensorPipeTensorPipeAgentCudaRpcTest",
    "TensorPipeTensorPipeCudaDistAutogradTest",
    "test_cpp_rpc"
]
EXCLUDED_TESTS = [
]


# Test config names
TestConfigName = Enum('TestConfigName', ['default', 'distributed', 'inductor'])

def _status_priority(test_case):
    """Return a numeric priority for deduplication of retried tests.
    PASSED/XFAILED are preferred over FAILED/ERROR/SKIPPED since a
    passing retry means the test is considered passing (flaky) in CI."""
    status = get_test_status(test_case)
    return {"PASSED": 4, "XFAILED": 3, "SKIPPED": 2, "FAILED": 1, "ERROR": 1, "MISSED": 0}.get(status, 0)

def _extract_shard(dirname):
    """Extract shard number from directory names like 'test-default-3-6'."""
    m = re.match(r'test-\w+-(\d+)-(\d+)', dirname)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return ""

def parse_xml_reports_as_dict(workflow_run_id, workflow_run_attempt, tag, path="."):
    test_config = ""
    test_cases = {}

    # download_testlogs writes the upstream pytorch CI workflow run id
    # into "_wf_run_id" alongside the shard dirs. We combine it with each
    # shard dir's trailing "_<job_id>" to form the URL
    # https://github.com/pytorch/pytorch/actions/runs/<wf>/job/<job_id>
    # surfaced as the "Job ID" column in the FAILED TESTS table.
    wf_run_id = ""
    wf_id_file = os.path.join(path, "_wf_run_id")
    if os.path.isfile(wf_id_file):
        with open(wf_id_file) as f:
            wf_run_id = f.read().strip()

    items_list = os.listdir(path)
    for dir in items_list:
        new_dir = path + '/' + dir + '/'
        if os.path.isdir(new_dir):
            if "test-default" in new_dir:
                test_config = TestConfigName.default.name
            elif "test-distributed" in new_dir:
                test_config = TestConfigName.distributed.name
            elif "test-inductor" in new_dir:
                test_config = TestConfigName.inductor.name
            shard = _extract_shard(dir)
            jid = re.search(r'_(\d+)$', dir)
            job_url = (
                f"https://github.com/pytorch/pytorch/actions/runs/{wf_run_id}/job/{jid.group(1)}"
                if wf_run_id and jid else ""
            )
            for xml_report in Path(new_dir).glob("**/*.xml"):
                try:
                    new_cases = parse_xml_report(
                        tag,
                        xml_report,
                        workflow_run_id,
                        workflow_run_attempt,
                        test_config
                    )
                except Exception as e:
                    print(f"WARNING: Skipping malformed XML {xml_report}: {e}")
                    continue
                for key, case in new_cases.items():
                    case["shard"] = shard
                    case["job_url"] = job_url
                    existing = test_cases.get(key)
                    if existing is None or _status_priority(case) > _status_priority(existing):
                        test_cases[key] = case
    return test_cases

def get_test_status(test_case):
  # In order of priority: S=skipped, F=failure, E=error, P=pass
  if not test_case:
    return "MISSED"
  elif "skipped" in test_case and test_case["skipped"]:
      type_message = test_case["skipped"]
      if type_message.__contains__('type') and type_message['type'] == "pytest.xfail":
          return "XFAILED"
      else:
          return "SKIPPED"
  elif "failure" in test_case and test_case["failure"]:
    return "FAILED"
  elif "error" in test_case and test_case["error"]:
    return "ERROR"
  else:
    return "PASSED"

def get_test_message(test_case, status=None):
  if status == "SKIPPED":
    return test_case["skipped"] if "skipped" in test_case else ""
  elif status == "FAILED":
    return test_case["failure"] if "failure" in test_case else ""
  elif status == "ERROR":
    return test_case["error"] if "error" in test_case else ""
  else:
    if "skipped" in test_case:
      return test_case["skipped"]
    elif "failure" in test_case:
      return test_case["failure"]
    elif "error" in test_case:
      return test_case["error"]
    else:
      return ""

def get_running_time(test_case):
  status = get_test_status(test_case)
  if test_case.__contains__('time'):
    return test_case["time"]
  return ""

def check_time_valid(time):
  if time == "":
    return False
  return True

def summarize_xml_files(args):
    # TODO: Add arguments and parse accordingly
    set1_path = args.set1 if args.set1 else "."
    set2_path = args.set2
    set1_name = args.set1_name
    set2_name = args.set2_name

    # statistics
    SKIPPED_DEFAULT = 0
    MISSED_DEFAULT = 0
    CUDA_DEFAULT = 0
    ROCM_DEFAULT = 0
    ROCMONLY_DEFAULT = 0

    SKIPPED_DISTRIBUTED = 0
    MISSED_DISTRIBUTED = 0
    CUDA_DISTRIBUTED = 0
    ROCM_DISTRIBUTED = 0
    ROCMONLY_DISTRIBUTED = 0

    SKIPPED_INDUCTOR = 0
    MISSED_INDUCTOR = 0
    CUDA_INDUCTOR = 0
    ROCM_INDUCTOR = 0
    ROCMONLY_INDUCTOR = 0

    TOTAL_CUDA_RUNNING_TIME = 0.0
    TOTAL_ROCM_RUNNING_TIME = 0.0

    # filter example: --filter SKIPPED-PASSED-MISSED-PASSED (tuples: set1 status1 - set2 status1, set1 status2 - set2 status2)
    ut_status_filter = args.filter if args.filter else "."
    list_of_status = ut_status_filter.split('-') if args.filter else []
    # assertion: should be an even number length
    assert len(list_of_status) % 2 == 0
    list_status_set1 = []
    list_status_set2 = []

    index = 0
    while index < len(list_of_status):
        # special handling for status-NOT_status scenario
        if "NOT" in list_of_status[index] or "NOT" in list_of_status[index+1]:
            if "NOT" in list_of_status[index]:
                items = list_of_status[index].split('_')
                not_item = items[1]
                for ind in range(len(UT_STATUS_LIST)):
                    if UT_STATUS_LIST[ind] != not_item:
                        list_status_set1.append(UT_STATUS_LIST[ind])
                        list_status_set2.append(list_of_status[index+1])
            else:
                items = list_of_status[index+1].split('_')
                not_item = items[1]
                for ind in range(len(UT_STATUS_LIST)):
                    if UT_STATUS_LIST[ind] != not_item:
                        list_status_set2.append(UT_STATUS_LIST[ind])
                        list_status_set1.append(list_of_status[index])
            index += 2
        else:
            list_status_set1.append(list_of_status[index])
            index += 1
            list_status_set2.append(list_of_status[index])
            index += 1

    assert len(list_status_set1) == len(list_status_set2), \
            "status_list not specified correctly, should be in pairs of two"
    len_status_filter = len(list_status_set1)

    # define column list
    column_list = ['set1', 'set2', 'skip_reason', 'assignee', 'comments']

    # function location pattern
    pattern = "at 0x"

    #parse the xml files
    test_cases_set1_running_time = parse_xml_reports_as_dict(-1, -1, 'testsuite', set1_path)
    # TODO: Does it matter what the workflow_run_attempt is set to below??
    # test_cases is dict of dicts, with keys as tuple of test_file, test_class, test_name and test_config
    test_cases_set1 = parse_xml_reports_as_dict(-1, -1, 'testcase', set1_path)
    for (k,v) in list(test_cases_set1.items()):
        if v['test_config'] == TestConfigName.default.name:
            ROCM_DEFAULT += 1
        elif v['test_config'] == TestConfigName.distributed.name:
            ROCM_DISTRIBUTED += 1
        elif v['test_config'] == TestConfigName.inductor.name:
            ROCM_INDUCTOR += 1

    # start with creating empty dicts for set2 for each test tuple
    # for rocm/cuda comparison(with valid set2_path), sometimes parity sheet has inaccurate resutls due to different function string but with same test names,
    # such as test_np_argmin_argmax_keepdims_size_(1, 2, 3, 4)_axis_-4_method_<function argmax at 0x7f1e411e6a70>
    test_cases_set1_new: Dict[Tuple[str], Dict[str, Any]] = {}
    if set2_path:
      for (k,v) in list(test_cases_set1.items()):
        if pattern in k[2]:
          values = list(k)
          index = k[2].find(pattern)
          values[2] = k[2][0 : index]
          k_new = tuple(values)
          test_cases_set1_new[k_new] = v
          del test_cases_set1[k]
      #combine two dict
      test_cases_set1_combined = {**test_cases_set1, **test_cases_set1_new}
      test_cases = { k:[v, {}] for (k,v) in test_cases_set1_combined.items() }
    else:
      test_cases = { k:[v, {}] for (k,v) in test_cases_set1.items() }

    test_cases_set2_running_time = {}
    if set2_path:
      assert set2_path != set1_path, \
              "set2 path not specified correctly, should be different from set1 path"
      test_cases_set2_running_time = parse_xml_reports_as_dict(-1, -1, 'testsuite', set2_path)
      test_cases_set2 = parse_xml_reports_as_dict(-1, -1, 'testcase', set2_path)
      for (k,v) in list(test_cases_set2.items()):
          if v['test_config'] == TestConfigName.default.name:
              CUDA_DEFAULT += 1
          elif v['test_config'] == TestConfigName.distributed.name:
              CUDA_DISTRIBUTED += 1
          elif v['test_config'] == TestConfigName.inductor.name:
              CUDA_INDUCTOR += 1

      # for rocm/cuda comparison, sometimes parity sheet has inaccurate resutls due to different function string but with same test names,
      # such as test_np_argmin_argmax_keepdims_size_(1, 2, 3, 4)_axis_-4_method_<function argmax at 0x7f1e411e6a70>
      test_cases_set2_new: Dict[Tuple[str], Dict[str, Any]] = {}
      for (k,v) in list(test_cases_set2.items()):
        if pattern in k[2]:
          values = list(k)
          index = k[2].find(pattern)
          values[2] = k[2][0 : index]
          k_new = tuple(values)
          test_cases_set2_new[k_new] = v
          del test_cases_set2[k]
      #combine two dict
      test_cases_set2_combined = {**test_cases_set2, **test_cases_set2_new}

      # repopulate set2 dicts for test_tuples from test_cases_set2, 
      # creating empty dicts for set1 if test_tuple doesn't exist in test_cases
      for test_case in test_cases_set2_combined:
        test_cases[test_case] = [test_cases_set1_combined[test_case] if test_case in test_cases_set1_combined else {}, test_cases_set2_combined[test_case]]

    # expand with skip_reason, assignee and comments
    for (k,v) in list(test_cases.items()):
        # set1, set2, skip_reason, assignee and comments
        while len(v) < len(column_list):
            v.append('')

    # get running time statistics before any exclusion and filter since they are only for comparison
    # total running time: ROCm and CUDA
    for (k,v) in list(test_cases_set1_running_time.items()):
          TOTAL_ROCM_RUNNING_TIME += v["running_time_xml"]
    for (k,v) in list(test_cases_set2_running_time.items()):
          TOTAL_CUDA_RUNNING_TIME += v["running_time_xml"]

    # test file level running time: ROCm and CUDA
    test_file_level_ROCm: Dict[Tuple[str], float] = {}
    test_file_level_CUDA: Dict[Tuple[str], float] = {}
    test_file_shards_ROCm: Dict[Tuple[str], set] = {}
    test_file_shards_CUDA: Dict[Tuple[str], set] = {}
    for (k,v) in list(test_cases_set1_running_time.items()):
          test_file_name = k[0]
          test_config_name = k[2]
          tar_tup_rocm = (test_file_name, test_config_name,)
          test_file_shards_ROCm.setdefault(tar_tup_rocm, set())
          if v.get("shard"):
              test_file_shards_ROCm[tar_tup_rocm].add(v["shard"])
          if test_file_level_ROCm.get(tar_tup_rocm) == None:
              test_file_level_ROCm[ ( test_file_name, test_config_name ) ] = v["running_time_xml"]
          else:
              test_file_level_ROCm[ ( test_file_name, test_config_name ) ] += v["running_time_xml"]
    for (k,v) in list(test_cases_set2_running_time.items()):
          test_file_name = k[0]
          test_config_name = k[2]
          tar_tup_cuda = (test_file_name, test_config_name)
          test_file_shards_CUDA.setdefault(tar_tup_cuda, set())
          if v.get("shard"):
              test_file_shards_CUDA[tar_tup_cuda].add(v["shard"])
          if test_file_level_CUDA.get(tar_tup_cuda) == None:
              test_file_level_CUDA[ ( test_file_name, test_config_name ) ] = v["running_time_xml"]
          else:
              test_file_level_CUDA[ ( test_file_name, test_config_name ) ] += v["running_time_xml"]

    # test file level counts: ROCm tests run, passed, skipped, missed; CUDA tests run
    test_file_counts_ROCm: Dict[Tuple[str], Dict[str, int]] = {}
    test_file_counts_CUDA: Dict[Tuple[str], int] = {}
    for (k,v) in list(test_cases_set1.items()):
        test_file_name = k[0]
        test_config_name = v['test_config']
        tar_tup = (test_file_name, test_config_name)
        if tar_tup not in test_file_counts_ROCm:
            test_file_counts_ROCm[tar_tup] = {'tests_run': 0, 'passed': 0, 'skipped': 0, 'missed': 0}
        test_file_counts_ROCm[tar_tup]['tests_run'] += 1
        status = get_test_status(v)
        if status == "PASSED":
            test_file_counts_ROCm[tar_tup]['passed'] += 1
        elif status == "SKIPPED":
            test_file_counts_ROCm[tar_tup]['skipped'] += 1
        elif status == "MISSED":
            test_file_counts_ROCm[tar_tup]['missed'] += 1
    for (k,v) in list(test_cases_set2.items()) if set2_path else []:
        test_file_name = k[0]
        test_config_name = v['test_config']
        tar_tup = (test_file_name, test_config_name)
        if tar_tup not in test_file_counts_CUDA:
            test_file_counts_CUDA[tar_tup] = 0
        test_file_counts_CUDA[tar_tup] += 1

    # exclude certain tests for comparison
    if set2_path:
      for (k,v) in list(test_cases.items()):
          if k[0] in EXCLUDED_TEST_SUITES:
              test_cases.pop(k)
          elif k[1] in EXCLUDED_TEST_CLASSES:
              test_cases.pop(k)
          elif (k[0], k[1], k[2]) in EXCLUDED_TESTS:
              test_cases.pop(k)

    # remove unmatched items if user specified ut status filters
    if len_status_filter > 0:
        case_matched = True
        for (k,v) in list(test_cases.items()):
            case_matched = False
            status_set_1 = get_test_status(v[0])
            status_set_2 = get_test_status(v[1]) if set2_path else ""
            for index in range(len_status_filter):
                if status_set_1 == list_status_set1[index] and status_set_2 == list_status_set2[index]:
                    case_matched = True
                    break

            if not case_matched:
                test_cases.pop(k)

    # insert skip_reason, assignee and comments info for the cases that: rocm-missed+cuda-passed OR rocm-skipped+cuda-passed
    # To do: assume set1 is ROCm currently. Should insert another arg for ROCm and CUDA order?
    skip_reasons_stat_default = dict()
    skip_reasons_stat_distributed = dict()
    skip_reasons_stat_inductor = dict()
    if args.skip_reasons:
        # read skip reasons csv file
        known_skips = pd.read_csv(args.skip_reasons, sep='\t')
        known_skips = known_skips.to_dict(orient="records")

    # Load previous week's CSV to check if tests existed and get skip reasons
    prev_week_tests = set()
    prev_week_skip_reasons = {}  # Maps (test_file, test_class, test_name) -> (skip_reason, assignee, comments)
    if args.prev_week_csv:
        prev_week_df = pd.read_csv(args.prev_week_csv)
        for _, row in prev_week_df.iterrows():
            test_key = (row['test_file'], row['test_class'], row['test_name'])
            prev_week_tests.add(test_key)
            # Also extract skip_reason, assignee, comments if they exist
            skip_reason = row.get('skip_reason', '') if 'skip_reason' in row and not pd.isna(row.get('skip_reason', '')) else ''
            assignee = row.get('assignee', '') if 'assignee' in row and not pd.isna(row.get('assignee', '')) else ''
            comments = row.get('comments', '') if 'comments' in row and not pd.isna(row.get('comments', '')) else ''
            if skip_reason or assignee or comments:
                prev_week_skip_reasons[test_key] = (skip_reason, assignee, comments)

    for (k,v) in list(test_cases.items()):
        status_set_1 = get_test_status(v[0])
        status_set_2 = get_test_status(v[1]) if set2_path else ""
        test_file_name = k[0]
        test_info = v[0]
        test_info_set2 = []
        if status_set_1 == "SKIPPED" and status_set_2 != "SKIPPED":
            if test_info['test_config'] == TestConfigName.default.name:
                SKIPPED_DEFAULT += 1
            elif test_info['test_config'] == TestConfigName.distributed.name:
                SKIPPED_DISTRIBUTED += 1
            elif test_info['test_config'] == TestConfigName.inductor.name:
                SKIPPED_INDUCTOR += 1
        elif set2_path:
            test_info_set2 = v[1]
            if status_set_1 == "MISSED" and status_set_2 != "MISSED":
              if test_info_set2['test_config'] == TestConfigName.default.name:
                MISSED_DEFAULT += 1
              elif test_info_set2['test_config'] == TestConfigName.distributed.name:
                MISSED_DISTRIBUTED += 1
              elif test_info_set2['test_config'] == TestConfigName.inductor.name:
                MISSED_INDUCTOR += 1


        if args.skip_reasons:
            if (status_set_1 == "SKIPPED" and status_set_2 != "SKIPPED") or status_set_1 == "MISSED":
              for known_skip in known_skips:
                  if test_file_name == known_skip['test_file'] and k[1] == known_skip['test_class'] and k[2] == known_skip['test_name']:
                      v[2] = known_skip['skip_reason'] if known_skip.__contains__('skip_reason') and not pd.isna(known_skip['skip_reason']) else ' '
                      if (test_info.__contains__('test_config') and test_info['test_config'] == TestConfigName.default.name) or (test_info_set2.__contains__('test_config') and test_info_set2['test_config'] == TestConfigName.default.name):
                          if not skip_reasons_stat_default.__contains__(v[2]):
                              skip_reasons_stat_default[v[2]] = 1
                          else:
                              skip_reasons_stat_default[v[2]] += 1
                      elif (test_info.__contains__('test_config') and test_info['test_config'] == TestConfigName.distributed.name) or (test_info_set2.__contains__('test_config') and test_info_set2['test_config'] == TestConfigName.distributed.name):
                          if not skip_reasons_stat_distributed.__contains__(v[2]):
                              skip_reasons_stat_distributed[v[2]] = 1
                          else:
                              skip_reasons_stat_distributed[v[2]] += 1
                      elif (test_info.__contains__('test_config') and test_info['test_config'] == TestConfigName.inductor.name) or (test_info_set2.__contains__('test_config') and test_info_set2['test_config'] == TestConfigName.inductor.name):
                          if not skip_reasons_stat_inductor.__contains__(v[2]):
                              skip_reasons_stat_inductor[v[2]] = 1
                          else:
                              skip_reasons_stat_inductor[v[2]] += 1
                      v[3] = known_skip['assignee'] if known_skip.__contains__('assignee') and not pd.isna(known_skip['assignee']) else ' '
                      v[4] = known_skip['comments'] if known_skip.__contains__('comments') and not pd.isna(known_skip['comments']) else ' '
                      break

        if status_set_1 == "PASSED" and status_set_2 != "PASSED" and set2_path:
            if test_info['test_config'] == TestConfigName.default.name:
                ROCMONLY_DEFAULT += 1
            elif test_info['test_config'] == TestConfigName.distributed.name:
                ROCMONLY_DISTRIBUTED += 1
            elif test_info['test_config'] == TestConfigName.inductor.name:
                ROCMONLY_INDUCTOR += 1

    skip_reasons_stat_default.pop(' ', None)
    skip_reasons_stat_distributed.pop(' ', None)

    test_cases_for_csv = {}
    # k is test_tuple, v is list of rocm and cuda info for that test_tuple
    skip_reason_file_specified = False
    if args.skip_reasons:
        skip_reason_file_specified = True
    for (k,v) in test_cases.items():
        item_values = {}
        item_values["test_file"] = k[0]
        item_values["test_class"] = k[1]
        item_values["test_name"] = k[2]
        item_values[f"status_{set1_name}"] = get_test_status(v[0])
        item_values[f"status_{set2_name}"] = get_test_status(v[1]) if set2_path else ""
        # get test config info
        v_values = v[0]
        v1_values = v[1] if set2_path else []
        config_name = ""
        item_values["test_config"] = ""
        if item_values[f"status_{set1_name}"] != "MISSED":
            config_name = v_values['test_config']
        elif item_values[f"status_{set2_name}"] != "MISSED" and item_values[f"status_{set2_name}"] != "":
            config_name = v1_values['test_config']
        item_values["test_config"] = config_name
        item_values[f"shard_{set1_name}"] = v_values.get('shard', '') if v_values else ''
        item_values[f"shard_{set2_name}"] = v1_values.get('shard', '') if v1_values else ''
        item_values[f"job_url_{set1_name}"] = v_values.get('job_url', '') if v_values else ''
        item_values[f"job_url_{set2_name}"] = v1_values.get('job_url', '') if v1_values else ''
        # get test related info
        item_values[f"message_{set1_name}"] = get_test_message(v[0])
        item_values[f"message_{set2_name}"] = get_test_message(v[1]) if set2_path else ""
        # Get skip_reason, assignee, comments from --skip_reasons file if specified
        if skip_reason_file_specified:
            item_values["skip_reason"] = v[2]
            item_values["assignee"] = v[3]
            item_values["comments"] = v[4]
        # Check if test existed in previous week's CSV and get skip reasons from there
        if args.prev_week_csv:
            test_key = (k[0], k[1], k[2])  # (test_file, test_class, test_name)
            item_values["existed_last_week"] = "yes" if test_key in prev_week_tests else "no"
            # If skip_reason not set by --skip_reasons, try to get from prev_week_csv
            if not skip_reason_file_specified:
                if test_key in prev_week_skip_reasons:
                    prev_skip_reason, prev_assignee, prev_comments = prev_week_skip_reasons[test_key]
                    item_values["skip_reason"] = prev_skip_reason
                    item_values["assignee"] = prev_assignee
                    item_values["comments"] = prev_comments
                else:
                    item_values["skip_reason"] = ""
                    item_values["assignee"] = ""
                    item_values["comments"] = ""
        if not skip_reason_file_specified and not args.prev_week_csv:
            item_values["skip_reason"] = ""
            item_values["assignee"] = ""
            item_values["comments"] = ""
        running_time1 = get_running_time(v[0])
        item_values[f"running_time_{set1_name}"] = running_time1
        running_time2 = get_running_time(v[1])
        item_values[f"running_time_{set2_name}"] = running_time2
        item_values["abs_time_diff"] = ""
        item_values["relative_time_diff"] = ""
        if check_time_valid(running_time1) and check_time_valid(running_time2):
          item_values["abs_time_diff"] = running_time1 - running_time2
          if get_running_time(v[1]) != 0.0:
            item_values["relative_time_diff"] = 100 * (running_time1 - running_time2) / running_time2
        test_cases_for_csv[k] = item_values

    test_cases_for_csv = dict(sorted(test_cases_for_csv.items()))

    #store test_cases in csv
    tests_from_xml_filename = args.output_csv
    keys_list = list(set(chain.from_iterable(sub.keys() for sub in test_cases_for_csv.values())))

    def sorting_key(e):
        if e == "invoking_file":
          return 0
        elif e == "test_file":
          return 1
        elif e == "test_class":
          return 2
        elif e == "test_name":
          return 3
        elif e == "test_config":
          return 4
        elif e == "skip_reason":
          return 5
        elif e == "assignee":
          return 6
        elif e == "comments":
          return 7
        elif e == f"status_{set1_name}":
          return 8
        elif e == f"message_{set1_name}":
          return 9
        elif e == f"running_time_{set1_name}":
          return 10
        elif e == f"status_{set2_name}":
          return 11
        elif e == f"message_{set2_name}":
          return 12
        elif e == f"running_time_{set2_name}":
          return 13
        elif e == "abs_time_diff":
          return 14
        elif e == "relative_time_diff":
          return 15
        elif e == "skipped":
          return 16
        elif e == "failure":
          return 17
        elif e == "error":
          return 18
        elif e == "system-out":
          return 19
        elif e == "existed_last_week":
          return 20
        elif e == f"shard_{set1_name}":
          return 21
        elif e == f"shard_{set2_name}":
          return 22
        elif e == f"job_url_{set1_name}":
          return 23
        elif e == f"job_url_{set2_name}":
          return 24
        elif e == "workflow_run_attempt" or e == "job_id":
          return 1000
        else:
          return 100

    keys_list.sort(key=sorting_key)

    with open(tests_from_xml_filename, "w") as outfile:
         writer = csv.DictWriter(outfile, fieldnames = keys_list)
         writer.writeheader()
         writer.writerows(test_cases_for_csv.values())
    ## TODO - usage yet to be identified
    #pytest_parallel_times = get_pytest_parallel_times()
    ##extract test cases summary and save them in csv file
    #test_cases_summary = summarize_test_cases(test_cases)
    #testcases_summary_filename = "testcases_summary.csv"
    #keys_list = list(set(chain.from_iterable(sub.keys() for sub in test_cases_summary)))
    #with open(testcases_summary_filename, "w") as outfile:
    #     writer = csv.DictWriter(outfile, fieldnames = keys_list)
    #     writer.writeheader()
    #     writer.writerows(test_cases_summary)

    # write test file running time to file
    test_file_running_time_for_csv = {}
    set1_running_time_col = f"{set1_name}_running_time"
    set2_running_time_col = f"{set2_name}_running_time"
    set1_tests_run_col = f"{set1_name}_tests_run"
    set2_tests_run_col = f"{set2_name}_tests_run"
    set1_test_shards_col = f"{set1_name}_test_shards"
    set2_test_shards_col = f"{set2_name}_test_shards"
    set1_passed_col = f"{set1_name}_passed"
    set1_skipped_col = f"{set1_name}_skipped"
    set1_missed_col = f"{set1_name}_missed"
    for key_rocm in test_file_level_ROCm.keys():
        item_values = {}
        item_values["test_file"] = key_rocm[0]
        item_values["test_config"] = key_rocm[1]
        item_values[set1_running_time_col] = test_file_level_ROCm[key_rocm]
        item_values[set2_running_time_col] = 0.0
        if key_rocm in test_file_level_CUDA.keys():
            item_values[set2_running_time_col] = test_file_level_CUDA[key_rocm]
        item_values["abs_time_diff"] = item_values[set1_running_time_col] - item_values[set2_running_time_col]
        item_values["relative_time_diff"] = 0.0
        if item_values[set2_running_time_col] != 0.0:
            item_values["relative_time_diff"] = 100 * (item_values[set1_running_time_col] - item_values[set2_running_time_col]) / item_values[set2_running_time_col]
        # Add test counts
        item_values[set1_tests_run_col] = test_file_counts_ROCm.get(key_rocm, {}).get('tests_run', 0)
        item_values[set2_tests_run_col] = test_file_counts_CUDA.get(key_rocm, 0)
        item_values[set1_test_shards_col] = len(test_file_shards_ROCm.get(key_rocm, set()))
        item_values[set2_test_shards_col] = len(test_file_shards_CUDA.get(key_rocm, set()))
        item_values[set1_passed_col] = test_file_counts_ROCm.get(key_rocm, {}).get('passed', 0)
        item_values[set1_skipped_col] = test_file_counts_ROCm.get(key_rocm, {}).get('skipped', 0)
        item_values[set1_missed_col] = test_file_counts_ROCm.get(key_rocm, {}).get('missed', 0)
        test_file_running_time_for_csv[key_rocm] = item_values

    for key_cuda in test_file_level_CUDA.keys():
        if not key_cuda in test_file_level_ROCm.keys():
            item_values = {}
            item_values["test_file"] = key_cuda[0]
            item_values["test_config"] = key_cuda[1]
            item_values[set1_running_time_col] = 0.0
            item_values[set2_running_time_col] = test_file_level_CUDA[key_cuda]
            item_values["abs_time_diff"] = item_values[set1_running_time_col] - item_values[set2_running_time_col]
            item_values["relative_time_diff"] = 0.0
            if item_values[set2_running_time_col] != 0.0:
                item_values["relative_time_diff"] = 100 * (item_values[set1_running_time_col] - item_values[set2_running_time_col]) / item_values[set2_running_time_col]
            # Add test counts
            item_values[set1_tests_run_col] = test_file_counts_ROCm.get(key_cuda, {}).get('tests_run', 0)
            item_values[set2_tests_run_col] = test_file_counts_CUDA.get(key_cuda, 0)
            item_values[set1_test_shards_col] = len(test_file_shards_ROCm.get(key_cuda, set()))
            item_values[set2_test_shards_col] = len(test_file_shards_CUDA.get(key_cuda, set()))
            item_values[set1_passed_col] = test_file_counts_ROCm.get(key_cuda, {}).get('passed', 0)
            item_values[set1_skipped_col] = test_file_counts_ROCm.get(key_cuda, {}).get('skipped', 0)
            item_values[set1_missed_col] = test_file_counts_ROCm.get(key_cuda, {}).get('missed', 0)
            test_file_running_time_for_csv[key_cuda] = item_values

    test_file_running_time_for_csv = dict(sorted(test_file_running_time_for_csv.items()))
    keys_list_running_time = list(set(chain.from_iterable(sub.keys() for sub in test_file_running_time_for_csv.values())))
    def sorting_key_running_time(e):
        if e == "test_file":
          return 0
        elif e == "test_config":
          return 1
        elif e == set1_running_time_col:
          return 2
        elif e == set2_running_time_col:
          return 3
        elif e == "abs_time_diff":
          return 4
        elif e == "relative_time_diff":
          return 5
        elif e == set1_tests_run_col:
          return 6
        elif e == set2_tests_run_col:
          return 7
        elif e == set1_test_shards_col:
          return 8
        elif e == set2_test_shards_col:
          return 9
        elif e == set1_passed_col:
          return 10
        elif e == set1_skipped_col:
          return 11
        elif e == set1_missed_col:
          return 12
        else:
          return 100

    keys_list_running_time.sort(key=sorting_key_running_time)
    tests_from_xml_file_running_time = args.test_file_running_time_output_csv
    with open(tests_from_xml_file_running_time, "w") as outfile:
         writer = csv.DictWriter(outfile, fieldnames = keys_list_running_time)
         writer.writeheader()
         writer.writerows(test_file_running_time_for_csv.values())

    # print summary
    # User-facing labels follow the same set1/set2 naming as the CSV columns
    # (set1_name/set2_name default to rocm/cuda, or are commit SHAs in
    # commit-vs-commit mode) so the printed summary stays consistent with them.
    set1_disp = set1_name.upper()
    set2_disp = set2_name.upper()
    print( " " )
    print( "_____________________________________" )
    print( "Test-results" )
    print( " " )
    print( "=====Single GPU Number=====" )
    print( f"SKIPPED_DEFAULT, MISSED_DEFAULT, {set1_disp}ONLY_DEFAULT, {set2_disp}_DEFAULT, {set1_disp}_DEFAULT" )
    print( str(SKIPPED_DEFAULT) + ", " + str(MISSED_DEFAULT) + ", " + str(ROCMONLY_DEFAULT) + ", " + str(CUDA_DEFAULT) + ", " + str(ROCM_DEFAULT) )
    print( " " )
    print( "=====Distributed GPU Number=====" )
    print( f"SKIPPED_DISTRIBUTED, MISSED_DISTRIBUTED, {set1_disp}ONLY_DISTRIBUTED, {set2_disp}_DISTRIBUTED, {set1_disp}_DISTRIBUTED" )
    print( str(SKIPPED_DISTRIBUTED) + ", " + str(MISSED_DISTRIBUTED) + ", " + str(ROCMONLY_DISTRIBUTED) + ", " + str(CUDA_DISTRIBUTED) + ", " + str(ROCM_DISTRIBUTED) )
    print( " " )
    print( "=====Inductor GPU Number=====" )
    print( f"SKIPPED_INDUCTOR, MISSED_INDUCTOR, {set1_disp}ONLY_INDUCTOR, {set2_disp}_INDUCTOR, {set1_disp}_INDUCTOR" )
    print( str(SKIPPED_INDUCTOR) + ", " + str(MISSED_INDUCTOR) + ", " + str(ROCMONLY_INDUCTOR) + ", " + str(CUDA_INDUCTOR) + ", " + str(ROCM_INDUCTOR) )
    print( " " )
    print( "SELECTED CAUSES SUMMARY" )
    print( " " )
    print( "=====================" )
    print( "Single GPU test" )
    sorted_skip_reasons_statistics_default = sorted(skip_reasons_stat_default.keys(), key = lambda x : x.lower())
    for skip_reason_entry in sorted_skip_reasons_statistics_default:
        print( skip_reason_entry, ": ", skip_reasons_stat_default[skip_reason_entry] )
    print( " " )
    print( "=====================" )
    print( "Distributed test" )
    sorted_skip_reasons_distributed_statistics = sorted(skip_reasons_stat_distributed.keys(), key = lambda x : x.lower())
    for skip_reason_entry in sorted_skip_reasons_distributed_statistics:
        print( skip_reason_entry, ": ", skip_reasons_stat_distributed[skip_reason_entry] )
    print( " " )
    print( "=====================" )
    print( "Inductor test" )
    sorted_skip_reasons_statistics_inductor = sorted(skip_reasons_stat_inductor.keys(), key = lambda x : x.lower())
    for skip_reason_entry in sorted_skip_reasons_statistics_inductor:
        print( skip_reason_entry, ": ", skip_reasons_stat_inductor[skip_reason_entry] )
    print( " " )
    print( "=====================" )
    print( "Time statistics" )
    print( f"{set1_disp}_RUNNING_TIME, {set2_disp}_RUNNING_TIME" )
    print( str(TOTAL_ROCM_RUNNING_TIME) + ", " + str(TOTAL_CUDA_RUNNING_TIME) )
    #print( "ROCm test file level time statistics" )
    #for (k,v) in list(test_file_level_ROCm.items()):
      #print( k[0] + ", " + k[1] + ", " + k[2] + ", " + str(v) )
    #print( "CUDA test file level time statistics" )
    #for (k,v) in list(test_file_level_CUDA.items()):
      #print( k[0] + ", " + k[1] + ", " + k[2] + ", " + str(v) )

def parse_args():
    parser = argparse.ArgumentParser(description='Parse xml test-reports')
    parser.add_argument("--set1", required=False, type=str, help="absolute or relative path to first test-reports dir")
    parser.add_argument("--set2", required=False, type=str, help="absolute or relative path to second test-reports dir")
    parser.add_argument("--set1_name", required=False, type=str, default="set1", help="display name for set1 in CSV column headers (default: set1)")
    parser.add_argument("--set2_name", required=False, type=str, default="set2", help="display name for set2 in CSV column headers (default: set2)")
    parser.add_argument("--output_csv", required=False, type=str, help="output csv filename", default="tests_from_xml.csv")
    parser.add_argument("--filter", required=False, type=str, help="ut status filter flag")
    parser.add_argument("--skip_reasons", required=False, type=str, help='skip reasons file')
    parser.add_argument("--test_file_running_time_output_csv", required=False, type=str, help="file running time output csv filename", default="file_running_time_output.csv")
    parser.add_argument("--prev_week_csv", required=False, type=str, help="previous week's all tests status CSV file to check if tests existed")
    return parser.parse_args()

def main():
    global args
    args = parse_args()
    summarize_xml_files(args)

if __name__ == "__main__":
    main()

