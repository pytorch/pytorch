#!/usr/bin/env python

"""
Script to launch all caffe2 MobileLab tests
MobieLab uses `flow-cli` to launch jobs
To install flow-cli on your devserver, run
```
sudo feature install fblearner_flow
```

Example usage:

- Run all tests: ~/fbsource/xplat/caffe2/scripts/run_mobilelab.py --diff_id 56041720 --base_commit 583c0d7396c8d0469a225bf2bb19d8ecd25ba3af
- Run all android tests: ~/fbsource/xplat/caffe2/scripts/run_mobilelab.py --diff_id 56041720 --base_commit <commit_id> --android
- Run all android tests: ~/fbsource/xplat/caffe2/scripts/run_mobilelab.py --diff_id 56041720 --base_commit <commit_id> --iOS
- Run all tests and let LabBot post lab run results on your diff:  ~/fbsource/xplat/caffe2/scripts/run_mobilelab.py --diff_id 56041720 --base_commit 583c0d7396c8d0469a225bf2bb19d8ecd25ba3af --comment_on_revision D8756131

"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import os
import subprocess
import sys


# comment out the tests you don't need to run
android_tests = [
    "android_microbenchmark.aml.microbenchmark_caffe2.correctness.nexus6",
    "android_microbenchmark.aml.microbenchmark_caffe2.body_tracking_cold_delay.nexus6",
    "android_microbenchmark.aml.microbenchmark_caffe2.body_tracking_delay.nexus6",
    "android_microbenchmark.aml.microbenchmark_caffe2.body_tracking_memory.nexus6",
    "android_microbenchmark.aml.microbenchmark_caffe2.body_tracking_operator_delay.nexus6",
    "android_microbenchmark.aml.microbenchmark_caffe2.hand_tracking_cold_delay.nexus6",
    "android_microbenchmark.aml.microbenchmark_caffe2.hand_tracking_delay.nexus6",
    "android_microbenchmark.aml.microbenchmark_caffe2.hand_tracking_memory.nexus6",
    "android_microbenchmark.aml.microbenchmark_caffe2.hand_tracking_operator_delay.nexus6",
    "android_microbenchmark.aml.microbenchmark_caffe2.person_segmentation_cold_delay.nexus6",
    "android_microbenchmark.aml.microbenchmark_caffe2.person_segmentation_delay.nexus6",
    "android_microbenchmark.aml.microbenchmark_caffe2.person_segmentation_memory.nexus6",
    "android_microbenchmark.aml.microbenchmark_caffe2.person_segmentation_operator_delay.nexus6",
    "android_microbenchmark.aml.microbenchmark_caffe2.xray_mobile_42_cold_delay.nexus6",
    "android_microbenchmark.aml.microbenchmark_caffe2.xray_mobile_42_delay.nexus6",
    "android_microbenchmark.aml.microbenchmark_caffe2.xray_mobile_42_memory.nexus6",
    "android_microbenchmark.aml.microbenchmark_caffe2.xray_mobile_42_operator_delay.nexus6",
    "android_microbenchmark.aml.microbenchmark_caffe2.xray_mobile_84_v1_cold_delay.nexus6",
    "android_microbenchmark.aml.microbenchmark_caffe2.xray_mobile_84_v1_delay.nexus6",
    "android_microbenchmark.aml.microbenchmark_caffe2.xray_mobile_84_v1_memory.nexus6",
    "android_microbenchmark.aml.microbenchmark_caffe2.xray_mobile_84_v1_operator_delay.nexus6",
]

iOS_tests = [
    "xcui.caffe2.correctness.ios10.iPhone7",
    "xcui.caffe2.body_tracking_cold_delay.ios10.iPhone7",
    "xcui.caffe2.body_tracking_delay.ios10.iPhone7",
    "xcui.caffe2.body_tracking_memory.ios10.iPhone7",
    "xcui.caffe2.hand_tracking_cold_delay.ios10.iPhone7",
    "xcui.caffe2.hand_tracking_delay.ios10.iPhone7",
    "xcui.caffe2.hand_tracking_memory.ios10.iPhone7",
    "xcui.caffe2.person_segmentation_cold_delay.ios10.iPhone7",
    "xcui.caffe2.person_segmentation_delay.ios10.iPhone7",
    "xcui.caffe2.person_segmentation_memory.ios10.iPhone7",
    "xcui.caffe2.xray_mobile_42_cold_delay.ios10.iPhone7",
    "xcui.caffe2.xray_mobile_42_delay.ios10.iPhone7",
    "xcui.caffe2.xray_mobile_42_memory.ios10.iPhone7",
    "xcui.caffe2.xray_mobile_84_v1_cold_delay.ios10.iPhone7",
    "xcui.caffe2.xray_mobile_84_v1_delay.ios10.iPhone7",
    "xcui.caffe2.xray_mobile_84_v1_memory.ios10.iPhone7",
]

android_job_inputs = json.loads(
    """{
  "test_config_name": "android_microbenchmark.aml.microbenchmark_caffe2.correctness.nexus6",
  "workflow_inputs": {
    "gating_overrides": null,
    "android_microbenchmark": {
      "experiment_store_config": {
        "metadata": {
          "entry_point": "frontend"
        }
      },
      "one_world_priority": -10,
      "experiment_request": {
        "treatment": {
          "build_request": {
            "build_spec": {
              "diff_id": "replaceme"
            },
            "priority": 0
          }
        },
        "control": {
          "build_request": {
            "build_spec": {
              "commit_hash": "replaceme"
            },
            "priority": 0
          }
        }
      },
      "comment_on_revision": "replaceme"
    }
  }
}"""
)

iOS_job_inputs = json.loads(
    """{
    "test_config_name": "xcui.caffe2.correctness.ios10.iPhone7",
    "workflow_inputs": {
    "xcui": {
      "gating_overrides": null,
      "experiment_store_config": {
        "metadata": {
          "entry_point": "frontend"
        }
      },
      "one_world_priority": -10,
      "experiment_request": {
        "treatment": {
          "build_request": {
            "build_spec": {
              "diff_id": "replaceme"
            },
            "priority": 0
          }
        },
        "control": {
          "build_request": {
            "build_spec": {
              "commit_hash": "replaceme"
            },
            "priority": 0
          }
        }
      },
      "comment_on_revision": "replaceme"
    }
  }
}"""
)


parser = argparse.ArgumentParser(description="Launch all caffe2 mobilelab jobs")
parser.add_argument(
    "--diff_id",
    required=True,
    help="the diff version (not the DNNNNNNNN revision) for your change, e.g. 55434539",
)
parser.add_argument(
    "--base_commit",
    required=True,
    help="base commit hash, e.g. 583c0d7396c8d0469a225bf2bb19d8ecd25ba3af",
)
parser.add_argument(
    "--comment_on_revision",
    default=None,
    help="the phabricator revision (the DNNNNNNNN revision), eg. D8756131 or D8756131",
)
parser.add_argument("--android", action="store_true", help="Run all android tests")
parser.add_argument("--iOS", action="store_true", help="Run all iOS tests")


def main(args):
    options = parser.parse_args(args)

    diff_id = options.diff_id
    base_commit = options.base_commit
    comment_on_revision = options.comment_on_revision
    if comment_on_revision and comment_on_revision.startswith("D"):
        comment_on_revision = comment_on_revision[1:]

    # Launch all jobs if the user didn't specify android or iOS
    run_all_jobs = True if not (options.android) and not (options.iOS) else False

    # Must launch jobs from fbcode
    os.system("cd ~/fbsource/fbcode")

    if run_all_jobs or options.android:
        for t in android_tests:
            android_job_inputs["workflow_inputs"]["android_microbenchmark"][
                "experiment_request"
            ]["treatment"]["build_request"]["build_spec"]["diff_id"] = diff_id
            android_job_inputs["workflow_inputs"]["android_microbenchmark"][
                "experiment_request"
            ]["control"]["build_request"]["build_spec"]["commit_hash"] = base_commit
            android_job_inputs["workflow_inputs"]["android_microbenchmark"][
                "comment_on_revision"
            ] = comment_on_revision
            android_job_inputs["test_config_name"] = t
            subprocess.check_call(
                [
                    "flow-cli",
                    "launch",
                    "lab.run_test.entry",
                    "--parameters-json",
                    json.dumps(android_job_inputs),
                    "--entitlement",
                    "oneworld_prn",
                    "--name",
                    "Run with test config " + t,
                ]
            )
    if run_all_jobs or options.iOS:
        for t in iOS_tests:
            iOS_job_inputs["workflow_inputs"]["xcui"]["experiment_request"][
                "treatment"
            ]["build_request"]["build_spec"]["diff_id"] = diff_id
            iOS_job_inputs["workflow_inputs"]["xcui"]["experiment_request"]["control"][
                "build_request"
            ]["build_spec"]["commit_hash"] = base_commit
            iOS_job_inputs["workflow_inputs"]["xcui"][
                "comment_on_revision"
            ] = comment_on_revision
            iOS_job_inputs["test_config_name"] = t
            subprocess.check_call(
                [
                    "flow-cli",
                    "launch",
                    "lab.run_test.entry",
                    "--parameters-json",
                    json.dumps(iOS_job_inputs),
                    "--entitlement",
                    "oneworld_prn",
                    "--name",
                    "Run with test config " + t,
                ]
            )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
