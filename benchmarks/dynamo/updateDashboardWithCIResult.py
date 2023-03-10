#!/usr/bin/env python3
import glob
import os
import shutil
import subprocess
import tempfile

import pandas as pd

from rockset import RocksetClient


def getLatestNightlyRunID():
    rs = RocksetClient(
        api_key=f"{os.environ['ROCKSET_APIKEY']}",
        host="https://api.usw2a1.rockset.com",
    )

    params = list()
    response = rs.QueryLambdas.execute_query_lambda(
        query_lambda="queryPerfNigthlyRunID",
        version="458ede14e8877943",
        workspace="commons",
        parameters=params,
    )
    assert (
        response is not None
        and response.results is not None
        and len(response.results) != 0
    ), "Not a valid Rockset response"
    return response.results[0]["workflow_id"]


def downloadArtifacts(runID, download_dir):
    try:
        print(f"Downloading nightly artifacts with runID:{runID} into {download_dir}")
        subprocess.check_call(
            [
                "/data/home/anijain/miniconda/bin/gh",
                "run",
                "download",
                f"{runID}",
                "--pattern",
                "test-reports*",
            ],
            cwd=download_dir,
        )
    except Exception:
        print("Downloading failed")

    return download_dir


def combineTimmShards(download_dir, combined_dir, dtype):
    os.makedirs(combined_dir, exist_ok=False)
    timm_test_reports_dirs = []
    for shard in [
        "inductor_timm_perf-1-2",
        "inductor_timm_perf-2-2",
    ]:
        test_reports_dir = os.path.join(
            download_dir,
            f"test-reports-runattempt1-test-{shard}-linux.gcp.a100.large_*",
            "test-reports",
            dtype,
        )
        test_reports_dir = glob.glob(test_reports_dir)
        assert len(test_reports_dir) == 1, "test-reports directory not found"
        timm_test_reports_dirs.append(test_reports_dir[0])

    for csv_file in [
        os.path.basename(file)
        for file in glob.glob(os.path.join(test_reports_dir[0], "*.csv"))
    ]:
        if csv_file in {"comp_time.csv", "geomean.csv", "memory.csv", "passrate.csv"}:
            continue
        df0 = pd.read_csv(os.path.join(timm_test_reports_dirs[0], csv_file))
        df1 = pd.read_csv(os.path.join(timm_test_reports_dirs[1], csv_file))
        pd.concat([df0, df1], ignore_index=True).to_csv(
            os.path.join(combined_dir, csv_file),
            index=False,
        )


def combineAllSuites(download_dir, combined_dir, dtype):
    for shard in [
        "inductor_huggingface_perf-1-1",
        "inductor_torchbench_perf-1-1",
    ]:
        test_reports_dir = os.path.join(
            download_dir,
            f"test-reports-runattempt1-test-{shard}-linux.gcp.a100.large_*",
            "test-reports",
            dtype,
        )
        test_reports_dir = glob.glob(test_reports_dir)
        assert len(test_reports_dir) == 1, "test-reports directory not found"
        test_reports_dir = test_reports_dir[0]
        for csv_file in [
            os.path.basename(file)
            for file in glob.glob(os.path.join(test_reports_dir, "*.csv"))
        ]:
            if csv_file in {
                "comp_time.csv",
                "geomean.csv",
                "memory.csv",
                "passrate.csv",
            }:
                continue
            shutil.copy(os.path.join(test_reports_dir, csv_file), combined_dir)


if __name__ == "__main__":
    runID = getLatestNightlyRunID()
    with tempfile.TemporaryDirectory(
        dir=os.path.dirname(os.path.realpath(__file__))
    ) as download_dir:
        downloadArtifacts(runID, download_dir)
        for dtype in ["float32", "amp"]:
            combined_dir = os.path.join(download_dir, dtype)
            combineTimmShards(download_dir, combined_dir, dtype)
            combineAllSuites(download_dir, combined_dir, dtype)

            print(f"Updating {dtype} dashboard")
            subprocess.check_call(
                [
                    "python",
                    os.path.join(
                        os.path.dirname(os.path.realpath(__file__)), "runner.py"
                    ),
                    "--visualize_logs",
                    "--update-dashboard",
                    "--training",
                    "--output-dir",
                    combined_dir,
                    "--dtypes",
                    dtype,
                    "--dashboard-archive-path",
                    "/data/home/binbao/cluster/cron_logs",
                    "--runner-url",
                    f"https://github.com/pytorch/pytorch/actions/runs/{runID}",
                    "--github-issue-number",
                    "96191" if dtype == "amp" else "96192",
                ],
                stderr=subprocess.STDOUT,
            )
