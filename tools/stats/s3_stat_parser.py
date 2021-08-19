import bz2
import json
import logging
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any, cast
from typing_extensions import Literal, TypedDict

try:
    import boto3  # type: ignore[import]
    import botocore  # type: ignore[import]
    HAVE_BOTO3 = True
except ImportError:
    HAVE_BOTO3 = False


logger = logging.getLogger(__name__)


OSSCI_METRICS_BUCKET = 'ossci-metrics'

Commit = str  # 40-digit SHA-1 hex string
Status = Optional[Literal['errored', 'failed', 'skipped']]


class CaseMeta(TypedDict):
    seconds: float


class Version1Case(CaseMeta):
    name: str
    errored: bool
    failed: bool
    skipped: bool


class Version1Suite(TypedDict):
    total_seconds: float
    cases: List[Version1Case]


class ReportMetaMeta(TypedDict):
    build_pr: str
    build_tag: str
    build_sha1: Commit
    build_base_commit: Commit
    build_branch: str
    build_job: str
    build_workflow_id: str
    build_start_time_epoch: str


class ReportMeta(ReportMetaMeta):
    total_seconds: float


class Version1Report(ReportMeta):
    suites: Dict[str, Version1Suite]


class Version2Case(CaseMeta):
    status: Status


class Version2Suite(TypedDict):
    total_seconds: float
    cases: Dict[str, Version2Case]


class Version2File(TypedDict):
    total_seconds: float
    suites: Dict[str, Version2Suite]


class VersionedReport(ReportMeta):
    format_version: int


# report: Version2Report implies report['format_version'] == 2
class Version2Report(VersionedReport):
    files: Dict[str, Version2File]


Report = Union[Version1Report, VersionedReport]

if HAVE_BOTO3:
    S3_RESOURCE_READ_ONLY = boto3.resource("s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED))
    S3_RESOURCE = boto3.resource('s3')


def get_S3_bucket_readonly(bucket_name: str) -> Any:
    return S3_RESOURCE_READ_ONLY.Bucket(bucket_name)


def get_S3_object_from_bucket(bucket_name: str, object: str) -> Any:
    return S3_RESOURCE.Object(bucket_name, object)


def case_status(case: Version1Case) -> Status:
    for k in {'errored', 'failed', 'skipped'}:
        if case[k]:  # type: ignore[misc]
            return cast(Status, k)
    return None


def newify_case(case: Version1Case) -> Version2Case:
    return {
        'seconds': case['seconds'],
        'status': case_status(case),
    }


def get_cases(
    *,
    data: Report,
    filename: Optional[str],
    suite_name: Optional[str],
    test_name: Optional[str],
) -> List[Version2Case]:
    cases: List[Version2Case] = []
    if 'format_version' not in data:  # version 1 implicitly
        v1report = cast(Version1Report, data)
        suites = v1report['suites']
        for sname, v1suite in suites.items():
            if not suite_name or sname == suite_name:
                for v1case in v1suite['cases']:
                    if not test_name or v1case['name'] == test_name:
                        cases.append(newify_case(v1case))
    else:
        v_report = cast(VersionedReport, data)
        version = v_report['format_version']
        if version == 2:
            v2report = cast(Version2Report, v_report)
            for fname, v2file in v2report['files'].items():
                if fname == filename or not filename:
                    for sname, v2suite in v2file['suites'].items():
                        if sname == suite_name or not suite_name:
                            for cname, v2case in v2suite['cases'].items():
                                if not test_name or cname == test_name:
                                    cases.append(v2case)
        else:
            raise RuntimeError(f'Unknown format version: {version}')
    return cases


def _parse_master_summaries(summaries: Any, jobs: List[str]) -> Dict[str, List[Report]]:
    summary_dict = defaultdict(list)
    for summary in summaries:
        # master summary format: "test_time/{sha}/{job}/file"
        summary_job = summary.key.split('/')[2]
        if summary_job in jobs or len(jobs) == 0:
            binary = summary.get()["Body"].read()
            string = bz2.decompress(binary).decode("utf-8")
            summary_dict[summary_job].append(json.loads(string))
    return summary_dict

def _parse_pr_summaries(summaries: Any, job_prefix: str) -> Dict[str, List[Tuple[Report, str]]]:
    summary_dict = defaultdict(list)
    for summary in summaries:
        # PR summary format: "pr_test_time/{pr}/{sha}/{job}/file"
        summary_job = summary.key.split('/')[3]
        summary_timestamp = summary.key.split('/')[4][:len("YYYY-MM-ddTHH:mm:ss")]
        if not job_prefix or len(job_prefix) == 0 or summary_job.startswith(job_prefix):
            binary = summary.get()["Body"].read()
            string = bz2.decompress(binary).decode("utf-8")
            summary_dict[summary_job].append((json.loads(string), summary_timestamp))
    return summary_dict


# Collect and decompress S3 test stats summaries into JSON.
# data stored on S3 buckets are pathed by {sha}/{job} so we also allow
# optional jobs filter
def get_test_stats_summaries(*, sha: str, jobs: Optional[List[str]] = None) -> Dict[str, List[Report]]:
    bucket = get_S3_bucket_readonly(OSSCI_METRICS_BUCKET)
    summaries = bucket.objects.filter(Prefix=f"test_time/{sha}")
    return _parse_master_summaries(summaries, jobs=list(jobs or []))


def get_test_stats_summaries_for_job(*, sha: str, job_prefix: str) -> Dict[str, List[Report]]:
    bucket = get_S3_bucket_readonly(OSSCI_METRICS_BUCKET)
    summaries = bucket.objects.filter(Prefix=f"test_time/{sha}/{job_prefix}")
    return _parse_master_summaries(summaries, jobs=list())

def get_test_stats_summaries_for_pr(*, pr: str, job_prefix: str) -> Dict[str, List[Tuple[Report, str]]]:
    bucket = get_S3_bucket_readonly(OSSCI_METRICS_BUCKET)
    summaries = bucket.objects.filter(Prefix=f"pr_test_time/{pr}/")
    return _parse_pr_summaries(summaries, job_prefix=job_prefix)


# This function returns a list of S3 test time reports. This function can run into errors if HAVE_BOTO3 = False
# or the S3 bucket is somehow unavailable. Even though this function goes through ten commits' reports to find a
# non-empty report, it is still conceivable (though highly unlikely) for this function to return no reports.
def get_previous_reports_for_branch(branch: str, ci_job_prefix: str = "") -> List[Report]:
    commit_date_ts = subprocess.check_output(
        ['git', 'show', '-s', '--format=%ct', 'HEAD'],
        encoding="ascii").strip()
    commit_date = datetime.fromtimestamp(int(commit_date_ts))
    # We go a day before this current commit to avoiding pulling incomplete reports
    day_before_commit = str(commit_date - timedelta(days=1)).split(' ')[0]
    # something like git rev-list --before="2021-03-04" --max-count=10 --remotes="*origin/nightly"
    commits = subprocess.check_output(
        ["git", "rev-list", f"--before={day_before_commit}", "--max-count=10", f"--remotes=*{branch}"],
        encoding="ascii").splitlines()

    reports: List[Report] = []
    commit_index = 0
    while len(reports) == 0 and commit_index < len(commits):
        commit = commits[commit_index]
        logger.info(f'Grabbing reports from commit: {commit}')
        summaries = get_test_stats_summaries_for_job(sha=commit, job_prefix=ci_job_prefix)
        for job_name, summary in summaries.items():
            reports.append(summary[0])
            if len(summary) > 1:
                logger.warning(f'WARNING: Multiple summary objects found for {commit}/{job_name}')
        commit_index += 1
    return reports


def get_previous_reports_for_pr(pr: str, ci_job_prefix: str = "") -> List[Tuple[Report, str]]:
    reports: List[Tuple[Report, str]] = []
    logger.info(f'Grabbing reports from PR: {[pr]}')
    summaries = get_test_stats_summaries_for_pr(pr=pr, job_prefix=ci_job_prefix)
    for _, summary in summaries.items():
        reports.extend(summary)
    # sort by summary_timestamp
    reports.sort(reverse=True, key=lambda s: s[1])
    return reports
