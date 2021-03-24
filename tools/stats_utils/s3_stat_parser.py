import bz2
import json
from collections import defaultdict

from typing import Dict, List, Optional, Union, Any, cast
from typing_extensions import Literal, TypedDict

try:
    import boto3  # type: ignore[import]
    import botocore  # type: ignore[import]
    HAVE_BOTO3 = True
except ImportError:
    HAVE_BOTO3 = False


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
    build_branch: str
    build_job: str
    build_workflow_id: str


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


def get_S3_bucket_readonly(bucket_name: str) -> Any:
    s3 = boto3.resource("s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED))
    return s3.Bucket(bucket_name)


def get_S3_object_from_bucket(bucket_name: str, object: str) -> Any:
    s3 = boto3.resource('s3')
    return s3.Object(bucket_name, object)


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
    test_name: str,
) -> List[Version2Case]:
    cases: List[Version2Case] = []
    if 'format_version' not in data:  # version 1 implicitly
        v1report = cast(Version1Report, data)
        suites = v1report['suites']
        for sname, v1suite in suites.items():
            if sname == suite_name or not suite_name:
                for v1case in v1suite['cases']:
                    if v1case['name'] == test_name:
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
                            v2case = v2suite['cases'].get(test_name)
                            if v2case:
                                cases.append(v2case)
        else:
            raise RuntimeError(f'Unknown format version: {version}')
    return cases


def _parse_s3_summaries(summaries: Any, jobs: List[str]) -> Dict[str, List[Any]]:
    summary_dict = defaultdict(list)
    for summary in summaries:
        summary_job = summary.key.split('/')[2]
        if summary_job in jobs or len(jobs) == 0:
            binary = summary.get()["Body"].read()
            string = bz2.decompress(binary).decode("utf-8")
            summary_dict[summary_job].append(json.loads(string))
    return summary_dict

# Collect and decompress S3 test stats summaries into JSON.
# data stored on S3 buckets are pathed by {sha}/{job} so we also allow
# optional jobs filter
def get_test_stats_summaries(*, sha: str, jobs: Optional[List[str]] = None) -> Dict[str, List[Any]]:
    bucket = get_S3_bucket_readonly(OSSCI_METRICS_BUCKET)
    summaries = list(bucket.objects.filter(Prefix=f"test_time/{sha}"))
    return _parse_s3_summaries(summaries, jobs=list(jobs or []))


def get_test_stats_summaries_for_job(*, sha: str, job_prefix: str) -> Dict[str, List[Any]]:
    bucket = get_S3_bucket_readonly(OSSCI_METRICS_BUCKET)
    summaries = list(bucket.objects.filter(Prefix=f"test_time/{sha}/{job_prefix}"))
    return _parse_s3_summaries(summaries, jobs=list())
