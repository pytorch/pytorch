from typing import Callable, List, Union
from typing_extensions import TypeAlias


try:
    from fbscribelogger import (  # type: ignore[import-untyped, import-not-found, unused-ignore]
        make_scribe_logger,
    )
except ImportError:
    TAtom: TypeAlias = Union[int, float, bool, str]
    TField: TypeAlias = Union[TAtom, List[TAtom]]
    TLazyField: TypeAlias = Union[TField, Callable[[], TField]]

    def make_scribe_logger(name: str, thrift_src: str) -> Callable[..., None]:
        def inner(**kwargs: TLazyField) -> None:
            pass

        return inner


open_source_signpost = make_scribe_logger(
    "TorchOpenSourceSignpost",
    """
struct TorchOpenSourceSignpostLogEntry {

  # The commit SHA that triggered the workflow, e.g., 02a6b1d30f338206a71d0b75bfa09d85fac0028a. Derived from GITHUB_SHA.
  4: optional string commit_sha;

  # Commit date (not author date) of the commit in commit_sha as timestamp, e.g., 1724208105.  Increasing if merge bot is used, though not monotonic; duplicates occur when stack is landed.
  5: optional i64 commit_date;

  # The fully-formed ref of the branch or tag that triggered the workflow run, e.g., refs/pull/133891/merge or refs/heads/main. Derived from GITHUB_REF.
  6: optional string github_ref;

  # Indicates if branch protections or rulesets are configured for the ref that triggered the workflow run. Derived from GITHUB_REF_PROTECTED.
  7: optional bool github_ref_protected;

  # A unique number for each attempt of a particular workflow run in a repository, e.g., 1. Derived from GITHUB_RUN_ATTEMPT.
  8: optional string github_run_attempt;

  # A unique number for each workflow run within a repository, e.g., 19471190684. Derived from GITHUB_RUN_ID.
  9: optional string github_run_id;

  # A unique number for each run of a particular workflow in a repository, e.g., 238742. Derived from GITHUB_RUN_NUMBER.
  10: optional string github_run_number_str;

  # The name of the current job. Derived from JOB_NAME, e.g., linux-jammy-py3.8-gcc11 / test (default, 3, 4, linux.2xlarge).
  11: optional string job_name;

  # The GitHub user who triggered the job.  Derived from GITHUB_TRIGGERING_ACTOR.
  12: optional string github_triggering_actor;
  13: optional string name; # Event name
  14: optional string parameters; # Parameters (JSON data)
  16: optional string subsystem; # Subsystem the event is associated with

  # The unit timestamp in second for the Scuba Time Column override
  17: optional i64 time;

  # The weight of the record according to current sampling rate
  18: optional i64 weight;
}
""",  # noqa: B950
)
