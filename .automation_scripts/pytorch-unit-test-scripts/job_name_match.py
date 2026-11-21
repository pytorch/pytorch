import difflib
import re


_TEST_JOB = re.compile(
    r"^(?P<prefix>.+) / (?P<kind>test(?:-osdc)?) "
    r"\((?P<config>[^,]+), (?P<shard>\d+), (?P<total>\d+)"
)
_SPECIAL_FAMILIES = ("build-only", "debug", "no-ops", "slow-gradcheck", "smoke")


def _matches_platform(prefix, platform):
    normalized = prefix.lower()
    if platform == "rocm":
        return "rocm" in normalized and "cuda" not in normalized
    if platform == "cuda":
        return "cuda" in normalized and "rocm" not in normalized
    raise ValueError(f"Unsupported platform: {platform}")


def choose_test_job_family(jobs, test_config, platform, configured_prefix):
    """Return the best matching sharded test family from actual workflow jobs."""
    families = {}
    for job in jobs:
        match = _TEST_JOB.match(job.get("name", ""))
        if not match or match.group("config") != test_config:
            continue
        prefix = match.group("prefix")
        if not _matches_platform(prefix, platform):
            continue
        key = (prefix, match.group("kind"), int(match.group("total")))
        families.setdefault(key, []).append(job)

    if not families:
        return None

    def score(item):
        (prefix, _, total), matching_jobs = item
        shards = {
            int(_TEST_JOB.match(job["name"]).group("shard"))
            for job in matching_jobs
        }
        complete = shards == set(range(1, total + 1))
        normalized = prefix.lower()
        return (
            not any(marker in normalized for marker in _SPECIAL_FAMILIES),
            complete,
            prefix == configured_prefix,
            difflib.SequenceMatcher(None, prefix, configured_prefix).ratio(),
            len(shards),
        )

    (prefix, kind, total), matching_jobs = max(families.items(), key=score)
    shards = {
        int(_TEST_JOB.match(job["name"]).group("shard"))
        for job in matching_jobs
    }
    return {
        "prefix": prefix,
        "kind": kind,
        "total": total,
        "jobs": matching_jobs,
        "complete": shards == set(range(1, total + 1)),
    }
