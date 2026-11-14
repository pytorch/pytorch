"""How long tests take in CI, for sizing the shards ramjet test runs: each test's mean
over the last 7 days of main, from ClickHouse's ci.health. The ci-metrics skill's
wrapper runs the query, which takes a gh login that HUD accepts."""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

from tools.ramjet.common import repo_root
from tools.ramjet.logger import log


WRAPPER = ".agents/skills/ci-metrics/gcx-wrapper.sh"
DATASOURCE = "ceczcsck1b20wb"  # ClickHouse, by the uid the skill gives it
# Tests a query may return: the wrapper stops a response at 50 MB, about 400,000
BATCH = 100_000
TIMEOUT = 120
TAIL = 1000  # characters of a failed query's output to show
# Each test's mean seconds on machines with these GPUs, where a skip counts as none
# since the test would skip here too, and its mean on any machine when it ran. Only
# runs with no variable but TEST_CONFIG count: the others wrap the test, in Dynamo
# for one. PYTORCH_TEST_EXCLUDE_HW_CLASSIFICATION doesn't: CI's GPU jobs set it to
# leave some test classes out, so their runs count as well
QUERY = """
SELECT concat(t.file, '::', if(t.suite = '', '', concat(t.suite, '::')), t.case_name),
       sumIf(h.total_duration_milliseconds, {here})
           / sumIf(h.attempts + h.skipped_runs, {here}) / 1000,
       sum(h.total_duration_milliseconds) / greatest(sum(h.attempts), 1) / 1000
FROM ci.health AS h
JOIN ci.tests AS t FINAL ON h.tests_id = t.id
JOIN ci.environments AS e FINAL ON h.environments_id = e.id
JOIN ci.env_vars AS v FINAL ON h.env_vars_id = v.id
WHERE h.window = '7d' AND h.ran_on = 'main' AND t.repo = 'pytorch/pytorch'
  AND t.file IN ({files})
  AND arrayFilter(k -> k != 'PYTORCH_TEST_EXCLUDE_HW_CLASSIFICATION',
                  mapKeys(v.env_vars)) = ['TEST_CONFIG']
GROUP BY 1
"""

Seconds = tuple[float | None, float]  # on machines with these GPUs, and on any


def literal(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def query(files: list[str], device: str, gpus: int) -> dict[str, Seconds] | str:
    """Each test's seconds by node id, or why CI gave none."""
    here = f"e.device_name = {literal(device)} AND e.device_count = {gpus}"
    sql = QUERY.format(files=", ".join(map(literal, files)), here=here)
    wrapper = [repo_root() / WRAPPER, "datasources", "clickhouse", "query"]
    options = ["-d", DATASOURCE, "--limit", "0", "-o", "json"]
    output = {"capture_output": True, "text": True, "timeout": TIMEOUT}
    try:
        done = subprocess.run([*wrapper, *options, sql], **output)
    except OSError as error:
        return str(error)
    except subprocess.TimeoutExpired:
        return f"no answer in {TIMEOUT} s"
    if done.returncode != 0:
        # All it printed: gcx tells a person why on stderr, and an agent on stdout
        return (done.stderr + done.stdout).strip()[-TAIL:]
    rows = json.loads(done.stdout)["rows"] or []  # null when CI ran none of the files
    return {nodeid: (same, anywhere) for nodeid, same, anywhere in rows}


def fetch(files: dict[str, int], device: str, gpus: int) -> dict[str, Seconds]:
    """CI's seconds for each test of these files, by node id; files maps each to how
    many tests it has here. Empty, with a warning, if CI can't be asked."""
    batches: list[list[str]] = [[]]
    room = BATCH
    for file, tests in files.items():
        if batches[-1] and tests > room:
            batches.append([])
            room = BATCH
        batches[-1].append(file)
        room -= tests
    times: dict[str, Seconds] = {}
    with ThreadPoolExecutor() as pool:
        for answer in pool.map(lambda batch: query(batch, device, gpus), batches):
            if isinstance(answer, str):
                log.warning("no test times from CI:\n%s", answer)
                return {}
            times.update(answer)
    return times
