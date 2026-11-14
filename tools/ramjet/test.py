"""ramjet test: run the tests under the paths you name on the GPUs you name, or list
them. The work is calls to a Modal Function, which Modal spreads over as many
containers as it gives you, each taking one call at a time. The first calls name
the test files, CALL_FILES of them each at most: a call's container collects the
tests of those that run_test.py runs there, worker.AT_ONCE files at a time, and
leaves out the tests whose tci constraints (torch/testing/_internal/tci) its machine
doesn't meet. The rest are cut into shards that should each take a container about
SHARD_SECONDS, by CI's time for each test and with as many test processes at once
as fit in a container. Each shard is one more call, which a container runs through
run_test.py: the containers that collected take the first."""

import contextlib
import gzip
import hashlib
import itertools
import json
import shlex
import statistics
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import modal

from tools.ramjet import progress, sync, times, worker
from tools.ramjet.common import count, repo_root
from tools.ramjet.lineage import Build
from tools.ramjet.logger import bold, CONTINUED, dim, log, magenta, red, RING
from tools.ramjet.run import CPU_CHOICES, gpu_name, MEMORY_CHOICES, Resources
from tools.testing.discover_tests import TESTS


# A container's resources unless --cpu and --memory say otherwise: CORES_PER_GPU per
# GPU, or as many without one, with MIB_PER_CORE of memory each
CORES_PER_GPU = 4
MIB_PER_CORE = 4096
# What a container gives each test process, which is how many it runs at once: of
# its cores and memory, what one of the default size gives each of three, and of a
# GPU's memory, what CI's L4 gives each of its three. A process has THREADS threads
# at least, as in CI: with fewer, some tests' sums round differently
PROCESSES_PER_CORE = Fraction(3, 4)
PROCESS_MIB = 5 * 1024
PROCESS_GPU_GIB = 7
THREADS = 4
# The variables of yours that go to the containers, by how their names start:
# those that torch and its tests read, such as PYTORCH_TEST_WITH_SLOW and TORCH_LOGS
PASSED = ("PYTORCH_", "TORCH")
# The test files a call names at most, which one container collects: a container
# gets through about two a second, however many cores it has. More files are more
# calls, which as many containers take at once. With 64 a call the suite's files
# took 98 s, and 72 s with 32: a container takes about as long to start
CALL_FILES = 32
# How long a container waits for a call before it stops. When listing, the least
# Modal allows. When running, long enough for a container that collected to get
# a shard: they are cut once CI has given the tests' times
SCALEDOWN_SECONDS = 2
HANDOVER_SECONDS = 15
SHARD_SECONDS = 150
QUEUE_ITEMS = 5000  # what a Modal Queue holds at most: all there is, in one call
# A container's seconds per second of CI, and its seconds to start a test file: fitted
# on L4 containers running test/dynamo
SLOWDOWN = 2.28
START_SECONDS = 5.4
# CI's mean for a test, for the tests of a file it has timed none of
GUESS_SECONDS = 0.3
# A test's outcomes, as torchci's reports name them, with ramjet's own flaky: those
# that need no look, and those that fail a run, as does a test that did not run. The
# summary counts each, in this order
FINE = ("passed", "flaky", "xfailed", "skipped")
FAILED = ("failed", "error", "crashed", "timed_out", "xpassed")

Tests = list[tuple[str, float]]  # node ids, each with the seconds it should take


@dataclass
class Shard:
    """What one container runs: tests of some files, each file started once."""

    seconds: float = 0.0  # what it should take the container
    tests: dict[str, list[str]] = field(default_factory=dict)  # node ids, by their file
    # The seconds run_test.py is told each file takes: past those it splits a file
    # at, it runs the file as several processes
    times: dict[str, float] = field(default_factory=dict)
    # The test process that runs each test, by node id
    where: dict[str, worker.Process] = field(default_factory=dict)


@dataclass
class Found:
    """What the containers made of the files asked for."""

    # The tests asked for that the containers can run, by file
    kept: dict[str, list[str]] = field(default_factory=dict)
    # Each file's tests marked @serialTest, kept or not
    serial: dict[str, set[str]] = field(default_factory=dict)
    alone: set[str] = field(default_factory=set)  # the files run_test.py runs alone
    # The containers, counted by what they read of themselves
    machines: Counter[str] = field(default_factory=Counter)
    failed: int = 0  # files that couldn't be collected


@dataclass
class Run:
    """Consecutive tests of a file, as run_test.py runs them in a shard."""

    name: str
    ids: list[str]
    beside: list[float]  # the seconds of each process it runs beside others
    alone: float  # the seconds of those it runs with nothing beside them

    @property
    def seconds(self) -> float:
        """What it takes a container that runs nothing else."""
        return max(self.beside, default=0.0) + self.alone


def names(paths: list[str]) -> dict[str, list[str]]:
    """run_test.py's name for each test file the paths cover, with what the paths
    ask of it: the file, for all its tests, or node ids in it. A path is a test
    file, a directory of them or a node id, from the directory you are in."""
    test_dir = repo_root() / "test"
    asked: dict[str, list[str]] = {}
    for path in paths:
        file, _, node = path.partition("::")
        target = Path(file).resolve()
        found: list[str] = []
        if target.is_dir() and not node and target.is_relative_to(test_dir):
            under = "" if target == test_dir else f"{target.relative_to(test_dir)}/"
            # TESTS also names C++ binaries and tests that have no file of their own
            found = [name for name in TESTS if name.startswith(under)]
            found = [name for name in found if (test_dir / f"{name}.py").is_file()]
        elif target.is_file() and target.is_relative_to(test_dir):
            name = target.relative_to(test_dir).with_suffix("").as_posix()
            found = [name] if name in TESTS else []
        if not found:
            log.error("%s is not a test file or directory that run_test.py runs", path)
            log.error("with pytest instead: python -m tools.ramjet run pytest %s", path)
            sys.exit(1)
        for name in found:
            file_id = f"test/{name}.py"
            asked.setdefault(name, []).append(f"{file_id}::{node}" if node else file_id)
    return asked


@contextlib.contextmanager
def fleet(
    image: Build,
    commit: str,
    resources: Resources,
    ramjet_dir: Path,
    idle: int,
    env: dict[str, str],
) -> Iterator[tuple[modal.Cls, str]]:
    """For the block, the class whose containers boot from the build image, take
    your changes since its commit and have env in their environment, and the
    address of its app on Modal's dashboard. A container takes one call at a time
    and stops once it has had none for idle seconds."""
    changes = sync.diff(image.commit, commit)
    changed = [path for path, status in changes.items() if status != "D"]
    deleted = [path for path, status in changes.items() if status == "D"]
    with (
        tempfile.NamedTemporaryFile(dir=ramjet_dir, suffix=".tar") as tar,
        tempfile.NamedTemporaryFile("w", dir=ramjet_dir) as gone,
    ):
        if changed:  # given no paths, git archives the whole tree
            sync.archive(commit, changed, Path(tar.name))
        gone.write("\0".join(deleted))
        gone.flush()
        files = (
            modal.Image.from_id(image.image_id)
            # The layer Modal's builder adds: on the build image alone, which is a
            # sandbox's snapshot, a Function's container can't start
            .run_commands("true")
            # Mounted as each container starts, so runs share the image
            .add_local_file(tar.name, worker.CHANGES)
            .add_local_file(gone.name, worker.DELETED)
        )
        app = modal.App("ramjet-test")
        # A shard runs for longer than Modal lets a call by default
        limits = {"scaledown_window": idle, "timeout": worker.CALL_SECONDS, "env": env}
        workers = app.cls(image=files, **limits, **resources._asdict())(worker.Worker)
        stopped: KeyboardInterrupt | None = None
        with app.run():  # its containers stop when ramjet does, Ctrl-C included
            try:
                yield workers, app.get_dashboard_url()
            except KeyboardInterrupt as stop:
                stopped = stop
                raise
        if stopped:  # Modal ends its app on Ctrl-C, and then swallows it
            raise stopped


def collect(
    workers: modal.Cls, names: list[str], pytest_args: list[str], meter: progress.Meter
) -> list[worker.Collected]:
    """Each test file's tests, in the order of names, but for the files of a call
    that Modal loses. A call names CALL_FILES files at most, for one container to
    collect, and the calls go out at once. Each gets files that follow each other
    in names: those of a directory import much the same, which their container
    then imports once. Meanwhile, show how far the files are: each container says
    on a queue which it has collected."""
    method = workers().collect
    calls = -(-len(names) // CALL_FILES)
    cuts = [len(names) * call // calls for call in range(calls + 1)]
    shares = [names[start:end] for start, end in itertools.pairwise(cuts)]
    found: dict[str, worker.Collected] = {}
    state = progress.Collecting(meter, len(names), calls)
    with modal.Queue.ephemeral() as said:

        def ask() -> None:
            meter.told(method.get_current_stats().num_total_runners, "collecting")
            state.back.update(said.get_many(QUEUE_ITEMS, block=False))

        with progress.shown(state, ask):
            given = [(share, pytest_args, said.object_id) for share in shares]
            options = {"order_outputs": False, "return_exceptions": True}
            try:
                for files in method.starmap(given, **options):
                    state.calls -= 1
                    if isinstance(files, Exception):
                        log.error("collecting failed:\n%r", files)
                        continue
                    for file in files:  # which the queue may not have said
                        found[file["name"]] = file
                        state.back.add(file["name"])
            except Exception as lost:
                log.error("collecting failed:\n%r", lost)
    return [found[name] for name in names if name in found]


def timed(kept: dict[str, list[str]], machine: dict[str, Any]) -> dict[str, Tests]:
    """Each test with the seconds it should take in a container: SLOWDOWN times CI's
    time for it on machines with the same GPUs, else on any machine, else the median
    of its file's tests."""
    files = {f"test/{name}.py": len(ids) for name, ids in kept.items()}
    device, gpus = machine["device"], machine["gpus"]
    ci = times.fetch(files, device, gpus)
    same = guessed = 0
    estimates = {}
    for name, ids in kept.items():
        seconds = []  # CI's for each test, or None for one it has never run
        for nodeid in ids:
            here, anywhere = ci.get(nodeid, (None, None))
            seconds.append(anywhere if here is None else here)
            same += here is not None
        known = [s for s in seconds if s is not None]
        typical = statistics.median(known) if known else GUESS_SECONDS
        guessed += len(ids) - len(known)
        scaled = [SLOWDOWN * (typical if s is None else s) for s in seconds]
        estimates[name] = list(zip(ids, scaled))
    others = sum(files.values()) - same - guessed
    on = f"{gpus} x {device}" if gpus else "no GPU"
    sources = count(same, "test"), on, others, guessed
    log.debug("CI's times: %s with %s, %d on other machines, %d guessed", *sources)
    return estimates


def lane(nodeid: str, processes: int) -> int:
    """Which of the processes that run_test.py runs a file as gets a test, from 0:
    test/pytest_shard_custom.py goes by the hash of the node id."""
    digest = hashlib.sha256(nodeid.encode()).digest()
    return int.from_bytes(digest, "little") % processes


def paced(name: str, tests: Tests, serial: Collection[str], processes: int) -> Run:
    """Tests of a file that run_test.py starts as so many processes beside others:
    none, for a file it runs alone. The tests marked @serialTest run after those,
    alone: serial names all of the file's, and each process that the hash gives
    one of them is started once more for its own."""
    beside = [START_SECONDS] * processes
    again = {lane(nodeid, processes) for nodeid in serial} if processes else {0}
    alone = START_SECONDS * len(again)
    for nodeid, seconds in tests:
        if processes and nodeid not in serial:
            beside[lane(nodeid, processes)] += seconds
        else:
            alone += seconds
    return Run(name, [nodeid for nodeid, _ in tests], beside, alone)


def placed(run: Run, serial: Collection[str]) -> dict[str, worker.Process]:
    """The test process that runs each test of a run, as paced has them: one of
    those beside others or, for a test marked @serialTest, the same one started
    again. A file that run_test.py runs alone is one process."""
    width = len(run.beside)
    if not width:
        return dict.fromkeys(run.ids, (run.name, 0, False))
    return {i: (run.name, lane(i, width), i in serial) for i in run.ids}


def cut(name: str, tests: Tests, serial: Collection[str], width: int) -> Iterator[Run]:
    """A file as runs of consecutive tests. A run ends where one more test would
    keep a container that runs it as width processes past SHARD_SECONDS, and then
    gets the fewest processes that keep it within them."""
    fresh = paced(name, [], serial, width)
    chunks: list[Tests] = [[]]
    loads = [*fresh.beside, fresh.alone]
    for nodeid, seconds in tests:
        at = lane(nodeid, width) if width and nodeid not in serial else -1
        loads[at] += seconds
        if chunks[-1] and max(loads[:-1], default=0.0) + loads[-1] > SHARD_SECONDS:
            chunks.append([])
            loads = [*fresh.beside, fresh.alone]
            loads[at] += seconds
        chunks[-1].append((nodeid, seconds))
    for chunk in chunks:
        ways = [paced(name, chunk, serial, n) for n in range(min(width, 1), width + 1)]
        # None, for a test that is longer than a shard: it gets one process
        yield next((run for run in ways if run.seconds <= SHARD_SECONDS), ways[0])


def length(runs: list[Run], procs: int) -> float:
    """What runs of different files take a container. run_test.py goes by the files'
    names: it starts the processes that run beside others in that order, procs at
    once, and runs the rest one after another."""
    ends = [0.0] * procs
    for run in sorted(runs, key=lambda run: run.name):
        for seconds in run.beside:
            ends[ends.index(min(ends))] += seconds
    return max(ends) + sum(run.alone for run in runs)


def sharded(
    timed: dict[str, Tests],
    alone: Collection[str],
    serial: dict[str, set[str]],
    procs: int,
    split: float,
) -> list[Shard]:
    """Shards that should each take a container about SHARD_SECONDS, the longest
    first. The container runs procs test processes at once, so a file that takes
    longer is cut into runs of consecutive tests that fill that many, and shorter
    files share a shard: one run of a file per shard, so that a shard starts each
    of its files once. The files in alone run as one process, with nothing beside
    it. run_test.py is told that a file takes split seconds for each process but
    the last, which makes it run the file as that many."""
    runs: list[Run] = []
    for name, tests in timed.items():
        width = 0 if name in alone else procs
        runs += cut(name, tests, serial.get(name, ()), width)
    shards: list[list[Run]] = []
    for run in sorted(runs, key=lambda run: -(sum(run.beside) + run.alone)):
        fits = (
            shard
            for shard in shards
            if all(other.name != run.name for other in shard)
            and length([*shard, run], procs) <= SHARD_SECONDS
        )
        shard = next(fits, None)  # the first with room, of the most work so far
        if shard is None:
            shard = []
            shards.append(shard)
        shard.append(run)
    planned = []
    for shard in shards:
        ids = {run.name: run.ids for run in shard}
        told = {run.name: split * (max(len(run.beside), 1) - 0.5) for run in shard}
        where: dict[str, worker.Process] = {}
        for run in shard:
            where.update(placed(run, serial.get(run.name, ())))
        planned.append(Shard(length(shard, procs), ids, told, where))
    return sorted(planned, key=lambda shard: -shard.seconds)


def gather(
    workers: modal.Cls,
    resources: Resources,
    names: dict[str, list[str]],
    pytest_args: list[str],
    meter: progress.Meter,
) -> Found:
    """What these containers make of the files asked for: the tests of each that
    they can run, and how run_test.py runs them there."""
    on = resources.gpu or "a CPU container"
    log.debug("collecting %s on %s", count(len(names), "file"), on)
    files = collect(workers, list(names), pytest_args, meter)
    found = Found(failed=len(names) - len(files))  # those of the calls Modal lost
    left_out: Counter[str] = Counter()  # tests, by the tci constraint that fails
    skipped = 0  # files that run_test.py doesn't run in these containers
    for result in files:
        name, asked = result["name"], names[result["name"]]
        if not result["runs"]:
            log.debug("left out test/%s.py: run_test.py doesn't run it", name)
            skipped += 1
            continue
        took = result["seconds"], result["container"]
        log.debug("test/%s.py took %.1f s on container %s", name, *took)
        found.machines[json.dumps(result["machine"])] += 1
        found.serial[name] = set(result["serial"])
        if result["alone"]:
            found.alone.add(name)
        # A node id also asks for the tests under it and for its parametrizations
        under = tuple(f"{a}{mark}" for a in asked for mark in ("::", "["))
        # A test that can run here has None; one left out, its constraint
        verdicts = {**dict.fromkeys(result["ids"]), **result["left_out"]}
        for nodeid, unmet in verdicts.items():
            if nodeid not in asked and not nodeid.startswith(under):
                continue
            if unmet:
                log.debug("left out %s: %s", nodeid, unmet)
                left_out[unmet] += 1
            else:
                found.kept.setdefault(name, []).append(nodeid)
        if error := result["error"].rstrip():
            log.error("collecting test/%s.py failed:\n%s", name, error)
            found.failed += 1
    if not found.kept:  # otherwise plan says how many, with what it makes of them
        log.info("found 0 tests in %s", count(len(names) - found.failed, "file"))
    if skipped:
        some = count(skipped, "file")
        log.info("left out %s that run_test.py doesn't run on %s", some, on)
    if left_out:
        tests = count(left_out.total(), "test")
        log.info("left out %s that %s can't run:", tests, on)
        for unmet, n in left_out.most_common():
            log.info("  %d with %s", n, unmet)
    return found


def roomier(resources: Resources, room: int) -> tuple[str, int]:
    """The --cpu and --memory to ask for, so that a container runs more of the test
    processes its GPU has room for, and how many it then runs: the cores that fit
    the nearest to as many and, if the memory is short of them, the least that fits
    them. Memory that came with the cores, as it does by default, follows them."""
    fits = {cpu: int(cpu * PROCESSES_PER_CORE) for cpu in CPU_CHOICES}
    cpu = max(min(fits, key=lambda cpu: abs(fits[cpu] - room)), resources.cpu)
    procs = min(fits[cpu], room)
    flags = [f"--cpu {cpu}"] if cpu > resources.cpu else []
    memory = resources.memory
    if memory == MIB_PER_CORE * resources.cpu:
        memory = MIB_PER_CORE * cpu
    if memory // PROCESS_MIB < procs:
        memory = next(m for m in MEMORY_CHOICES if m // PROCESS_MIB >= procs)
        flags.append(f"--memory {memory}")
    return " ".join(flags), procs


def at_once(resources: Resources, machine: dict[str, Any]) -> int:
    """How many test processes a container runs at once: those that fit in its
    cores, in its memory and, with GPUs, in the memory of the first, which they all
    use. When that GPU has room for more of them, say what to ask for."""
    fit = {
        count(resources.cpu, "core"): int(resources.cpu * PROCESSES_PER_CORE),
        f"{resources.memory // 1024} GiB": resources.memory // PROCESS_MIB,
    }
    least = min(fit, key=fit.__getitem__)
    procs = max(fit[least], 1)
    said = [f"{n} in {what}" for what, n in fit.items()]
    if machine["gpus"] and machine["gpu_memory"]:
        room = int(machine["gpu_memory"] // PROCESS_GPU_GIB)
        said.append(f"{room} in a GPU's {machine['gpu_memory']:.0f} GiB")
        flags, more = roomier(resources, room)
        if more > procs:
            processes = f"{procs} test process{'' if procs == 1 else 'es'}"
            verb = "fits" if least == "1 core" else "fit"
            log.info("running %s at once: %s %s no more", processes, least, verb)
            gpu = gpu_name(resources.gpu or "")
            log.info("%s fits %d with %s", gpu, more, flags, extra=CONTINUED)
        procs = max(min(procs, room), 1)
    log.debug("test processes that fit: %s", ", ".join(said))
    return procs


def plan(found: Found, resources: Resources, rate: Decimal) -> tuple[list[Shard], int]:
    """The shards to run the tests in, and how many test processes a container
    runs at once, both for the machine most containers are. Say what the run comes
    to: its tests, its shards, about how long it takes and what it costs, at the
    rate a container bills an hour."""
    if len(found.machines) > 1:  # Modal may serve an H100 with an H200
        log.warning("containers differ: %s", ", ".join(found.machines))
    machine = json.loads(found.machines.most_common(1)[0][0])
    procs = at_once(resources, machine)
    # One process at a time is how run_test.py runs a file alone: then all are
    alone = [name for name in found.kept if procs == 1 or name in found.alone]
    facts = procs, len(alone)
    log.debug("test processes at once: %d, and %d of the files alone", *facts)
    estimates = timed(found.kept, machine)
    shards = sharded(estimates, alone, found.serial, procs, machine["split"])
    for number, shard in enumerate(shards, 1):
        tests = count(sum(map(len, shard.tests.values())), "test")
        sizes = tests, count(len(shard.tests), "file")
        log.debug("shard %d, about %.0f s: %s of %s", number, shard.seconds, *sizes)
    # By CI's times: the run takes as long as its longest shard, which is the first
    total = count(sum(map(len, found.kept.values())), "test")
    long = f"about {progress.span(shards[0].seconds)}"
    cost = rate * Decimal(sum(shard.seconds for shard in shards)) / 3600
    price = f"estimate {magenta(f'${cost:.2f}')}"
    log.debug("%s", dim(" • ").join((total, count(len(shards), "shard"), long, price)))
    return shards, procs


def run(
    workers: modal.Cls, told: dict[str, Any], state: progress.Running
) -> Iterator[worker.Ran | Exception]:
    """What the container of each of the state's shards returns, as shards finish. A
    call that Modal lost comes back as its exception. Each is told what the shards
    share: pytest's arguments, how many test processes to run at once, and with how
    many threads each. Meanwhile, show how far the shards are."""
    method = workers().run
    # Where each container says how far its shard's processes are, every few seconds
    with modal.Queue.ephemeral() as said:

        def ask() -> None:
            state.meter.told(method.get_current_stats().num_total_runners, "running")
            for word in said.get_many(QUEUE_ITEMS, block=False):
                state.report(*word)

        with progress.shown(state, ask):
            queue = said.object_id
            shards = enumerate(state.shards, 1)
            calls = [(n, p.tests, p.times, p.where, queue) for n, p in shards]
            options = {"kwargs": told, "return_exceptions": True}
            for result in method.starmap(calls, order_outputs=False, **options):
                state.back.append(result)
                yield result


def outcomes(reports: dict[str, str], sent: list[str]) -> dict[str, tuple[str, str]]:
    """Each test sent to a shard, with its outcome and what its last failing run
    said, from torchci's reports: the outcome of its last run, flaky if that one
    passed after one that failed, and not_run without any."""
    keys = {worker.identity(nodeid): nodeid for nodeid in sent}
    runs: dict[str, list[tuple[int, int, str, str]]] = {}
    for report in reports.values():
        for line in report.split("\n"):  # splitlines also breaks inside strings
            try:
                run = json.loads(line)
            except ValueError:  # empty, or torn as its process died
                continue
            key = run.get("file"), run.get("suite"), run.get("case_name")
            if run["type"] == "run" and key in keys:
                when = run["started_at"], run["rerun_number"]
                attempt = *when, run["outcome"], run["outcome_summary"]
                runs.setdefault(keys[key], []).append(attempt)
    verdicts = {}
    for nodeid in sent:
        attempts = sorted(runs.get(nodeid, []))
        failures = [said for *_, outcome, said in attempts if outcome in FAILED]
        outcome = attempts[-1][2] if attempts else "not_run"
        if outcome == "passed" and failures:
            outcome = "flaky"
        said = failures[-1] if failures else attempts[-1][3] if attempts else ""
        verdicts[nodeid] = outcome, said
    return verdicts


def beyond(used: dict[int, float], asked: float, unit: str, option: str) -> None:
    """Warn of the shards that used more than a container asks for, which Modal
    bills, and gives only while the machine has it to spare."""
    over = {number: most for number, most in used.items() if most > asked}
    if over:
        shards, worst = count(len(over), "shard"), max(over, key=over.__getitem__)
        log.warning("%s used more than a container's %g %s", shards, asked, unit)
        most = worst, over[worst], unit.partition(" ")[0], option
        log.warning("shard %d the most, %.1f %s: see %s", *most, extra=CONTINUED)


def execute(
    workers: modal.Cls,
    shards: list[Shard],
    told: dict[str, Any],
    directory: Path,
    meter: progress.Meter,
    resources: Resources,
    address: str,
) -> int:
    """Run the shards, logging each failure as its shard finishes, with every shard's
    output and reports kept in the directory. Print each test that failed or didn't
    run, a JSON object per line, and return how many. The summary ends the run: what
    it cost, how many tests it ran and for how long, how many ended in each state,
    the directory, and the address of the run's app on Modal. told is what run tells
    every shard's container, and resources are what a container asks for. Ctrl-C
    ends the run with the summary of the shards handled until then."""
    (directory / "reports").mkdir(parents=True)
    state = progress.Running(meter, shards)
    containers = set()
    cores: dict[int, float] = {}  # what each shard used, by its number
    memory: dict[int, float] = {}  # in GiB
    verdicts: dict[str, tuple[str, str]] = {}
    stopped: KeyboardInterrupt | None = None
    try:
        for ran in run(workers, told, state):
            if isinstance(ran, Exception):  # which doesn't say whose shard it was
                log.error("Modal lost a shard: %r", ran)
                continue
            number, code = ran["shard"], ran["code"]
            ended = number, ran["seconds"], ran["container"], code
            log.debug("shard %d took %.0f s on container %s: exit code %s", *ended)
            cores[number], memory[number] = ran["cores"], ran["memory"] / 1024
            used = number, ran["cores"], ran["memory"], ran["gpu_memory"]
            log.debug("shard %d used %.1f cores, %d MiB, and %d MiB of a GPU", *used)
            containers.add(ran["container"])
            output = directory / f"shard-{number}.log"
            output.write_bytes(gzip.decompress(ran["output"]))
            for name, report in ran["reports"].items():
                (directory / "reports" / name).write_text(report)
            sent = [i for ids in shards[number - 1].tests.values() for i in ids]
            found = outcomes(ran["reports"], sent)
            for nodeid, (outcome, said) in found.items():
                if outcome in FAILED:
                    # The run's first failure rings the bell: you may be elsewhere
                    ring = None if state.failed else RING
                    log.error("failed: %s", nodeid, extra=ring)
                    log.error("  %s", said.partition("\n")[0])
                    state.failed += 1
                elif outcome == "flaky":
                    log.warning("flaky: %s", nodeid)
            if missing := sum(outcome == "not_run" for outcome, _ in found.values()):
                gave_up = f"stopped after {worker.GIVE_UP_SECONDS // 60} min"
                how = gave_up if code is None else f"exited {code}"
                lost = number, how, count(missing, "test"), output
                log.error("shard %d %s: %s didn't run, see %s", *lost)
                state.failed += missing
            verdicts.update(found)  # last: a shard counts once all of this is done
    except KeyboardInterrupt as stop:  # the summary of the shards that count follows
        stopped = stop
    failing = 0
    for nodeid in (i for shard in shards for ids in shard.tests.values() for i in ids):
        if stopped and nodeid not in verdicts:  # its shard was still to come
            continue
        outcome, _ = verdicts.setdefault(nodeid, ("not_run", ""))
        if outcome in FAILED or outcome == "not_run":
            # Flushed: the summary is the last a run writes, wherever stdout goes
            print(json.dumps({"nodeid": nodeid, "outcome": outcome}), flush=True)
            failing += 1
    if stopped:
        total = sum(len(ids) for shard in shards for ids in shard.tests.values())
        left = f"{total - len(verdicts):,}", count(total, "test")
        log.warning("stopped by Ctrl-C: %s of %s did not run", *left)
    beyond(cores, resources.cpu, "cores", "--cpu")
    beyond(memory, resources.memory / 1024, "GiB of memory", "--memory")
    now, used = time.monotonic(), containers | set(state.started)
    log.debug("on %s: %s", meter.named(len(used)), meter.where())
    tests, took = count(len(verdicts), "test"), progress.span(now - meter.began)
    words = magenta(meter.cost(now)), dim("for"), bold(tests), dim("in"), took
    ring = None if stopped else RING  # the run's end rings the bell as well
    log.info("%s%s", " ".join(words), dim(":") if verdicts else "", extra=ring)
    tally = Counter(outcome for outcome, _ in verdicts.values())
    counts = {o: f"{n:,} {o.replace('_', ' ')}" for o, n in tally.items()}
    # Under it the states that need no look, then in red those that fail the run
    fine = [counts[o] for o in FINE if o in counts]
    bad = [red(counts[o].upper()) for o in (*FAILED, "not_run") if o in counts]
    for states in filter(None, (fine, bad)):
        log.info("%s", dim(" • ").join(states), extra=CONTINUED)
    for name, where in (("Logs", directory), ("Modal", address)):
        log.info("%s", dim(str(where)), extra={**CONTINUED, "margin": name})
    if stopped:
        raise stopped
    return failing


def run_tests(
    image: Build,
    commit: str,
    resources: Resources,
    names: dict[str, list[str]],
    pytest_args: list[str],
    ramjet_dir: Path,
    environ: Mapping[str, str],
    listing: bool,
    rate: Decimal,
    built: Decimal,
) -> int:
    """Run the tests asked for that these containers can run, or print each with its
    shard, one JSON object per line. Return the exit code: 1 if a test failed or
    didn't run, or if a file's tests couldn't be collected. environ is your
    environment, of which the containers get the variables that PASSED names. rate
    is what a container bills an hour, and built the dollars this run's build billed,
    where its price starts."""
    idle = SCALEDOWN_SECONDS if listing else HANDOVER_SECONDS
    env = {name: environ[name] for name in sorted(environ) if name.startswith(PASSED)}
    if env:
        pairs = (f"{name}={shlex.quote(value)}" for name, value in env.items())
        log.info("passing %s", " ".join(pairs))
    with fleet(image, commit, resources, ramjet_dir, idle, env) as (workers, address):
        meter = progress.Meter(resources.gpu, rate, built)
        found = gather(workers, resources, names, pytest_args, meter)
        failed = found.failed
        if not found.kept:
            return 1 if failed else 0
        shards, procs = plan(found, resources, rate)
        if listing:
            shard_of: dict[str, int] = {}
            for number, shard in enumerate(shards, 1):
                for ids in shard.tests.values():
                    shard_of.update(dict.fromkeys(ids, number))
            for ids in found.kept.values():
                for nodeid in ids:
                    print(json.dumps({"nodeid": nodeid, "shard": shard_of[nodeid]}))
        else:
            directory = ramjet_dir / time.strftime("test-%Y%m%d-%H%M%S")
            # The cores that the processes at once leave each, for its threads
            threads = max(resources.cpu // procs, THREADS)
            told = {"args": pytest_args, "procs": procs, "threads": threads}
            work = (workers, shards, told, directory)
            failed += execute(*work, meter, resources, address)
    return 1 if failed else 0
