"""What ramjet test shows while its containers collect and run. On a terminal it is a
block under the log's lines, drawn anew in place: a row of facts and, while tests
run, a row as wide as the terminal for each test process. Elsewhere it is a line of
the log when there is news. Both show what a Collecting or a Running knows, which
test.py keeps current."""

import contextlib
import os
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from typing import Protocol, TextIO

from tools.ramjet.common import count
from tools.ramjet.logger import (
    bold,
    dim,
    green,
    log,
    magenta,
    MARGIN,
    red,
    terminal,
    through,
)
from tools.ramjet.worker import Process, Ran, Test


COLUMNS = 80  # what the lines keep to, as the rest of the log does
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"  # beside the time: it turns while containers work
FRAME_SECONDS = 0.09  # between two drawings of the block
POLL_SECONDS = 2  # between two questions to Modal
ROWS = 8  # of test processes at most, and fewer in a short terminal
TIP_SECONDS = 180  # without a container, before the block says what you can do
# Where the log is not on a terminal: a line after QUIET_SECONDS without one, and no
# two within APART_SECONDS
QUIET_SECONDS = 300
APART_SECONDS = 15
# Modal's GPUs that PyTorch's CI tests on as well, for the tip to name
CI_GPUS = ("L4", "A10", "T4")
# What a run's seconds go to: "waiting" while there is no container
DOING = ("waiting", "collecting", "running")
# The facts of the block's first row that give way where it is too narrow for all,
# the first first: how far the files or the tests are, which the count says too
GIVE_WAY = ("share", "cost", "took", "fleet")
# What a terminal takes (ECMA-48, and xterm's control sequences)
UP = "\r\033[{}A"  # to the start of the row so many rows up
END = "\033[K\n"  # erase the rest of the row, and go to the next
BELOW = "\033[J"  # erase from here down
WRAP = "\033[?7{}"  # h: a row too long goes on in the next, l: it is cut
CURSOR = "\033[?25{}"  # h: shown, l: hidden
TITLE = "\033]2;{}\a"  # name the window
TITLES = "\033[{};2t"  # 22: keep the window's name, 23: bring it back

Style = Callable[[str], str]
Row = list[tuple[str, Style | None]]  # a row's text, in parts that may have a style


def span(seconds: float) -> str:
    """A length of time in its largest units: 40 s, 6 min, 1 h 12 min."""
    if round(seconds) < 60:
        return f"{max(round(seconds), 1)} s"
    hours, minutes = divmod(round(seconds / 60), 60)
    if not hours:
        return f"{minutes} min"
    return f"{hours} h {minutes} min" if minutes else f"{hours} h"


def clock(seconds: float) -> str:
    """Minutes and seconds, as a timer shows them: 2:29."""
    return "{}:{:02}".format(*divmod(int(seconds), 60))


def paint(row: Row) -> str:
    return "".join(style(text) if style else text for text, style in row)


def fitted(facts: dict[str, Row], width: int) -> Row:
    """The facts, each by its name, in one row, so wide at most: those in GIVE_WAY
    go, in that order, until the rest fit. A bullet parts them, but for what the
    containers are doing: the fact after it goes on from that word."""
    for gone in (None, *GIVE_WAY):
        facts = {name: fact for name, fact in facts.items() if name != gone}
        row: Row = []
        for name, fact in facts.items():
            row += [*fact, (" ", None) if name == "doing" else (" • ", dim)]
        row = row[:-1]  # without the last fact's
        if sum(len(text) for text, _ in row) <= width:
            break
    return row


@dataclass
class Meter:
    """A run's containers: how many Modal says there are, what they and the run's
    build have billed, and where the run's seconds went."""

    gpu: str | None  # each container's, as --gpu names them
    rate: Decimal  # dollars a container bills an hour
    built: Decimal = Decimal(0)  # dollars the run's build billed, if it built
    began: float = field(default_factory=time.monotonic)
    asked: float = field(default_factory=time.monotonic)  # when Modal last said
    containers: int = 0
    billed: float = 0.0  # container-seconds
    spent: dict[str, float] = field(default_factory=lambda: dict.fromkeys(DOING, 0.0))

    def told(self, containers: int, doing: str) -> None:
        """Modal says how many containers there are now: until now there were as many
        as it said last, doing this."""
        now = time.monotonic()
        seconds, self.asked = now - self.asked, now
        self.billed += self.containers * seconds
        self.spent[doing if self.containers else "waiting"] += seconds
        self.containers = containers

    def cost(self, now: float) -> str:
        """What the run has billed so far, to the cent: its build and its containers."""
        seconds = self.billed + self.containers * (now - self.asked)
        return f"${self.built + self.rate * Decimal(seconds) / 3600:.2f}"

    def named(self, containers: int) -> str:
        """So many containers, by their GPUs if they have any: 2 L4, no L4."""
        if not containers:
            return f"no {self.gpu or 'container'}"
        if self.gpu is None:
            return count(containers, "container")
        return f"{containers} {self.gpu}"

    def of(self, wanted: int) -> Row:
        """The containers there are, of those the run can use now: 7/141 L4. One
        with nothing left to take is about to stop, and isn't counted."""
        most = wanted or self.containers
        if not most:
            return [(self.named(0), None)]
        have = f"{min(self.containers, most):,}"
        unit = self.gpu or count(most, "container").partition(" ")[2]
        return [(f"{have}/{wanted:,}" if wanted else have, None), (f" {unit}", dim)]

    def where(self) -> str:
        """Where the run's seconds went: 2 s with no L4 • 16 s collecting • 1 min
        running."""
        names = {"waiting": f"with {self.named(0)}"}
        said = [
            f"{span(seconds)} {names.get(doing, doing)}"
            for doing, seconds in self.spent.items()
            if seconds
        ]
        return " • ".join(said)


def under_way(
    meter: Meter,
    doing: str,
    far: tuple[int, int],
    unit: str,
    wanted: int,
    since: float,
    now: float,
    failed: int = 0,
) -> dict[str, Row]:
    """What to say of work under way, each fact by the name fitted knows it by: what
    the containers are doing, how far it is, how many of its files or tests are
    done, how many tests failed if any did, the containers there are of those it
    can use now, how long it has taken and what the run has cost. Collecting and
    testing say the same of themselves, in this one shape."""
    done, total = far
    said: dict[str, Row] = {"doing": [(doing, None)]}
    said["share"] = [(f"{100 * done // total}%", bold)]
    said["count"] = [(f"{done:,}", None), (f"/{count(total, unit)}", dim)]
    if failed:
        said["failed"] = [(f"{failed:,} FAILED", red)]
    said["fleet"] = meter.of(wanted)
    said["took"] = [(clock(now - since), None)]
    return {**said, "cost": [(meter.cost(now), magenta)]}


@dataclass
class Collecting:
    """What ramjet knows while test files are collected."""

    meter: Meter
    files: int
    calls: int = 1  # those that aren't back: a container collects the files of each
    back: set[str] = field(default_factory=set)  # the files that are collected

    def working(self, now: float) -> bool:
        return self.meter.containers > 0

    def facts(self, now: float) -> dict[str, Row]:
        """What to say of the files, each fact by the name fitted knows it by: what
        under_way says of collecting them, since the wait for a container began. Or,
        while Modal has given none, for how long."""
        meter, back = self.meter, len(self.back)
        if not meter.containers and not back:
            state: Row = [(f"{meter.named(0)} yet", None)]
            waited: Row = [(clock(now - meter.began), None)]
            bill: Row = [("nothing billed", dim)]
            # By the run's build, or by containers that came and went
            if meter.billed or meter.built:
                bill = [(meter.cost(now), magenta)]
            return {"state": state, "waited": waited, "cost": bill}
        far = back, self.files
        return under_way(meter, "collecting", far, "file", self.calls, meter.began, now)

    def rows(self, now: float, width: int) -> list[Row]:
        """The block, so wide: its facts and, after a long wait that has cost
        nothing, what you can do about it."""
        meter = self.meter
        rows = [fitted(self.facts(now), width)]
        idle = not meter.containers and not self.back and not meter.billed
        if idle and now - meter.began >= TIP_SECONDS:
            others = [gpu for gpu in CI_GPUS if gpu != meter.gpu]
            tip = f"try --gpu {' or '.join(others)}: " if meter.gpu in CI_GPUS else ""
            rows.append([(f"{tip}Ctrl-C is free while nothing runs", dim)])
        return rows

    def news(self) -> tuple[int, int]:
        """What a new line is for when it grows: the containers, and the tenths of
        the files that are back."""
        return self.meter.containers, 10 * len(self.back) // self.files

    def title(self) -> str:
        if not self.meter.containers and not self.back:
            return f"ramjet: {self.meter.named(0)} yet"
        return "ramjet: collecting"


class Planned(Protocol):
    """A shard, as test.py cuts it."""

    seconds: float
    tests: dict[str, list[str]]
    times: dict[str, float]
    where: dict[str, Process]


@dataclass(frozen=True)
class Flight:
    """A container on a shard, as it last said."""

    shard: int  # its number, or 0 once another container started it
    since: float  # when the container said it started
    done: Mapping[Process, int]  # how many tests of each test process have ended
    # The processes that run: the test each runs or ran last, if it has had one,
    # and since when it has been heard of
    running: Mapping[Process, Test | None]
    heard: Mapping[Process, float]


@dataclass
class Running:
    """What ramjet knows while shards run."""

    meter: Meter
    shards: Sequence[Planned]  # a shard's number is its place here, from 1
    back: list[Ran | Exception] = field(default_factory=list)  # as they come
    # By Modal's container id: a container's number is its place here, from 1
    started: dict[str, Flight] = field(default_factory=dict)
    failed: int = 0  # tests of the shards that are back
    began: float = field(default_factory=time.monotonic)  # when the shards went out
    sizes: list[int] = field(init=False)  # how many tests each shard has
    # How many of them each of the shard's test processes runs
    shares: list[Counter[Process]] = field(init=False)

    def __post_init__(self) -> None:
        self.sizes = [sum(map(len, shard.tests.values())) for shard in self.shards]
        self.shares = [Counter(shard.where.values()) for shard in self.shards]

    def report(
        self,
        shard: int,
        container: str,
        done: Mapping[Process, int],
        running: Mapping[Process, Test | None],
    ) -> None:
        """A container says how far the test processes of a shard are, which no
        other container runs any more."""
        now, last = time.monotonic(), self.started.get(container)
        if last is None or last.shard != shard:  # it starts the shard
            last = Flight(shard, now, {}, {}, {})
        heard = {process: last.heard.get(process, now) for process in running}
        moved = {
            other: replace(flight, shard=0)
            for other, flight in self.started.items()
            if flight.shard == shard and other != container
        }
        mine = {container: Flight(shard, last.since, done, running, heard)}
        self.started = {**self.started, **moved, **mine}  # anew: other threads read

    def returned(self) -> dict[int, float]:
        """The seconds each shard that is back took, by its number."""
        return {
            ran["shard"]: ran["seconds"]
            for ran in self.back
            if not isinstance(ran, Exception)
        }

    def flying(self) -> list[Flight]:
        """The containers on a shard that isn't back, the longest on its shard
        first."""
        back = self.returned()
        on = [f for f in self.started.values() if f.shard and f.shard not in back]
        return sorted(on, key=lambda flight: flight.since)

    def working(self, now: float) -> bool:
        return bool(self.flying())

    def tests(self) -> tuple[int, int]:
        """How many tests have ended: those of the shards that are back, and those
        the containers say have on the shards they run. And how many there are."""
        back = sum(self.sizes[shard - 1] for shard in self.returned())
        flying = sum(
            min(done, self.shares[flight.shard - 1][process])
            for flight in self.flying()
            for process, done in flight.done.items()
        )
        return back + flying, sum(self.sizes)

    def facts(self, now: float) -> dict[str, Row]:
        """What to say of the tests, each fact by the name fitted knows it by: what
        under_way says of running them, since the shards went out, with how many
        failed in the shards that are back."""
        meter, far, since = self.meter, self.tests(), self.began
        # A shard runs in a container of its own
        wanted = len(self.shards) - len(self.back)
        return under_way(meter, "testing", far, "test", wanted, since, now, self.failed)

    def rows(self, now: float, width: int) -> list[Row]:
        """The block's facts, so wide."""
        return [fitted(self.facts(now), width)]

    def wide(self, now: float, most: int, width: int) -> list[Row]:
        """A row as wide as the terminal for each test process that runs and has
        tests left, up to the most and the longest on its file first: its container,
        how long it has been on its file, how many of its tests have ended, and the
        test it runs, or ran last. A container is named as the facts name it, or CPU
        without a GPU, with its number: A10 3. A name that doesn't fit is cut. The
        processes past the most are counted."""
        back, running = self.returned(), []
        for number, flight in enumerate(self.started.values(), 1):
            if not flight.shard or flight.shard in back:
                continue
            shares = self.shares[flight.shard - 1]
            for process, test in flight.running.items():
                done, tests = flight.done.get(process, 0), shares[process]
                if done < tests:
                    since = flight.heard[process]
                    running.append((since, number, done, tests, process, test))
        running.sort(key=lambda on: on[0])
        shown = running[:most]
        if not shown:
            return []
        said = [(str(c), clock(now - t), str(d), str(n)) for t, c, d, n, *_ in shown]
        # As wide in every row, so that the names start in one column. The tests
        # that have ended get the width of the tests there are, which they grow to
        digits, on, _, size = (max(map(len, column)) for column in zip(*said))
        runner = self.meter.gpu or "CPU"
        rows: list[Row] = []
        for (*_, process, test), (number, timer, ended, tests) in zip(shown, said):
            # torchci's file is a path: test/test_nn.py. Before the process's first
            # test, its file alone
            file, suite, case = test or (f"{process[0]}.py", "", "")
            name = "::".join(filter(None, (Path(file).name, suite, case)))
            where, start = f" {runner} {number:>{digits}} ", f" {timer:>{on}} "
            progress = f"[{ended:>{size}}/{tests:>{size}}] "
            room = width - len(where + start + progress) - 1  # a space at its end
            if len(name) > room:
                name = f"{name[: max(room - 1, 0)]}…"
            rows.append([(where, dim), (start, None), (progress, dim), (name, None)])
        if len(shown) < len(running):
            # It starts in the column the rows' containers do
            more = f" … and {len(running) - len(shown):,} more running"
            rows.append([(more, dim)])
        return rows

    def news(self) -> tuple[int, int]:
        """What a new line is for when it grows: the containers, and the tenths of
        the tests that have ended."""
        ended, tests = self.tests()
        return self.meter.containers, 10 * ended // tests

    def title(self) -> str:
        ended, tests = self.tests()
        failed = f" • {self.failed:,} failed" if self.failed else ""
        return f"ramjet {100 * ended // tests}%{failed}"


def drawn(
    state: Collecting | Running, now: float, turn: int, most: int, width: int
) -> list[str]:
    """The block's rows, as a terminal so wide takes them. The first starts as the
    log's lines do, with a mark in the level's place that turns while containers
    work, and goes on with the facts, to a space short of the terminal's end at
    most. There the rows of the test processes end, at most so many."""
    first, *rest = state.rows(now, max(width - MARGIN - 1, 1))
    mark = green(SPINNER[turn % len(SPINNER)]) if state.working(now) else dim("·")
    start = dim(f"ramjet {time.strftime('%H:%M:%S')}".ljust(MARGIN - 2))
    under = [" " * MARGIN + paint(row) for row in rest]
    if isinstance(state, Running):
        under += [paint(row) for row in state.wide(now, most, width)]
    return [f"{start}{mark} {paint(first)}", *under]


def line(state: Collecting | Running, now: float) -> str:
    """The block's facts as a line of the log, which starts two columns in: those
    that keep it within COLUMNS."""
    row = fitted(state.facts(now), COLUMNS - MARGIN - 2)
    return "".join(text for text, _ in row)


class Block:
    """Rows of a terminal that stay under the log's lines: a line written through the
    block goes above them."""

    def __init__(self, terminal: TextIO) -> None:
        self.terminal = terminal
        self.rows: list[str] = []
        self.title = ""
        self.lock = threading.Lock()

    def __enter__(self) -> "Block":
        self.terminal.write(CURSOR.format("l") + TITLES.format(22))
        return self

    def __exit__(self, *error: object) -> None:
        self.draw([], self.title)
        self.terminal.write(TITLES.format(23) + CURSOR.format("h"))
        self.terminal.flush()

    def room(self) -> tuple[int, int]:
        """How many rows of test processes fit, as the block takes half the terminal
        at most, and how wide a row can be."""
        try:
            columns, lines = os.get_terminal_size(self.terminal.fileno())
        except OSError:
            columns, lines = COLUMNS, 24
        return max(min(lines // 2 - 2, ROWS), 0), columns

    def draw(self, rows: list[str], title: str, above: str = "") -> None:
        """Draw the rows where the last ones are, under the text if there is any,
        and name the window."""
        with self.lock:
            if (rows, title, above) == (self.rows, self.title, ""):
                return
            up = UP.format(len(self.rows)) if self.rows else ""
            named = TITLE.format(title) if title != self.title else ""
            self.rows, self.title = rows, title
            cut = WRAP.format("l") + "".join(row + END for row in rows)
            text = up + named + above.replace("\n", END) + cut + WRAP.format("h")
            self.terminal.write(text + BELOW)
            self.terminal.flush()

    def write(self, text: str) -> None:
        self.draw(self.rows, self.title, text)

    def flush(self) -> None:
        self.terminal.flush()


@contextlib.contextmanager
def shown(state: Collecting | Running, ask: Callable[[], None]) -> Iterator[None]:
    """Until the block ends, ask Modal every POLL_SECONDS, which keeps the state
    current, and show the state: as a block on the log's terminal or, without one, as
    a line of the log when there is news."""
    done = threading.Event()
    screen = terminal()

    def poll() -> None:
        said, seen = time.monotonic(), (0, 0)
        while not done.is_set():
            done.wait(POLL_SECONDS)  # and once more as the block ends
            try:
                ask()
            except Exception as error:  # Modal didn't say: it may the next time
                log.debug("no word from Modal: %r", error)
            quiet, news = time.monotonic() - said, state.news()
            more = any(new > old for new, old in zip(news, seen))
            if screen or done.is_set() or quiet < APART_SECONDS:
                continue
            if more or quiet >= QUIET_SECONDS:
                log.info("  %s", line(state, time.monotonic()))
                said, seen = time.monotonic(), news

    def draw(block: Block) -> None:
        turn = 0
        while not done.wait(FRAME_SECONDS):
            rows = drawn(state, time.monotonic(), turn, *block.room())
            block.draw(rows, state.title())
            turn += 1

    threads = [threading.Thread(target=poll, name="ramjet-progress", daemon=True)]
    with contextlib.ExitStack() as stack:
        if screen:
            block = stack.enter_context(Block(screen))
            stack.enter_context(through(block))
            threads.append(threading.Thread(target=draw, args=(block,), daemon=True))
        for thread in threads:
            thread.start()
        try:
            yield
        finally:
            done.set()
            for thread in threads:
                thread.join()
