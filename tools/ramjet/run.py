"""Runners: a sandbox with the GPUs, cores and memory you ask for, booted from the
build image. One stays warm per size, so the next run skips the start and gets only
the files that changed."""

import json
import re
import sys
import tempfile
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import NamedTuple

import modal
from modal.stream_type import StreamType

from tools.ramjet import sync
from tools.ramjet.lineage import Build
from tools.ramjet.logger import log, red


# A warm runner stops after this many idle seconds, or once idling has cost
# IDLE_BUDGET dollars, whichever comes first: the budget binds above $10/h
IDLE_MAX_SECONDS = 15 * 60
IDLE_BUDGET = Decimal("2.50")
# Modal's largest sandbox, and the longest one lives
MAX_CORES = 64
MAX_GIB = 336
MAX_SECONDS = 24 * 3600
# A runner's resources unless --cpu and --memory say otherwise: CORES_PER_GPU per
# GPU, or as many without one, with MIB_PER_CORE of memory each
CORES_PER_GPU = 8
MIB_PER_CORE = 8192
# --cpu's choices in cores, and --memory's in MiB, up to Modal's largest sandbox
CPU_CHOICES = [1, 2, 4, 8, 16, 32, 64]
MEMORY_CHOICES = [8192, 16384, 32768, 65536, 131072, 262144, 344064]
PID_FILE = "/tmp/ramjet.pid"  # in the runner: the command's process group leader


class GPU(NamedTuple):
    capability: tuple[int, int]  # which Modal's API doesn't report
    max_count: int  # per sandbox
    rate_name: str  # in Modal's rate card


# Modal's GPUs, by the names Modal takes (https://modal.com/docs/guide/gpu). Its
# guide gives no limit for RTX-PRO-6000, which gets the 8 most others have
GPUS = {
    "T4": GPU((7, 5), 8, "t4"),
    "L4": GPU((8, 9), 8, "l4"),
    "A10": GPU((8, 6), 4, "a10g"),
    "L40S": GPU((8, 9), 8, "l40s"),
    "A100": GPU((8, 0), 8, "a100_40gb"),
    "A100-40GB": GPU((8, 0), 8, "a100_40gb"),
    "A100-80GB": GPU((8, 0), 8, "a100_80gb"),
    "RTX-PRO-6000": GPU((12, 0), 8, "rtx6000"),
    "H100": GPU((9, 0), 8, "h100"),
    "H100!": GPU((9, 0), 8, "h100"),  # never upgraded to an H200
    "H200": GPU((9, 0), 8, "h200"),
    "B200": GPU((10, 0), 8, "b200"),
    "B200+": GPU((10, 0), 8, "b200"),  # a B200 or a B300, billed as a B200
    "B300": GPU((10, 3), 8, "b300"),
}
# --gpu's choices: a GPU, with :N for N of them, as Modal takes them
GPU_CHOICES = [
    name if count == 1 else f"{name}:{count}"
    for name, gpu in GPUS.items()
    for count in range(1, gpu.max_count + 1)
]


class Resources(NamedTuple):
    """A runner's, as modal.Sandbox.create takes them."""

    gpu: str | None
    cpu: int  # physical cores
    memory: int  # MiB


def gpu_name(gpu: str) -> str:
    """The GPU in a request such as H100!:2."""
    return gpu.partition(":")[0]


def gpu_count(gpu: str | None) -> int:
    return 0 if gpu is None else int(gpu.partition(":")[2] or 1)


def resources(gpu: str | None, cpu: int | None, memory: int | None) -> Resources:
    """The runner's resources from --gpu, --cpu and --memory, the last two by
    default sized by the GPUs, within Modal's largest sandbox."""
    cpu = cpu or min(CORES_PER_GPU * max(gpu_count(gpu), 1), MAX_CORES)
    memory = memory or min(MIB_PER_CORE * cpu, MAX_GIB * 1024)
    return Resources(gpu, cpu, memory)


def runs_on(capability: tuple[int, int], archs: str) -> bool:
    """Whether a build for these TORCH_CUDA_ARCH_LIST archs runs on a GPU: a cubin
    runs on its major version from its minor up, and PTX on anything from its
    version up. An arch given by name, such as Hopper, passes unjudged."""
    for arch in archs.replace(";", " ").split():
        match = re.fullmatch(r"(\d+)\.(\d+)[af]?(\+PTX)?", arch)
        if match is None:
            return True
        built = (int(match[1]), int(match[2]))
        if built[0] == capability[0] and built[1] <= capability[1]:
            return True
        if match[3] and built <= capability:
            return True
    return False


def check_archs(gpu: str | None, archs: str) -> None:
    """Exit, before any build, if the build can't run on the GPU."""
    if gpu is None:
        return
    capability = GPUS[gpu_name(gpu)].capability
    if not runs_on(capability, archs):
        sm = "".join(map(str, capability))
        log.error("a build for %s can't run on %s, sm_%s", archs, gpu, sm)
        log.error("pick another --build-cuda or TORCH_CUDA_ARCH_LIST")
        sys.exit(1)


def hourly_rate(resources: Resources, rates: Mapping[str, Decimal]) -> Decimal:
    """What the runner bills per hour while idle: its GPUs plus the CPU and memory
    it requested."""
    rate = resources.cpu * rates["cpu_hour_cost_sandbox"]
    rate += Decimal(resources.memory) / 1024 * rates["mem_gib_hour_cost_sandbox"]
    if resources.gpu is not None:
        gpu = GPUS[gpu_name(resources.gpu)]
        name = f"gpu_hour_cost_{gpu.rate_name}"
        if name not in rates:
            log.error("Modal's rate card has no GPU named %s", gpu.rate_name)
            sys.exit(1)
        rate += gpu_count(resources.gpu) * rates[name]
    return rate


def idle_timeout(resources: Resources, rates: Mapping[str, Decimal]) -> int:
    affordable = int(IDLE_BUDGET / hourly_rate(resources, rates) * 3600)
    return min(affordable, IDLE_MAX_SECONDS)


def start(
    app: modal.App, image: Build, resources: Resources, idle: int
) -> modal.Sandbox:
    gpu, cpu, memory = resources
    sandbox = modal.Sandbox.create(
        app=app,
        image=modal.Image.from_id(image.image_id),
        gpu=gpu,
        cpu=cpu,
        memory=memory,
        idle_timeout=idle,
        timeout=MAX_SECONDS,
        workdir="/pytorch",
    )
    log.debug("runner: sandbox %s", sandbox.object_id)
    return sandbox


def warm(
    app: modal.App, image: Build, resources: Resources, idle: int, saved: Path
) -> tuple[modal.Sandbox, str, bool]:
    """The warm runner noted in saved if it is up on the same build image, else a
    new one, with the commit its /pytorch holds and whether it was warm."""
    if (state := noted(saved)) is not None:
        try:
            sandbox = modal.Sandbox.from_id(state["sandbox"])
        except modal.exception.NotFoundError:
            pass
        else:
            if sandbox.poll() is None:
                if state["image"] == image.image_id:
                    return sandbox, state["commit"], True
                log.info("warm sandbox %s has an older build", sandbox.object_id)
            sandbox.terminate()  # on an older build image, or its idle window closed
    return start(app, image, resources, idle), image.commit, False


def noted(saved: Path) -> dict[str, str] | None:
    """What remember noted about the warm runner, if anything."""
    return json.loads(saved.read_text()) if saved.exists() else None


def show(sandbox_id: str, reused: bool, resources: Resources, rate: Decimal) -> None:
    """Log the runner: its id, whether it is the warm one, and what it costs and
    has."""
    gpu = "no GPU"
    if resources.gpu is not None:
        name, count = gpu_name(resources.gpu), gpu_count(resources.gpu)
        gpu = name if count == 1 else f"{name} × {count}"
    cores, memory = f"{resources.cpu} cores", f"{resources.memory // 1024} GiB"
    facts = " • ".join([red(f"${rate:.2f}/h"), gpu, cores, memory])
    log.info("sandbox %s (%s): %s", sandbox_id, "warm" if reused else "new", facts)


def remember(saved: Path, runner: modal.Sandbox, image: Build, commit: str) -> None:
    """Note the warm runner, its build image and the commit its /pytorch holds."""
    state = {"sandbox": runner.object_id, "image": image.image_id, "commit": commit}
    saved.write_text(json.dumps(state))


def checked(sandbox: modal.Sandbox, *args: str) -> None:
    process = sandbox.exec(*args, workdir="/pytorch")
    if process.wait() != 0:
        log.error("runner: %s failed: %s", args[0], process.stderr.read().strip())
        sys.exit(1)


def sync_files(
    sandbox: modal.Sandbox, synced: str, commit: str, ramjet_dir: Path
) -> None:
    """Bring the runner's /pytorch from the commit it holds to this one: rm for the
    files deleted, and a tar of those changed. No git runs in the runner."""
    changes = sync.diff(synced, commit)
    # Deletions first, in case a directory takes a file's place
    if deleted := [path for path, status in changes.items() if status == "D"]:
        checked(sandbox, "rm", "-f", "--", *deleted)
    if changed := [path for path, status in changes.items() if status != "D"]:
        with tempfile.NamedTemporaryFile(dir=ramjet_dir, suffix=".tar") as tar:
            sync.archive(commit, changed, Path(tar.name))
            sandbox.filesystem.copy_from_local(tar.name, tar.name)  # same path
            checked(sandbox, "tar", "-xf", tar.name)


def execute(sandbox: modal.Sandbox, command: list[str], cwd: str) -> int:
    """Run the command, as given, in the runner's /pytorch/<cwd>, and return its
    exit code."""
    # In its own process group, whose leader notes its pid for Ctrl-C before it
    # becomes the command; -w waits for the command if setsid has to fork
    leader = f'echo $$ > {PID_FILE} && exec "$@"'
    args = ["setsid", "-w", "bash", "-c", leader, "ramjet", *command]
    workdir, output = f"/pytorch/{cwd}", StreamType.STDOUT  # to your terminal
    process = sandbox.exec(*args, workdir=workdir, stdout=output, stderr=output)
    try:
        return process.wait()
    except KeyboardInterrupt:
        # Like a terminal's Ctrl-C, interrupt the command's process group, which
        # takes the processes it started too; a warm runner stays up for the next run
        sandbox.exec("bash", "-c", f'kill -INT -- -"$(cat {PID_FILE})"').wait()
        return process.wait()
