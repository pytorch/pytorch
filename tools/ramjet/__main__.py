"""Build this worktree on Modal and run a command on a runner you size:

    python -m tools.ramjet [OPTIONS] -- COMMAND [ARGS...]

As in modal shell, --gpu, such as H100 or B200:8, --cpu and --memory size the runner;
without --gpu it has no GPU. A 64-core builder keeps an incremental build of your
worktree, and the runner stays warm between runs.
"""

import argparse
import fcntl
import os
import signal
import sys
import time
from pathlib import Path

import modal

from tools.ramjet import build, common, lineage, run, sync
from tools.ramjet.logger import LEVEL_NAMES, log, setup_logger


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """ramjet's options, and the command after the first --, as given."""
    split = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(
        prog="python -m tools.ramjet",
        usage="%(prog)s [OPTIONS] -- COMMAND [ARGS...]",
    )
    # Named and given as modal shell takes them
    runner_group = parser.add_argument_group("runner (the sandbox COMMAND runs in)")
    runner_group.add_argument(
        "--gpu",
        type=str.upper,  # as Modal does
        choices=run.GPU_CHOICES,
        help="GPUs as Modal names them, with :N for N of them (default: none)",
    )
    cores = f"{run.CORES_PER_GPU} per GPU, or {run.CORES_PER_GPU} without one"
    runner_group.add_argument(
        "--cpu",
        type=int,
        choices=run.CPU_CHOICES,
        help=f"physical cores (default: {cores})",
    )
    runner_group.add_argument(
        "--memory",
        type=int,
        choices=run.MEMORY_CHOICES,
        help=f"memory in MiB (default: {run.MIB_PER_CORE} per core)",
    )
    build_group = parser.add_argument_group("build")
    build_group.add_argument(
        "--build-cuda",
        choices=build.ci_images(),
        default=build.stable_cuda(),
        help="the CUDA version to build with (default: %(default)s)",
    )
    parser.add_argument(
        "--python-version",
        choices=build.python_versions(),
        help="the Python version of the builder and the runner (default: CI's image's)",
    )
    parser.add_argument(
        "--log-level",
        choices=LEVEL_NAMES,
        default="info",
        help="the least severe messages to show (default: %(default)s)",
    )
    options, command = parser.parse_args(argv[:split]), argv[split + 1 :]
    if not command:
        parser.error("the command to run goes after --")
    return options, command


def image_for(
    app: modal.App,
    commit: str,
    ramjet_dir: Path,
    cuda: str,
    python: str | None,
    config: dict[str, str],
) -> lineage.Build:
    """The build image to run the commit on, built with config. Python None is
    CI's image's own."""
    ci_image = build.ci_images()[cuda]
    python = python or ci_image.python
    key = lineage.key(build.toolchain_id(ci_image, python), config)
    state, inputs = ramjet_dir / "lineage.json", ramjet_dir / f"{key}.inputs"
    builds = lineage.load(state)
    now = time.time()
    last = lineage.usable(builds.get(key), now) if inputs.exists() else None
    head = common.git("rev-parse", f"{commit}^")  # the snapshot's parent
    log.debug("snapshot %.7s on HEAD %.7s", commit, head)
    if last is None:
        why = "no build image for this build config"
        if key in builds and inputs.exists():
            why = "last build image expired"
        log.warning("building cold pytorch: %s", why)
        since = sync.merge_base(commit)
        # CI publishes its images for main's .ci/docker, not for your edits to it
        start = build.toolchain(since, ci_image, python)
    else:
        changes = sync.diff(last.commit, commit)
        build_inputs = frozenset(inputs.read_text().splitlines())
        used, due = lineage.read_by_build(changes, build_inputs), lineage.due(last, now)
        n = len(changes)
        if used:
            more = f" and {len(used) - 1} more" if len(used) > 1 else ""
            reason = f"build affected by {used[0]}{more}"
        elif due:
            days = int((now - last.taken) // (24 * 3600))
            reason = f"renewing {days}-day-old image before it expires"
        elif n:
            reason = f"build not affected by {n} changed file{'' if n == 1 else 's'}"
        else:
            reason = "no changes since last build"
        building = bool(used) or due
        log.info("%s pytorch: %s", "building" if building else "not building", reason)
        if not building:
            return last
        start, since = modal.Image.from_id(last.image_id), last.commit
    bundle = sync.bundle(commit, since, ramjet_dir)
    image_id = build.build(app, start, bundle, config, inputs, clone=last is None)
    builds[key] = lineage.Build(image_id, commit, time.time())
    lineage.save(state, builds)
    log.debug("saved build image %s", image_id)
    return builds[key]


def main() -> int:
    setup_logger()  # first: reading the options can fail
    options, command = parse_args(sys.argv[1:])
    log.setLevel(LEVEL_NAMES[options.log_level])
    cuda = options.build_cuda
    resources = run.resources(options.gpu, options.cpu, options.memory)
    config = {**build.default_config(cuda), **lineage.build_config(os.environ)}
    # Before any build, so a GPU missing from Modal's rate card, or one the build
    # can't run on, fails fast
    rates = modal.Workspace.from_context().billing.rates()
    idle = run.idle_timeout(resources, rates)
    run.check_archs(resources.gpu, config["TORCH_CUDA_ARCH_LIST"])
    root = common.repo_root()
    cwd = Path.cwd().resolve().relative_to(root)  # the command's, under /pytorch
    os.chdir(root)
    app = modal.App.lookup("ramjet", create_if_missing=True)
    ramjet_dir = common.ramjet_dir()
    with open(ramjet_dir / "lock", "w") as lock:
        try:  # one sync and build at a time per worktree
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log.info("waiting for another ramjet run in this worktree")
            fcntl.flock(lock, fcntl.LOCK_EX)
        commit = sync.snapshot_tree()
        image = image_for(app, commit, ramjet_dir, cuda, options.python_version, config)
    # The warm runner of this size
    gpu, cpu, memory = resources
    saved = ramjet_dir / f"runner-{gpu or 'no-gpu'}-{cpu}-{memory}.json"
    with open(saved.with_suffix(".lock"), "w") as busy:
        try:
            fcntl.flock(busy, fcntl.LOCK_EX | fcntl.LOCK_NB)
            keep = True
        except BlockingIOError:
            keep = False
        if keep:
            runner, synced, reused = run.warm(app, image, resources, idle, saved)
        else:
            if note := run.noted(saved):
                log.warning("warm sandbox %s is busy", note["sandbox"])
            else:
                log.warning("warm sandbox is busy")
            runner, synced = run.start(app, image, resources, idle), image.commit
            reused = False
        run.show(runner.object_id, reused, resources, run.hourly_rate(resources, rates))
        try:
            run.sync_files(runner, synced, commit, ramjet_dir)
            if keep:
                run.remember(saved, runner, image, commit)
            code = run.execute(runner, command, str(cwd))
        finally:
            if not keep:
                runner.terminate()
    log.debug("%s exited %d", command[0], code)
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(128 + signal.SIGINT)  # the shell's code for Ctrl-C
    except modal.exception.Error as error:  # ramjet logs its own errors and exits
        log.error("%s", error)
        sys.exit(1)
