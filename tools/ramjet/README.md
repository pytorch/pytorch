# ramjet

ramjet builds your worktree on [Modal](https://modal.com) and runs a command, such as pytest, on a runner with the GPUs you name, so you can test on GPUs your machine does not have:

```
python -m tools.ramjet --gpu H100 -- pytest test/test_nn.py -k linear
```

The runner, the sandbox your command runs in, takes `modal shell`'s options. `--gpu` takes GPUs as Modal names them, in any case, with `:N` for N of them: `L4`, `A100-80GB`, `H100`, `B200:8`; without it the runner has no GPU. `--cpu` sets its physical cores, 1 to 64 in powers of two, and `--memory` its memory in MiB, 8192 to 344064; by default it gets 8 cores per GPU, or 8 without one, and 8192 MiB per core. Modal has no CPU types, so the CPU model and its instruction set can't be chosen. The builder is always the same: 64 cores and no GPU. `--help` lists every choice. `--build-cuda` picks the CUDA version to build with: any that CI builds an image for and that has an official arch list, 12.6, 13.2 and 13.4 today. The default is the version of PyTorch's default wheel. `--python-version` picks the Python version, the same in the builder and the runner, which boots from the builder's image: any PyTorch ships wheels for, 3.11 to 3.15 with the free-threaded `t` builds. The default is the one in CI's image, 3.11 today. `--log-level` sets the least severe messages ramjet shows: `debug`, `info` (the default), `warn` or `error`. Everything after `--` is the command, run as given in the directory you run ramjet from: `pytest test/test_nn.py -k linear`, `nvidia-smi`, or `bash -c '...'` for pipes and `&&`.

## Setup

```
pip install modal
modal token new
```

## How it works

1. Your working tree, edits and untracked files included, becomes a commit. HEAD, the index and the stash are left alone.
2. If the build reads none of the files that changed since the last build, there is no build. Otherwise a 64-core builder boots from the last build image, fetches a git bundle of the new commits, rebuilds incrementally and saves itself as the next build image. The first build clones pytorch/pytorch into CI's CUDA image and takes 1-1.5 hours when the compile cache has none of its compiles. Builders share that cache: ccache looks up each compile in the build image's own cache, then in the `ramjet-ccache` Modal Volume, which every builder in your Modal workspace reads and adds to. Each build deletes the Volume's entries written more than 14 days ago.
3. A runner of the size you ask for boots from the build image, or the warm one from your last run is reused. It gets a tar of the files that changed and runs your command.

## Costs

The builder bills about $13 an hour, so a first build costs $13-20 and an incremental one much less. A runner bills its GPUs plus its cores and memory: 8 cores and 64 GiB per GPU unless `--cpu` and `--memory` say otherwise. After a run it stays warm for 15 minutes, or until idling has cost $2.50, whichever comes first (`IDLE_MAX_SECONDS` and `IDLE_BUDGET` in `run.py`). A full build adds about 1.2 GB to the compile cache's Volume; Modal bills Volume storage past 1 TiB, at $0.09/GiB a month.

## Files

ramjet keeps its state in `/tmp/ramjet-<user>/<worktree folder>-<hash of its path>`, or under `$TMPDIR` if you set it: the build images per build config (`lineage.json`), the last build's log (`build.log`) and a warm runner per runner size. Build images expire after 5 days; ramjet rebuilds images older than 4 days so a lineage outlives them.

## Limits

- Builds use the official arch list for their CUDA version. Before building, ramjet stops if that list, or your `TORCH_CUDA_ARCH_LIST`, can't run on `--gpu`: 12.6's stops at sm_90, so it can't run on Blackwell GPUs such as B200, B300 or RTX PRO 6000. `--build-cuda` and your `BUILD_*`, `USE_*`, `CMAKE_BUILD_TYPE`, `DEBUG`, `REL_WITH_DEB_INFO` and `TORCH_CUDA_ARCH_LIST` variables change the build; each combination gets its own lineage.
- CI's CUDA images come with one Python. ramjet installs another `--python-version` the way CI's Dockerfile installs Python (`install_python.sh` into a venv first on `PATH`), as its own lineage. CI runs CUDA tests only on 3.11 and 3.12, so other versions with CUDA are untested there.
- A lineage stays on the version of CI's image it started on, so pulling a `.ci/docker` change doesn't restart it. To move to the current version, delete `lineage.json`.
- The command gets no terminal and no input, so interactive programs such as a bare `bash` or `python` don't work.
- Your whole Modal workspace shares the compile cache: a build links whatever objects other builders put in it.
