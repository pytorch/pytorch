# ramjet

ramjet builds your worktree on [Modal](https://modal.com) and runs a command or tests there, on GPUs your machine does not have. It takes the worktree as it is, edits and untracked files included, so there is nothing to commit or push, and you need no local build.

## Setup

```
pip install modal
modal token new
```

## Run a command

```
python -m tools.ramjet --gpu H100 run pytest test/test_nn.py -k linear
```

Everything after `run` is the command. It runs in the directory you run ramjet from, and ramjet exits with the command's exit code. The command gets no terminal and no input, so use `bash -c '...'` for pipes and `&&`. The sandbox stays warm for up to 15 minutes, so your next run skips the start.

## Run tests

```
python -m tools.ramjet --gpu L4 test test/dynamo
python -m tools.ramjet --gpu L4 test test/test_nn.py -- -k linear
python -m tools.ramjet --gpu L4 test --list test/dynamo
```

`test` runs the tests under each path through `run_test.py`, as CI does, on as many containers as Modal gives you. A path is a test file, a directory of them, or a node id such as `test/test_nn.py::TestNN`. Arguments after `--` go to pytest. `--list` prints the tests instead of running them, with an estimate of the run's time and cost.

A run ends with a summary: what it cost, how its tests ended and where their logs are. The same result goes to stdout as JSON, and the exit code is 1 if a test failed or did not run.

Your `PYTORCH_*` and `TORCH*` variables go to the containers, so `PYTORCH_TEST_WITH_SLOW=1` also runs the slow tests.

## Options

Options go before `run` or `test`. `python -m tools.ramjet --help` lists every choice.

- `--gpu`: the GPUs, as Modal names them, such as `L4`, `A100-80GB`, `H100`, or `B200:8` for eight. Without it there is no GPU.
- `--cpu`, `--memory`: physical cores, and memory in MiB.
- `--build-cuda`, `--python-version`: the CUDA and Python versions.
- `--log-level`: `debug` shows more.

Build variables in your environment, such as `DEBUG=1` or `USE_*`, apply to the build as they do locally.

## Cost

Modal bills your workspace for what ramjet uses.

- The first build of a worktree takes about an hour and costs about $15. So does the first with another CUDA version, Python version or set of build variables, and the first after a build image expires, 5 days after it was built.
- Later builds are incremental, and changes that the build does not read, such as most Python files, need no build.
- After `run`, the sandbox keeps billing while it stays warm.
