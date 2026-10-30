# CI Dependency Guidelines

This document covers package installs that run inside Linux CI jobs. Apply it whenever a PR changes code that runs in a Linux CI build or test job: anything under `.ci/` other than `.ci/docker/`, anything under `.github/`, and scripts those jobs run, such as `benchmarks/dynamo/` and `test/`.

## Why

A package resolved from an index while a job runs picks up whatever was released that day, so CI can break with no PyTorch commit to blame or revert. The CI images are the controlled place for upgrades. Image contents change only when `.ci/docker` changes, and PR CI runs on the image built for the PR's `.ci/docker` tree, so an upgrade is exercised before it lands.

## What to flag

Flag an added install that resolves packages from an index during a Linux build or test job:

- `pip install`, `pip_install`, `uv pip install`, `conda install`
- `pip install -e .` of a checked-out project, which resolves that project's dependencies
- `pip install --upgrade pip` (or `-U`) at test time
- a helper or `subprocess` call that runs one of these
- an install-on-`ImportError` fallback (`try: import x` / `except ImportError: pip install ...`)

The fix is to add the package to the CI image: `.ci/docker/requirements-ci.txt`, or a script under `.ci/docker/common/`. Pin it there with `==`: upgrades then come through a PR that bumps the pin, which rebuilds the image and runs CI on it. An exact `==` pin at the install site is not a fix, because the package's own dependencies still resolve fresh on every run.

Adding a package to `requirements-ci.txt` is the fix even though some jobs still resolve it on each run: s390x installs that file at test time into an image that doesn't preinstall it, and macOS and Windows have no image.

When the changed dependency is used only by workflows that don't run on PRs by default, ask for a representative ciflow label that exercises it (for example `ciflow/inductor-periodic` for a benchmark-only dependency). A shared change, such as to `requirements-ci.txt` or a base Dockerfile, does not by itself call for every consumer workflow's label; the image builds are already exercised by `ciflow/docker`.

A job that needs its own venv should layer it over the image with `create_venv_over_image` (`.ci/pytorch/common_utils.sh`) instead of resolving everything again in a fresh one.

## Exempt

- Installing wheels built earlier in the same workflow, when the image already has their dependencies.
- Pinned domain libraries, such as torchvision, that must be built against the PR's torch: built with `--no-deps --no-build-isolation` (`pip_build_and_install`) or installed from their built wheels (`pip_install_whl`). Their ordinary prerequisites still belong in the image.
- Binary smoke tests, which install the built wheel from the index on purpose to check what users get (`.ci/pytorch/binary_linux_test.sh`).
- Best-effort steps that cannot fail the job, such as a `continue-on-error` tool install into a throwaway venv after the tests.
- Deliberately testing an alternative version, such as the sequential einops installs in `test_einops` or the `numpy_2_x` config.
- Jobs not run on an image built with `requirements-ci.txt`.

## Reporting

One finding per install site, anchored on the install line, filed under Infrastructure. Name the package and the image file it belongs in. A non-exempt install always means **Request Changes**.
