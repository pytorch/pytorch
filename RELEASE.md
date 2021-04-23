# Releasing PyTorch

<!-- toc -->

  - [General Overview](#general-overview)
  - [Cutting release branches](#cutting-release-branches)
    - [Making release branch specific changes](#making-release-branch-specific-changes)
    - [Getting CI signal on release branches:](#getting-ci-signal-on-release-branches)
  - [Drafting RCs (Release Candidates)](#drafting-rcs-release-candidates)
    - [Release Candidate Storage](#release-candidate-storage)
    - [Cherry Picking Fixes](#cherry-picking-fixes)
  - [Promoting RCs to Stable](#promoting-rcs-to-stable)
- [Special Topics](#special-topics)
  - [Updating submodules for a release](#updating-submodules-for-a-release)

<!-- tocstop -->

## General Overview

Releasing a new version of PyTorch generally entails 3 major steps:

1. Cutting a release branch and making release branch specific changes
2. Drafting RCs (Release Candidates), and merging cherry picks
3. Promoting RCs to stable

## Cutting release branches

Release branches are typically cut from the branch [`viable/strict`](https://github.com/pytorch/pytorch/tree/viable/strict) as to ensure that tests are passing on the release branch.

Release branches *should* be prefixed like so:
```
release/{MAJOR}.{MINOR}
```

An example of this would look like:
```
release/1.8
```

Please make sure to create branch that pins divergent point of release branch from the main thunk, i.e. `orig/release/{MAJOR}.{MINOR}`
### Making release branch specific changes

These are examples of changes that should be made to release branches so that CI / tooling can function normally on
them:

* Update target determinator to use release branch:
  * Example: https://github.com/pytorch/pytorch/pull/40712
* Cutting a release branch on [`pytorch/xla`](https://github.com/pytorch/xla)
  * Example: https://github.com/pytorch/pytorch/pull/40721
* Update backwards compatibility tests to use RC binaries instead of nightlies
  * Example: https://github.com/pytorch/pytorch/pull/40706
* Add `release/{MAJOR}.{MINOR}` to list of branches in [`browser-extension.json`](https://github.com/pytorch/pytorch/blob/fb-config/browser-extension.json) for FaceHub integrated setups
  * Example: https://github.com/pytorch/pytorch/commit/f99fbd94d18627bae776ea2448e075ca4d5e37b2

> TODO: Create release branch in [`pytorch/builder`](https://github.com/pytorch/builder) repo and pin release CI to use that branch rather than HEAD of builder repo.

### Getting CI signal on release branches:
Create a PR from `release/{MAJOR}.{MINOR}` to `orig/release/{MAJOR}.{MINOR}` in order to start CI testing for cherry-picks into release branch.

Example:
* https://github.com/pytorch/pytorch/pull/51995

## Drafting RCs (Release Candidates)

To draft RCs, a user with the necessary permissions can push a git tag to the main `pytorch/pytorch` git repository.

The git tag for a release candidate must follow the following format:
```
v{MAJOR}.{MINOR}.{PATCH}-rc{RC_NUMBER}
```

An example of this would look like:
```
v1.8.1-rc1
```

Pushing a release candidate should trigger the `binary_builds` workflow within CircleCI using [`pytorch/pytorch-probot`](https://github.com/pytorch/pytorch-probot)'s [`trigger-circleci-workflows`](trigger-circleci-workflows) functionality.

This trigger functionality is configured here: [`pytorch-circleci-labels.yml`](https://github.com/pytorch/pytorch/blob/master/.github/pytorch-circleci-labels.yml)

### Release Candidate Storage

Release candidates are currently stored in the following places:

* Wheels: https://download.pytorch.org/whl/test/
* Conda: https://anaconda.org/pytorch-test
* Libtorch: https://download.pytorch.org/libtorch/test

Backups are stored in a non-public S3 bucket at [`s3://pytorch-backup`](https://s3.console.aws.amazon.com/s3/buckets/pytorch-backup?region=us-east-1&tab=objects)

### Cherry Picking Fixes

Typically within a release cycle fixes are necessary for regressions, test fixes, etc.

For fixes that are to go into a release after the release branch has been cut we typically employ the use of a cherry pick tracker.

An example of this would look like:
* https://github.com/pytorch/pytorch/issues/51886

**NOTE**: The cherry pick process is not an invitation to add new features, it is mainly there to fix regressions

## Promoting RCs to Stable

Promotion of RCs to stable is done with this script:
[`pytorch/builder:release/promote.sh`](https://github.com/pytorch/builder/blob/master/release/promote.sh)

Users of that script should take care to update the versions necessary for the specific packages you are attempting to promote.

Promotion should occur in two steps:
* Promote S3 artifacts (wheels, libtorch) and Conda packages
* Promote S3 wheels to PyPI

**NOTE**: The promotion of wheels to PyPI can only be done once so take caution when attempting to promote wheels to PyPI, (see https://github.com/pypa/warehouse/issues/726 for a discussion on potential draft releases within PyPI)

# Special Topics

## Updating submodules for a release

In the event a submodule cannot be fast forwarded and a patch must be applied we can take two different approaches:

* (preferred) Fork the said repository under the pytorch Github organization, apply the patches we need there, and then switch our submodule to accept our fork.
* Get the dependencies maintainers to support a release branch for us

Editing submodule remotes can be easily done with: (running from the root of the git repository)
```
git config --file=.gitmodules -e
```

An example of this process can be found here:

* https://github.com/pytorch/pytorch/pull/48312
