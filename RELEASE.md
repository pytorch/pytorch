# Releasing PyTorch

<!-- toc -->

  - [Release Compatibility Matrix](#release-compatibility-matrix)
  - [General Overview](#general-overview)
    - [Frequently Asked Questions](#frequently-asked-questions)
  - [Cutting a release branch preparations](#cutting-a-release-branch-preparations)
  - [Cutting release branches](#cutting-release-branches)
    - [`pytorch/pytorch`](#pytorchpytorch)
    - [`pytorch/builder` / PyTorch domain libraries](#pytorchbuilder--pytorch-domain-libraries)
    - [Making release branch specific changes for PyTorch](#making-release-branch-specific-changes-for-pytorch)
    - [Making release branch specific changes for domain libraries](#making-release-branch-specific-changes-for-domain-libraries)
  - [Drafting RCs (Release Candidates) for PyTorch and domain libraries](#drafting-rcs-release-candidates-for-pytorch-and-domain-libraries)
    - [Release Candidate Storage](#release-candidate-storage)
    - [Release Candidate health validation](#release-candidate-health-validation)
    - [Cherry Picking Fixes](#cherry-picking-fixes)
  - [Promoting RCs to Stable](#promoting-rcs-to-stable)
  - [Additional Steps to prepare for release day](#additional-steps-to-prepare-for-release-day)
    - [Modify release matrix](#modify-release-matrix)
    - [Open Google Colab issue](#open-google-colab-issue)
- [Patch Releases](#patch-releases)
  - [Patch Release Criteria](#patch-release-criteria)
  - [Patch Release Process](#patch-release-process)
    - [Patch Release Process Description](#patch-release-process-description)
    - [Triage](#triage)
    - [Issue Tracker for Patch releases](#issue-tracker-for-patch-releases)
    - [Building a release schedule / cherry picking](#building-a-release-schedule--cherry-picking)
    - [Building Binaries / Promotion to Stable](#building-binaries--promotion-to-stable)
- [Hardware / Software Support in Binary Build Matrix](#hardware--software-support-in-binary-build-matrix)
  - [Python](#python)
    - [TL;DR](#tldr)
  - [Accelerator Software](#accelerator-software)
    - [Special support cases](#special-support-cases)
- [Special Topics](#special-topics)
  - [Updating submodules for a release](#updating-submodules-for-a-release)

<!-- tocstop -->

## Release Compatibility Matrix

Following is the Release Compatibility Matrix for PyTorch releases:

| PyTorch version | Python | Stable CUDA | Experimental CUDA |
| --- | --- | --- | --- |
| 2.0 | >=3.8, <=3.11 | CUDA 11.7, CUDNN 8.5.0.96 | CUDA 11.8, CUDNN 8.7.0.84 |
| 1.13 | >=3.7, <=3.10 | CUDA 11.6, CUDNN 8.3.2.44 | CUDA 11.7, CUDNN 8.5.0.96 |
| 1.12 | >=3.7, <=3.10 | CUDA 11.3, CUDNN 8.3.2.44 | CUDA 11.6, CUDNN 8.3.2.44 |

## General Overview

Releasing a new version of PyTorch generally entails 3 major steps:

0. Cutting a release branch preparations
1. Cutting a release branch and making release branch specific changes
2. Drafting RCs (Release Candidates), and merging cherry picks
3. Promoting RCs to stable and performing release day tasks

### Frequently Asked Questions

* Q: What is release branch cut  ?
  * A: When bulk of the tracked features merged into the main branch, the primary release engineer starts the release process of cutting the release branch by creating a new git branch based off of the current `main` development branch of PyTorch. This allows PyTorch development flow on `main` to continue uninterrupted, while the release engineering team focuses on stabilizing the release branch in order to release a series of release candidates (RC). The activities in the release branch include both regression and performance testing as well as polishing new features and fixing release-specific bugs. In general, new features *are not* added to the release branch after it was created.

* Q: What is cherry-pick ?
  * A: A cherry pick is a process of propagating commits from the main into the release branch, utilizing git's built in [cherry-pick feature](https://git-scm.com/docs/git-cherry-pick). These commits are typically limited to small fixes or documentation updates to ensure that the release engineering team has sufficient time to complete a thorough round of testing on the release branch. To nominate a fix for cherry-picking, a separate pull request must be created against the respective release branch and then mentioned in the Release Tracker issue (example: https://github.com/pytorch/pytorch/issues/94937) following the template from the issue description. The comment nominating a particular cherry-pick for inclusion in the release should include the committed PR against main branch, the newly created cherry-pick PR, as well as the acceptance criteria for why the cherry-pick is needed in the first place.

## Cutting a release branch preparations

Following Requirements needs to be met prior to final RC Cut:

* Resolve all outstanding issues in the milestones(for example [1.11.0](https://github.com/pytorch/pytorch/milestone/28))before first RC cut is completed. After RC cut is completed following script should be executed from builder repo in order to validate the presence of the fixes in the release branch :
``` python github_analyze.py --repo-path ~/local/pytorch --remote upstream --branch release/1.11 --milestone-id 26 --missing-in-branch ```
* Validate that all new workflows have been created in the PyTorch and domain libraries included in the release. Validate it against all dimensions of release matrix, including operating systems(Linux, MacOS, Windows), Python versions as well as CPU architectures(x86 and arm) and accelerator versions(CUDA, ROCm).
* All the nightly jobs for pytorch and domain libraries should be green. Validate this using following HUD links:
  * [Pytorch](https://hud.pytorch.org/hud/pytorch/pytorch/nightly)
  * [TorchVision](https://hud.pytorch.org/hud/pytorch/vision/nightly)
  * [TorchAudio](https://hud.pytorch.org/hud/pytorch/audio/nightly)
  * [TorchText](https://hud.pytorch.org/hud/pytorch/text/nightly)

## Cutting release branches

### `pytorch/pytorch`

Release branches are typically cut from the branch [`viable/strict`](https://github.com/pytorch/pytorch/tree/viable/strict) as to ensure that tests are passing on the release branch.

There's a convenience script to create release branches from current `viable/strict`. Perform following actions :
* Perform a fresh clone of pytorch repo using
```bash
git clone git@github.com:pytorch/pytorch.git
```

* Execute following command from PyTorch repository root folder:
```bash
DRY_RUN=disabled scripts/release/cut-release-branch.sh
```
This script should create 2 branches:
* `release/{MAJOR}.{MINOR}`
* `orig/release/{MAJOR}.{MINOR}`

### `pytorch/builder` / PyTorch domain libraries

*Note*:  Release branches for individual domain libraries should be created after first release candidate build of PyTorch is available in staging channels (which happens about a week after PyTorch release branch has been created). This is absolutely required to allow sufficient testing time for each of the domain library. Domain libraries branch cut is performed by Domain Library POC.
Builder branch cut should be performed at the same time as Pytorch core branch cut. Convenience script can also be used domains as well as `pytorch/builder`

> NOTE: RELEASE_VERSION only needs to be specified if version.txt is not available in root directory

```bash
DRY_RUN=disabled GIT_BRANCH_TO_CUT_FROM=main RELEASE_VERSION=1.11 scripts/release/cut-release-branch.sh
```

### Making release branch specific changes for PyTorch

These are examples of changes that should be made to release branches so that CI / tooling can function normally on
them:

* Update backwards compatibility tests to use RC binaries instead of nightlies
  * Example: https://github.com/pytorch/pytorch/pull/77983 and https://github.com/pytorch/pytorch/pull/77986
* A release branches should also be created in [`pytorch/xla`](https://github.com/pytorch/xla) and [`pytorch/builder`](https://github.com/pytorch/builder) repos and pinned in `pytorch/pytorch`
  * Example: https://github.com/pytorch/pytorch/pull/86290 and https://github.com/pytorch/pytorch/pull/90506
* Update branch used in composite actions from trunk to release (for example, can be done by running `for i in .github/workflows/*.yml; do sed -i -e s#@master#@release/2.0# $i; done`
  * Example: https://github.com/pytorch/pytorch/commit/51b42d98d696a9a474bc69f9a4c755058809542f

These are examples of changes that should be made to the *default* branch after a release branch is cut

* Nightly versions should be updated in all version files to the next MINOR release (i.e. 0.9.0 -> 0.10.0) in the default branch:
  * Example: https://github.com/pytorch/pytorch/pull/77984

### Making release branch specific changes for domain libraries

Domain library branch cut is done a week after branch cut for the `pytorch/pytorch`. The branch cut is performed by the Domain Library POC.
After the branch cut is performed, the Pytorch Dev Infra member should be informed of the branch cut and Domain Library specific change is required before Drafting RC for this domain library.

Follow these examples of PR that updates the version and sets RC Candidate upload channel:
* torchvision : https://github.com/pytorch/vision/pull/5400
* torchaudio: https://github.com/pytorch/audio/pull/2210

## Drafting RCs (Release Candidates) for PyTorch and domain libraries

To draft RCs, a user with the necessary permissions can push a git tag to the main `pytorch/pytorch` git repository. Please note: exactly same process is used for each of the domain library

The git tag for a release candidate must follow the following format:
```
v{MAJOR}.{MINOR}.{PATCH}-rc{RC_NUMBER}
```

An example of this would look like:
```
v1.12.0-rc1
```
You can use following commands to perform tag from pytorch core repo (not fork):
* Checkout and validate the repo history before tagging
```
git checkout release/1.12
git log --oneline
```
* Perform tag and push it to github (this will trigger the binary release build)
```
git tag -f  v1.12.0-rc2
git push origin  v1.12.0-rc2
```

Pushing a release candidate should trigger the `binary_builds` workflow within CircleCI using [`pytorch/pytorch-probot`](https://github.com/pytorch/pytorch-probot)'s [`trigger-circleci-workflows`](trigger-circleci-workflows) functionality.

This trigger functionality is configured here: [`pytorch-circleci-labels.yml`](https://github.com/pytorch/pytorch/blob/main/.github/pytorch-circleci-labels.yml)

To view the state of the release build, please navigate to [HUD](https://hud.pytorch.org/hud/pytorch/pytorch/release%2F1.12). And make sure all binary builds are successful.
### Release Candidate Storage

Release candidates are currently stored in the following places:

* Wheels: https://download.pytorch.org/whl/test/
* Conda: https://anaconda.org/pytorch-test
* Libtorch: https://download.pytorch.org/libtorch/test

Backups are stored in a non-public S3 bucket at [`s3://pytorch-backup`](https://s3.console.aws.amazon.com/s3/buckets/pytorch-backup?region=us-east-1&tab=objects)

### Release Candidate health validation

Validate the release jobs for pytorch and domain libraries should be green. Validate this using following HUD links:
  * [Pytorch](https://hud.pytorch.org/hud/pytorch/pytorch/release%2F1.12)
  * [TorchVision](https://hud.pytorch.org/hud/pytorch/vision/release%2F1.12)
  * [TorchAudio](https://hud.pytorch.org/hud/pytorch/audio/release%2F1.12)
  * [TorchText](https://hud.pytorch.org/hud/pytorch/text/release%2F1.12)

Validate that the documentation build has completed and generated entry corresponding to the release in  [docs folder](https://github.com/pytorch/pytorch.github.io/tree/site/docs/) of pytorch.github.io repository

### Cherry Picking Fixes

Typically, within a release cycle fixes are necessary for regressions, test fixes, etc.

For fixes that are to go into a release after the release branch has been cut we typically employ the use of a cherry pick tracker.

An example of this would look like:
* https://github.com/pytorch/pytorch/issues/51886

Please also make sure to add milestone target to the PR/issue, especially if it needs to be considered for inclusion into the dot release.

**NOTE**: The cherry pick process is not an invitation to add new features, it is mainly there to fix regressions


## Promoting RCs to Stable

Promotion of RCs to stable is done with this script:
[`pytorch/builder:release/promote.sh`](https://github.com/pytorch/builder/blob/main/release/promote.sh)

Users of that script should take care to update the versions necessary for the specific packages you are attempting to promote.

Promotion should occur in two steps:
* Promote S3 artifacts (wheels, libtorch) and Conda packages
* Promote S3 wheels to PyPI

**NOTE**: The promotion of wheels to PyPI can only be done once so take caution when attempting to promote wheels to PyPI, (see https://github.com/pypa/warehouse/issues/726 for a discussion on potential draft releases within PyPI)

## Additional Steps to prepare for release day

The following should be prepared for the release day

### Modify release matrix

Need to modify release matrix for get started page. See following [PR](https://github.com/pytorch/pytorch.github.io/pull/959) as reference.

After modifying published_versions.json you will need to regenerate the quick-start-module.js file run following command
```
python3 scripts/gen_quick_start_module.py >assets/quick-start-module.js
```
Please note: This PR needs to be merged on the release day and hence it should be absolutely free of any failures. To test this PR, open another test PR but pointing to the Release candidate location as above [Release Candidate Storage](RELEASE.md#release-candidate-storage)

### Open Google Colab issue

This is normally done right after the release is completed. We would need to create Google Colab Issue see following [PR](https://github.com/googlecolab/colabtools/issues/2372)

# Patch Releases

A patch release is a maintenance release of PyTorch that includes fixes for regressions found in a previous minor release. Patch releases typically will bump the `patch` version from semver (i.e. `[major].[minor].[patch]`)

## Patch Release Criteria

Patch releases should be considered if a regression meets the following criteria:

1. Does the regression break core functionality (stable / beta features) including functionality in first party domain libraries?
    * First party domain libraries:
        * [pytorch/vision](https://github.com/pytorch/vision)
        * [pytorch/audio](https://github.com/pytorch/audio)
        * [pytorch/text](https://github.com/pytorch/text)
3. Is there not a viable workaround?
    * Can the regression be solved simply or is it not overcomable?

> *NOTE*: Patch releases should only be considered when functionality is broken, documentation does not typically fall within this category

## Patch Release Process

### Patch Release Process Description

> Main POC: Patch Release Managers, Triage Reviewers

Patch releases should follow these high-level phases. This process starts immediately after the previous release has completed.
Minor release process takes around 6-7 weeks to complete.

1. Triage, is a process where issues are identified, graded, compared to Patch Release Criteria and added to Patch Release milestone. This process normally takes 2-3 weeks after the release completion.
2. Patch Release: Go/No Go meeting between PyTorch Releng, PyTorch Core and Project Managers where potential issues triggering a release in milestones are reviewed, and following decisions are made:
  * Should the new patch Release be created ?
  * Timeline execution for the patch release
3. Cherry picking phase starts after the decision is made to create patch release. At this point a new release tracker for the patch release is created, and an announcement will be made on official channels [example announcement](https://dev-discuss.pytorch.org/t/pytorch-release-2-0-1-important-information/1176). The authors of the fixes to regressions will be asked to create their own cherry picks. This process normally takes 2 weeks.
4. Building Binaries, Promotion to Stable and testing. After all cherry picks have been merged, Release Managers trigger new build and produce new release candidate. Announcement is made on the official channel about the RC availability at this point. This process normally takes 2 weeks.
5. General Availability

### Triage

> Main POC: Triage Reviewers

1. Tag issues / pull requests that are candidates for a potential patch release with `triage review`
    * ![adding triage review label](https://user-images.githubusercontent.com/1700823/132589089-a9210a14-6159-409d-95e5-f79067f6fa38.png)
2. Triage reviewers will then check if the regression / fix identified fits within above mentioned [Patch Release Criteria](#patch-release-criteria)
3. Triage reviewers will then add the issue / pull request to the related milestone (i.e. `1.9.1`) if the regressions is found to be within the [Patch Release Criteria](#patch-release-criteria)
    * ![adding to milestone](https://user-images.githubusercontent.com/1700823/131175980-148ff38d-44c3-4611-8a1f-cd2fd1f4c49d.png)

### Issue Tracker for Patch releases

For patch releases issue tracker needs to be created. For patch release, we require all cherry-pick changes to have links to either a high-priority GitHub issue or a CI failure from previous RC. An example of this would look like:
* https://github.com/pytorch/pytorch/issues/51886

Only following issues are accepted:
1. Fixes to regressions against previous major version (e.g. regressions introduced in 1.13.0 from 1.12.0 are pickable for 1.13.1)
2. Low risk critical fixes for: silent correctness, backwards compatibility, crashes, deadlocks, (large) memory leaks
3. Fixes to new features being introduced in this release
4. Documentation improvements
5. Release branch specific changes (e.g. blocking ci fixes, change version identifiers)

### Building a release schedule / cherry picking

> Main POC: Patch Release Managers

1. After regressions / fixes have been triaged Patch Release Managers will work together and build /announce a schedule for the patch release
    * *NOTE*: Ideally this should be ~2-3 weeks after a regression has been identified to allow other regressions to be identified
2. Patch Release Managers will work with the authors of the regressions / fixes to cherry pick their change into the related release branch (i.e. `release/1.9` for `1.9.1`)
    * *NOTE*: Patch release managers should notify authors of the regressions to post a cherry picks for their changes. It is up to authors of the regressions to post a cherry pick. If cherry pick is not posted the issue will not be included in the release.
3. If cherry picking deadline is missed by cherry pick author, patch release managers will not accept any requests after the fact.

### Building Binaries / Promotion to Stable

> Main POC: Patch Release managers

1. Patch Release Managers will follow the process of [Drafting RCs (Release Candidates)](#drafting-rcs-release-candidates-for-pytorch-and-domain-libraries)
2. Patch Release Managers will follow the process of [Promoting RCs to Stable](#promoting-rcs-to-stable)

# Hardware / Software Support in Binary Build Matrix

PyTorch has a support matrix across a couple of different axis. This section should be used as a decision making framework to drive hardware / software support decisions

## Python

For versions of Python that we support we follow the [NEP 29 policy](https://numpy.org/neps/nep-0029-deprecation_policy.html), which was originally drafted by numpy.

### TL;DR

* All minor versions of Python released 42 months prior to the project, and at minimum the two latest minor versions.
* All minor versions of numpy released in the 24 months prior to the project, and at minimum the last three minor versions.

## Accelerator Software

For accelerator software like CUDA and ROCm we will typically use the following criteria:
* Support latest 2 minor versions

### Special support cases

In some instances support for a particular version of software will continue if a need is found. For example, our CUDA 11 binaries do not currently meet
the size restrictions for publishing on PyPI so the default version that is published to PyPI is CUDA 10.2.

These special support cases will be handled on a case by case basis and support may be continued if current PyTorch maintainers feel as though there may still be a
need to support these particular versions of software.

# Special Topics

## Updating submodules for a release

In the event a submodule cannot be fast forwarded, and a patch must be applied we can take two different approaches:

* (preferred) Fork the said repository under the pytorch GitHub organization, apply the patches we need there, and then switch our submodule to accept our fork.
* Get the dependencies maintainers to support a release branch for us

Editing submodule remotes can be easily done with: (running from the root of the git repository)
```
git config --file=.gitmodules -e
```

An example of this process can be found here:

* https://github.com/pytorch/pytorch/pull/48312
