# PyTorch release scripts performing branch cut and applying release only changes

These are a collection of scripts that are to be used for release activities.

> NOTE: All scripts should do no actual work unless the `DRY_RUN` environment variable is set
>       to `disabled`.
>       The basic idea being that there should be no potential to do anything dangerous unless
>       `DRY_RUN` is explicitly set to `disabled`.

### Order of Execution

1. Run cut-release-branch.sh to cut the release branch
2. Run apply-release-changes.sh to apply release only changes to create a PR with release only changes similar to this [PR](https://github.com/pytorch/pytorch/pull/149056)

#### Promoting packages

 Scripts for Promotion of PyTorch packages are under test-infra repository. Please follow [README.md](https://github.com/pytorch/test-infra/blob/main/release/README.md)
