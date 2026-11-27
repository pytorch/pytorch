# PyTorch release scripts performing branch cut and applying release only changes

These are a collection of scripts that are to be used for release activities.

> NOTE: All scripts should do no actual work unless the `DRY_RUN` environment variable is set
>       to `disabled`.
>       The basic idea being that there should be no potential to do anything dangerous unless
>       `DRY_RUN` is explicitly set to `disabled`.

### Order of Execution

1. Run cut-release-branch.sh to cut the release branch
2. Run apply-release-changes.sh to apply release only changes to create a PR with release only changes similar to this [PR](https://github.com/pytorch/pytorch/pull/149056)

#### XPU dependency selection

`third_party/xpu.txt` pins a SHA on main and tracks `release/X.Y` on release branches.
Create the matching torch-xpu-ops branch from the branch-cut XPU pin before building XPU.
`apply-release-changes.sh` writes its name into `xpu.txt` and warns but continues if it cannot resolve the branch.
CMake fetches the full branch ref and checks out its resolved SHA.
`torch.version.xpu_ops_commit` records the dependency SHA, or `None` when unavailable.

#### Promoting packages

 Scripts for Promotion of PyTorch packages are under test-infra repository. Please follow [README.md](https://github.com/pytorch/test-infra/blob/main/release/README.md)
