# PyTorch Release Scripts

These are a collection of scripts that are to be used for release activities.

> NOTE: All scripts should do no actual work unless the `DRY_RUN` environment variable is set
>       to `disabled`.
>       The basic idea being that there should be no potential to do anything dangerous unless
>       `DRY_RUN` is explicitly set to `disabled`.

## Requirements to actually run these scripts
* AWS access to pytorch account
* Access to upload conda packages to the `pytorch` conda channel
* Access to the PyPI repositories


## Promote

These are scripts related to promotion of release candidates to GA channels, these
can actually be used to promote pytorch, libtorch, and related domain libraries.

### Usage

Usage should be fairly straightforward and should actually require no extra variables
if you are running from the correct git tags. (i.e. the GA tag to promote is currently
checked out)

`PACKAGE_TYPE` and `PACKAGE_NAME` can be swapped out to promote other packages.

#### Promoting pytorch wheels
```bash
promote/s3_to_s3.sh
```

#### Promoting libtorch archives
```bash
PACKAGE_TYPE=libtorch PACKAGE_NAME=libtorch promote/s3_to_s3.sh
```

#### Promoting conda packages
```bash
promote/conda_to_conda.sh
```

#### Promoting wheels to PyPI
**WARNING**: These can only be run once and cannot be undone, run with caution
```
promote/wheel_to_pypi.sh
```

## Restoring backups

All release candidates are currently backed up to `s3://pytorch-backup/${TAG_NAME}` and
can be restored to the test channels with the `restore-backup.sh` script.

Which backup to restore from is dictated by the `RESTORE_FROM` environment variable.

### Usage
```bash
RESTORE_FROM=v1.5.0-rc5 ./restore-backup.sh
```
