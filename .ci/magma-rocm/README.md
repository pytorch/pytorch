# Magma ROCm

This folder contains the scripts and configurations to build libmagma.so, linked for various versions of ROCm.

## Building

Look in the `Makefile` for available targets to build. To build any target, for example `magma-rocm63`, run

```
# Using `docker`
make magma-rocm63

# Using `podman`
DOCKER_CMD=podman make magma-rocm63
```

This spawns a `pytorch/manylinux-rocm<version>` docker image, which has the required `devtoolset` and ROCm versions installed.
Within the docker image, it runs `build_magma.sh` with the correct environment variables set, which package the necessary files
into a tarball, with the following structure:

```
.
├── include       # header files
├── lib           # libmagma.so
├── info
│   ├── licenses  # license file
│   └── recipe    # build script
```

More specifically, `build_magma.sh` copies over the relevant files from the `package_files` directory depending on the ROCm version.
Outputted binaries should be in the `output` folder.


## Pushing

Packages can be uploaded to an S3 bucket using:

```
aws s3 cp output/*/magma-cuda*.bz2 <bucket-with-path>
```

If you do not have upload permissions, please ping @seemethere or @soumith to gain access

## New versions

New ROCm versions can be added by creating a new make target with the next desired version. For ROCm version N.n, the target should be named `magma-rocmNn`.

Make sure to edit the appropriate environment variables (e.g., DESIRED_ROCM) in the `Makefile` accordingly. Remember also to check `build_magma.sh` to ensure the logic for copying over the files remains correct.
