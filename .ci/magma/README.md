# Magma

This folder contains the scripts and configurations to build magma, statically linked for various versions of CUDA.

## Building

Look in the `Makefile` for available targets to build. To build any target, for example `magma-cuda118`, run

```
# Using `docker`
make magma-cuda118

# Using `podman`
DOCKER_CMD=podman make magma-cuda118
```

This spawns a `pytorch/manylinux-cuda<version>` docker image, which has the required `devtoolset` and CUDA versions installed.
Within the docker image, it runs `build_magma.sh` with the correct environment variables set, which package the necessary files
into a tarball, with the following structure:

```
.
├── include       # header files
├── lib           # libmagma.a
├── info
│   ├── licenses  # license file
│   └── recipe    # build script and patches
```

More specifically, `build_magma.sh` copies over the relevant files from the `package_files` directory depending on the CUDA version.
Outputted binaries should be in the `output` folder.


## Pushing

Packages can be uploaded to an S3 bucket using:

```
aws s3 cp output/*/magma-cuda*.bz2 <bucket-with-path>
```

If you do not have upload permissions, please ping @seemethere or @soumith to gain access

## New versions

New CUDA versions can be added by creating a new make target with the next desired version. For CUDA version NN.n, the target should be named `magma-cudaNNn`.

Make sure to edit the appropriate environment variables (e.g., DESIRED_CUDA, CUDA_ARCH_LIST) in the `Makefile` accordingly. Remember also to check `build_magma.sh` to ensure the logic for copying over the files remains correct.

New patches can be added by editing `Makefile` and`build_magma.sh` the same way `getrf_nbparam.patch` is implemented.
