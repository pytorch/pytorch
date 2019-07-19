Structure of CI
===============

setup job:
1. Does a git checkout
2. Persists CircleCI scripts (everything in `.circleci`) into a workspace.  Why?
   We don't always do a Git checkout on all subjobs, but we usually
   still want to be able to call scripts one way or another in a subjob.
   Persisting files this way lets us have access to them without doing a
   checkout.  This workspace is conventionally mounted on `~/workspace`
   (this is distinguished from `~/project`, which is the conventional
   working directory that CircleCI will default to starting your jobs
   in.)
3. Write out the commit message to `.circleci/COMMIT_MSG`.  This is so
   we can determine in subjobs if we should actually run the jobs or
   not, even if there isn't a Git checkout.




CircleCI configuration generator
================================

One may no longer make changes to the `.circleci/config.yml` file directly.
Instead, one must edit these Python scripts or files in the `verbatim-sources/` directory.


Usage
----------

1. Make changes to these scripts.
2. Run the `regenerate.sh` script in this directory and commit the script changes and the resulting change to `config.yml`.

You'll see a build failure on TravisCI if the scripts don't agree with the checked-in version.


Motivation
----------

These scripts establish a single, authoritative source of documentation for the CircleCI configuration matrix.
The documentation, in the form of diagrams, is automatically generated and cannot drift out of sync with the YAML content.

Furthermore, consistency is enforced within the YAML config itself, by using a single source of data to generate
multiple parts of the file.

* Facilitates one-off culling/enabling of CI configs for testing PRs on special targets

Also see https://github.com/pytorch/pytorch/issues/17038


Future direction
----------------

### Declaring sparse config subsets
See comment [here](https://github.com/pytorch/pytorch/pull/17323#pullrequestreview-206945747):

In contrast with a full recursive tree traversal of configuration dimensions,
> in the future future I think we actually want to decrease our matrix somewhat and have only a few mostly-orthogonal builds that taste as many different features as possible on PRs, plus a more complete suite on every PR and maybe an almost full suite nightly/weekly (we don't have this yet). Specifying PR jobs in the future might be easier to read with an explicit list when we come to this.

----------------
----------------

# How do the binaries / nightlies / releases work?

### What is a binary?

A binary or package (used interchangeably) is a pre-built collection of c++ libraries, header files, python bits, and other files. We build these and distribute them so that users do not need to install from source.

A **binary configuration** is a collection of

* release or nightly
    * releases are stable, nightlies are beta and built every night
* python version
    * linux: 2.7m, 2.7mu, 3.5m, 3.6m 3.7m (mu is wide unicode or something like that. It usually doesn't matter but you should know that it exists)
    * macos and windows: 2.7, 3.5, 3.6, 3.7
* cpu version
    * cpu, cuda 9.0, cuda 10.0
    * The supported cuda versions occasionally change
* operating system
    * Linux - these are all built on CentOS. There haven't been any problems in the past building on CentOS and using on Ubuntu
    * MacOS
    * Windows - these are built on Azure pipelines
* devtoolset version (gcc compiler version)
    * This only matters on Linux cause only Linux uses gcc. tldr is gcc made a backwards incompatible change from gcc 4.8 to gcc 5, because it had to change how it implemented std::vector and std::string 

### Where are the binaries?

The binaries are built in CircleCI. There are nightly binaries built every night at 9pm PST (midnight EST) and release binaries corresponding to Pytorch releases, usually every few months.

We have 3 types of binary packages

* pip packages - nightlies are stored on s3 (pip install -f <a s3 url>). releases are stored in a pip repo (pip install torch) (ask Soumith about this)
* conda packages - nightlies and releases are both stored in a conda repo. Nighty packages have a '_nightly' suffix
* libtorch packages - these are zips of all the c++ libraries, header files, and sometimes dependencies. These are c++ only
    * shared with dependencies
    * static with dependencies
    * shared without dependencies
    * static without dependencies

All binaries are built in CircleCI workflows. There are checked-in workflows (committed into the .circleci/config.yml) to build the nightlies every night. Releases are built by manually pushing a PR that builds the suite of release binaries (overwrite the config.yml to build the release)

# CircleCI structure of the binaries

Some quick vocab: 

* A\**workflow** is a CircleCI concept; it is a DAG of '**jobs**'. ctrl-f 'workflows' on\https://github.com/pytorch/pytorch/blob/master/.circleci/config.yml to see the workflows. 
* **jobs** are a sequence of '**steps**'
* **steps** are usually just a bash script or a builtin CircleCI command.* All steps run in new environments, environment variables declared in one script DO NOT persist to following steps*
* CircleCI has a **workspace**, which is essentially a cache between steps of the *same job* in which you can store artifacts between steps. 

## How are the workflows structured?

The nightly binaries have 3 workflows. We have one job (actually 3 jobs:  build, test, and upload) per binary configuration

1. binarybuilds
    1. every day midnight EST
    2. linux: https://github.com/pytorch/pytorch/blob/master/.circleci/verbatim-sources/linux-binary-build-defaults.yml
    3. macos: https://github.com/pytorch/pytorch/blob/master/.circleci/verbatim-sources/macos-binary-build-defaults.yml
    4. For each binary configuration, e.g. linux_conda_3.7_cpu there is a 
        1. binary_linux_conda_3.7_cpu_build
            1. Builds the build. On linux jobs this uses the 'docker executor'. 
            2. Persists the package to the workspace
        2. binary_linux_conda_3.7_cpu_test
            1. Loads the package to the workspace
            2. Spins up a docker image (on Linux), mapping the package and code repos into the docker
            3. Runs some smoke tests in the docker
            4. (Actually, for macos this is a step rather than a separate job)
        3. binary_linux_conda_3.7_cpu_upload
            1. Logs in to aws/conda
            2. Uploads the package
2. update_s3_htmls
    1. every day 5am EST
    2. https://github.com/pytorch/pytorch/blob/master/.circleci/verbatim-sources/binary_update_htmls.yml
    3. See below for what these are for and why they're needed
    4. Three jobs that each examine the current contents of aws and the conda repo and update some html files in s3
3. binarysmoketests
    1. every day 
    2. https://github.com/pytorch/pytorch/blob/master/.circleci/verbatim-sources/nightly-build-smoke-tests-defaults.yml
    3. For each binary configuration, e.g. linux_conda_3.7_cpu there is a 
        1. smoke_linux_conda_3.7_cpu
            1. Downloads the package from the cloud, e.g. using the official pip or conda instructions
            2. Runs the smoke tests

## How are the jobs structured?

The jobs are in https://github.com/pytorch/pytorch/tree/master/.circleci/verbatim-sources . Jobs are made of multiple steps. There are some shared steps used by all the binaries/smokes. Steps of these jobs are all delegated to scripts in https://github.com/pytorch/pytorch/tree/master/.circleci/scripts . 

* Linux jobs: https://github.com/pytorch/pytorch/blob/master/.circleci/verbatim-sources/linux-binary-build-defaults.yml
    * binary_linux_build.sh
    * binary_linux_test.sh
    * binary_linux_upload.sh
* MacOS jobs: https://github.com/pytorch/pytorch/blob/master/.circleci/verbatim-sources/macos-binary-build-defaults.yml
    * binary_macos_build.sh
    * binary_macos_test.sh
    * binary_macos_upload.sh
* Update html jobs: https://github.com/pytorch/pytorch/blob/master/.circleci/verbatim-sources/binary_update_htmls.yml
    * These delegate from the pytorch/builder repo
    * https://github.com/pytorch/builder/blob/master/cron/update_s3_htmls.sh
    * https://github.com/pytorch/builder/blob/master/cron/upload_binary_sizes.sh
* Smoke jobs (both linux and macos): https://github.com/pytorch/pytorch/blob/master/.circleci/verbatim-sources/nightly-build-smoke-tests-defaults.yml
    * These delegate from the pytorch/builder repo
    * https://github.com/pytorch/builder/blob/master/run_tests.sh
    * https://github.com/pytorch/builder/blob/master/smoke_test.sh
    * https://github.com/pytorch/builder/blob/master/check_binary.sh
* Common shared code (shared across linux and macos): https://github.com/pytorch/pytorch/blob/master/.circleci/verbatim-sources/nightly-binary-build-defaults.yml
    * binary_checkout.sh - checks out pytorch/builder repo. Right now this also checks out pytorch/pytorch, but it shouldn't. pytorch/pytorch should just be shared through the workspace. This can handle being run before binary_populate_env.sh
    * binary_populate_env.sh - parses BUILD_ENVIRONMENT into the separate env variables that make up a binary configuration. Also sets lots of default values, the date, the version strings, the location of folders in s3, all sorts of things. This generally has to be run before other steps.
    * binary_install_miniconda.sh - Installs miniconda, cross platform. Also hacks this for the update_binary_sizes job that doesn't have the right env variables
    * binary_run_in_docker.sh - Takes a bash script file (the actual test code) from a hardcoded location, spins up a docker image, and runs the script inside the docker image

### **Why do the steps all refer to scripts?**

CircleCI creates a  final yaml file by inlining every <<* segment, so if we were to keep all the code in the config.yml itself then the config size would go over 4 MB and cause infra problems.

### **What is binary_run_in_docker for?**

So, CircleCI has several executor types: macos, machine, and docker are the ones we use. The 'machine' executor gives you two cores on some linux vm. The 'docker' executor gives you considerably more cores (nproc was 32 instead of 2 back when I tried in February). Since the dockers are faster, we try to run everything that we can in dockers. Thus

* linux build jobs use the docker executor. Running them on the docker executor was at least 2x faster than running them on the machine executor
* linux test jobs use the machine executor and spin up their own docker. Why this nonsense? It's cause we run nvidia-docker for our GPU tests; any code that calls into the CUDA runtime needs to be run on nvidia-docker. To run a nvidia-docker you need to install some nvidia packages on the host machine and then call docker with the '—runtime nvidia' argument. CircleCI doesn't support this, so we have to do it ourself. 
    * This is not just a mere inconvenience. **This blocks all of our linux tests from using more than 2 cores.** But there is nothing that we can do about it, but wait for a fix on circleci's side. Right now, we only run some smoke tests (some simple imports) on the binaries, but this also affects non-binary test jobs.
* linux upload jobs use the machine executor. The upload jobs are so short that it doesn't really matter what they use
* linux smoke test jobs use the machine executor for the same reason as the linux test jobs

binary_run_in_docker.sh is a way to share the docker start-up code between the binary test jobs and the binary smoke test jobs

### **Why does binary_checkout also checkout pytorch? Why shouldn't it?**

We want all the nightly binary jobs to run on the exact same git commit, so we wrote our own checkout logic to ensure that the same commit was always picked. Later circleci changed that to use a single pytorch checkout and persist it through the workspace (they did this because our config file was too big, so they wanted to take a lot of the setup code into scripts, but the scripts needed the code repo to exist to be called, so they added a prereq step called 'setup' to checkout the code and persist the needed scripts to the workspace). The changes to the binary jobs were not properly tested, so they all broke from missing pytorch code no longer existing. We hotfixed the problem by adding the pytorch checkout back to binary_checkout, so now there's two checkouts of pytorch on the binary jobs. This problem still needs to be fixed, but it takes careful tracing of which code is being called where.

# Code structure of the binaries (circleci agnostic)

## Overview

The code that runs the binaries lives in two places, in the normal [github.com/pytorch/pytorch](http://github.com/pytorch/pytorch), but also in [github.com/pytorch/builder](http://github.com/pytorch/builder) , which is a repo that defines how all the binaries are built. The relevant code is


```
# All code needed to set-up environments for build code to run in,
# but only code that is specific to the current CI system
pytorch/pytorch
- .circleci/                # Folder that holds all circleci related stuff
  - config.yml              # GENERATED file that actually controls all circleci behavior
  - verbatim-sources        # Used to generate job/workflow sections in ^
  - scripts/                # Code needed to prepare circleci environments for binary build scripts

- setup.py                  # Builds pytorch. This is wrapped in pytorch/builder
- cmake files               # used in normal building of pytorch

# All code needed to prepare a binary build, given an environment
# with all the right variables/packages/paths.
pytorch/builder

# Given an installed binary and a proper python env, runs some checks
# to make sure the binary was built the proper way. Checks things like
# the library dependencies, symbols present, etc.
- check_binary.sh

# Given an installed binary, runs python tests to make sure everything
# is in order. These should be de-duped. Right now they both run smoke
# tests, but are called from different places. Usually just call some
# import statements, but also has overlap with check_binary.sh above
- run_tests.sh
- smoke_test.sh

# Folders that govern how packages are built. See paragraphs below

- conda/
  - build_pytorch.sh          # Entrypoint. Delegates to proper conda build folder
  - switch_cuda_version.sh    # Switches activate CUDA installation in Docker
  - pytorch-nightly/          # Build-folder
- manywheel/
  - build_cpu.sh              # Entrypoint for cpu builds
  - build.sh                  # Entrypoint for CUDA builds
  - build_common.sh           # Actual build script that ^^ call into
- wheel/
  - build_wheel.sh            # Entrypoint for wheel builds
```

Every type of package has an entrypoint build script that handles the all the important logic.

## Conda

Both Linux and MacOS use the same code flow for the conda builds.

Conda packages are built with conda-build, see https://conda.io/projects/conda-build/en/latest/resources/commands/conda-build.html 

Basically, you pass `conda build` a build folder (pytorch-nightly/ above) that contains a build script and a meta.yaml. The meta.yaml specifies in what python environment to build the package in, and what dependencies the resulting package should have, and the build script gets called in the env to build the thing.
tldr; on conda-build is

1. Creates a brand new conda environment, based off of deps in the meta.yaml
    1. Note that environment variables do not get passed into this build env unless they are specified in the meta.yaml
    2. If the build fails this environment will stick around. You can activate it for much easier debugging. The “General Python” section below explains what exactly a python “environment” is.
2. Calls build.sh in the environment
3. Copies the finished package to a new conda env, also specified by the meta.yaml
4. Runs some simple import tests (if specified in the meta.yaml)
5. Saves the finished package as a tarball

The build.sh we use is essentially a wrapper around ```python setup.py build``` , but it also manually copies in some of our dependent libraries into the resulting tarball and messes with some rpaths.

The entrypoint file `builder/conda/build_conda.sh` is complicated because

* It works for both Linux and MacOS
    * The mac builds used to create their own environments, since they all used to be on the same machine. There’s now a lot of extra logic to handle conda envs. This extra machinery could be removed
* It used to handle testing too, which adds more logic messing with python environments too. This extra machinery could be removed.

## Manywheels (linux pip and libtorch packages)

Manywheels are pip packages for linux distros. Note that these manywheels are not actually manylinux compliant. 

`builder/manywheel/build_cpu.sh` and `builder/manywheel/build.sh` (for CUDA builds) just set different env vars and then call into `builder/manywheel/build_common.sh`

The entrypoint file `builder/manywheel/build_common.sh` is really really complicated because

* This used to handle building for several different python versions at the same time. This is why there are loops everywhere
    * The script is never used this way anymore. This extra machinery could be removed.
* This used to handle testing the pip packages too. This is why there’s testing code at the end that messes with python installations and stuff
    * The script is never used this way anymore. This extra machinery could be removed.
* This also builds libtorch packages
    * This should really be separate. libtorch packages are c++ only and have no python. They should not share infra with all the python specific stuff in this file.
* There is a lot of messing with rpaths. This is necessary, but could be made much much simpler if the loops for libtorch and separate python versions were removed.

## Wheels (MacOS pip and libtorch packages)

The entrypoint file `builder/wheel/build_wheel.sh` is complicated because

* The mac builds used to all run on one machine (we didn’t have autoscaling mac machines till circleci). So this script handled siloing itself by setting-up and tearing-down its build env and siloing itself into its own build directory.
    * The script is never used this way anymore. This extra machinery could be removed.
* This also builds libtorch packages
    * Ditto the comment above. This should definitely be separated out.

Note that the MacOS Python wheels are still built in conda environments. Some of the dependencies present during build also come from conda.

## General notes

### Note on run_tests.sh, smoke_test.sh, and check_binary.sh

* These should all be consolidated
* These must run on all OS types: MacOS, Linux, and Windows
* These all run smoke tests at the moment. They inspect the packages some, maybe run a few import statements. They DO NOT run the python tests nor the cpp tests. The idea is that python tests on master and PR merges will catch all breakages. All these tests have to do is make sure the special binary machinery didn’t mess anything up.
* There are separate run_tests.sh and smoke_test.sh because one used to be called by the smoke jobs and one used to be called by the binary test jobs (see circleci structure section above). This is still true actually, but these could be united into a single script that runs these checks, given an installed pytorch package.

### Note on libtorch

Libtorch packages are built in the wheel build scripts: manywheel/build_*.sh for linux and build_wheel.sh for mac. There are several things wrong with this

* It’s confusinig. Most of those scripts deal with python specifics.
* The extra conditionals everywhere severely complicate the wheel build scripts
* The process for building libtorch is different from the official instructions (a plain call to cmake, or a call to a script)
* For Linux specifically, the job is set up to build all libtorch varieties in a single go. This leads to 9+ hour builds times for CUDA 10.0 libtorch. This is more of a problem with the circleci setup though.

### Note on docker images / Dockerfiles

All linux builds occur in docker images. The docker images are

* soumith/conda-cuda
    * Has ALL CUDA versions installed. The script pytorch/builder/conda/switch_cuda_version.sh sets /usr/local/cuda to a symlink to e.g. /usr/local/cuda-8.0 to enable different CUDA builds
    * Also used for cpu builds
* soumith/manylinux-cuda80
    * Also used for cpu builds
* soumith/manylinux-cuda90
* soumith/manylinux-cuda92
* soumith/manylinux-cuda100

The Dockerfiles are available in pytorch/builder, but there is no circleci job or script to build these docker images, and they cannot be run locally (unless you have the correct local packages/paths). Only Soumith can build them right now.

### General Python

* This is still a good explanation of python installations https://caffe2.ai/docs/faq.html#why-do-i-get-import-errors-in-python-when-i-try-to-use-caffe2

# How to manually rebuild the binaries

tldr; make a PR that looks like https://github.com/pytorch/pytorch/pull/21159

Sometimes we want to push a change to master and then rebuild all of today's binaries after that change. As of May 30, 2019 there isn't a way to manually run a workflow in the UI. You can manually re-run a workflow, but it will use the exact same git commits as the first run and will not include any changes. So we have to make a PR and then force circleci to run the binary workflow instead of the normal tests. The above PR is an example of how to do this; essentially you copy-paste the binarybuilds workflow steps into the default workflow steps. If you need to point the builder repo to a different commit then you'd need to change https://github.com/pytorch/pytorch/blob/master/.circleci/scripts/binary_checkout.sh#L42-L45 to checkout what you want.

## How to test changes to the binaries via .circleci

Writing PRs that test the binaries is annoying, since the default circleci jobs that run on PRs are not the jobs that you want to run. Likely, changes to the binaries will touch something under .circleci/ and require that .circleci/config.yml be regenerated (.circleci/config.yml controls all .circleci behavior, and is generated using ```.circleci/regenerate.sh``` in python 3.7). But you also need to manually hardcode the binary jobs that you want to test into the .circleci/config.yml workflow, so you should actually make at least two commits, one for your changes and one to temporarily hardcode jobs. See https://github.com/pytorch/pytorch/pull/22928 as an example of how to do this.

```
# Make your changes
touch .circleci/verbatim-sources/nightly-binary-build-defaults.yml

# Regenerate the yaml, has to be in python 3.7
.circleci/regenerate.sh

# Make a commit
git add .circleci *
git commit -m "My real changes"
git push origin my_branch

# Now hardcode the jobs that you want in the .circleci/config.yml workflows section
# Also eliminate ensure-consistency and should_run_job checks
# e.g. https://github.com/pytorch/pytorch/commit/2b3344bfed8772fe86e5210cc4ee915dee42b32d

# Make a commit you won't keep
git add .circleci
git commit -m "[DO NOT LAND] testing binaries for above changes"
git push origin my_branch

# Now you need to make some changes to the first commit.
git rebase -i HEAD~2 # mark the first commit as 'edit'

# Make the changes
touch .circleci/verbatim-sources/nightly-binary-build-defaults.yml
.circleci/regenerate.sh

# Ammend the commit and recontinue
git add .circleci
git commit --amend
git rebase --continue

# Update the PR, need to force since the commits are different now
git push origin my_branch --force
```

The advantage of this flow is that you can make new changes to the base commit and regenerate the .circleci without having to re-write which binary jobs you want to test on. The downside is that all updates will be force pushes.

## How to build a binary locally

### Linux

You can build Linux binaries locally easily using docker. 

```
# Run the docker
# Use the correct docker image, soumith/conda-cuda used here as an example
#
# -v path/to/foo:path/to/bar makes path/to/foo on your local machine (the
#    machine that you're running the command on) accessible to the docker
#    container at path/to/bar. So if you then run `touch path/to/bar/baz`
#    in the docker container then you will see path/to/foo/baz on your local
#    machine. You could also clone the pytorch and builder repos in the docker.
#
# If you're building a CUDA binary then use `nvidia-docker run` instead, see below.
#
# If you know how, add ccache as a volume too and speed up everything
docker run \
    -v your/pytorch/repo:/pytorch \
    -v your/builder/repo:/builder \
    -v where/you/want/packages/to/appear:/final_pkgs \
    -it soumith/conda-cuda /bin/bash
    
# Export whatever variables are important to you. All variables that you'd
# possibly need are in .circleci/scripts/binary_populate_env.sh
# You should probably always export at least these 3 variables
export PACKAGE_TYPE=conda
export DESIRED_PYTHON=3.6
export DESIRED_CUDA=cpu

# Call the entrypoint
# `|& tee foo.log` just copies all stdout and stderr output to foo.log
# The builds generate lots of output so you probably need this when 
# building locally.
/builder/conda/build_pytorch.sh |& tee build_output.log
```

**Building CUDA binaries on docker**

To build a CUDA binary you need to use `nvidia-docker run` instead of just `docker run` (or you can manually pass `--runtime=nvidia`). This adds some needed libraries and things to build CUDA stuff. 

You can build CUDA binaries on CPU only machines, but you can only run CUDA binaries on CUDA machines. This means that you can build a CUDA binary on a docker on your laptop if you so choose (though it’s gonna take a loong time).

For Facebook employees, ask about beefy machines that have docker support and use those instead of your laptop; it will be 5x as fast.

### MacOS

There’s no easy way to generate reproducible hermetic MacOS environments. If you have a Mac laptop then you can try emulating the .circleci environments as much as possible, but you probably have packages in /usr/local/, possibly installed by brew, that will probably interfere with the build. If you’re trying to repro an error on a Mac build in .circleci and you can’t seem to repro locally, then my best advice is actually to iterate on .circleci    :/

But if you want to try, then I’d recommend

```
# Create a new terminal
# Clear your LD_LIBRARY_PATH and trim as much out of your PATH as you 
# know how to do

# Install a new miniconda
# First remove any other python or conda installation from your PATH
# Always install miniconda 3, even if building for Python <3
new_conda="~/my_new_conda"
conda_sh="$new_conda/install_miniconda.sh"
curl -o "$conda_sh" https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
chmod +x "$conda_sh"
"$conda_sh" -b -p "$MINICONDA_ROOT"
rm -f "$conda_sh"
export PATH="~/my_new_conda/bin:$PATH"

# Create a clean python env
# All MacOS builds use conda to manage the python env and dependencies
# that are built with, even the pip packages
conda create -yn binary python=2.7
conda activate binary

# Export whatever variables are important to you. All variables that you'd
# possibly need are in .circleci/scripts/binary_populate_env.sh
# You should probably always export at least these 3 variables
export PACKAGE_TYPE=conda
export DESIRED_PYTHON=3.6
export DESIRED_CUDA=cpu

# Call the entrypoint you want
path/to/builder/wheel/build_wheel.sh
```

N.B. installing a brand new miniconda is important. This has to do with how conda installations work. See the “General Python” section above, but tldr; is that 

1. You make the ‘conda’ command accessible by prepending `path/to/conda_root/bin` to your PATH.
2. You make a new env and activate it, which then also gets prepended to your PATH. Now you have `path/to/conda_root/envs/new_env/bin:path/to/conda_root/bin:$PATH` 
3. Now say you (or some code that you ran) call python executable `foo` 
    1. if you installed `foo` in `new_env`, then `path/to/conda_root/envs/new_env/bin/foo` will get called, as expected.
    2. But if you forgot to installed `foo` in `new_env` but happened to previously install it in your root conda env (called ‘base’), then unix/linux will still find `path/to/conda_root/bin/foo` . This is dangerous, since `foo` can be a different version than you want; `foo` can even be for an incompatible python version!

Newer conda versions and proper python hygeine can prevent this, but just install a new miniconda to be safe.

### Windows

Maybe @peterjc123 can fill this section in.

