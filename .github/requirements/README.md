### Cached requirements and consolidation of conda and pip installation

At the moment, the installation of conda and pip dependencies happens at
different places in the CI depending at the whim of different
developers, which makes it very challenging to handle issues like
network flakiness or upstream dependency failures gracefully. So, this
center directory is created to gradually include all the conda environment
and pip requirement files that are used to setup CI jobs. Not only it
gives a clear picture of all the dependencies required by different CI
jobs, but it also allows them to be cached properly to improve CI
reliability.

The list of support files are as follows:
* Pip:
  * pip-requirements-macOS.txt. This is used by MacOS build and test jobs to
    setup the pip environment
