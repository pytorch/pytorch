# Benchmark Files

There are two files in this folder:
- dlrm_s_criteo_kaggle.sh
- dlrm_s_pytorch.py

These two files are from the DLRM repository ([repo link](https://github.com/facebookresearch/dlrm)) and were modified from the original
versions found in this repository to help test the ModelReport API on a real internal model.

The purpose of this folder is to just have a record of the experiments that were performed to show that
you could use the ModelReport API on a submodule of a larger model and still get better results.

At a high level, if you want to recreate this, clone the dlrm repo, replace the two files in the repo with the copies in this folder,
and you should be able to recreate the experiments done.

The lines to perform some of the tests (for example using a dynamic vs static config) are commented out, and some may need to be
un-commented to be used.

For the `.sh` file, you may need to manually add an arg as needed to recreate some of the tests, although these are straighforward to
add looking at the list of params and just selecting the correct option.
