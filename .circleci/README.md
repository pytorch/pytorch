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

See https://github.com/pytorch/pytorch/issues/17038
