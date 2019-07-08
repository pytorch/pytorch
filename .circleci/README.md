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
