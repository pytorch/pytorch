% PyTorch documentation master file, created by
%  sphinx-quickstart on Fri Dec 23 13:31:47 2016.
%  You can adapt this file completely to your liking, but it should at least
%  contain the root `toctree` directive.

% :github_url: https://github.com/pytorch/pytorch

PyTorch documentation
===================================

PyTorch is an optimized tensor library for deep learning using GPUs and CPUs.

Features described in this documentation are classified by release status:

  *Stable:*  These features will be maintained long-term and there should generally
  be no major performance limitations or gaps in documentation.
  We also expect to maintain backwards compatibility (although
  breaking changes can happen and notice will be given one release ahead
  of time).

  *Beta:*  These features are tagged as Beta because the API may change based on
  user feedback, because the performance needs to improve, or because
  coverage across operators is not yet complete. For Beta features, we are
  committing to seeing the feature through to the Stable classification.
  We are not, however, committing to backwards compatibility.

  *Prototype:*  These features are typically not available as part of
  binary distributions like PyPI or Conda, except sometimes behind run-time
  flags, and are at an early stage for feedback and testing.

```{toctree}
:glob:
:maxdepth: 2

pytorch-api
notes
```

```{toctree}
:glob:
:hidden:
:maxdepth: 2

community/index
C++ <https://docs.pytorch.org/cppdocs/>
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
