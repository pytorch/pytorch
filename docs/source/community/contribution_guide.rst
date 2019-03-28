PyTorch Contribution Guide
==========================

PyTorch is a GPU-accelerated Python tensor computation package for
building deep neural networks built on tape-based autograd systems.

The PyTorch Contribution Process
--------------------------------

The PyTorch organization is governed by `PyTorch
Governance </docs/community/governance.html>`__.

The PyTorch development process involves a healthy amount of open
discussions between the core development team and the community.

PyTorch operates similar to most open source projects on GitHub.
However, if you've never contributed to an open source project before,
here is the basic process.

-  **Figure out what you're going to work on.** The majority of open
   source contributions come from people scratching their own itches.
   However, if you don't know what you want to work on, or are just
   looking to get more acquainted with the project, here are some tips
   for how to find appropriate tasks:

   -  Look through the `issue
      tracker <https://github.com/pytorch/pytorch/issues/>`__ and see if
      there are any issues you know how to fix. Issues that are
      confirmed by other contributors tend to be better to investigate.
      We also maintain some labels for issues which are likely to be
      good for new people, e.g., **bootcamp** and **1hr**, although
      these labels are less well maintained.
   -  Join us on Slack and let us know you're interested in getting to
      know PyTorch. We're very happy to help out researchers and
      partners get up to speed with the codebase.

-  **Figure out the scope of your change and reach out for design
   comments on a GitHub issue if it's large.** The majority of pull
   requests are small; in that case, no need to let us know about what
   you want to do, just get cracking. But if the change is going to be
   large, it's usually a good idea to get some design comments about it
   first.

   -  If you don't know how big a change is going to be, we can help you
      figure it out! Just post about it on issues or Slack.
   -  Some feature additions are very standardized; for example, lots of
      people add new operators or optimizers to PyTorch. Design
      discussion in these cases boils down mostly to, “Do we want this
      operator/optimizer?” Giving evidence for its utility, e.g., usage
      in peer reviewed papers, or existence in other frameworks, helps a
      bit when making this case.
   -  Core changes and refactors can be quite difficult to coordinate,
      as the pace of development on PyTorch master is quite fast.
      Definitely reach out about fundamental or cross-cutting changes;
      we can often give guidance about how to stage such changes into
      more easily reviewable pieces.

-  **Code it out!**

   -  See the technical guide for advice for working with PyTorch in a
      technical form.

-  **Open a pull request.**

   -  If you are not ready for the pull request to be reviewed, tag it
      with [WIP]. We will ignore it when doing review passes. If you are
      working on a complex change, it's good to start things off as WIP,
      because you will need to spend time looking at CI results to see
      if things worked out or not.
   -  Find an appropriate reviewer for your change. We have some folks
      who regularly go through the PR queue and try to review
      everything, but if you happen to know who the maintainer for a
      given subsystem affected by your patch is, feel free to include
      them directly on the pull request. You can learn more about this
      structure at PyTorch Subsystem Ownership.

-  **Iterate on the pull request until it's accepted!**

   -  We'll try our best to minimize the number of review roundtrips and
      block PRs only when there are major issues. For the most common
      issues in pull requests, take a look at `Common Mistakes <#common-mistakes-to-avoid>`__.
   -  Once a pull request is accepted and CI is passing, there is
      nothing else you need to do; we will merge the PR for you.

Getting Started
---------------

Proposing new features
~~~~~~~~~~~~~~~~~~~~~~

New feature ideas are best discussed on a specific issue. Please include
as much information as you can, any accompanying data, and your proposed
solution. The PyTorch team and community frequently reviews new issues
and comments where they think they can help. If you feel confident in
your solution, go ahead and implement it.

Reporting Issues
~~~~~~~~~~~~~~~~

If you've identified an issue, first search through the `list of
existing issues <https://github.com/pytorch/pytorch/issues>`__ on the
repo. If you are unable to find a similar issue, then create a new one.
Supply as much information you can to reproduce the problematic
behavior. Also, include any additional insights like the behavior you
expect.

Implementing Features or Fixing Bugs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to fix a specific issue, it's best to comment on the
individual issue with your intent. However, we do not lock or assign
issues except in cases where we have worked with the developer before.
It's best to strike up a conversation on the issue and discuss your
proposed solution. The PyTorch team can provide guidance that saves you
time.

Issues that are labeled first-new-issue, low, or medium priority provide
the best entrance point are great places to start.

Adding Tutorials
~~~~~~~~~~~~~~~~

A great deal of the tutorials on `pytorch.org <http://pytorch.org/>`__
come from the community itself and we welcome additional contributions.
To learn more about how to contribute a new tutorial you can learn more
here: `PyTorch.org Tutorial Contribution Guide on
Github <https://github.com/pytorch/tutorials/#contributing>`__

Improving Documentation & Tutorials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We aim to produce high quality documentation and tutorials. On rare
occasions that content includes typos or bugs. If you find something you
can fix, send us a pull request for consideration.

Take a look at the `Documentation <#on-documentation>`__ section to learn how our system
works.

Participating in online discussions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can find active discussions happening on the PyTorch Discussion
`forum <https://discuss.pytorch.org/>`__.

Submitting pull requests to fix open issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can view a list of all open issues
`here <https://github.com/pytorch/pytorch/issues>`__. Commenting on an
issue is a great way to get the attention of the team. From here you can
share your ideas and how you plan to resolve the issue.

For more challenging issues, the team will provide feedback and
direction for how to best solve the issue.

If you're not able to fix the issue itself, commenting and sharing
whether you can reproduce the issue can be useful for helping the team
identify problem areas.

Reviewing open pull requests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We appreciate your help reviewing and commenting on pull requests. Our
team strives to keep the number of open pull requests at a manageable
size, we respond quickly for more information if we need it, and we
merge PRs that we think are useful. However, due to the high level of
interest, additional eyes on pull requests is appreciated.

Improving code readability
~~~~~~~~~~~~~~~~~~~~~~~~~~

Improve code readability helps everyone. It is often better to submit a
small number of pull requests that touch few files versus a large pull
request that touches many files. Starting a discussion in the PyTorch
forum `here <https://discuss.pytorch.org/>`__ or on an issue related to
your improvement is the best way to get started.

Adding test cases to make the codebase more robust
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Additional test coverage is appreciated.

Promoting PyTorch
~~~~~~~~~~~~~~~~~

Your use of PyTorch in your projects, research papers, write ups, blogs,
or general discussions around the internet helps to raise awareness for
PyTorch and our growing community. Please reach out to
`pytorch-marketing@fb.com <http://mailto:pytorch-marketing@fb.com/>`__
for marketing support.

Triaging issues
~~~~~~~~~~~~~~~

If you feel that an issue could benefit from a particular tag or level
of complexity comment on the issue and share your opinion. If an you
feel an issue isn't categorized properly comment and let the team know.

About open source development
-----------------------------

If this is your first time contributing to an open source project, some
aspects of the development process may seem unusual to you.

-  **There is no way to “claim” issues.** People often want to “claim”
   an issue when they decide to work on it, to ensure that there isn't
   wasted work when someone else ends up working on it. This doesn't
   really work too well in open source, since someone may decide to work
   on something, and end up not having time to do it. Feel free to give
   information in an advisory fashion, but at the end of the day, we
   will take running code and rough consensus.
-  **There is a high bar for new functionality that is added.** Unlike
   in a corporate environment, where the person who wrote code
   implicitly “owns” it and can be expected to take care of it in the
   beginning of its lifetime, once a pull request is merged into an open
   source project, it immediately becomes the collective responsibility
   of all maintainers on the project. When we merge code, we are saying
   that we, the maintainers, are able to review subsequent changes and
   make a bugfix to the code. This naturally leads to a higher standard
   of contribution.

Common Mistakes To Avoid
------------------------

-  **Did you add tests?** (Or if the change is hard to test, did you
   describe how you tested your change?)

   -  We have a few motivations for why we ask for tests:

      1. to help us tell if we break it later
      2. to help us tell if the patch is correct in the first place
         (yes, we did review it, but as Knuth says, “beware of the
         following code, for I have not run it, merely proven it
         correct”)

   -  When is it OK not to add a test? Sometimes a change can't be
      conveniently tested, or the change is so obviously correct (and
      unlikely to be broken) that it's OK not to test it. On the
      contrary, if a change is seems likely (or is known to be likely)
      to be accidentally broken, it's important to put in the time to
      work out a testing strategy.

-  **Is your PR too long?**

   -  It's easier for us to review and merge small PRs. Difficulty of
      reviewing a PR scales nonlinearly with its size.
   -  When is it OK to submit a large PR? It helps a lot if there was a
      corresponding design discussion in an issue, with sign off from
      the people who are going to review your diff. We can also help
      give advice about how to split up a large change into individually
      shippable parts. Similarly, it helps if there is a complete
      description of the contents of the PR: it's easier to review code
      if we know what's inside!

-  **Comments for subtle things?** In cases where behavior of your code
   is nuanced, please include extra comments and documentation to allow
   us to better understand the intention of your code.
-  **Did you add a hack?** Sometimes a hack is the right answer. But
   usually we will have to discuss it.
-  **Do you want to touch a very core component?** In order to prevent
   major regressions, pull requests that touch core components receive
   extra scrutiny. Make sure you've discussed your changes with the team
   before undertaking major changes.
-  **Want to add a new feature?** If you want to add new features,
   comment your intention on the related issue. Our team tries to
   comment on and provide feedback to the community. It's better to have
   an open discussion with the team and the rest of the community prior
   to building new features. This helps us stay aware of what you're
   working on and increases the chance that it'll be merged.
-  **Did you touch unrelated code to the PR?** To aid in code review,
   please only include files in your pull request that are directly
   related to your changes.

Frequently asked questions

-  **How can I contribute as a reviewer?** There is lots of value if
   community developer reproduce issues, try out new functionality, or
   otherwise help us identify or troubleshoot issues. Commenting on
   tasks or pull requests with your enviroment details is helpful and
   appreciated.
-  **CI tests failed, what does it mean?** Maybe you need to merge with
   master or rebase with latest changes. Pushing your changes should
   re-trigger CI tests. If the tests persist, you'll want to trace
   through the error messages and resolve the related issues.
-  **What are the most high risk changes?** Anything that touches build
   configuration is an risky area. Please avoid changing these unless
   you've had a discussion with the team beforehand.
-  **Hey, a commit showed up on my branch, what's up with that?**
   Sometimes another community member will provide a patch or fix to
   your pull request or branch. This is often needed for getting CI tests
   to pass.

On Documentation
----------------

Python Docs
~~~~~~~~~~~

PyTorch documentation is generated from python source using
`Sphinx <http://www.sphinx-doc.org/en/master/>`__. Generated HTML is
copied to the docs folder in the master branch of
`pytorch.github.io <https://github.com/pytorch/pytorch.github.io/tree/master/docs>`__,
and is served via GitHub pages.

-  Site: http://pytorch.org/docs
-  GitHub: http://github.com/pytorch/pytorch/docs
-  Served from:
   `https://github.com/pytorch/pytorch.github.io/tree/master/doc <https://github.com/pytorch/pytorch.github.io/tree/master/docs>`__

C++ Docs
~~~~~~~~

For C++ code we use Doxygen to generate the content files. The C++ docs
are built on a special server and the resulting files are copied to the
https://github.com/pytorch/cppdocs repo, and are served from GitHub
pages.

-  Site: http://pytorch.org/cppdocs
-  GitHub: https://github.com/pytorch/pytorch/tree/master/docs/cpp
-  Served from: https://github.com/pytorch/cppdocs

Tutorials
---------

PyTorch tutorials are documents used to help understand using PyTorch to
accomplish specific tasks or to understand more holistic concepts.
Tutorials are built using
`Sphinx-Gallery <https://sphinx-gallery.readthedocs.io/en/latest/index.html>`__
from executable python sources files, or from restructured-text (rst)
files.

-  Site: http://pytorch.org/tutorials
-  GitHub: http://github.com/pytorch/tutorials

Tutorials Build Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For tutorials, `pull
requests <https://github.com/pytorch/tutorials/pulls>`__ trigger a
rebuild the entire site using CircleCI to test the effects of the
change. This build is sharded into 9 worker builds and takes around 40
minutes total. At the same time, we do a Netlify build using *make
html-noplot*, which builds the site without rendering the notebook
output into pages for quick review.

After a PR is accepted, the site is rebuilt and deployed from CircleCI.

Contributing a new Tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`PyTorch.org Tutorial Contribution
Guide <https://github.com/pytorch/tutorials/#contributing>`__

Code Style
~~~~~~~~~~

**Python style**

**C++ style**

Submitting a Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~

PyTorch development happens publicly on our Github repo.

To have your feature or fix added to PyTorch, please submit a Pull
Request.

Running Tests
~~~~~~~~~~~~~

Show examples for running all tests, just one individual...

Technical Process
-----------------

Developing PyTorch
~~~~~~~~~~~~~~~~~~

To develop PyTorch on your machine, here are some tips:

1. Uninstall all existing PyTorch installs:

::

    conda uninstall pytorch
    pip uninstall torch
    pip uninstall torch # run this command twice

2. Clone a copy of PyTorch from source:

::

    git clone https://github.com/pytorch/pytorch
    cd pytorch

3. Install PyTorch in ``build develop`` mode:

A full set of instructions on installing PyTorch from source is here:
https://github.com/pytorch/pytorch#from-source

The change you have to make is to replace

::

    python setup.py install

with

::

    python setup.py build develop

This is especially useful if you are only changing Python files.

This mode will symlink the Python files from the current local source
tree into the Python install.

Hence, if you modify a Python file, you do not need to reinstall PyTorch
again and again.

For example:

-  Install local PyTorch in ``build develop`` mode
-  modify your Python file ``torch/__init__.py`` (for example)
-  test functionality
-  modify your Python file ``torch/__init__.py``
-  test functionality
-  modify your Python file ``torch/__init__.py``
-  test functionality

You do not need to repeatedly install after modifying Python files.

In case you want to reinstall, make sure that you uninstall PyTorch
first by running ``pip uninstall torch`` and ``python setup.py clean``.
Then you can install in ``build develop`` mode again.

Codebase structure
------------------

-  `c10 <https://github.com/pytorch/pytorch/blob/master/c10>`__ - Core
   library files that work everywhere, both server and mobile. We are
   slowly moving pieces from
   `ATen/core <https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/core>`__
   here. This library is intended only to contain essential
   functionality, and appropriate to use in settings where binary size
   matters. (But you'll have a lot of missing functionality if you try
   to use it directly.)
-  `aten <https://github.com/pytorch/pytorch/blob/master/aten>`__ - C++
   tensor library for PyTorch (no autograd support)

   -  `src <https://github.com/pytorch/pytorch/blob/master/aten/src>`__

      -  `TH <https://github.com/pytorch/pytorch/blob/master/aten/src/TH>`__
         `THC <https://github.com/pytorch/pytorch/blob/master/aten/src/THC>`__
         `THNN <https://github.com/pytorch/pytorch/blob/master/aten/src/THNN>`__
         `THCUNN <https://github.com/pytorch/pytorch/blob/master/aten/src/THCUNN>`__
         - Legacy library code from the original Torch. Try not to add
         things here; we're slowly porting these to
         `native <https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native>`__.

         -  generic - Contains actual implementations of operators,
            parametrized over ``scalar_t``. Files here get compiled N
            times per supported scalar type in PyTorch.

      -  `ATen <https://github.com/pytorch/pytorch/blob/master/aten/src/ATen>`__

         -  `core <https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/core>`__
            - Core functionality of ATen. This is migrating to top-level
            c10 folder.
         -  `native <https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native>`__
            - Modern implementations of operators. If you want to write
            a new operator, here is where it should go. Most CPU
            operators go in the top level directory, except for
            operators which need to be compiled specially; see cpu
            below.

            -  `cpu <https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cpu>`__
               - Not actually CPU implementations of operators, but
               specifically implementations which are compiled with
               processor-specific instructions, like AVX. See the
               `README <https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cpu/README.md>`__
               for more details.
            -  `cuda <https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda>`__
               - CUDA implementations of operators.
            -  `sparse <https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/sparse>`__
               - CPU and CUDA implementations of COO sparse tensor
               operations
            -  `mkl <https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/mkl>`__
               `mkldnn <https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/mkldnn>`__
               `miopen <https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/miopen>`__
               `cudnn <https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cudnn>`__

               -  implementations of operators which simply bind to some
                  backend library.

-  `torch <https://github.com/pytorch/pytorch/blob/master/torch>`__ -
   The actual PyTorch library. Everything that is not in
   `csrc <https://github.com/pytorch/pytorch/blob/master/torch/csrc>`__
   is a Python module, following the PyTorch Python frontend module
   structure.

   -  `csrc <https://github.com/pytorch/pytorch/blob/master/torch/csrc>`__
      - C++ files composing the PyTorch library. Files in this directory
      tree are a mix of Python binding code, and C++ heavy lifting.
      Consult ``setup.py`` for the canonical list of Python binding
      files; conventionally, they are often prefixed with ``python_``.

      -  `jit <https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit>`__
         - Compiler and frontend for TorchScript JIT frontend.
      -  `autograd <https://github.com/pytorch/pytorch/blob/master/torch/csrc/autograd>`__
         - Implementation of reverse-mode automatic differentiation.
      -  `api <https://github.com/pytorch/pytorch/blob/master/torch/csrc/api>`__
         - The PyTorch C++ frontend.
      -  `distributed <https://github.com/pytorch/pytorch/blob/master/torch/csrc/distributed>`__
         - Distributed training support for PyTorch.

-  `tools <https://github.com/pytorch/pytorch/blob/master/tools>`__ -
   Code generation scripts for the PyTorch library. See
   `README <https://github.com/pytorch/pytorch/blob/master/tools/README.md>`__
   of this directory for more details.
-  `test <https://github.com/pytorch/pytorch/blob/master/tests>`__ -
   Python unit tests for PyTorch Python frontend.

   -  `test\_torch.py <https://github.com/pytorch/pytorch/blob/master/test/test_torch.py>`__
      - Basic tests for PyTorch functionality.
   -  `test\_autograd.py <https://github.com/pytorch/pytorch/blob/master/test/test_autograd.py>`__
      - Tests for non-NN automatic differentiation support.
   -  `test\_nn.py <https://github.com/pytorch/pytorch/blob/master/test/test_nn.py>`__
      - Tests for NN operators and their automatic differentiation.
   -  `test\_jit.py <https://github.com/pytorch/pytorch/blob/master/test/test_jit.py>`__
      - Tests for the JIT compiler and TorchScript.
   -  ...
   -  `cpp <https://github.com/pytorch/pytorch/blob/master/test/cpp>`__
      - C++ unit tests for PyTorch C++ frontend.
   -  `expect <https://github.com/pytorch/pytorch/blob/master/test/expect>`__
      - Automatically generated "expect" files which are used to compare
      against expected output.
   -  `onnx <https://github.com/pytorch/pytorch/blob/master/test/onnx>`__
      - Tests for ONNX export functionality, using both PyTorch and
      Caffe2.

-  `caffe2 <https://github.com/pytorch/pytorch/blob/master/caffe2>`__ -
   The Caffe2 library.

   -  `core <https://github.com/pytorch/pytorch/blob/master/caffe2/core>`__
      - Core files of Caffe2, e.g., tensor, workspace, blobs, etc.
   -  `operators <https://github.com/pytorch/pytorch/blob/master/caffe2/operators>`__
      - Operators of Caffe2.
   -  `python <https://github.com/pytorch/pytorch/blob/master/caffe2/python>`__
      - Python bindings to Caffe2.
   -  ...

Unit Testing
------------

PyTorch's testing is located under ``test/``. Run the entire test suite
with

::

    python test/run_test.py

or run individual test files, like ``python test/test_nn.py``, for
individual test suites.

Better local unit tests with pytest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We don't officially support ``pytest``, but it works well with our
``unittest`` tests and offers a number of useful features for local
developing. Install it via ``pip install pytest``.

If you want to just run tests that contain a specific substring, you can
use the ``-k`` flag:

::

    pytest test/test_nn.py -k Loss -v

The above is an example of testing a change to Loss functions: this
command runs tests such as ``TestNN.test_BCELoss``\ and
``TestNN.test_MSELoss`` and can be useful to save keystrokes.

Writing documentation
---------------------

PyTorch uses `Google
style <http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`__
for formatting docstrings. Length of line inside docstrings block must
be limited to 80 characters to fit into Jupyter documentation popups.

For C++ documentation (https://pytorch.org/cppdocs), we use
`Doxygen <http://www.doxygen.nl/>`__ and then convert it to
`Sphinx <http://www.sphinx-doc.org/>`__ via
`Breathe <https://github.com/michaeljones/breathe>`__
and\ `Exhale <https://github.com/svenevs/exhale>`__. Check the `Doxygen
reference <http://www.stack.nl/~dimitri/doxygen/manual/index.html>`__
for more information on the documentation syntax. To build the
documentation locally, ``cd`` into ``docs/cpp`` and then ``make html``.

We run Doxygen in CI (Travis) to verify that you do not use invalid
Doxygen commands. To run this check locally, run ``./check-doxygen.sh``
from inside ``docs/cpp``.

Managing multiple build trees
-----------------------------

One downside to using ``python setup.py develop`` is that your
development version of PyTorch will be installed globally on your
account (e.g., if you run ``import torch`` anywhere else, the
development version will be used.

If you want to manage multiple builds of PyTorch, you can make use of
`conda environments <https://conda.io/docs/using/envs.html>`__ to
maintain separate Python package environments, each of which can be tied
to a specific build of PyTorch. To set one up:

::

    conda create -n pytorch-myfeaturesource activate pytorch-myfeature# if you run python now, torch will NOT be installed
    python setup.py build develop

C++ Development tips
--------------------

If you are working on the C++ code, there are a few important things
that you will want to keep in mind:

1. How to rebuild only the code you are working on.
2. How to make rebuilds in the absence of changes go faster.

Build only what you need.
~~~~~~~~~~~~~~~~~~~~~~~~~

``python setup.py build`` will build everything, but since our build
system is not very optimized for incremental rebuilds, this will
actually be very slow. Far better is to only request rebuilds of the
parts of the project you are working on:

-  Working on the Python bindings? Run ``python setup.py develop`` to
   rebuild (NB: no ``build`` here!)
-  Working on ``torch/csrc`` or ``aten``? Run
   ``python setup.py rebuild_libtorch`` to rebuild and avoid having to
   rebuild other dependent libraries we depend on.
-  Working on one of the other dependent libraries? The other valid
   targets are listed in ``dep_libs`` in ``setup.py``. prepend
   ``build_`` to get a target, and run as e.g.
   ``python setup.py build_gloo``.
-  Working on a test binary? Run
   ``(cd build && ninja bin/test_binary_name)`` to rebuild only that
   test binary (without rerunning cmake). (Replace ``ninja`` with
   ``make`` if you don't have ninja installed).

On the initial build, you can also speed things up with the environment
variables ``DEBUG`` and ``NO_CUDA``.

-  ``DEBUG=1`` will enable debug builds (-g -O0)
-  ``REL_WITH_DEB_INFO=1`` will enable debug symbols with optimizations
   (-g -O3)
-  ``NO_CUDA=1`` will disable compiling CUDA (in case you are developing
   on something not CUDA related), to save compile time.

For example:

::

    NO_CUDA=1 DEBUG=1 python setup.py build develop

Make sure you continue to pass these flags on subsequent builds.

Code completion and IDE support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using ``python setup.py develop``, PyTorch will generate a
``compile_commands.json`` file that can be used by many editors to
provide command completion and error highlighting for PyTorch's C++
code. You need to ``pip install ninja`` to generate accurate information
for the code in ``torch/csrc``. More information at:

-  https://sarcasm.github.io/notes/dev/compilation-database.html

Make no-op build fast.
~~~~~~~~~~~~~~~~~~~~~~

Use Ninja
~~~~~~~~~

Python ``setuptools`` is pretty dumb, and always rebuilds every C file
in a project. If you install the ninja build system with
``pip install ninja``, then PyTorch will use it to track dependencies
correctly. If PyTorch was already built, you will need to run
``python setup.py clean`` once after installing ninja for builds to
succeed.

Use CCache
~~~~~~~~~~

Even when dependencies are tracked with file modification, there are
many situations where files get rebuilt when a previous compilation was
exactly the same.

Using ccache in a situation like this is a real time-saver. However, by
default, ccache does not properly support CUDA stuff, so here are the
instructions for installing a custom ccache fork that has CUDA support:

::

    # install and export ccacheif ! ls ~/ccache/bin/ccachethen
        sudo apt-get update
        sudo apt-get install -y automake autoconf
        sudo apt-get install -y asciidoc
        mkdir -p ~/ccache
        pushd /tmp
        rm -rf ccache
        git clone https://github.com/colesbury/ccache -b ccbin
        pushd ccache
        ./autogen.sh
        ./configure
        make install prefix=~/ccache
        popdpopd

        mkdir -p ~/ccache/lib
        mkdir -p ~/ccache/cuda
        ln -s ~/ccache/bin/ccache ~/ccache/lib/cc
        ln -s ~/ccache/bin/ccache ~/ccache/lib/c++
        ln -s ~/ccache/bin/ccache ~/ccache/lib/gcc
        ln -s ~/ccache/bin/ccache ~/ccache/lib/g++
        ln -s ~/ccache/bin/ccache ~/ccache/cuda/nvcc

        ~/ccache/bin/ccache -M 25Gifiexport PATH=~/ccache/lib:$PATHexport CUDA_NVCC_EXECUTABLE=~/ccache/cuda/nvcc

CUDA Development tips
---------------------

If you are working on the CUDA code, here are some useful CUDA debugging
tips:

1. ``CUDA_DEVICE_DEBUG=1`` will enable CUDA device function debug
   symbols (``-g -G``). This will be particularly helpful in debugging
   device code. However, it will slow down the build process for about
   50% (compared to only ``DEBUG=1``), so use wisely.
2. ``cuda-gdb`` and ``cuda-memcheck`` are your best CUDA debugging
   friends. Unlike\ ``gdb``, ``cuda-gdb`` can display actual values in a
   CUDA tensor (rather than all zeros).

Hope this helps, and thanks for considering to contribute.

Windows development tips
------------------------

Occasionally, you will write a patch which works on Linux, but fails CI
on Windows. There are a few aspects in which MSVC (the Windows compiler
toolchain we use) is stricter than Linux, which are worth keeping in
mind when fixing these problems.

1. Symbols are NOT exported by default on Windows; instead, you have to
   explicitly mark a symbol as exported/imported in a header file with
   ``__declspec(dllexport)`` / ``__declspec(dllimport)``. We have
   codified this pattern into a set of macros which follow the
   convention ``*_API``, e.g., ``CAFFE2_API`` inside Caffe2 and ATen.
   (Every separate shared library needs a unique macro name, because
   symbol visibility is on a per shared library basis. See
   c10/macros/Macros.h for more details.) The upshot is if you see an
   "unresolved external" error in your Windows build, this is probably
   because you forgot to mark a function with ``*_API``. However, there
   is one important counterexample to this principle: if you want a
   *templated* function to be instantiated at the call site, do NOT mark
   it with ``*_API`` (if you do mark it, you'll have to explicitly
   instantiate all of the specializations used by the call sites.)
2. If you link against a library, this does not make its dependencies
   transitively visible. You must explicitly specify a link dependency
   against every library whose symbols you use. (This is different from
   Linux where in most environments, transitive dependencies can be used
   to fulfill unresolved symbols.)
3. If you have a Windows box (we have a few on EC2 which you can request
   access to) and you want to run the build, the easiest way is to just
   run ``.jenkins/pytorch/win-build.sh``. If you need to rebuild, run
   ``REBUILD=1 .jenkins/pytorch/win-build.sh`` (this will avoid blowing
   away your Conda environment.)

Even if you don't know anything about MSVC, you can use cmake to build
simple programs on Windows; this can be helpful if you want to learn
more about some peculiar linking behavior by reproducing it on a small
example. Here's a simple example cmake file that defines two dynamic
libraries, one linking with the other:

::

    project(myproject CXX)set(CMAKE_CXX_STANDARD 11)add_library(foo SHARED foo.cpp)add_library(bar SHARED bar.cpp)# NB: don't forget to __declspec(dllexport) at least one symbol from foo,# otherwise foo.lib will not be created.target_link_libraries(bar PUBLIC foo)

You can build it with:

::

    mkdir buildcd build
    cmake ..
    cmake --build .

Known MSVC (and MSVC with NVCC) bugs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PyTorch codebase sometimes likes to use exciting C++ features, and
these exciting features lead to exciting bugs in Windows compilers. To
add insult to injury, the error messages will often not tell you which
line of code actually induced the erroring template instantiation. We've
found the most effective way to debug these problems is to carefully
read over diffs, keeping in mind known bugs in MSVC/NVCC. Here are a few
well known pitfalls and workarounds:

-  This is not actually a bug per se, but in general, code generated by
   MSVC is more sensitive to memory errors; you may have written some
   code that does a use-after-free or stack overflows; on Linux the code
   might work, but on Windows your program will crash. ASAN may not
   catch all of these problems: stay vigilant to the possibility that
   your crash is due to a real memory problem.
-  (NVCC) ``c10::optional`` does not work when used from device code.
   Don't use it from kernels. Upstream issue:
   https://github.com/akrzemi1/Optional/issues/58 and our local issue
   #10329.
-  ``constexpr`` generally works less well on MSVC.

   -  The idiom ``static_assert(f() == f())`` to test if ``f`` is
      constexpr does not work; you'll get "error C2131: expression did
      not evaluate to a constant". Don't use these asserts on Windows.
      (Example: ``c10/util/intrusive_ptr.h``)

-  (NVCC) Code you access inside a ``static_assert`` will eagerly be
   evaluated as if it were device code, and so you might get an error
   that the code is "not accessible".

::

    class A {
      static A singleton_;
      static constexpr inline A* singleton() {
        return &singleton_;
      }
    };static_assert(std::is_same(A*, decltype(A::singleton()))::value, "hmm");

-  The compiler will run out of heap space if you attempt to compile
   files that are too large. Splitting such files into separate files
   helps. (Example: ``THTensorMath``, ``THTensorMoreMath``,
   ``THTensorEvenMoreMath``.)
-  MSVC's preprocessor (but not the standard compiler) has a bug where
   it incorrectly tokenizes raw string literals, ending when it sees a
   ``"``. This causes preprocessor tokens inside the literal like
   an\ ``#endif`` to be incorrectly treated as preprocessor directives.
   See https://godbolt.org/z/eVTIJq as an example.

Running Clang-Tidy
~~~~~~~~~~~~~~~~~~

`Clang-Tidy <https://clang.llvm.org/extra/clang-tidy/index.html>`__ is a
C++ linter and static analysis tool based on the clang compiler. We run
clang-tidy in our CI to make sure that new C++ code is safe, sane and
efficient. See our
`.travis.yml <https://github.com/pytorch/pytorch/blob/master/.travis.yml>`__
file for the simple commands we use for this. To run clang-tidy locally,
follow these steps:

1. Install clang-tidy. First, check if you already have clang-tidy by
   simply writing ``clang-tidy`` in your terminal. If you don't yet have
   clang-tidy, you should be able to install it easily with your package
   manager, e.g. by writing ``apt-get install clang-tidy`` on Ubuntu.
   See `https://apt.llvm.org <https://apt.llvm.org/>`__ for details on
   how to install the latest version. Note that newer versions of
   clang-tidy will have more checks than older versions. In our CI, we
   run clang-tidy-6.0.
2. Use our driver script to run clang-tidy over any changes relative to
   some git revision (you may want to replace ``HEAD~1`` with ``HEAD``
   to pick up uncommitted changes). Changes are picked up based on a
   ``git diff`` with the given revision:

::

    python tools/clang_tidy.py -d build -p torch/csrc --diff 'HEAD~1'

Above, it is assumed you are in the PyTorch root folder.
``path/to/build`` should be the path to where you built PyTorch from
source, e.g. ``build`` in the PyTorch root folder if you used
``setup.py build``. You can use ``-c <clang-tidy-binary>``\ to change
the clang-tidy this script uses. Make sure you have PyYaml installed,
which is in PyTorch's ``requirements.txt``.

Pre-commit Tidy/Linting Hook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use clang-tidy and flake8 to perform additional formatting and
semantic checking of code. We provide a pre-commit git hook for
performing these checks, before a commit is created:

::

    ln -s ../../tools/git-pre-commit .git/hooks/pre-commit

Caffe2 notes
------------

In 2018, we merged Caffe2 into the PyTorch source repository. While the
steady state aspiration is that Caffe2 and PyTorch share code freely, in
the meantime there will be some separation. If you submit a PR to only
PyTorch or only Caffe2 code, CI will only run for the project you
edited. The logic for this is implemented in
``.jenkins/pytorch/dirty.sh`` and ``.jenkins/caffe2/dirty.sh``; you can
look at this to see what path prefixes constitute changes. This also
means if you ADD a new top-level path, or you start sharing code between
projects, you need to modify these files. There are a few "unusual"
directories which, for historical reasons, are Caffe2/PyTorch specific.
Here they are:

-  ``CMakeLists.txt``, ``Makefile``, ``binaries``, ``cmake``, ``conda``,
   ``modules``, ``scripts`` are Caffe2-specific. Don't put PyTorch code
   in them without extra coordination.
-  ``mypy*``, ``requirements.txt``, ``setup.py``, ``test``, ``tools``
   are PyTorch-specific. Don't put Caffe2 code in them without extra
   coordination.
