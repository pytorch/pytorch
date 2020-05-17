On Documentation
----------------

Python Docs
~~~~~~~~~~~

PyTorch documentation is generated from python source using
`Sphinx <https://www.sphinx-doc.org/en/master/>`__. Generated HTML is
copied to the docs folder in the master branch of
`pytorch.github.io <https://github.com/pytorch/pytorch.github.io/tree/master/docs>`__,
and is served via GitHub pages.

-  Site: https://pytorch.org/docs
-  GitHub: https://github.com/pytorch/pytorch/tree/master/docs
-  Served from:
   `https://github.com/pytorch/pytorch.github.io/tree/master/doc <https://github.com/pytorch/pytorch.github.io/tree/master/docs>`__

C++ Docs
~~~~~~~~

For C++ code we use Doxygen to generate the content files. The C++ docs
are built on a special server and the resulting files are copied to the
https://github.com/pytorch/cppdocs repo, and are served from GitHub
pages.

-  Site: https://pytorch.org/cppdocs
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

-  Site: https://pytorch.org/tutorials
-  GitHub: https://github.com/pytorch/tutorials

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
