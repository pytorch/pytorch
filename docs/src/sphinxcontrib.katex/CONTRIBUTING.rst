Contributing
------------

If you find errors, omissions, inconsistencies or other things that need
improvement, please create an issue or a pull request at
https://github.com/hagenw/sphinxcontrib-katex/.
Contributions are always welcome!


Development Installation
^^^^^^^^^^^^^^^^^^^^^^^^

Instead of pip-installing the latest release from PyPI, you should get the
newest development version from Github_::

   git clone https://github.com/hagenw/sphinxcontrib-katex.git
   cd sphinxcontrib-katex
   # Create virtual environment
   python setup.py develop

.. _Github: https://github.com/hagenw/sphinxcontrib-katex/

This way, your installation always stays up-to-date, even if you pull new
changes from the Github repository.

If you prefer, you can also replace the last command with::

   pip install -e .

... where ``-e`` stands for ``--editable``.


Building the Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you make changes to the documentation, you can re-create the HTML pages
using Sphinx_.
You can install it and a few other necessary packages with::

   pip install -r docs/requirements.txt

To create the HTML pages, use::

   python -m sphinx docs/ build/sphinx/ -b html

The generated files will be available in the directory ``build/sphinx/``.

It is also possible to automatically check if all links are still valid::

   python -m sphinx docs/ build/sphinx/ -b linkcheck

.. _Sphinx: http://sphinx-doc.org/


Running Tests
^^^^^^^^^^^^^

``sphinxcontrib.katex`` is supposed to work for all versions ``sphinx>=1.6``.
To test that you have to use a stripped down version of the documentation that
is provided in the ``tests/`` folder, as the documentation under ``docs/`` uses
features that are only supported by ``sphinx>=1.8``.

To test that everything works as expected, please execute:

.. code-block:: bash

   python -m sphinx tests/ tests/_build/ -c docs/ -b html -W
   python -m sphinx tests/ tests/_build/ -c docs/ -b latex -W

The same tests are automatically performed by Travis_ once you create a pull
request on Github_.

.. _Travis: https://travis-ci.org/hagenw/sphinxcontrib-katex/


Creating a New Release
^^^^^^^^^^^^^^^^^^^^^^

New releases are made using the following steps:

#. Bump version number in ``sphinxcontrib/katex.py``
#. Update ``NEWS.rst``
#. Commit those changes as "Release x.y.z"
#. Create an (annotated) tag with ``git tag -a x.y.z``
#. Clear the ``dist/`` directory
#. Create a source distribution with ``python setup.py sdist``
#. Create a wheel distribution with ``python setup.py bdist_wheel --universal``
#. Check that both files have the correct content
#. Upload them to PyPI with twine_: ``twine upload dist/*``
#. Push the commit and the tag to Github and `add release notes`_ containing a
   link to PyPI and the bullet points from ``NEWS.rst``
#. Check that the new release was built correctly on RTD_, delete the "stable"
   version and select the new release as default version

.. _twine: https://pypi.org/project/twine/
.. _add release notes: https://github.com/hagenw/sphinxcontrib-katex/tags
.. _RTD: http://readthedocs.org/projects/sphinxcontrib-katex/builds/
