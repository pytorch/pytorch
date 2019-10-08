sphinxcontrib-katex
===================

|tests| |docs| |license|


A `Sphinx extension`_ for rendering math in HTML pages.

The extension uses `KaTeX`_ for rendering of math in HTML pages. It is designed
as a replacement for the built-in extension `sphinx.ext.mathjax`_, which uses
`MathJax`_ for rendering.

* **Documentation**: https://sphinxcontrib-katex.readthedocs.io/

* **Download**: https://pypi.org/project/sphinxcontrib-katex/#files

* **Development**: https://github.com/hagenw/sphinxcontrib-katex/

.. _Sphinx extension: http://www.sphinx-doc.org/en/master/extensions.html
.. _MathJax: https://www.mathjax.org
.. _KaTeX: https://khan.github.io/KaTeX/
.. _sphinx.ext.mathjax:
    https://github.com/sphinx-doc/sphinx/blob/master/sphinx/ext/mathjax.py

.. |tests| image:: https://travis-ci.org/hagenw/sphinxcontrib-katex.svg?branch=master
    :target: https://travis-ci.org/hagenw/sphinxcontrib-katex/
    :alt: sphinxcontrib.katex on TravisCI
.. |docs| image:: https://readthedocs.org/projects/sphinxcontrib-katex/badge/
    :target: https://sphinxcontrib-katex.readthedocs.io/
    :alt: sphinxcontrib.katex's documentation on Read the Docs
.. |license| image:: https://img.shields.io/badge/license-MIT-green.svg
    :target: https://github.com/hagenw/sphinxcontrib-katex/blob/master/LICENSE
    :alt: sphinxcontrib.katex's MIT license


Installation
------------

To install ``sphinxcontrib.katex`` into your Python virtual environment run:

.. code-block:: bash

    $ pip install sphinxcontrib-katex

If you want to pre-render the math by running Javascript on your server instead
of running it in the browsers of the users, you have to install ``katex`` on
your server and add it to your path:

.. code-block:: bash

    $ npm install katex
    $ PATH="${PATH}:$(pwd)/node_modules/.bin"


Usage
-----

In ``conf.py`` of your Sphinx project, add the extension with:

.. code-block:: python

    extensions = ['sphinxcontrib.katex']

For enable server side pre-rendering add in addition:

.. code-block:: python

    katex_prerender = True

See the Configuration section for all availble settings.


Configuration
-------------

The behavior of ``sphinxcontrib.katex`` can be changed by configuration
entries in ``conf.py`` of your documentation project. In the following
all configuration entries are listed and their default values are shown.

.. code-block:: python

    katex_css_path = \
        'https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.css'
    katex_js_path = \
        'https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.js'
    katex_autorender_path = \
        'https://cdn.jsdelivr.net/npm/katex@0.10.2/contrib/auto-render.min.js'
    katex_inline = [r'\(', r'\)']
    katex_display = [r'\[', r'\]']
    katex_prerender = False
    katex_options = ''

The specific delimiters written to HTML when math mode is encountered are
controlled by the two lists ``katex_inline`` and ``katex_display``.

If ``katex_prerender`` is set to ``True`` the equations will be pre-rendered on
the server and loading of the page in the browser will be faster.
On your server you must have a ``katex`` executable installed and in your PATH
as described in the Installation section.

The string variable ``katex_options`` allows you to change all available
official `KaTeX rendering options`_, e.g.

.. code-block:: python

    katex_options = r'''{
        displayMode: true,
        macros: {
            "\\RR": "\\mathbb{R}"
        }
    }'''

You can also add `KaTeX auto-rendering options`_ to ``katex_options``, but be
aware that the ``delimiters`` entry should contain the entries of
``katex_inline`` and ``katex_display``.

.. _KaTeX rendering options:
    https://khan.github.io/KaTeX/docs/options.html
.. _KaTeX auto-rendering options:
    https://khan.github.io/KaTeX/docs/autorender.html


LaTeX Macros
------------

Most probably you want to add some of your LaTeX math commands for the
rendering. In KaTeX this is supported by LaTeX macros (``\def``).
You can use the ``katex_options`` configuration setting to add those:

.. code-block:: python

    katex_options = r'''macros: {
            "\\i": "\\mathrm{i}",
            "\\e": "\\mathrm{e}^{#1}",
            "\\vec": "\\mathbf{#1}",
            "\\x": "\\vec{x}",
            "\\d": "\\operatorname{d}\\!{}",
            "\\dirac": "\\operatorname{\\delta}\\left(#1\\right)",
            "\\scalarprod": "\\left\\langle#1,#2\\right\\rangle",
        }'''

The disadvantage of this option is that those macros will be only available in
the HTML based `Sphinx builders`_. If you want to use them in the LaTeX based
builders as well you have to add them as the ``latex_macros`` setting in your
``conf.py`` and specify them using proper LaTeX syntax. Afterwards you can
include them via the ``sphinxcontrib.katex.latex_defs_to_katex_macros``
function into ``katex_options`` and add them to the LaTeX preamble:

.. code-block:: python

    import sphinxcontrib.katex as katex

    latex_macros = r"""
        \def \i                {\mathrm{i}}
        \def \e              #1{\mathrm{e}^{#1}}
        \def \vec            #1{\mathbf{#1}}
        \def \x                {\vec{x}}
        \def \d                {\operatorname{d}\!}
        \def \dirac          #1{\operatorname{\delta}\left(#1\right)}
        \def \scalarprod   #1#2{\left\langle#1,#2\right\rangle}
    """

    # Translate LaTeX macros to KaTeX and add to options for HTML builder
    katex_macros = katex.latex_defs_to_katex_macros(latex_macros)
    katex_options = 'macros: {' + katex_macros + '}'

    # Add LaTeX macros for LATEX builder
    latex_elements = {'preamble': latex_macros}

.. _Sphinx builders: http://www.sphinx-doc.org/en/master/builders.html
