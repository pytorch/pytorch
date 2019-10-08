
*************
Configuration
*************

You can configure different parts of the theme.

Project-wide Configuration
==========================

HTML Theme Options
------------------

The theme's project-wide options are defined in the ``pytorch_sphinx_theme/theme.conf``
file of this repository, and can be defined in your project's ``conf.py`` via
``html_theme_options``. For example:

.. code:: python

    html_theme_options = {
        'canonical_url': '',
        'analytics_id': '',
        'logo_only': False,
        'display_version': True,
        'prev_next_buttons_location': 'bottom',
        'style_external_links': False,
        'vcs_pageview_mode': '',
        # Toc options
        'collapse_navigation': True,
        'sticky_navigation': True,
        'navigation_depth': 4,
        'includehidden': True,
        'titles_only': False
    }

The following options are available:

Base options
~~~~~~~~~~~~

* ``canonical_url`` String. This will specify a `canonical url <https://en.wikipedia.org/wiki/Canonical_link_element>`__
  to let search engines know they should give higher ranking to latest version of the docs.
  The url points to the root of the documentation and requires a trailing slash.
* ``analytics_id`` String. Change the Google Analytics ID that is included on pages.
* ``display_version`` Bool. With this disabled, the version number isn't shown at the top of the sidebar.
* ``prev_next_buttons_location`` String. can take the value ``bottom``, ``top``, ``both`` , or ``None``
  and will display the "Next" and "Previous" buttons accordingly.
* ``style_external_links`` Bool. Add an icon next to external links. Defaults to ``False``.
* ``vcs_pageview_mode`` String. Changes how to view files when using `display_github`, `display_gitlab`, etc.
  When using Github or Gitlab this can be: `blob` (default), `edit`, or `raw`,
  on Bitbucket, this can be either: `view` (default) or `edit`.

TOC Options
~~~~~~~~~~~

These effect how we display the Table of Contents in the side bar. You can read more about them here: http://www.sphinx-doc.org/en/stable/templating.html#toctree

* ``collapse_navigation`` Bool. With this enabled, you will lose the ``[+]`` drop downs next to each section in the sidebar.
* ``sticky_navigation`` Bool. This causes the sidebar to scroll with the main page content as you scroll the page.
* ``navigation_depth`` Int. Indicate the max depth of the tree; by default, 4 levels are included;
  set it to -1 to allow unlimited depth.
* ``includehidden`` Bool. Specifies if the sidebar includes toctrees marked with the ``:hidden:`` option
* ``titles_only`` Bool. If True, removes headers within a page from the sidebar.

.. note::

   Setting ``collapse_navigation`` to False and using a high ``navigation_depth``
   can cause projects with many files and a deep file structure to generate HTML files
   that are significantly larger in file size and much longer compilation times.


HTML Context Options
--------------------

TODO.


Page-level Configuration
========================

Pages support metadata that changes how the theme renders.
You can currently add the following:

* ``:github_url:`` This will force the "Edit on GitHub" to the configured URL
* ``:bitbucket_url:`` This will force the "Edit on Bitbucket" to the configured URL
* ``:gitlab_url:`` This will force the "Edit on GitLab" to the configured URL


How the Table of Contents builds
================================

Currently the left menu will build based upon any ``toctree(s)`` defined in your ``index.rst`` file.
It outputs 2 levels of depth, which should give your visitors a high level of access to your
docs. If no toctrees are set the theme reverts to sphinx's usual local toctree.

It's important to note that if you don't follow the same styling for your rST headers across
your documents, the toctree will misbuild, and the resulting menu might not show the correct
depth when it renders.

Also note that by default the table of contents is set with ``includehidden=True``. This allows you
to set a hidden toc in your index file with the `:hidden: <hidden_>`_ property that will allow you
to build a toc without it rendering in your index.

By default, the navigation will "stick" to the screen as you scroll. However if your toc
is vertically too large, it will revert to static positioning. To disable the sticky nav
altogether change the setting in ``conf.py``.

.. _hidden: http://sphinx-doc.org/markup/toctree.html
