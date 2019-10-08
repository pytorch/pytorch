# -*- coding: utf-8 -*-

import subprocess

import sphinxcontrib.katex as katex


# -- GENERAL -------------------------------------------------------------

project = 'sphinxcontrib-katex'
author = 'Hagen Wierstorf'
copyright = '2017-2019, ' + author

needs_sphinx = '1.6'   # minimal sphinx version
extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.viewcode',
        'sphinxcontrib.katex',
]
master_doc = 'index'
source_suffix = '.rst'
exclude_patterns = ['_build']

# The full version, including alpha/beta/rc tags.
# release = version
try:
    release = subprocess.check_output(
            ['git', 'describe', '--tags', '--always'])
    release = release.decode().strip()
except Exception:
    release = '<unknown>'

# Code syntax highlighting style
pygments_style = 'tango'


# -- ACRONYMS AND MATH ---------------------------------------------------
latex_macros = r"""
    \def \x                {\mathbf{x}}
    \def \w                {\omega}
    \def \d                {\operatorname{d}\!}
"""
katex_macros = katex.latex_defs_to_katex_macros(latex_macros)
katex_options = '{macros: {' + katex_macros + '}, strict: false}'
katex_prerender = False


# -- HTML ----------------------------------------------------------------

html_title = project
html_short_title = ""
htmlhelp_basename = project
html_theme_options = {
    'code_font_size': '0.8em',
}


# -- LATEX ---------------------------------------------------------------

# Add arydshln package for dashed lines in array
latex_macros += r'\usepackage{arydshln}'

latex_elements = {
        'papersize': 'a4paper',
        'pointsize': '10pt',
        'preamble': latex_macros,  # command definitions
        'figure_align': 'htbp',
        'sphinxsetup': ('TitleColor={rgb}{0,0,0}, '
                        'verbatimwithframe=false, '
                        'VerbatimColor={rgb}{.96,.96,.96}'),
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc,
     'sphinxcontrib-katex.tex',
     'sphinxcontrib-katex',
     'Hagen Wierstorf',
     'manual',
     True),
]
