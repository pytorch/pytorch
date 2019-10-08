# -*- coding: utf-8 -*-
"""
    sphinxcontrib.katex
    ~~~~~~~~~~~~~~~~~~~

    Allow `KaTeX <khan.github.io/KaTeX/>`_ to be used to display math in
    Sphinx's HTML writer.

    :copyright: Copyright 2017 by Hagen Wierstorf.
    :license: MIT, see LICENSE for details.
"""

import os
import re
import shutil
from docutils import nodes
from tempfile import mkdtemp
from textwrap import dedent
import subprocess

from sphinx.locale import _
from sphinx.errors import ExtensionError
from sphinx.util.osutil import copyfile


__version__ = '0.5.1'
katex_version = '0.10.2'
filename_css = 'katex-math.css'
filename_autorenderer = 'katex_autorenderer.js'


def latex_defs_to_katex_macros(defs):
    r'''Converts LaTeX \def statements to KaTeX macros.

    This is a helper function that can be used in conf.py to translate your
    already specified LaTeX definitions.

    https://github.com/Khan/KaTeX#rendering-options, e.g.
    `\def \e #1{\mathrm{e}^{#1}}` => `"\\e:" "\\mathrm{e}^{#1}"`'

    Example
    -------
    import sphinxcontrib.katex as katex
    # Get your LaTeX defs into `latex_defs` and then do
    latex_macros = katex.import_macros_from_latex(latex_defs)
    katex_options = 'macros: {' + latex_macros + '}'
    '''
    # Remove empty lines
    defs = defs.strip()
    tmp = []
    for line in defs.splitlines():
        # Remove spaces from every line
        line = line.strip()
        # Remove "\def" at the beginning of line
        line = re.sub(r'^\\def[ ]?', '', line)
        # Remove optional #1 parameter before {} command brackets
        line = re.sub(r'(#[0-9])+', '', line, 1)
        # Remove outer {} command brackets with ""
        line = re.sub(r'( {)|(}$)', '"', line)
        # Add "": to the new command
        line = re.sub(r'(^\\[A-Za-z]+)', r'"\1":', line, 1)
        # Add , at end of line
        line = re.sub(r'$', ',', line, 1)
        # Duplicate all \
        line = re.sub(r'\\', r'\\\\', line)
        tmp.append(line)
    macros = '\n'.join(tmp)
    return macros


def get_latex(node):
    if 'latex' in node.attributes:
        return node['latex']  # for Sphinx < 1.8.0
    else:
        return node.astext()  # for Sphinx >= 1.8.0


def run_katex(latex, *options):
    p = subprocess.Popen(
        ('katex', ) + options,
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE
    )
    stdout, stderr = p.communicate(latex.encode('utf-8'))
    return stdout.decode('utf-8')


def html_visit_math(self, node):
    self.body.append(self.starttag(node, 'span', '', CLASS='math'))

    if self.builder.config.katex_prerender:
        self.body.append(run_katex(get_latex(node)))
    else:
        self.body.append(self.builder.config.katex_inline[0] +
                         self.encode(get_latex(node)) +
                         self.builder.config.katex_inline[1])

    self.body.append('</span>')
    raise nodes.SkipNode


def html_visit_displaymath(self, node):
    self.body.append(self.starttag(node, 'div', CLASS='math'))

    # necessary to e.g. set the id property correctly
    if node['number']:
        number = get_node_equation_number(self, node)
        self.body.append('<span class="eqno">(%s)' % number)
        self.add_permalink_ref(node, _('Permalink to this equation'))
        self.body.append('</span>')

    if self.builder.config.katex_prerender:
        # NB: nowrap is always "on" when using prerendering
        self.body.append(run_katex(get_latex(node), '--display-mode'))
        self.body.append('</div>')
    elif node['nowrap']:
        self.body.append(self.encode(get_latex(node)))
        self.body.append('</div>')
    else:
        self.body.append(self.builder.config.katex_display[0])
        self.body.append(get_latex(node))
        self.body.append(self.builder.config.katex_display[1])
        self.body.append('</div>\n')

    raise nodes.SkipNode


def builder_inited(app):
    if not (app.config.katex_js_path and app.config.katex_css_path and
            app.config.katex_autorender_path):
        raise ExtensionError('KaTeX paths not set')
    # Sphinx 1.8 renamed `add_stylesheet` to `add_css_file` and
    # `add_javascript` to `add_js_file`.
    add_css = getattr(app, 'add_css_file', getattr(app, 'add_stylesheet'))
    add_js = getattr(app, 'add_js_file', getattr(app, 'add_javascript'))
    add_css(app.config.katex_css_path)
    # Ensure the static path is setup to hold KaTeX CSS and autorender files
    setup_static_path(app)
    if not app.config.katex_prerender:
        add_js(app.config.katex_js_path)
        # Automatic math rendering and custom CSS
        # https://github.com/Khan/KaTeX/blob/master/contrib/auto-render/README.md
        add_js(app.config.katex_autorender_path)
        write_katex_autorenderer_file(app, filename_autorenderer)
        add_js(filename_autorenderer)
    # sphinxcontrib.katex custom CSS
    copy_katex_css_file(app, filename_css)
    add_css(filename_css)


def builder_finished(app, exception):
    # Delete temporary dir used for _static file
    shutil.rmtree(app._katex_static_path)


def write_katex_autorenderer_file(app, filename):
    filename = os.path.join(
        app.builder.srcdir, app._katex_static_path, filename
    )
    content = katex_autorenderer_content(app)
    with open(filename, 'w') as file:
        file.write(content)


def copy_katex_css_file(app, css_file_name):
    pwd = os.path.abspath(os.path.dirname(__file__))
    source = os.path.join(pwd, css_file_name)
    dest = os.path.join(app._katex_static_path, css_file_name)
    copyfile(source, dest)


def katex_autorenderer_content(app):
    content = dedent('''\
        document.addEventListener("DOMContentLoaded", function() {
          renderMathInElement(document.body, katex_options);
        });
        ''')
    prefix = 'katex_options = {'
    suffix = '}'
    options = katex_rendering_options(app)
    delimiters = katex_rendering_delimiters(app)
    return '\n'.join([prefix, options, delimiters, suffix, content])


def katex_rendering_delimiters(app):
    """Delimiters for rendering KaTeX math.

    If no delimiters are specified in katex_options, add the
    katex_inline and katex_display delimiters. See also
    https://khan.github.io/KaTeX/docs/autorender.html
    """
    # Return if we have user defined rendering delimiters
    if 'delimiters' in app.config.katex_options:
        return ''
    katex_inline = [d.replace('\\', '\\\\') for d in app.config.katex_inline]
    katex_display = [d.replace('\\', '\\\\') for d in app.config.katex_display]
    katex_delimiters = {'inline': katex_inline, 'display': katex_display}
    # Set chosen delimiters for the auto-rendering options of KaTeX
    delimiters = r'''delimiters: [
        {{ left: "{inline[0]}", right: "{inline[1]}", display: false }},
        {{ left: "{display[0]}", right: "{display[1]}", display: true }}
        ]'''.format(**katex_delimiters)
    return delimiters


def katex_rendering_options(app):
    """Strip katex_options from enclosing {} and append ,"""
    options = trim(app.config.katex_options)
    # Remove surrounding {}
    if options.startswith('{') and options.endswith('}'):
        options = trim(options[1:-1])
    # If options is not empty, ensure it ends with ','
    if options and not options.endswith(','):
        options += ','
    return options


def trim(text):
    """Remove whitespace from both sides of a string."""
    return text.lstrip().rstrip()


def setup_static_path(app):
    app._katex_static_path = mkdtemp()
    if app._katex_static_path not in app.config.html_static_path:
        app.config.html_static_path.append(app._katex_static_path)


def setup(app):
    try:
        app.add_html_math_renderer(
            'katex',
            inline_renderers=(html_visit_math, None),
            block_renderers=(html_visit_displaymath, None)
        )
    except AttributeError:
        # Versions of sphinx<1.8 require setup_math instead
        from sphinx.ext.mathbase import setup_math
        setup_math(app, (html_visit_math, None),
                   (html_visit_displaymath, None))

    # Include KaTex CSS and JS files
    katex_url = 'https://cdn.jsdelivr.net/npm/katex@{version}/dist/'.format(
        version=katex_version)
    app.add_config_value('katex_css_path',
                         katex_url + 'katex.min.css',
                         False)
    app.add_config_value('katex_js_path',
                         katex_url + 'katex.min.js',
                         False)
    app.add_config_value('katex_autorender_path',
                         katex_url + 'contrib/auto-render.min.js',
                         False)
    app.add_config_value('katex_inline', [r'\(', r'\)'], 'html')
    app.add_config_value('katex_display', [r'\[', r'\]'], 'html')
    app.add_config_value('katex_options', '', 'html')
    app.add_config_value('katex_prerender', False, 'html')
    app.connect('builder-inited', builder_inited)
    app.connect('build-finished', builder_finished)

    return {'version': __version__, 'parallel_read_safe': True}


# This function is copied from Sphinx 1.8 as it is not available in Sphinx 1.6
def get_node_equation_number(writer, node):
    if writer.builder.config.math_numfig and writer.builder.config.numfig:
        figtype = 'displaymath'
        if writer.builder.name == 'singlehtml':
            key = u"%s/%s" % (writer.docnames[-1], figtype)
        else:
            key = figtype

        id = node['ids'][0]
        number = writer.builder.fignumbers.get(key, {}).get(id, ())
        number = '.'.join(map(str, number))
    else:
        number = node['number']

    return number
