import os
from os.path import join
import shutil
import tempfile
from pathlib import Path

try:
    from subprocess import STDOUT, CalledProcessError, check_output
except ImportError:
    pass

from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.misc import debug
from .latex import latex

__doctest_requires__ = {('preview',): ['pyglet']}


def _check_output_no_window(*args, **kwargs):
    # Avoid showing a cmd.exe window when running this
    # on Windows
    if os.name == 'nt':
        creation_flag = 0x08000000 # CREATE_NO_WINDOW
    else:
        creation_flag = 0 # Default value
    return check_output(*args, creationflags=creation_flag, **kwargs)


def system_default_viewer(fname, fmt):
    """ Open fname with the default system viewer.

    In practice, it is impossible for python to know when the system viewer is
    done. For this reason, we ensure the passed file will not be deleted under
    it, and this function does not attempt to block.
    """
    # copy to a new temporary file that will not be deleted
    with tempfile.NamedTemporaryFile(prefix='sympy-preview-',
                                     suffix=os.path.splitext(fname)[1],
                                     delete=False) as temp_f:
        with open(fname, 'rb') as f:
            shutil.copyfileobj(f, temp_f)

    import platform
    if platform.system() == 'Darwin':
        import subprocess
        subprocess.call(('open', temp_f.name))
    elif platform.system() == 'Windows':
        os.startfile(temp_f.name)
    else:
        import subprocess
        subprocess.call(('xdg-open', temp_f.name))


def pyglet_viewer(fname, fmt):
    try:
        from pyglet import window, image, gl
        from pyglet.window import key
        from pyglet.image.codecs import ImageDecodeException
    except ImportError:
        raise ImportError("pyglet is required for preview.\n visit https://pyglet.org/")

    try:
        img = image.load(fname)
    except ImageDecodeException:
        raise ValueError("pyglet preview does not work for '{}' files.".format(fmt))

    offset = 25

    config = gl.Config(double_buffer=False)
    win = window.Window(
        width=img.width + 2*offset,
        height=img.height + 2*offset,
        caption="SymPy",
        resizable=False,
        config=config
    )

    win.set_vsync(False)

    try:
        def on_close():
            win.has_exit = True

        win.on_close = on_close

        def on_key_press(symbol, modifiers):
            if symbol in [key.Q, key.ESCAPE]:
                on_close()

        win.on_key_press = on_key_press

        def on_expose():
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            img.blit(
                (win.width - img.width) / 2,
                (win.height - img.height) / 2
            )

        win.on_expose = on_expose

        while not win.has_exit:
            win.dispatch_events()
            win.flip()
    except KeyboardInterrupt:
        pass

    win.close()


def _get_latex_main(expr, *, preamble=None, packages=(), extra_preamble=None,
                    euler=True, fontsize=None, **latex_settings):
    """
    Generate string of a LaTeX document rendering ``expr``.
    """
    if preamble is None:
        actual_packages = packages + ("amsmath", "amsfonts")
        if euler:
            actual_packages += ("euler",)
        package_includes = "\n" + "\n".join(["\\usepackage{%s}" % p
                                             for p in actual_packages])
        if extra_preamble:
            package_includes += extra_preamble

        if not fontsize:
            fontsize = "12pt"
        elif isinstance(fontsize, int):
            fontsize = "{}pt".format(fontsize)
        preamble = r"""\documentclass[varwidth,%s]{standalone}
%s

\begin{document}
""" % (fontsize, package_includes)
    else:
        if packages or extra_preamble:
            raise ValueError("The \"packages\" or \"extra_preamble\" keywords"
                             "must not be set if a "
                             "custom LaTeX preamble was specified")

    if isinstance(expr, str):
        latex_string = expr
    else:
        latex_string = ('$\\displaystyle ' +
                        latex(expr, mode='plain', **latex_settings) +
                        '$')

    return preamble + '\n' + latex_string + '\n\n' + r"\end{document}"


@doctest_depends_on(exe=('latex', 'dvipng'), modules=('pyglet',),
            disable_viewers=('evince', 'gimp', 'superior-dvi-viewer'))
def preview(expr, output='png', viewer=None, euler=True, packages=(),
            filename=None, outputbuffer=None, preamble=None, dvioptions=None,
            outputTexFile=None, extra_preamble=None, fontsize=None,
            **latex_settings):
    r"""
    View expression or LaTeX markup in PNG, DVI, PostScript or PDF form.

    If the expr argument is an expression, it will be exported to LaTeX and
    then compiled using the available TeX distribution.  The first argument,
    'expr', may also be a LaTeX string.  The function will then run the
    appropriate viewer for the given output format or use the user defined
    one. By default png output is generated.

    By default pretty Euler fonts are used for typesetting (they were used to
    typeset the well known "Concrete Mathematics" book). For that to work, you
    need the 'eulervm.sty' LaTeX style (in Debian/Ubuntu, install the
    texlive-fonts-extra package). If you prefer default AMS fonts or your
    system lacks 'eulervm' LaTeX package then unset the 'euler' keyword
    argument.

    To use viewer auto-detection, lets say for 'png' output, issue

    >>> from sympy import symbols, preview, Symbol
    >>> x, y = symbols("x,y")

    >>> preview(x + y, output='png')

    This will choose 'pyglet' by default. To select a different one, do

    >>> preview(x + y, output='png', viewer='gimp')

    The 'png' format is considered special. For all other formats the rules
    are slightly different. As an example we will take 'dvi' output format. If
    you would run

    >>> preview(x + y, output='dvi')

    then 'view' will look for available 'dvi' viewers on your system
    (predefined in the function, so it will try evince, first, then kdvi and
    xdvi). If nothing is found, it will fall back to using a system file
    association (via ``open`` and ``xdg-open``). To always use your system file
    association without searching for the above readers, use

    >>> from sympy.printing.preview import system_default_viewer
    >>> preview(x + y, output='dvi', viewer=system_default_viewer)

    If this still does not find the viewer you want, it can be set explicitly.

    >>> preview(x + y, output='dvi', viewer='superior-dvi-viewer')

    This will skip auto-detection and will run user specified
    'superior-dvi-viewer'. If ``view`` fails to find it on your system it will
    gracefully raise an exception.

    You may also enter ``'file'`` for the viewer argument. Doing so will cause
    this function to return a file object in read-only mode, if ``filename``
    is unset. However, if it was set, then 'preview' writes the generated
    file to this filename instead.

    There is also support for writing to a ``io.BytesIO`` like object, which
    needs to be passed to the ``outputbuffer`` argument.

    >>> from io import BytesIO
    >>> obj = BytesIO()
    >>> preview(x + y, output='png', viewer='BytesIO',
    ...         outputbuffer=obj)

    The LaTeX preamble can be customized by setting the 'preamble' keyword
    argument. This can be used, e.g., to set a different font size, use a
    custom documentclass or import certain set of LaTeX packages.

    >>> preamble = "\\documentclass[10pt]{article}\n" \
    ...            "\\usepackage{amsmath,amsfonts}\\begin{document}"
    >>> preview(x + y, output='png', preamble=preamble)

    It is also possible to use the standard preamble and provide additional
    information to the preamble using the ``extra_preamble`` keyword argument.

    >>> from sympy import sin
    >>> extra_preamble = "\\renewcommand{\\sin}{\\cos}"
    >>> preview(sin(x), output='png', extra_preamble=extra_preamble)

    If the value of 'output' is different from 'dvi' then command line
    options can be set ('dvioptions' argument) for the execution of the
    'dvi'+output conversion tool. These options have to be in the form of a
    list of strings (see ``subprocess.Popen``).

    Additional keyword args will be passed to the :func:`~sympy.printing.latex.latex` call,
    e.g., the ``symbol_names`` flag.

    >>> phidd = Symbol('phidd')
    >>> preview(phidd, symbol_names={phidd: r'\ddot{\varphi}'})

    For post-processing the generated TeX File can be written to a file by
    passing the desired filename to the 'outputTexFile' keyword
    argument. To write the TeX code to a file named
    ``"sample.tex"`` and run the default png viewer to display the resulting
    bitmap, do

    >>> preview(x + y, outputTexFile="sample.tex")


    """
    # pyglet is the default for png
    if viewer is None and output == "png":
        try:
            import pyglet  # noqa: F401
        except ImportError:
            pass
        else:
            viewer = pyglet_viewer

    # look up a known application
    if viewer is None:
        # sorted in order from most pretty to most ugly
        # very discussable, but indeed 'gv' looks awful :)
        candidates = {
            "dvi": [ "evince", "okular", "kdvi", "xdvi" ],
            "ps": [ "evince", "okular", "gsview", "gv" ],
            "pdf": [ "evince", "okular", "kpdf", "acroread", "xpdf", "gv" ],
        }

        for candidate in candidates.get(output, []):
            path = shutil.which(candidate)
            if path is not None:
                viewer = path
                break

    # otherwise, use the system default for file association
    if viewer is None:
        viewer = system_default_viewer

    if viewer == "file":
        if filename is None:
            raise ValueError("filename has to be specified if viewer=\"file\"")
    elif viewer == "BytesIO":
        if outputbuffer is None:
            raise ValueError("outputbuffer has to be a BytesIO "
                             "compatible object if viewer=\"BytesIO\"")
    elif not callable(viewer) and not shutil.which(viewer):
        raise OSError("Unrecognized viewer: %s" % viewer)

    latex_main = _get_latex_main(expr, preamble=preamble, packages=packages,
                                 euler=euler, extra_preamble=extra_preamble,
                                 fontsize=fontsize, **latex_settings)

    debug("Latex code:")
    debug(latex_main)
    with tempfile.TemporaryDirectory() as workdir:
        Path(join(workdir, 'texput.tex')).write_text(latex_main, encoding='utf-8')

        if outputTexFile is not None:
            shutil.copyfile(join(workdir, 'texput.tex'), outputTexFile)

        if not shutil.which('latex'):
            raise RuntimeError("latex program is not installed")

        try:
            _check_output_no_window(
                ['latex', '-halt-on-error', '-interaction=nonstopmode',
                 'texput.tex'],
                cwd=workdir,
                stderr=STDOUT)
        except CalledProcessError as e:
            raise RuntimeError(
                "'latex' exited abnormally with the following output:\n%s" %
                e.output)

        src = "texput.%s" % (output)

        if output != "dvi":
            # in order of preference
            commandnames = {
                "ps": ["dvips"],
                "pdf": ["dvipdfmx", "dvipdfm", "dvipdf"],
                "png": ["dvipng"],
                "svg": ["dvisvgm"],
            }
            try:
                cmd_variants = commandnames[output]
            except KeyError:
                raise ValueError("Invalid output format: %s" % output) from None

            # find an appropriate command
            for cmd_variant in cmd_variants:
                cmd_path = shutil.which(cmd_variant)
                if cmd_path:
                    cmd = [cmd_path]
                    break
            else:
                if len(cmd_variants) > 1:
                    raise RuntimeError("None of %s are installed" % ", ".join(cmd_variants))
                else:
                    raise RuntimeError("%s is not installed" % cmd_variants[0])

            defaultoptions = {
                "dvipng": ["-T", "tight", "-z", "9", "--truecolor"],
                "dvisvgm": ["--no-fonts"],
            }

            commandend = {
                "dvips": ["-o", src, "texput.dvi"],
                "dvipdf": ["texput.dvi", src],
                "dvipdfm": ["-o", src, "texput.dvi"],
                "dvipdfmx": ["-o", src, "texput.dvi"],
                "dvipng": ["-o", src, "texput.dvi"],
                "dvisvgm": ["-o", src, "texput.dvi"],
            }

            if dvioptions is not None:
                cmd.extend(dvioptions)
            else:
                cmd.extend(defaultoptions.get(cmd_variant, []))
            cmd.extend(commandend[cmd_variant])

            try:
                _check_output_no_window(cmd, cwd=workdir, stderr=STDOUT)
            except CalledProcessError as e:
                raise RuntimeError(
                    "'%s' exited abnormally with the following output:\n%s" %
                    (' '.join(cmd), e.output))


        if viewer == "file":
            shutil.move(join(workdir, src), filename)
        elif viewer == "BytesIO":
            s = Path(join(workdir, src)).read_bytes()
            outputbuffer.write(s)
        elif callable(viewer):
            viewer(join(workdir, src), fmt=output)
        else:
            try:
                _check_output_no_window(
                    [viewer, src], cwd=workdir, stderr=STDOUT)
            except CalledProcessError as e:
                raise RuntimeError(
                    "'%s %s' exited abnormally with the following output:\n%s" %
                    (viewer, src, e.output))
