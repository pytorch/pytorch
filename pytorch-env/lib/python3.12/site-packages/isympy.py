"""
Python shell for SymPy.

This is just a normal Python shell (IPython shell if you have the
IPython package installed), that executes the following commands for
the user:

    >>> from __future__ import division
    >>> from sympy import *
    >>> x, y, z, t = symbols('x y z t')
    >>> k, m, n = symbols('k m n', integer=True)
    >>> f, g, h = symbols('f g h', cls=Function)
    >>> init_printing()

So starting 'isympy' is equivalent to starting Python (or IPython) and
executing the above commands by hand.  It is intended for easy and quick
experimentation with SymPy.  isympy is a good way to use SymPy as an
interactive calculator. If you have IPython and Matplotlib installed, then
interactive plotting is enabled by default.

COMMAND LINE OPTIONS
--------------------

-c CONSOLE, --console=CONSOLE

     Use the specified shell (Python or IPython) shell as the console
     backend instead of the default one (IPython if present, Python
     otherwise), e.g.:

        $isympy -c python

    CONSOLE must be one of 'ipython' or 'python'

-p PRETTY, --pretty PRETTY

    Setup pretty-printing in SymPy. When pretty-printing is enabled,
    expressions can be printed with Unicode or ASCII. The default is
    to use pretty-printing (with Unicode if the terminal supports it).
    When this option is 'no', expressions will not be pretty-printed
    and ASCII will be used:

        $isympy -p no

    PRETTY must be one of 'unicode', 'ascii', or 'no'

-t TYPES, --types=TYPES

    Setup the ground types for the polys.  By default, gmpy ground types
    are used if gmpy2 or gmpy is installed, otherwise it falls back to python
    ground types, which are a little bit slower.  You can manually
    choose python ground types even if gmpy is installed (e.g., for
    testing purposes):

        $isympy -t python

    TYPES must be one of 'gmpy', 'gmpy1' or 'python'

    Note that the ground type gmpy1 is primarily intended for testing; it
    forces the use of gmpy version 1 even if gmpy2 is available.

    This is the same as setting the environment variable
    SYMPY_GROUND_TYPES to the given ground type (e.g.,
    SYMPY_GROUND_TYPES='gmpy')

    The ground types can be determined interactively from the variable
    sympy.polys.domains.GROUND_TYPES.

-o ORDER, --order ORDER

    Setup the ordering of terms for printing.  The default is lex, which
    orders terms lexicographically (e.g., x**2 + x + 1). You can choose
    other orderings, such as rev-lex, which will use reverse
    lexicographic ordering (e.g., 1 + x + x**2):

        $isympy -o rev-lex

    ORDER must be one of 'lex', 'rev-lex', 'grlex', 'rev-grlex',
    'grevlex', 'rev-grevlex', 'old', or 'none'.

    Note that for very large expressions, ORDER='none' may speed up
    printing considerably but the terms will have no canonical order.

-q, --quiet

    Print only Python's and SymPy's versions to stdout at startup.

-d, --doctest

    Use the same format that should be used for doctests.  This is
    equivalent to -c python -p no.

-C, --no-cache

    Disable the caching mechanism.  Disabling the cache may slow certain
    operations down considerably.  This is useful for testing the cache,
    or for benchmarking, as the cache can result in deceptive timings.

    This is equivalent to setting the environment variable
    SYMPY_USE_CACHE to 'no'.

-a, --auto-symbols (requires at least IPython 0.11)

    Automatically create missing symbols.  Normally, typing a name of a
    Symbol that has not been instantiated first would raise NameError,
    but with this option enabled, any undefined name will be
    automatically created as a Symbol.

    Note that this is intended only for interactive, calculator style
    usage. In a script that uses SymPy, Symbols should be instantiated
    at the top, so that it's clear what they are.

    This will not override any names that are already defined, which
    includes the single character letters represented by the mnemonic
    QCOSINE (see the "Gotchas and Pitfalls" document in the
    documentation). You can delete existing names by executing "del
    name".  If a name is defined, typing "'name' in dir()" will return True.

    The Symbols that are created using this have default assumptions.
    If you want to place assumptions on symbols, you should create them
    using symbols() or var().

    Finally, this only works in the top level namespace. So, for
    example, if you define a function in isympy with an undefined
    Symbol, it will not work.

    See also the -i and -I options.

-i, --int-to-Integer (requires at least IPython 0.11)

    Automatically wrap int literals with Integer.  This makes it so that
    things like 1/2 will come out as Rational(1, 2), rather than 0.5.  This
    works by preprocessing the source and wrapping all int literals with
    Integer.  Note that this will not change the behavior of int literals
    assigned to variables, and it also won't change the behavior of functions
    that return int literals.

    If you want an int, you can wrap the literal in int(), e.g. int(3)/int(2)
    gives 1.5 (with division imported from __future__).

-I, --interactive (requires at least IPython 0.11)

    This is equivalent to --auto-symbols --int-to-Integer.  Future options
    designed for ease of interactive use may be added to this.

-D, --debug

    Enable debugging output.  This is the same as setting the
    environment variable SYMPY_DEBUG to 'True'.  The debug status is set
    in the variable SYMPY_DEBUG within isympy.

-- IPython options

    Additionally you can pass command line options directly to the IPython
    interpreter (the standard Python shell is not supported).  However you
    need to add the '--' separator between two types of options, e.g the
    startup banner option and the colors option. You need to enter the
    options as required by the version of IPython that you are using, too:

    in IPython 0.11,

        $isympy -q -- --colors=NoColor

    or older versions of IPython,

        $isympy -q -- -colors NoColor

See also isympy --help.
"""

import os
import sys

# DO NOT IMPORT SYMPY HERE! Or the setting of the sympy environment variables
# by the command line will break.

def main() -> None:
    from argparse import ArgumentParser, RawDescriptionHelpFormatter

    VERSION = None
    if '--version' in sys.argv:
        # We cannot import sympy before this is run, because flags like -C and
        # -t set environment variables that must be set before SymPy is
        # imported. The only thing we need to import it for is to get the
        # version, which only matters with the --version flag.
        import sympy
        VERSION = sympy.__version__

    usage = 'isympy [options] -- [ipython options]'
    parser = ArgumentParser(
        usage=usage,
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter,
    )

    parser.add_argument('--version', action='version', version=VERSION)

    parser.add_argument(
        '-c', '--console',
        dest='console',
        action='store',
        default=None,
        choices=['ipython', 'python'],
        metavar='CONSOLE',
        help='select type of interactive session: ipython | python; defaults '
        'to ipython if IPython is installed, otherwise python')

    parser.add_argument(
        '-p', '--pretty',
        dest='pretty',
        action='store',
        default=None,
        metavar='PRETTY',
        choices=['unicode', 'ascii', 'no'],
        help='setup pretty printing: unicode | ascii | no; defaults to '
        'unicode printing if the terminal supports it, otherwise ascii')

    parser.add_argument(
        '-t', '--types',
        dest='types',
        action='store',
        default=None,
        metavar='TYPES',
        choices=['gmpy', 'gmpy1', 'python'],
        help='setup ground types: gmpy | gmpy1 | python; defaults to gmpy if gmpy2 '
        'or gmpy is installed, otherwise python')

    parser.add_argument(
        '-o', '--order',
        dest='order',
        action='store',
        default=None,
        metavar='ORDER',
        choices=['lex', 'grlex', 'grevlex', 'rev-lex', 'rev-grlex', 'rev-grevlex', 'old', 'none'],
        help='setup ordering of terms: [rev-]lex | [rev-]grlex | [rev-]grevlex | old | none; defaults to lex')

    parser.add_argument(
        '-q', '--quiet',
        dest='quiet',
        action='store_true',
        default=False,
        help='print only version information at startup')

    parser.add_argument(
        '-d', '--doctest',
        dest='doctest',
        action='store_true',
        default=False,
        help='use the doctest format for output (you can just copy and paste it)')

    parser.add_argument(
        '-C', '--no-cache',
        dest='cache',
        action='store_false',
        default=True,
        help='disable caching mechanism')

    parser.add_argument(
        '-a', '--auto-symbols',
        dest='auto_symbols',
        action='store_true',
        default=False,
        help='automatically construct missing symbols')

    parser.add_argument(
        '-i', '--int-to-Integer',
        dest='auto_int_to_Integer',
        action='store_true',
        default=False,
        help="automatically wrap int literals with Integer")

    parser.add_argument(
        '-I', '--interactive',
        dest='interactive',
        action='store_true',
        default=False,
        help="equivalent to -a -i")

    parser.add_argument(
        '-D', '--debug',
        dest='debug',
        action='store_true',
        default=False,
        help='enable debugging output')

    (options, ipy_args) = parser.parse_known_args()
    if '--' in ipy_args:
        ipy_args.remove('--')

    if not options.cache:
        os.environ['SYMPY_USE_CACHE'] = 'no'

    if options.types:
        os.environ['SYMPY_GROUND_TYPES'] = options.types

    if options.debug:
        os.environ['SYMPY_DEBUG'] = str(options.debug)

    if options.doctest:
        options.pretty = 'no'
        options.console = 'python'

    session = options.console

    if session is not None:
        ipython = session == 'ipython'
    else:
        try:
            import IPython
            ipython = True
        except ImportError:
            if not options.quiet:
                from sympy.interactive.session import no_ipython
                print(no_ipython)
            ipython = False

    args = {
        'pretty_print': True,
        'use_unicode':  None,
        'use_latex':    None,
        'order':        None,
        'argv':         ipy_args,
    }

    if options.pretty == 'unicode':
        args['use_unicode'] = True
    elif options.pretty == 'ascii':
        args['use_unicode'] = False
    elif options.pretty == 'no':
        args['pretty_print'] = False

    if options.order is not None:
        args['order'] = options.order

    args['quiet'] = options.quiet
    args['auto_symbols'] = options.auto_symbols or options.interactive
    args['auto_int_to_Integer'] = options.auto_int_to_Integer or options.interactive

    from sympy.interactive import init_session
    init_session(ipython, **args)

if __name__ == "__main__":
    main()
