import os
import sys
import textwrap
import types
import re
import warnings

from numpy.core.numerictypes import issubclass_, issubsctype, issubdtype
from numpy.core.overrides import set_module
from numpy.core import ndarray, ufunc, asarray
import numpy as np

# getargspec and formatargspec were removed in Python 3.6
from numpy.compat import getargspec, formatargspec

__all__ = [
    'issubclass_', 'issubsctype', 'issubdtype', 'deprecate',
    'deprecate_with_doc', 'get_include', 'info', 'source', 'who',
    'lookfor', 'byte_bounds', 'safe_eval'
    ]

def get_include():
    """
    Return the directory that contains the NumPy \\*.h header files.

    Extension modules that need to compile against NumPy should use this
    function to locate the appropriate include directory.

    Notes
    -----
    When using ``distutils``, for example in ``setup.py``.
    ::

        import numpy as np
        ...
        Extension('extension_name', ...
                include_dirs=[np.get_include()])
        ...

    """
    import numpy
    if numpy.show_config is None:
        # running from numpy source directory
        d = os.path.join(os.path.dirname(numpy.__file__), 'core', 'include')
    else:
        # using installed numpy core headers
        import numpy.core as core
        d = os.path.join(os.path.dirname(core.__file__), 'include')
    return d


def _set_function_name(func, name):
    func.__name__ = name
    return func


class _Deprecate:
    """
    Decorator class to deprecate old functions.

    Refer to `deprecate` for details.

    See Also
    --------
    deprecate

    """

    def __init__(self, old_name=None, new_name=None, message=None):
        self.old_name = old_name
        self.new_name = new_name
        self.message = message

    def __call__(self, func, *args, **kwargs):
        """
        Decorator call.  Refer to ``decorate``.

        """
        old_name = self.old_name
        new_name = self.new_name
        message = self.message

        if old_name is None:
            try:
                old_name = func.__name__
            except AttributeError:
                old_name = func.__name__
        if new_name is None:
            depdoc = "`%s` is deprecated!" % old_name
        else:
            depdoc = "`%s` is deprecated, use `%s` instead!" % \
                     (old_name, new_name)

        if message is not None:
            depdoc += "\n" + message

        def newfunc(*args,**kwds):
            """`arrayrange` is deprecated, use `arange` instead!"""
            warnings.warn(depdoc, DeprecationWarning, stacklevel=2)
            return func(*args, **kwds)

        newfunc = _set_function_name(newfunc, old_name)
        doc = func.__doc__
        if doc is None:
            doc = depdoc
        else:
            lines = doc.expandtabs().split('\n')
            indent = _get_indent(lines[1:])
            if lines[0].lstrip():
                # Indent the original first line to let inspect.cleandoc()
                # dedent the docstring despite the deprecation notice.
                doc = indent * ' ' + doc
            else:
                # Remove the same leading blank lines as cleandoc() would.
                skip = len(lines[0]) + 1
                for line in lines[1:]:
                    if len(line) > indent:
                        break
                    skip += len(line) + 1
                doc = doc[skip:]
            depdoc = textwrap.indent(depdoc, ' ' * indent)
            doc = '\n\n'.join([depdoc, doc])
        newfunc.__doc__ = doc
        try:
            d = func.__dict__
        except AttributeError:
            pass
        else:
            newfunc.__dict__.update(d)
        return newfunc


def _get_indent(lines):
    """
    Determines the leading whitespace that could be removed from all the lines.
    """
    indent = sys.maxsize
    for line in lines:
        content = len(line.lstrip())
        if content:
            indent = min(indent, len(line) - content)
    if indent == sys.maxsize:
        indent = 0
    return indent


def deprecate(*args, **kwargs):
    """
    Issues a DeprecationWarning, adds warning to `old_name`'s
    docstring, rebinds ``old_name.__name__`` and returns the new
    function object.

    This function may also be used as a decorator.

    Parameters
    ----------
    func : function
        The function to be deprecated.
    old_name : str, optional
        The name of the function to be deprecated. Default is None, in
        which case the name of `func` is used.
    new_name : str, optional
        The new name for the function. Default is None, in which case the
        deprecation message is that `old_name` is deprecated. If given, the
        deprecation message is that `old_name` is deprecated and `new_name`
        should be used instead.
    message : str, optional
        Additional explanation of the deprecation.  Displayed in the
        docstring after the warning.

    Returns
    -------
    old_func : function
        The deprecated function.

    Examples
    --------
    Note that ``olduint`` returns a value after printing Deprecation
    Warning:

    >>> olduint = np.deprecate(np.uint)
    DeprecationWarning: `uint64` is deprecated! # may vary
    >>> olduint(6)
    6

    """
    # Deprecate may be run as a function or as a decorator
    # If run as a function, we initialise the decorator class
    # and execute its __call__ method.

    if args:
        fn = args[0]
        args = args[1:]

        return _Deprecate(*args, **kwargs)(fn)
    else:
        return _Deprecate(*args, **kwargs)

deprecate_with_doc = lambda msg: _Deprecate(message=msg)


#--------------------------------------------
# Determine if two arrays can share memory
#--------------------------------------------

def byte_bounds(a):
    """
    Returns pointers to the end-points of an array.

    Parameters
    ----------
    a : ndarray
        Input array. It must conform to the Python-side of the array
        interface.

    Returns
    -------
    (low, high) : tuple of 2 integers
        The first integer is the first byte of the array, the second
        integer is just past the last byte of the array.  If `a` is not
        contiguous it will not use every byte between the (`low`, `high`)
        values.

    Examples
    --------
    >>> I = np.eye(2, dtype='f'); I.dtype
    dtype('float32')
    >>> low, high = np.byte_bounds(I)
    >>> high - low == I.size*I.itemsize
    True
    >>> I = np.eye(2); I.dtype
    dtype('float64')
    >>> low, high = np.byte_bounds(I)
    >>> high - low == I.size*I.itemsize
    True

    """
    ai = a.__array_interface__
    a_data = ai['data'][0]
    astrides = ai['strides']
    ashape = ai['shape']
    bytes_a = asarray(a).dtype.itemsize

    a_low = a_high = a_data
    if astrides is None:
        # contiguous case
        a_high += a.size * bytes_a
    else:
        for shape, stride in zip(ashape, astrides):
            if stride < 0:
                a_low += (shape-1)*stride
            else:
                a_high += (shape-1)*stride
        a_high += bytes_a
    return a_low, a_high


#-----------------------------------------------------------------------------
# Function for output and information on the variables used.
#-----------------------------------------------------------------------------


def who(vardict=None):
    """
    Print the NumPy arrays in the given dictionary.

    If there is no dictionary passed in or `vardict` is None then returns
    NumPy arrays in the globals() dictionary (all NumPy arrays in the
    namespace).

    Parameters
    ----------
    vardict : dict, optional
        A dictionary possibly containing ndarrays.  Default is globals().

    Returns
    -------
    out : None
        Returns 'None'.

    Notes
    -----
    Prints out the name, shape, bytes and type of all of the ndarrays
    present in `vardict`.

    Examples
    --------
    >>> a = np.arange(10)
    >>> b = np.ones(20)
    >>> np.who()
    Name            Shape            Bytes            Type
    ===========================================================
    a               10               80               int64
    b               20               160              float64
    Upper bound on total bytes  =       240

    >>> d = {'x': np.arange(2.0), 'y': np.arange(3.0), 'txt': 'Some str',
    ... 'idx':5}
    >>> np.who(d)
    Name            Shape            Bytes            Type
    ===========================================================
    x               2                16               float64
    y               3                24               float64
    Upper bound on total bytes  =       40

    """
    if vardict is None:
        frame = sys._getframe().f_back
        vardict = frame.f_globals
    sta = []
    cache = {}
    for name in vardict.keys():
        if isinstance(vardict[name], ndarray):
            var = vardict[name]
            idv = id(var)
            if idv in cache.keys():
                namestr = name + " (%s)" % cache[idv]
                original = 0
            else:
                cache[idv] = name
                namestr = name
                original = 1
            shapestr = " x ".join(map(str, var.shape))
            bytestr = str(var.nbytes)
            sta.append([namestr, shapestr, bytestr, var.dtype.name,
                        original])

    maxname = 0
    maxshape = 0
    maxbyte = 0
    totalbytes = 0
    for k in range(len(sta)):
        val = sta[k]
        if maxname < len(val[0]):
            maxname = len(val[0])
        if maxshape < len(val[1]):
            maxshape = len(val[1])
        if maxbyte < len(val[2]):
            maxbyte = len(val[2])
        if val[4]:
            totalbytes += int(val[2])

    if len(sta) > 0:
        sp1 = max(10, maxname)
        sp2 = max(10, maxshape)
        sp3 = max(10, maxbyte)
        prval = "Name %s Shape %s Bytes %s Type" % (sp1*' ', sp2*' ', sp3*' ')
        print(prval + "\n" + "="*(len(prval)+5) + "\n")

    for k in range(len(sta)):
        val = sta[k]
        print("%s %s %s %s %s %s %s" % (val[0], ' '*(sp1-len(val[0])+4),
                                        val[1], ' '*(sp2-len(val[1])+5),
                                        val[2], ' '*(sp3-len(val[2])+5),
                                        val[3]))
    print("\nUpper bound on total bytes  =       %d" % totalbytes)
    return

#-----------------------------------------------------------------------------


# NOTE:  pydoc defines a help function which works similarly to this
#  except it uses a pager to take over the screen.

# combine name and arguments and split to multiple lines of width
# characters.  End lines on a comma and begin argument list indented with
# the rest of the arguments.
def _split_line(name, arguments, width):
    firstwidth = len(name)
    k = firstwidth
    newstr = name
    sepstr = ", "
    arglist = arguments.split(sepstr)
    for argument in arglist:
        if k == firstwidth:
            addstr = ""
        else:
            addstr = sepstr
        k = k + len(argument) + len(addstr)
        if k > width:
            k = firstwidth + 1 + len(argument)
            newstr = newstr + ",\n" + " "*(firstwidth+2) + argument
        else:
            newstr = newstr + addstr + argument
    return newstr

_namedict = None
_dictlist = None

# Traverse all module directories underneath globals
# to see if something is defined
def _makenamedict(module='numpy'):
    module = __import__(module, globals(), locals(), [])
    thedict = {module.__name__:module.__dict__}
    dictlist = [module.__name__]
    totraverse = [module.__dict__]
    while True:
        if len(totraverse) == 0:
            break
        thisdict = totraverse.pop(0)
        for x in thisdict.keys():
            if isinstance(thisdict[x], types.ModuleType):
                modname = thisdict[x].__name__
                if modname not in dictlist:
                    moddict = thisdict[x].__dict__
                    dictlist.append(modname)
                    totraverse.append(moddict)
                    thedict[modname] = moddict
    return thedict, dictlist


def _info(obj, output=sys.stdout):
    """Provide information about ndarray obj.

    Parameters
    ----------
    obj : ndarray
        Must be ndarray, not checked.
    output
        Where printed output goes.

    Notes
    -----
    Copied over from the numarray module prior to its removal.
    Adapted somewhat as only numpy is an option now.

    Called by info.

    """
    extra = ""
    tic = ""
    bp = lambda x: x
    cls = getattr(obj, '__class__', type(obj))
    nm = getattr(cls, '__name__', cls)
    strides = obj.strides
    endian = obj.dtype.byteorder

    print("class: ", nm, file=output)
    print("shape: ", obj.shape, file=output)
    print("strides: ", strides, file=output)
    print("itemsize: ", obj.itemsize, file=output)
    print("aligned: ", bp(obj.flags.aligned), file=output)
    print("contiguous: ", bp(obj.flags.contiguous), file=output)
    print("fortran: ", obj.flags.fortran, file=output)
    print(
        "data pointer: %s%s" % (hex(obj.ctypes._as_parameter_.value), extra),
        file=output
        )
    print("byteorder: ", end=' ', file=output)
    if endian in ['|', '=']:
        print("%s%s%s" % (tic, sys.byteorder, tic), file=output)
        byteswap = False
    elif endian == '>':
        print("%sbig%s" % (tic, tic), file=output)
        byteswap = sys.byteorder != "big"
    else:
        print("%slittle%s" % (tic, tic), file=output)
        byteswap = sys.byteorder != "little"
    print("byteswap: ", bp(byteswap), file=output)
    print("type: %s" % obj.dtype, file=output)


@set_module('numpy')
def info(object=None, maxwidth=76, output=sys.stdout, toplevel='numpy'):
    """
    Get help information for a function, class, or module.

    Parameters
    ----------
    object : object or str, optional
        Input object or name to get information about. If `object` is a
        numpy object, its docstring is given. If it is a string, available
        modules are searched for matching objects.  If None, information
        about `info` itself is returned.
    maxwidth : int, optional
        Printing width.
    output : file like object, optional
        File like object that the output is written to, default is
        ``stdout``.  The object has to be opened in 'w' or 'a' mode.
    toplevel : str, optional
        Start search at this level.

    See Also
    --------
    source, lookfor

    Notes
    -----
    When used interactively with an object, ``np.info(obj)`` is equivalent
    to ``help(obj)`` on the Python prompt or ``obj?`` on the IPython
    prompt.

    Examples
    --------
    >>> np.info(np.polyval) # doctest: +SKIP
       polyval(p, x)
         Evaluate the polynomial p at x.
         ...

    When using a string for `object` it is possible to get multiple results.

    >>> np.info('fft') # doctest: +SKIP
         *** Found in numpy ***
    Core FFT routines
    ...
         *** Found in numpy.fft ***
     fft(a, n=None, axis=-1)
    ...
         *** Repeat reference found in numpy.fft.fftpack ***
         *** Total of 3 references found. ***

    """
    global _namedict, _dictlist
    # Local import to speed up numpy's import time.
    import pydoc
    import inspect

    if (hasattr(object, '_ppimport_importer') or
           hasattr(object, '_ppimport_module')):
        object = object._ppimport_module
    elif hasattr(object, '_ppimport_attr'):
        object = object._ppimport_attr

    if object is None:
        info(info)
    elif isinstance(object, ndarray):
        _info(object, output=output)
    elif isinstance(object, str):
        if _namedict is None:
            _namedict, _dictlist = _makenamedict(toplevel)
        numfound = 0
        objlist = []
        for namestr in _dictlist:
            try:
                obj = _namedict[namestr][object]
                if id(obj) in objlist:
                    print("\n     "
                          "*** Repeat reference found in %s *** " % namestr,
                          file=output
                          )
                else:
                    objlist.append(id(obj))
                    print("     *** Found in %s ***" % namestr, file=output)
                    info(obj)
                    print("-"*maxwidth, file=output)
                numfound += 1
            except KeyError:
                pass
        if numfound == 0:
            print("Help for %s not found." % object, file=output)
        else:
            print("\n     "
                  "*** Total of %d references found. ***" % numfound,
                  file=output
                  )

    elif inspect.isfunction(object):
        name = object.__name__
        arguments = formatargspec(*getargspec(object))

        if len(name+arguments) > maxwidth:
            argstr = _split_line(name, arguments, maxwidth)
        else:
            argstr = name + arguments

        print(" " + argstr + "\n", file=output)
        print(inspect.getdoc(object), file=output)

    elif inspect.isclass(object):
        name = object.__name__
        arguments = "()"
        try:
            if hasattr(object, '__init__'):
                arguments = formatargspec(
                        *getargspec(object.__init__.__func__)
                        )
                arglist = arguments.split(', ')
                if len(arglist) > 1:
                    arglist[1] = "("+arglist[1]
                    arguments = ", ".join(arglist[1:])
        except Exception:
            pass

        if len(name+arguments) > maxwidth:
            argstr = _split_line(name, arguments, maxwidth)
        else:
            argstr = name + arguments

        print(" " + argstr + "\n", file=output)
        doc1 = inspect.getdoc(object)
        if doc1 is None:
            if hasattr(object, '__init__'):
                print(inspect.getdoc(object.__init__), file=output)
        else:
            print(inspect.getdoc(object), file=output)

        methods = pydoc.allmethods(object)
        if methods != []:
            print("\n\nMethods:\n", file=output)
            for meth in methods:
                if meth[0] == '_':
                    continue
                thisobj = getattr(object, meth, None)
                if thisobj is not None:
                    methstr, other = pydoc.splitdoc(
                            inspect.getdoc(thisobj) or "None"
                            )
                print("  %s  --  %s" % (meth, methstr), file=output)

    elif inspect.ismethod(object):
        name = object.__name__
        arguments = formatargspec(
                *getargspec(object.__func__)
                )
        arglist = arguments.split(', ')
        if len(arglist) > 1:
            arglist[1] = "("+arglist[1]
            arguments = ", ".join(arglist[1:])
        else:
            arguments = "()"

        if len(name+arguments) > maxwidth:
            argstr = _split_line(name, arguments, maxwidth)
        else:
            argstr = name + arguments

        print(" " + argstr + "\n", file=output)
        print(inspect.getdoc(object), file=output)

    elif hasattr(object, '__doc__'):
        print(inspect.getdoc(object), file=output)


@set_module('numpy')
def source(object, output=sys.stdout):
    """
    Print or write to a file the source code for a NumPy object.

    The source code is only returned for objects written in Python. Many
    functions and classes are defined in C and will therefore not return
    useful information.

    Parameters
    ----------
    object : numpy object
        Input object. This can be any object (function, class, module,
        ...).
    output : file object, optional
        If `output` not supplied then source code is printed to screen
        (sys.stdout).  File object must be created with either write 'w' or
        append 'a' modes.

    See Also
    --------
    lookfor, info

    Examples
    --------
    >>> np.source(np.interp)                        #doctest: +SKIP
    In file: /usr/lib/python2.6/dist-packages/numpy/lib/function_base.py
    def interp(x, xp, fp, left=None, right=None):
        \"\"\".... (full docstring printed)\"\"\"
        if isinstance(x, (float, int, number)):
            return compiled_interp([x], xp, fp, left, right).item()
        else:
            return compiled_interp(x, xp, fp, left, right)

    The source code is only returned for objects written in Python.

    >>> np.source(np.array)                         #doctest: +SKIP
    Not available for this object.

    """
    # Local import to speed up numpy's import time.
    import inspect
    try:
        print("In file: %s\n" % inspect.getsourcefile(object), file=output)
        print(inspect.getsource(object), file=output)
    except Exception:
        print("Not available for this object.", file=output)


# Cache for lookfor: {id(module): {name: (docstring, kind, index), ...}...}
# where kind: "func", "class", "module", "object"
# and index: index in breadth-first namespace traversal
_lookfor_caches = {}

# regexp whose match indicates that the string may contain a function
# signature
_function_signature_re = re.compile(r"[a-z0-9_]+\(.*[,=].*\)", re.I)


@set_module('numpy')
def lookfor(what, module=None, import_modules=True, regenerate=False,
            output=None):
    """
    Do a keyword search on docstrings.

    A list of objects that matched the search is displayed,
    sorted by relevance. All given keywords need to be found in the
    docstring for it to be returned as a result, but the order does
    not matter.

    Parameters
    ----------
    what : str
        String containing words to look for.
    module : str or list, optional
        Name of module(s) whose docstrings to go through.
    import_modules : bool, optional
        Whether to import sub-modules in packages. Default is True.
    regenerate : bool, optional
        Whether to re-generate the docstring cache. Default is False.
    output : file-like, optional
        File-like object to write the output to. If omitted, use a pager.

    See Also
    --------
    source, info

    Notes
    -----
    Relevance is determined only roughly, by checking if the keywords occur
    in the function name, at the start of a docstring, etc.

    Examples
    --------
    >>> np.lookfor('binary representation') # doctest: +SKIP
    Search results for 'binary representation'
    ------------------------------------------
    numpy.binary_repr
        Return the binary representation of the input number as a string.
    numpy.core.setup_common.long_double_representation
        Given a binary dump as given by GNU od -b, look for long double
    numpy.base_repr
        Return a string representation of a number in the given base system.
    ...

    """
    import pydoc

    # Cache
    cache = _lookfor_generate_cache(module, import_modules, regenerate)

    # Search
    # XXX: maybe using a real stemming search engine would be better?
    found = []
    whats = str(what).lower().split()
    if not whats:
        return

    for name, (docstring, kind, index) in cache.items():
        if kind in ('module', 'object'):
            # don't show modules or objects
            continue
        doc = docstring.lower()
        if all(w in doc for w in whats):
            found.append(name)

    # Relevance sort
    # XXX: this is full Harrison-Stetson heuristics now,
    # XXX: it probably could be improved

    kind_relevance = {'func': 1000, 'class': 1000,
                      'module': -1000, 'object': -1000}

    def relevance(name, docstr, kind, index):
        r = 0
        # do the keywords occur within the start of the docstring?
        first_doc = "\n".join(docstr.lower().strip().split("\n")[:3])
        r += sum([200 for w in whats if w in first_doc])
        # do the keywords occur in the function name?
        r += sum([30 for w in whats if w in name])
        # is the full name long?
        r += -len(name) * 5
        # is the object of bad type?
        r += kind_relevance.get(kind, -1000)
        # is the object deep in namespace hierarchy?
        r += -name.count('.') * 10
        r += max(-index / 100, -100)
        return r

    def relevance_value(a):
        return relevance(a, *cache[a])
    found.sort(key=relevance_value)

    # Pretty-print
    s = "Search results for '%s'" % (' '.join(whats))
    help_text = [s, "-"*len(s)]
    for name in found[::-1]:
        doc, kind, ix = cache[name]

        doclines = [line.strip() for line in doc.strip().split("\n")
                    if line.strip()]

        # find a suitable short description
        try:
            first_doc = doclines[0].strip()
            if _function_signature_re.search(first_doc):
                first_doc = doclines[1].strip()
        except IndexError:
            first_doc = ""
        help_text.append("%s\n    %s" % (name, first_doc))

    if not found:
        help_text.append("Nothing found.")

    # Output
    if output is not None:
        output.write("\n".join(help_text))
    elif len(help_text) > 10:
        pager = pydoc.getpager()
        pager("\n".join(help_text))
    else:
        print("\n".join(help_text))

def _lookfor_generate_cache(module, import_modules, regenerate):
    """
    Generate docstring cache for given module.

    Parameters
    ----------
    module : str, None, module
        Module for which to generate docstring cache
    import_modules : bool
        Whether to import sub-modules in packages.
    regenerate : bool
        Re-generate the docstring cache

    Returns
    -------
    cache : dict {obj_full_name: (docstring, kind, index), ...}
        Docstring cache for the module, either cached one (regenerate=False)
        or newly generated.

    """
    # Local import to speed up numpy's import time.
    import inspect

    from io import StringIO

    if module is None:
        module = "numpy"

    if isinstance(module, str):
        try:
            __import__(module)
        except ImportError:
            return {}
        module = sys.modules[module]
    elif isinstance(module, list) or isinstance(module, tuple):
        cache = {}
        for mod in module:
            cache.update(_lookfor_generate_cache(mod, import_modules,
                                                 regenerate))
        return cache

    if id(module) in _lookfor_caches and not regenerate:
        return _lookfor_caches[id(module)]

    # walk items and collect docstrings
    cache = {}
    _lookfor_caches[id(module)] = cache
    seen = {}
    index = 0
    stack = [(module.__name__, module)]
    while stack:
        name, item = stack.pop(0)
        if id(item) in seen:
            continue
        seen[id(item)] = True

        index += 1
        kind = "object"

        if inspect.ismodule(item):
            kind = "module"
            try:
                _all = item.__all__
            except AttributeError:
                _all = None

            # import sub-packages
            if import_modules and hasattr(item, '__path__'):
                for pth in item.__path__:
                    for mod_path in os.listdir(pth):
                        this_py = os.path.join(pth, mod_path)
                        init_py = os.path.join(pth, mod_path, '__init__.py')
                        if (os.path.isfile(this_py) and
                                mod_path.endswith('.py')):
                            to_import = mod_path[:-3]
                        elif os.path.isfile(init_py):
                            to_import = mod_path
                        else:
                            continue
                        if to_import == '__init__':
                            continue

                        try:
                            old_stdout = sys.stdout
                            old_stderr = sys.stderr
                            try:
                                sys.stdout = StringIO()
                                sys.stderr = StringIO()
                                __import__("%s.%s" % (name, to_import))
                            finally:
                                sys.stdout = old_stdout
                                sys.stderr = old_stderr
                        # Catch SystemExit, too
                        except BaseException:
                            continue

            for n, v in _getmembers(item):
                try:
                    item_name = getattr(v, '__name__', "%s.%s" % (name, n))
                    mod_name = getattr(v, '__module__', None)
                except NameError:
                    # ref. SWIG's global cvars
                    #    NameError: Unknown C global variable
                    item_name = "%s.%s" % (name, n)
                    mod_name = None
                if '.' not in item_name and mod_name:
                    item_name = "%s.%s" % (mod_name, item_name)

                if not item_name.startswith(name + '.'):
                    # don't crawl "foreign" objects
                    if isinstance(v, ufunc):
                        # ... unless they are ufuncs
                        pass
                    else:
                        continue
                elif not (inspect.ismodule(v) or _all is None or n in _all):
                    continue
                stack.append(("%s.%s" % (name, n), v))
        elif inspect.isclass(item):
            kind = "class"
            for n, v in _getmembers(item):
                stack.append(("%s.%s" % (name, n), v))
        elif hasattr(item, "__call__"):
            kind = "func"

        try:
            doc = inspect.getdoc(item)
        except NameError:
            # ref SWIG's NameError: Unknown C global variable
            doc = None
        if doc is not None:
            cache[name] = (doc, kind, index)

    return cache

def _getmembers(item):
    import inspect
    try:
        members = inspect.getmembers(item)
    except Exception:
        members = [(x, getattr(item, x)) for x in dir(item)
                   if hasattr(item, x)]
    return members


def safe_eval(source):
    """
    Protected string evaluation.

    Evaluate a string containing a Python literal expression without
    allowing the execution of arbitrary non-literal code.

    Parameters
    ----------
    source : str
        The string to evaluate.

    Returns
    -------
    obj : object
       The result of evaluating `source`.

    Raises
    ------
    SyntaxError
        If the code has invalid Python syntax, or if it contains
        non-literal code.

    Examples
    --------
    >>> np.safe_eval('1')
    1
    >>> np.safe_eval('[1, 2, 3]')
    [1, 2, 3]
    >>> np.safe_eval('{"foo": ("bar", 10.0)}')
    {'foo': ('bar', 10.0)}

    >>> np.safe_eval('import os')
    Traceback (most recent call last):
      ...
    SyntaxError: invalid syntax

    >>> np.safe_eval('open("/home/user/.ssh/id_dsa").read()')
    Traceback (most recent call last):
      ...
    ValueError: malformed node or string: <_ast.Call object at 0x...>

    """
    # Local import to speed up numpy's import time.
    import ast
    return ast.literal_eval(source)


def _median_nancheck(data, result, axis, out):
    """
    Utility function to check median result from data for NaN values at the end
    and return NaN in that case. Input result can also be a MaskedArray.

    Parameters
    ----------
    data : array
        Input data to median function
    result : Array or MaskedArray
        Result of median function
    axis : {int, sequence of int, None}, optional
        Axis or axes along which the median was computed.
    out : ndarray, optional
        Output array in which to place the result.
    Returns
    -------
    median : scalar or ndarray
        Median or NaN in axes which contained NaN in the input.
    """
    if data.size == 0:
        return result
    data = np.moveaxis(data, axis, -1)
    n = np.isnan(data[..., -1])
    # masked NaN values are ok
    if np.ma.isMaskedArray(n):
        n = n.filled(False)
    if result.ndim == 0:
        if n == True:
            if out is not None:
                out[...] = data.dtype.type(np.nan)
                result = out
            else:
                result = data.dtype.type(np.nan)
    elif np.count_nonzero(n.ravel()) > 0:
        result[n] = np.nan
    return result

#-----------------------------------------------------------------------------
