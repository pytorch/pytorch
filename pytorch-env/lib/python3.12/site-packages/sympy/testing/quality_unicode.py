import re
import fnmatch


message_unicode_B = \
    "File contains a unicode character : %s, line %s. " \
    "But not in the whitelist. " \
    "Add the file to the whitelist in " + __file__
message_unicode_D = \
    "File does not contain a unicode character : %s." \
    "but is in the whitelist. " \
    "Remove the file from the whitelist in " + __file__


encoding_header_re = re.compile(
    r'^[ \t\f]*#.*?coding[:=][ \t]*([-_.a-zA-Z0-9]+)')

# Whitelist pattern for files which can have unicode.
unicode_whitelist = [
    # Author names can include non-ASCII characters
    r'*/bin/authors_update.py',
    r'*/bin/mailmap_check.py',

    # These files have functions and test functions for unicode input and
    # output.
    r'*/sympy/testing/tests/test_code_quality.py',
    r'*/sympy/physics/vector/tests/test_printing.py',
    r'*/physics/quantum/tests/test_printing.py',
    r'*/sympy/vector/tests/test_printing.py',
    r'*/sympy/parsing/tests/test_sympy_parser.py',
    r'*/sympy/printing/pretty/stringpict.py',
    r'*/sympy/printing/pretty/tests/test_pretty.py',
    r'*/sympy/printing/tests/test_conventions.py',
    r'*/sympy/printing/tests/test_preview.py',
    r'*/liealgebras/type_g.py',
    r'*/liealgebras/weyl_group.py',
    r'*/liealgebras/tests/test_type_G.py',

    # wigner.py and polarization.py have unicode doctests. These probably
    # don't need to be there but some of the examples that are there are
    # pretty ugly without use_unicode (matrices need to be wrapped across
    # multiple lines etc)
    r'*/sympy/physics/wigner.py',
    r'*/sympy/physics/optics/polarization.py',

    # joint.py uses some unicode for variable names in the docstrings
    r'*/sympy/physics/mechanics/joint.py',

    # lll method has unicode in docstring references and author name
    r'*/sympy/polys/matrices/domainmatrix.py',
    r'*/sympy/matrices/repmatrix.py',

    # Explanation of symbols uses greek letters
    r'*/sympy/core/symbol.py',
]

unicode_strict_whitelist = [
    r'*/sympy/parsing/latex/_antlr/__init__.py',
    # test_mathematica.py uses some unicode for testing Greek characters are working #24055
    r'*/sympy/parsing/tests/test_mathematica.py',
]


def _test_this_file_encoding(
    fname, test_file,
    unicode_whitelist=unicode_whitelist,
    unicode_strict_whitelist=unicode_strict_whitelist):
    """Test helper function for unicode test

    The test may have to operate on filewise manner, so it had moved
    to a separate process.
    """
    has_unicode = False

    is_in_whitelist = False
    is_in_strict_whitelist = False
    for patt in unicode_whitelist:
        if fnmatch.fnmatch(fname, patt):
            is_in_whitelist = True
            break
    for patt in unicode_strict_whitelist:
        if fnmatch.fnmatch(fname, patt):
            is_in_strict_whitelist = True
            is_in_whitelist = True
            break

    if is_in_whitelist:
        for idx, line in enumerate(test_file):
            try:
                line.encode(encoding='ascii')
            except (UnicodeEncodeError, UnicodeDecodeError):
                has_unicode = True

        if not has_unicode and not is_in_strict_whitelist:
            assert False, message_unicode_D % fname

    else:
        for idx, line in enumerate(test_file):
            try:
                line.encode(encoding='ascii')
            except (UnicodeEncodeError, UnicodeDecodeError):
                assert False, message_unicode_B % (fname, idx + 1)
