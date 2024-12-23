# This tests the compilation and execution of the source code generated with
# utilities.codegen. The compilation takes place in a temporary directory that
# is removed after the test. By default the test directory is always removed,
# but this behavior can be changed by setting the environment variable
# SYMPY_TEST_CLEAN_TEMP to:
#   export SYMPY_TEST_CLEAN_TEMP=always   : the default behavior.
#   export SYMPY_TEST_CLEAN_TEMP=success  : only remove the directories of working tests.
#   export SYMPY_TEST_CLEAN_TEMP=never    : never remove the directories with the test code.
# When a directory is not removed, the necessary information is printed on
# screen to find the files that belong to the (failed) tests. If a test does
# not fail, py.test captures all the output and you will not see the directories
# corresponding to the successful tests. Use the --nocapture option to see all
# the output.

# All tests below have a counterpart in utilities/test/test_codegen.py. In the
# latter file, the resulting code is compared with predefined strings, without
# compilation or execution.

# All the generated Fortran code should conform with the Fortran 95 standard,
# and all the generated C code should be ANSI C, which facilitates the
# incorporation in various projects. The tests below assume that the binary cc
# is somewhere in the path and that it can compile ANSI C code.

from sympy.abc import x, y, z
from sympy.external import import_module
from sympy.testing.pytest import skip
from sympy.utilities.codegen import codegen, make_routine, get_code_generator
import sys
import os
import tempfile
import subprocess


pyodide_js = import_module('pyodide_js')

# templates for the main program that will test the generated code.

main_template = {}
main_template['F95'] = """
program main
  include "codegen.h"
  integer :: result;
  result = 0

  %(statements)s

  call exit(result)
end program
"""

main_template['C89'] = """
#include "codegen.h"
#include <stdio.h>
#include <math.h>

int main() {
  int result = 0;

  %(statements)s

  return result;
}
"""
main_template['C99'] = main_template['C89']
# templates for the numerical tests

numerical_test_template = {}
numerical_test_template['C89'] = """
  if (fabs(%(call)s)>%(threshold)s) {
    printf("Numerical validation failed: %(call)s=%%e threshold=%(threshold)s\\n", %(call)s);
    result = -1;
  }
"""
numerical_test_template['C99'] = numerical_test_template['C89']

numerical_test_template['F95'] = """
  if (abs(%(call)s)>%(threshold)s) then
    write(6,"('Numerical validation failed:')")
    write(6,"('%(call)s=',e15.5,'threshold=',e15.5)") %(call)s, %(threshold)s
    result = -1;
  end if
"""
# command sequences for supported compilers

compile_commands = {}
compile_commands['cc'] = [
    "cc -c codegen.c -o codegen.o",
    "cc -c main.c -o main.o",
    "cc main.o codegen.o -lm -o test.exe"
]

compile_commands['gfortran'] = [
    "gfortran -c codegen.f90 -o codegen.o",
    "gfortran -ffree-line-length-none -c main.f90 -o main.o",
    "gfortran main.o codegen.o -o test.exe"
]

compile_commands['g95'] = [
    "g95 -c codegen.f90 -o codegen.o",
    "g95 -ffree-line-length-huge -c main.f90 -o main.o",
    "g95 main.o codegen.o -o test.exe"
]

compile_commands['ifort'] = [
    "ifort -c codegen.f90 -o codegen.o",
    "ifort -c main.f90 -o main.o",
    "ifort main.o codegen.o -o test.exe"
]

combinations_lang_compiler = [
    ('C89', 'cc'),
    ('C99', 'cc'),
    ('F95', 'ifort'),
    ('F95', 'gfortran'),
    ('F95', 'g95')
]


def try_run(commands):
    """Run a series of commands and only return True if all ran fine."""
    if pyodide_js:
        return False
    with open(os.devnull, 'w') as null:
        for command in commands:
            retcode = subprocess.call(command, stdout=null, shell=True,
                    stderr=subprocess.STDOUT)
            if retcode != 0:
                return False
    return True


def run_test(label, routines, numerical_tests, language, commands, friendly=True):
    """A driver for the codegen tests.

       This driver assumes that a compiler ifort is present in the PATH and that
       ifort is (at least) a Fortran 90 compiler. The generated code is written in
       a temporary directory, together with a main program that validates the
       generated code. The test passes when the compilation and the validation
       run correctly.
    """

    # Check input arguments before touching the file system
    language = language.upper()
    assert language in main_template
    assert language in numerical_test_template

    # Check that environment variable makes sense
    clean = os.getenv('SYMPY_TEST_CLEAN_TEMP', 'always').lower()
    if clean not in ('always', 'success', 'never'):
        raise ValueError("SYMPY_TEST_CLEAN_TEMP must be one of the following: 'always', 'success' or 'never'.")

    # Do all the magic to compile, run and validate the test code
    # 1) prepare the temporary working directory, switch to that dir
    work = tempfile.mkdtemp("_sympy_%s_test" % language, "%s_" % label)
    oldwork = os.getcwd()
    os.chdir(work)

    # 2) write the generated code
    if friendly:
        # interpret the routines as a name_expr list and call the friendly
        # function codegen
        codegen(routines, language, "codegen", to_files=True)
    else:
        code_gen = get_code_generator(language, "codegen")
        code_gen.write(routines, "codegen", to_files=True)

    # 3) write a simple main program that links to the generated code, and that
    #    includes the numerical tests
    test_strings = []
    for fn_name, args, expected, threshold in numerical_tests:
        call_string = "%s(%s)-(%s)" % (
            fn_name, ",".join(str(arg) for arg in args), expected)
        if language == "F95":
            call_string = fortranize_double_constants(call_string)
            threshold = fortranize_double_constants(str(threshold))
        test_strings.append(numerical_test_template[language] % {
            "call": call_string,
            "threshold": threshold,
        })

    if language == "F95":
        f_name = "main.f90"
    elif language.startswith("C"):
        f_name = "main.c"
    else:
        raise NotImplementedError(
            "FIXME: filename extension unknown for language: %s" % language)

    with open(f_name, "w") as f:
        f.write(
            main_template[language] % {'statements': "".join(test_strings)})

    # 4) Compile and link
    compiled = try_run(commands)

    # 5) Run if compiled
    if compiled:
        executed = try_run(["./test.exe"])
    else:
        executed = False

    # 6) Clean up stuff
    if clean == 'always' or (clean == 'success' and compiled and executed):
        def safe_remove(filename):
            if os.path.isfile(filename):
                os.remove(filename)
        safe_remove("codegen.f90")
        safe_remove("codegen.c")
        safe_remove("codegen.h")
        safe_remove("codegen.o")
        safe_remove("main.f90")
        safe_remove("main.c")
        safe_remove("main.o")
        safe_remove("test.exe")
        os.chdir(oldwork)
        os.rmdir(work)
    else:
        print("TEST NOT REMOVED: %s" % work, file=sys.stderr)
        os.chdir(oldwork)

    # 7) Do the assertions in the end
    assert compiled, "failed to compile %s code with:\n%s" % (
        language, "\n".join(commands))
    assert executed, "failed to execute %s code from:\n%s" % (
        language, "\n".join(commands))


def fortranize_double_constants(code_string):
    """
    Replaces every literal float with literal doubles
    """
    import re
    pattern_exp = re.compile(r'\d+(\.)?\d*[eE]-?\d+')
    pattern_float = re.compile(r'\d+\.\d*(?!\d*d)')

    def subs_exp(matchobj):
        return re.sub('[eE]', 'd', matchobj.group(0))

    def subs_float(matchobj):
        return "%sd0" % matchobj.group(0)

    code_string = pattern_exp.sub(subs_exp, code_string)
    code_string = pattern_float.sub(subs_float, code_string)

    return code_string


def is_feasible(language, commands):
    # This test should always work, otherwise the compiler is not present.
    routine = make_routine("test", x)
    numerical_tests = [
        ("test", ( 1.0,), 1.0, 1e-15),
        ("test", (-1.0,), -1.0, 1e-15),
    ]
    try:
        run_test("is_feasible", [routine], numerical_tests, language, commands,
                 friendly=False)
        return True
    except AssertionError:
        return False

valid_lang_commands = []
invalid_lang_compilers = []
for lang, compiler in combinations_lang_compiler:
    commands = compile_commands[compiler]
    if is_feasible(lang, commands):
        valid_lang_commands.append((lang, commands))
    else:
        invalid_lang_compilers.append((lang, compiler))

# We test all language-compiler combinations, just to report what is skipped

def test_C89_cc():
    if ("C89", 'cc') in invalid_lang_compilers:
        skip("`cc' command didn't work as expected (C89)")


def test_C99_cc():
    if ("C99", 'cc') in invalid_lang_compilers:
        skip("`cc' command didn't work as expected (C99)")


def test_F95_ifort():
    if ("F95", 'ifort') in invalid_lang_compilers:
        skip("`ifort' command didn't work as expected")


def test_F95_gfortran():
    if ("F95", 'gfortran') in invalid_lang_compilers:
        skip("`gfortran' command didn't work as expected")


def test_F95_g95():
    if ("F95", 'g95') in invalid_lang_compilers:
        skip("`g95' command didn't work as expected")

# Here comes the actual tests


def test_basic_codegen():
    numerical_tests = [
        ("test", (1.0, 6.0, 3.0), 21.0, 1e-15),
        ("test", (-1.0, 2.0, -2.5), -2.5, 1e-15),
    ]
    name_expr = [("test", (x + y)*z)]
    for lang, commands in valid_lang_commands:
        run_test("basic_codegen", name_expr, numerical_tests, lang, commands)


def test_intrinsic_math1_codegen():
    # not included: log10
    from sympy.core.evalf import N
    from sympy.functions import ln
    from sympy.functions.elementary.exponential import log
    from sympy.functions.elementary.hyperbolic import (cosh, sinh, tanh)
    from sympy.functions.elementary.integers import (ceiling, floor)
    from sympy.functions.elementary.miscellaneous import sqrt
    from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin, tan)
    name_expr = [
        ("test_fabs", abs(x)),
        ("test_acos", acos(x)),
        ("test_asin", asin(x)),
        ("test_atan", atan(x)),
        ("test_cos", cos(x)),
        ("test_cosh", cosh(x)),
        ("test_log", log(x)),
        ("test_ln", ln(x)),
        ("test_sin", sin(x)),
        ("test_sinh", sinh(x)),
        ("test_sqrt", sqrt(x)),
        ("test_tan", tan(x)),
        ("test_tanh", tanh(x)),
    ]
    numerical_tests = []
    for name, expr in name_expr:
        for xval in 0.2, 0.5, 0.8:
            expected = N(expr.subs(x, xval))
            numerical_tests.append((name, (xval,), expected, 1e-14))
    for lang, commands in valid_lang_commands:
        if lang.startswith("C"):
            name_expr_C = [("test_floor", floor(x)), ("test_ceil", ceiling(x))]
        else:
            name_expr_C = []
        run_test("intrinsic_math1", name_expr + name_expr_C,
                 numerical_tests, lang, commands)


def test_instrinsic_math2_codegen():
    # not included: frexp, ldexp, modf, fmod
    from sympy.core.evalf import N
    from sympy.functions.elementary.trigonometric import atan2
    name_expr = [
        ("test_atan2", atan2(x, y)),
        ("test_pow", x**y),
    ]
    numerical_tests = []
    for name, expr in name_expr:
        for xval, yval in (0.2, 1.3), (0.5, -0.2), (0.8, 0.8):
            expected = N(expr.subs(x, xval).subs(y, yval))
            numerical_tests.append((name, (xval, yval), expected, 1e-14))
    for lang, commands in valid_lang_commands:
        run_test("intrinsic_math2", name_expr, numerical_tests, lang, commands)


def test_complicated_codegen():
    from sympy.core.evalf import N
    from sympy.functions.elementary.trigonometric import (cos, sin, tan)
    name_expr = [
        ("test1", ((sin(x) + cos(y) + tan(z))**7).expand()),
        ("test2", cos(cos(cos(cos(cos(cos(cos(cos(x + y + z))))))))),
    ]
    numerical_tests = []
    for name, expr in name_expr:
        for xval, yval, zval in (0.2, 1.3, -0.3), (0.5, -0.2, 0.0), (0.8, 2.1, 0.8):
            expected = N(expr.subs(x, xval).subs(y, yval).subs(z, zval))
            numerical_tests.append((name, (xval, yval, zval), expected, 1e-12))
    for lang, commands in valid_lang_commands:
        run_test(
            "complicated_codegen", name_expr, numerical_tests, lang, commands)
