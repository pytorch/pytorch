dnl Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
dnl file LICENSE.rst or https://cmake.org/licensing for details.

# CMAKE_FIND_BINARY
# -----------------
# Finds the cmake command-line binary and sets its absolute path in the
# CMAKE_BINARY variable.
AC_DEFUN([CMAKE_FIND_BINARY],
[AC_ARG_VAR([CMAKE_BINARY], [path to the cmake binary])dnl

if test "x$ac_cv_env_CMAKE_BINARY_set" != "xset"; then
    AC_PATH_TOOL([CMAKE_BINARY], [cmake])dnl
fi
])dnl

# CMAKE_FIND_PACKAGE(package, lang, [compiler-id], [cmake-args],
#   [action-if-found], [action-if-not-found])
# --------------------------------------------------------------
# Finds a package with CMake.
#
# package:
#   The name of the package as called in CMake with find_package(package).
#
# lang:
#   The programming language to use (e.g., C, CXX, Fortran).
#   See https://cmake.org/cmake/help/latest/command/enable_language.html
#   for a complete list of supported languages.
#
# compiler-id:
#   (Optional) The compiler ID to use. Defaults to GNU.
#   See https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER_ID.html
#   for possible values.
#
# cmake-args:
#   (Optional) Additional arguments to pass to cmake command, e.g.,
#   -DCMAKE_SIZEOF_VOID_P=8.
#
# action-if-found:
#   (Optional) Commands to execute if the package is found.
#
# action-if-not-found:
#   (Optional) Commands to execute if the package is not found.
AC_DEFUN([CMAKE_FIND_PACKAGE], [
AC_REQUIRE([CMAKE_FIND_BINARY])dnl

AC_ARG_VAR([$1][_][$2][FLAGS], [$2 compiler flags for $1. This overrides the cmake output])dnl
AC_ARG_VAR([$1][_LIBS], [linker flags for $1. This overrides the cmake output])dnl

failed=false
AC_MSG_CHECKING([for $1])
if test -z "${$1[]_$2[]FLAGS}"; then
    $1[]_$2[]FLAGS=`$CMAKE_BINARY --find-package "-DNAME=$1" "-DCOMPILER_ID=m4_default([$3], [GNU])" "-DLANGUAGE=$2" -DMODE=COMPILE $4` || failed=true
fi
if test -z "${$1[]_LIBS}"; then
    $1[]_LIBS=`$CMAKE_BINARY --find-package "-DNAME=$1" "-DCOMPILER_ID=m4_default([$3], [GNU])" "-DLANGUAGE=$2" -DMODE=LINK $4` || failed=true
fi

if $failed; then
    unset $1[]_$2[]FLAGS
    unset $1[]_LIBS

    AC_MSG_RESULT([no])
    $6
else
    AC_MSG_RESULT([yes])
    $5
fi[]dnl
])
