# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# Do NOT include this module directly into any of your code. It is meant as
# a library for Check*CompilerFlag.cmake modules. It's content may change in
# any way between releases.

macro (CHECK_COMPILER_FLAG_COMMON_PATTERNS _VAR)
  set(${_VAR}
    FAIL_REGEX "[Uu]nrecogni[sz]ed [^\n]*option"                             # GNU, NAG, Fujitsu
    FAIL_REGEX "switch [^\n]* is no longer supported"                        # GNU
    FAIL_REGEX "unknown [^\n]*option"                                        # Clang
    FAIL_REGEX "optimization flag [^\n]* not supported"                      # Clang
    FAIL_REGEX "unknown argument ignored"                                    # Clang (cl)
    FAIL_REGEX "ignoring unknown option"                                     # MSVC, Intel
    FAIL_REGEX "warning D9002"                                               # MSVC, any lang
    FAIL_REGEX "option[^\n]*not supported"                                   # Intel
    FAIL_REGEX "invalid argument [^\n]*option"                               # Intel
    FAIL_REGEX "ignoring option [^\n]*argument required"                     # Intel
    FAIL_REGEX "ignoring option [^\n]*argument is of wrong type"             # Intel
    # noqa: spellcheck off
    FAIL_REGEX "[Uu]nknown option"                                           # HP
    # noqa: spellcheck on
    FAIL_REGEX "[Ww]arning: [Oo]ption"                                       # SunPro
    FAIL_REGEX "command option [^\n]* is not recognized"                     # XL
    FAIL_REGEX "command option [^\n]* contains an incorrect subargument"     # XL
    FAIL_REGEX "Option [^\n]* is not recognized.  Option will be ignored."   # XL
    FAIL_REGEX "not supported in this configuration. ignored"                # AIX
    FAIL_REGEX "File with unknown suffix passed to linker"                   # PGI
    # noqa: spellcheck off
    FAIL_REGEX "[Uu]nknown switch"                                           # PGI
    # noqa: spellcheck on
    FAIL_REGEX "WARNING: unknown flag:"                                      # Open64
    FAIL_REGEX "Incorrect command line option:"                              # Borland
    FAIL_REGEX "Warning: illegal option"                                     # SunStudio 12
    FAIL_REGEX "[Ww]arning: Invalid suboption"                               # Fujitsu
    FAIL_REGEX "An invalid option [^\n]* appears on the command line"        # Cray
    FAIL_REGEX "WARNING: invalid compiler option"                            # TI armcl
  )
endmacro ()
