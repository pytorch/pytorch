import re

QUOTE_INCLUDE_RE = re.compile(r'^#include "(.*)"')
ANGLE_INCLUDE_RE = re.compile(r"^#include <(.*)>")

# By default iwyu will pick the C include, but we prefer the C++ headers
STD_C_HEADER_MAP = {
    "<assert.h>": "<cassert>",
    "<complex.h>": "<ccomplex>",
    "<ctype.h>": "<cctype>",
    "<errno.h>": "<cerrno>",
    "<fenv.h>": "<cfenv>",
    "<float.h>": "<cfloat>",
    "<inttypes.h>": "<cinttypes>",
    "<iso646.h>": "<ciso646>",
    "<limits.h>": "<climits>",
    "<locale.h>": "<clocale>",
    "<math.h>": "<cmath>",
    "<setjmp.h>": "<csetjmp>",
    "<signal.h>": "<csignal>",
    "<stdalign.h>": "<cstdalign>",
    "<stdarg.h>": "<cstdarg>",
    "<stdbool.h>": "<cstdbool>",
    "<stddef.h>": "<cstddef>",
    "<stdint.h>": "<cstdint>",
    "<stdio.h>": "<cstdio>",
    "<stdlib.h>": "<cstdlib>",
    "<string.h>": "<cstring>",
    "<tgmath.h>": "<ctgmath>",
    "<time.h>": "<ctime>",
    "<uchar.h>": "<cuchar>",
    "<wchar.h>": "<cwchar>",
    "<wctype.h>": "<cwctype>",
}


def use_angled_includes(line: str) -> str:
    match = QUOTE_INCLUDE_RE.match(line)
    if match is None:
        return line

    return f"#include <{match.group(1)}>{line[match.end(0):]}"


def use_quotes_for_project_includes(line: str) -> str:
    match = ANGLE_INCLUDE_RE.match(line)
    if match is None:
        return line

    filename = match.group(1)
    if not filename.endswith(".h"):
        return line

    if (
        filename.startswith("c10")
        or filename.startswith("ATen")
        or filename.startswith("torch")
    ):
        return f'#include "{filename}"{line[match.end(0):]}'

    return line


def normalize_c_headers(line: str) -> str:
    match = ANGLE_INCLUDE_RE.match(line)
    if match is None:
        return line

    path = f"<{match.group(1)}>"
    new_path = STD_C_HEADER_MAP.get(path, None)
    if new_path is None:
        return line

    tail = line[match.end(0) :]
    if len(tail) > 1:
        tail = " " + tail
    return f"#include {new_path}{tail}"
