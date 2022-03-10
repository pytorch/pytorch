import sys
import re

QUOTE_INCLUDE_RE = re.compile(r'^#include "(.*)"')
ANGLE_INCLUDE_RE = re.compile(r'^#include <(.*)>')

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

def main() -> None:
    for line in sys.stdin:
        # Convert all quoted includes to angle brackets
        match = QUOTE_INCLUDE_RE.match(line)
        if match is not None:
            print(f"#include <{match.group(1)}>{line[match.end(0):]}", end='')
            continue

        match = ANGLE_INCLUDE_RE.match(line)
        if match is not None:
            path = f"<{match.group(1)}>"
            new_path = STD_C_HEADER_MAP.get(path, path)
            tail = line[match.end(0):]
            if len(tail) > 1:
                tail = ' ' + tail
            print(f"#include {new_path}{tail}", end='')
            continue

        print(line, end='')

if __name__ == "__main__":
    main()
