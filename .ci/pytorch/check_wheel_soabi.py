#!/usr/bin/env python3
"""Check that a built wheel is self-consistent about its extension module ABI.

FindPython does not run the interpreter to compute ``Python_SOABI`` while
cross-compiling, so ``Python_add_library(... WITH_SOABI)`` composes a name with
no ABI tag and the wheel ends up carrying a bare ``torch/_C.so``. That was
gh-189388, reported from conda-forge rather than caught here, because the cross
job builds a wheel and never imports it. An earlier form of the same bug dropped
the module from the wheel entirely, when ``setup.py``'s ``package_data`` glob was
built from ``EXT_SUFFIX`` and did not match the untagged file.

The expected ABI comes from the wheel itself, not from the interpreter running
this script. A wheel names its tags twice -- in the filename and in
``.dist-info/WHEEL`` -- and the extension module inside has to agree with them.
Deriving the answer from ``sysconfig`` instead would only give a true result when
run under the interpreter the wheel was built for, which for a cross build is the
cross interpreter rather than the obvious one; this avoids that coupling.

Checks, in order:

1. the filename tags and the ``Tag:`` lines in ``.dist-info/WHEEL`` agree;
2. the wheel ships exactly one ``torch/_C`` extension module;
3. it is not the untagged ``torch/_C.so``;
4. its ABI tag matches the wheel's -- ``cp312`` against ``cpython-312``.
"""

import argparse
import itertools
import posixpath
import re
import sys
import zipfile


# The suffixes CPython's import system recognises for extension modules, per
# importlib.machinery.EXTENSION_SUFFIXES (.so on POSIX including macOS, .pyd on
# Windows). Notably not .dylib: macOS uses that for shared libraries -- torch
# ships several -- but never for an importable module.
EXTENSION_SUFFIXES = (".so", ".pyd")

# torch/_C.<abi>.<ext> or the untagged torch/_C.<ext>. Deliberately not a plain
# prefix match: torch/_C/ is a directory of .pyi stubs, and a sibling module such
# as torch/_C_foo.so would not be this one.
MEMBER_RE = re.compile(r"torch/_C\.[^/]+")


class CheckFailed(Exception):
    pass


def filename_tags(wheel: str) -> tuple[str, str, str]:
    """Return the (python, abi, platform) tags from a wheel filename."""
    stem = posixpath.basename(wheel)
    if not stem.endswith(".whl"):
        raise CheckFailed(f"{wheel}: not a .whl filename")
    parts = stem[: -len(".whl")].split("-")
    if len(parts) < 5:
        raise CheckFailed(f"{wheel}: filename has too few '-' separated fields")
    python_tag, abi_tag, platform_tag = parts[-3:]
    return python_tag, abi_tag, platform_tag


def metadata_tags(archive: zipfile.ZipFile, wheel: str) -> set[str]:
    """Return the Tag: values declared in .dist-info/WHEEL."""
    wheel_files = [
        n for n in archive.namelist() if re.fullmatch(r"[^/]+\.dist-info/WHEEL", n)
    ]
    if len(wheel_files) != 1:
        raise CheckFailed(
            f"{wheel}: expected exactly one .dist-info/WHEEL, found {len(wheel_files)}"
        )
    text = archive.read(wheel_files[0]).decode("utf-8", "replace")
    return {
        line.split(":", 1)[1].strip()
        for line in text.splitlines()
        if line.lower().startswith("tag:")
    }


def extension_members(archive: zipfile.ZipFile) -> list[str]:
    return sorted(
        n
        for n in archive.namelist()
        if MEMBER_RE.fullmatch(n) and n.endswith(EXTENSION_SUFFIXES)
    )


def abi_from_member(member: str) -> str | None:
    """Return the ABI tag in torch/_C.<abi>.<ext>, or None when untagged."""
    base = posixpath.basename(member)
    stem = base[: base.rindex(".")]
    _, _, abi = stem.partition(".")
    return abi or None


def check_wheel(wheel: str) -> bool:
    """Check one wheel. Returns False if it ships no extension module."""
    python_tag, abi_tag, platform_tag = filename_tags(wheel)
    with zipfile.ZipFile(wheel) as archive:
        declared = metadata_tags(archive, wheel)
        members = extension_members(archive)

    if not members:
        # Nothing to be inconsistent about -- a libtorch wheel, say. A torch
        # wheel that lost its module is caught by the caller, once nothing at
        # all has been verified.
        return False

    joined = f"{python_tag}-{abi_tag}-{platform_tag}"
    # A filename may compress a tag set ("py2.py3-none-any"); WHEEL lists each
    # expansion on its own Tag: line. Compare the expansions, not the strings.
    expanded = {
        "-".join(combo)
        for combo in itertools.product(
            python_tag.split("."), abi_tag.split("."), platform_tag.split(".")
        )
    }
    if not expanded <= declared:
        raise CheckFailed(
            f"{wheel}: filename says {joined}, .dist-info/WHEEL says "
            f"{', '.join(sorted(declared)) or '(nothing)'}"
        )

    if len(members) > 1:
        raise CheckFailed(f"{wheel}: ships several torch/_C modules: {members}")

    member = members[0]
    abi = abi_from_member(member)
    if abi is None:
        raise CheckFailed(
            f"{wheel}: ships an untagged {member}, but declares {joined}. "
            "Python_SOABI was empty at configure time; see the "
            "CMAKE_CROSSCOMPILING block in cmake/Dependencies.cmake."
        )

    # cp312 -> cpython-312, cp313t -> cpython-313t; abi3 is left alone.
    accepted = [re.sub(r"^cp(?=\d)", "cpython-", a) for a in abi_tag.split(".")]
    if not any(abi.startswith(a) for a in accepted):
        raise CheckFailed(
            f"{wheel}: {member} carries ABI tag '{abi}', but the wheel declares "
            f"'{abi_tag}', so one of {accepted} was expected."
        )

    print(f"OK: {wheel} declares {joined} and ships {member}", flush=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+", help="wheel files to check")
    args = parser.parse_args()

    checked = 0
    for wheel in args.wheels:
        try:
            checked += check_wheel(wheel)
        except CheckFailed as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if not checked:
        print(
            f"ERROR: none of {', '.join(args.wheels)} ships a torch/_C extension "
            "module. It was dropped from the wheel rather than merely misnamed.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
