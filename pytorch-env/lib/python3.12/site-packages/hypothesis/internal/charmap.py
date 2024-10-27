# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import codecs
import gzip
import json
import os
import sys
import tempfile
import unicodedata
from functools import lru_cache

from hypothesis.configuration import storage_directory
from hypothesis.control import _current_build_context
from hypothesis.errors import InvalidArgument
from hypothesis.internal.intervalsets import IntervalSet

intervals = tuple[tuple[int, int], ...]
cache_type = dict[tuple[tuple[str, ...], int, int, intervals], IntervalSet]


def charmap_file(fname="charmap"):
    return storage_directory(
        "unicode_data", unicodedata.unidata_version, f"{fname}.json.gz"
    )


_charmap = None


def charmap():
    """Return a dict that maps a Unicode category, to a tuple of 2-tuples
    covering the codepoint intervals for characters in that category.

    >>> charmap()['Co']
    ((57344, 63743), (983040, 1048573), (1048576, 1114109))
    """
    global _charmap
    # Best-effort caching in the face of missing files and/or unwritable
    # filesystems is fairly simple: check if loaded, else try loading,
    # else calculate and try writing the cache.
    if _charmap is None:
        f = charmap_file()
        try:
            with gzip.GzipFile(f, "rb") as i:
                tmp_charmap = dict(json.load(i))

        except Exception:
            # This loop is reduced to using only local variables for performance;
            # indexing and updating containers is a ~3x slowdown.  This doesn't fix
            # https://github.com/HypothesisWorks/hypothesis/issues/2108 but it helps.
            category = unicodedata.category  # Local variable -> ~20% speedup!
            tmp_charmap = {}
            last_cat = category(chr(0))
            last_start = 0
            for i in range(1, sys.maxunicode + 1):
                cat = category(chr(i))
                if cat != last_cat:
                    tmp_charmap.setdefault(last_cat, []).append([last_start, i - 1])
                    last_cat, last_start = cat, i
            tmp_charmap.setdefault(last_cat, []).append([last_start, sys.maxunicode])

            try:
                # Write the Unicode table atomically
                tmpdir = storage_directory("tmp")
                tmpdir.mkdir(exist_ok=True, parents=True)
                fd, tmpfile = tempfile.mkstemp(dir=tmpdir)
                os.close(fd)
                # Explicitly set the mtime to get reproducible output
                with gzip.GzipFile(tmpfile, "wb", mtime=1) as o:
                    result = json.dumps(sorted(tmp_charmap.items()))
                    o.write(result.encode())

                os.renames(tmpfile, f)
            except Exception:
                pass

        # convert between lists and tuples
        _charmap = {
            k: tuple(tuple(pair) for pair in pairs) for k, pairs in tmp_charmap.items()
        }
        # each value is a tuple of 2-tuples (that is, tuples of length 2)
        # and that both elements of that tuple are integers.
        for vs in _charmap.values():
            ints = list(sum(vs, ()))
            assert all(isinstance(x, int) for x in ints)
            assert ints == sorted(ints)
            assert all(len(tup) == 2 for tup in vs)

    assert _charmap is not None
    return _charmap


@lru_cache(maxsize=None)
def intervals_from_codec(codec_name: str) -> IntervalSet:  # pragma: no cover
    """Return an IntervalSet of characters which are part of this codec."""
    assert codec_name == codecs.lookup(codec_name).name
    fname = charmap_file(f"codec-{codec_name}")
    try:
        with gzip.GzipFile(fname) as gzf:
            encodable_intervals = json.load(gzf)

    except Exception:
        # This loop is kinda slow, but hopefully we don't need to do it very often!
        encodable_intervals = []
        for i in range(sys.maxunicode + 1):
            try:
                chr(i).encode(codec_name)
            except Exception:  # usually _but not always_ UnicodeEncodeError
                pass
            else:
                encodable_intervals.append((i, i))

    res = IntervalSet(encodable_intervals)
    res = res.union(res)
    try:
        # Write the Unicode table atomically
        tmpdir = storage_directory("tmp")
        tmpdir.mkdir(exist_ok=True, parents=True)
        fd, tmpfile = tempfile.mkstemp(dir=tmpdir)
        os.close(fd)
        # Explicitly set the mtime to get reproducible output
        with gzip.GzipFile(tmpfile, "wb", mtime=1) as o:
            o.write(json.dumps(res.intervals).encode())
        os.renames(tmpfile, fname)
    except Exception:
        pass
    return res


_categories = None


def categories():
    """Return a tuple of Unicode categories in a normalised order.

    >>> categories() # doctest: +ELLIPSIS
    ('Zl', 'Zp', 'Co', 'Me', 'Pc', ..., 'Cc', 'Cs')
    """
    global _categories
    if _categories is None:
        cm = charmap()
        _categories = sorted(cm.keys(), key=lambda c: len(cm[c]))
        _categories.remove("Cc")  # Other, Control
        _categories.remove("Cs")  # Other, Surrogate
        _categories.append("Cc")
        _categories.append("Cs")
    return tuple(_categories)


def as_general_categories(cats, name="cats"):
    """Return a tuple of Unicode categories in a normalised order.

    This function expands one-letter designations of a major class to include
    all subclasses:

    >>> as_general_categories(['N'])
    ('Nd', 'Nl', 'No')

    See section 4.5 of the Unicode standard for more on classes:
    https://www.unicode.org/versions/Unicode10.0.0/ch04.pdf

    If the collection ``cats`` includes any elements that do not represent a
    major class or a class with subclass, a deprecation warning is raised.
    """
    if cats is None:
        return None
    major_classes = ("L", "M", "N", "P", "S", "Z", "C")
    cs = categories()
    out = set(cats)
    for c in cats:
        if c in major_classes:
            out.discard(c)
            out.update(x for x in cs if x.startswith(c))
        elif c not in cs:
            raise InvalidArgument(
                f"In {name}={cats!r}, {c!r} is not a valid Unicode category."
            )
    return tuple(c for c in cs if c in out)


category_index_cache = {(): ()}


def _category_key(cats):
    """Return a normalised tuple of all Unicode categories that are in
    `include`, but not in `exclude`.

    If include is None then default to including all categories.
    Any item in include that is not a unicode character will be excluded.

    >>> _category_key(exclude=['So'], include=['Lu', 'Me', 'Cs', 'So'])
    ('Me', 'Lu', 'Cs')
    """
    cs = categories()
    if cats is None:
        cats = set(cs)
    return tuple(c for c in cs if c in cats)


def _query_for_key(key):
    """Return a tuple of codepoint intervals covering characters that match one
    or more categories in the tuple of categories `key`.

    >>> _query_for_key(categories())
    ((0, 1114111),)
    >>> _query_for_key(('Zl', 'Zp', 'Co'))
    ((8232, 8233), (57344, 63743), (983040, 1048573), (1048576, 1114109))
    """
    context = _current_build_context.value
    if context is None or not context.data.provider.avoid_realization:
        try:
            return category_index_cache[key]
        except KeyError:
            pass
    elif not key:  # pragma: no cover  # only on alternative backends
        return ()
    assert key
    if set(key) == set(categories()):
        result = IntervalSet([(0, sys.maxunicode)])
    else:
        result = IntervalSet(_query_for_key(key[:-1])).union(
            IntervalSet(charmap()[key[-1]])
        )
    assert isinstance(result, IntervalSet)
    if context is None or not context.data.provider.avoid_realization:
        category_index_cache[key] = result.intervals
    return result.intervals


limited_category_index_cache: cache_type = {}


def query(
    *,
    categories=None,
    min_codepoint=None,
    max_codepoint=None,
    include_characters="",
    exclude_characters="",
):
    """Return a tuple of intervals covering the codepoints for all characters
    that meet the criteria.

    >>> query()
    ((0, 1114111),)
    >>> query(min_codepoint=0, max_codepoint=128)
    ((0, 128),)
    >>> query(min_codepoint=0, max_codepoint=128, categories=['Lu'])
    ((65, 90),)
    >>> query(min_codepoint=0, max_codepoint=128, categories=['Lu'],
    ...       include_characters='☃')
    ((65, 90), (9731, 9731))
    """
    if min_codepoint is None:
        min_codepoint = 0
    if max_codepoint is None:
        max_codepoint = sys.maxunicode
    catkey = _category_key(categories)
    character_intervals = IntervalSet.from_string(include_characters or "")
    exclude_intervals = IntervalSet.from_string(exclude_characters or "")
    qkey = (
        catkey,
        min_codepoint,
        max_codepoint,
        character_intervals.intervals,
        exclude_intervals.intervals,
    )
    context = _current_build_context.value
    if context is None or not context.data.provider.avoid_realization:
        try:
            return limited_category_index_cache[qkey]
        except KeyError:
            pass
    base = _query_for_key(catkey)
    result = []
    for u, v in base:
        if v >= min_codepoint and u <= max_codepoint:
            result.append((max(u, min_codepoint), min(v, max_codepoint)))
    result = (IntervalSet(result) | character_intervals) - exclude_intervals
    if context is None or not context.data.provider.avoid_realization:
        limited_category_index_cache[qkey] = result
    return result
