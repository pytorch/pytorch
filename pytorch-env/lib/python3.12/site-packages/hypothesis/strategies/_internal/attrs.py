# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from functools import reduce
from itertools import chain

import attr

from hypothesis import strategies as st
from hypothesis.errors import ResolutionFailed
from hypothesis.internal.compat import get_type_hints
from hypothesis.strategies._internal.core import BuildsStrategy
from hypothesis.strategies._internal.types import is_a_type, type_sorting_key
from hypothesis.utils.conventions import infer


def get_attribute_by_alias(fields, alias, *, target=None):
    """
    Get an attrs attribute by its alias, rather than its name (compare
    getattr(fields, name)).

    ``target`` is used only to provide a nicer error message, and can be safely
    omitted.
    """
    # attrs supports defining an alias for a field, which is the name used when
    # defining __init__. The init args are what we pull from when determining
    # what parameters we need to supply to the class, so it's what we need to
    # match against as well, rather than the class-level attribute name.
    matched_fields = [f for f in fields if f.alias == alias]
    if not matched_fields:
        raise TypeError(
            f"Unexpected keyword argument {alias} for attrs class"
            f"{f' {target}' if target else ''}. Expected one of "
            f"{[f.name for f in fields]}"
        )
    # alias is used as an arg in __init__, so it is guaranteed to be unique, if
    # it exists.
    assert len(matched_fields) == 1
    return matched_fields[0]


def from_attrs(target, args, kwargs, to_infer):
    """An internal version of builds(), specialised for Attrs classes."""
    fields = attr.fields(target)
    kwargs = {k: v for k, v in kwargs.items() if v is not infer}
    for name in to_infer:
        attrib = get_attribute_by_alias(fields, name, target=target)
        kwargs[name] = from_attrs_attribute(attrib, target)
    # We might make this strategy more efficient if we added a layer here that
    # retries drawing if validation fails, for improved composition.
    # The treatment of timezones in datetimes() provides a precedent.
    return BuildsStrategy(target, args, kwargs)


def from_attrs_attribute(attrib, target):
    """Infer a strategy from the metadata on an attr.Attribute object."""
    # Try inferring from the default argument.  Note that this will only help if
    # the user passed `...` to builds() for this attribute, but in that case
    # we use it as the minimal example.
    default = st.nothing()
    if isinstance(attrib.default, attr.Factory):
        if not attrib.default.takes_self:
            default = st.builds(attrib.default.factory)
    elif attrib.default is not attr.NOTHING:
        default = st.just(attrib.default)

    # Try inferring None, exact values, or type from attrs provided validators.
    null = st.nothing()  # updated to none() on seeing an OptionalValidator
    in_collections = []  # list of in_ validator collections to sample from
    validator_types = set()  # type constraints to pass to types_to_strategy()
    if attrib.validator is not None:
        validator = attrib.validator
        if isinstance(validator, attr.validators._OptionalValidator):
            null = st.none()
            validator = validator.validator
        if isinstance(validator, attr.validators._AndValidator):
            vs = validator._validators
        else:
            vs = [validator]
        for v in vs:
            if isinstance(v, attr.validators._InValidator):
                if isinstance(v.options, str):
                    in_collections.append(list(all_substrings(v.options)))
                else:
                    in_collections.append(v.options)
            elif isinstance(v, attr.validators._InstanceOfValidator):
                validator_types.add(v.type)

    # This is the important line.  We compose the final strategy from various
    # parts.  The default value, if any, is the minimal shrink, followed by
    # None (again, if allowed).  We then prefer to sample from values passed
    # to an in_ validator if available, but infer from a type otherwise.
    # Pick one because (sampled_from((1, 2)) | from_type(int)) would usually
    # fail validation by generating e.g. zero!
    if in_collections:
        sample = st.sampled_from(list(ordered_intersection(in_collections)))
        strat = default | null | sample
    else:
        strat = default | null | types_to_strategy(attrib, validator_types)

    # Better to give a meaningful error here than an opaque "could not draw"
    # when we try to get a value but have lost track of where this was created.
    if strat.is_empty:
        raise ResolutionFailed(
            "Cannot infer a strategy from the default, validator, type, or "
            f"converter for attribute={attrib!r} of class={target!r}"
        )
    return strat


def types_to_strategy(attrib, types):
    """Find all the type metadata for this attribute, reconcile it, and infer a
    strategy from the mess."""
    # If we know types from the validator(s), that's sufficient.
    if len(types) == 1:
        (typ,) = types
        if isinstance(typ, tuple):
            return st.one_of(*map(st.from_type, typ))
        return st.from_type(typ)
    elif types:
        # We have a list of tuples of types, and want to find a type
        # (or tuple of types) that is a subclass of all of of them.
        type_tuples = [k if isinstance(k, tuple) else (k,) for k in types]
        # Flatten the list, filter types that would fail validation, and
        # sort so that ordering is stable between runs and shrinks well.
        allowed = [
            t
            for t in set(sum(type_tuples, ()))
            if all(issubclass(t, tup) for tup in type_tuples)
        ]
        allowed.sort(key=type_sorting_key)
        return st.one_of([st.from_type(t) for t in allowed])

    # Otherwise, try the `type` attribute as a fallback, and finally try
    # the type hints on a converter (desperate!) before giving up.
    if is_a_type(getattr(attrib, "type", None)):
        # The convoluted test is because variable annotations may be stored
        # in string form; attrs doesn't evaluate them and we don't handle them.
        # See PEP 526, PEP 563, and Hypothesis issue #1004 for details.
        return st.from_type(attrib.type)

    converter = getattr(attrib, "converter", None)
    if isinstance(converter, type):
        return st.from_type(converter)
    elif callable(converter):
        hints = get_type_hints(converter)
        if "return" in hints:
            return st.from_type(hints["return"])

    return st.nothing()


def ordered_intersection(in_):
    """Set union of n sequences, ordered for reproducibility across runs."""
    intersection = reduce(set.intersection, in_, set(in_[0]))
    for x in chain.from_iterable(in_):
        if x in intersection:
            yield x
            intersection.remove(x)


def all_substrings(s):
    """Generate all substrings of `s`, in order of length then occurrence.
    Includes the empty string (first), and any duplicates that are present.

    >>> list(all_substrings('010'))
    ['', '0', '1', '0', '01', '10', '010']
    """
    yield s[:0]
    for n, _ in enumerate(s):
        for i in range(len(s) - n):
            yield s[i : i + n + 1]
