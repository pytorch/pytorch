# SPDX-License-Identifier: MIT

from attr import (
    NOTHING,
    Attribute,
    AttrsInstance,
    Converter,
    Factory,
    _make_getattr,
    assoc,
    cmp_using,
    define,
    evolve,
    field,
    fields,
    fields_dict,
    frozen,
    has,
    make_class,
    mutable,
    resolve_types,
    validate,
)
from attr._next_gen import asdict, astuple

from . import converters, exceptions, filters, setters, validators


__all__ = [
    "__author__",
    "__copyright__",
    "__description__",
    "__doc__",
    "__email__",
    "__license__",
    "__title__",
    "__url__",
    "__version__",
    "__version_info__",
    "asdict",
    "assoc",
    "astuple",
    "Attribute",
    "AttrsInstance",
    "cmp_using",
    "Converter",
    "converters",
    "define",
    "evolve",
    "exceptions",
    "Factory",
    "field",
    "fields_dict",
    "fields",
    "filters",
    "frozen",
    "has",
    "make_class",
    "mutable",
    "NOTHING",
    "resolve_types",
    "setters",
    "validate",
    "validators",
]

__getattr__ = _make_getattr(__name__)
