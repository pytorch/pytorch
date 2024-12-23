# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import re
import string
from datetime import datetime, timedelta
from decimal import Decimal
from functools import lru_cache
from typing import Any, Callable, TypeVar, Union

import django
from django import forms as df
from django.contrib.auth.forms import UsernameField
from django.core.validators import (
    validate_ipv4_address,
    validate_ipv6_address,
    validate_ipv46_address,
)
from django.db import models as dm

from hypothesis import strategies as st
from hypothesis.errors import InvalidArgument, ResolutionFailed
from hypothesis.internal.validation import check_type
from hypothesis.provisional import urls
from hypothesis.strategies import emails

AnyField = Union[dm.Field, df.Field]
F = TypeVar("F", bound=AnyField)


def numeric_bounds_from_validators(
    field, min_value=float("-inf"), max_value=float("inf")
):
    for v in field.validators:
        if isinstance(v, django.core.validators.MinValueValidator):
            min_value = max(min_value, v.limit_value)
        elif isinstance(v, django.core.validators.MaxValueValidator):
            max_value = min(max_value, v.limit_value)
    return min_value, max_value


def integers_for_field(min_value, max_value):
    def inner(field):
        return st.integers(*numeric_bounds_from_validators(field, min_value, max_value))

    return inner


@lru_cache
def timezones():
    # From Django 4.0, the default is to use zoneinfo instead of pytz.
    assert getattr(django.conf.settings, "USE_TZ", False)
    if django.VERSION < (5, 0, 0) and getattr(
        django.conf.settings, "USE_DEPRECATED_PYTZ", True
    ):
        from hypothesis.extra.pytz import timezones
    else:
        from hypothesis.strategies import timezones

    return timezones()


# Mapping of field types, to strategy objects or functions of (type) -> strategy
_FieldLookUpType = dict[
    type[AnyField],
    Union[st.SearchStrategy, Callable[[Any], st.SearchStrategy]],
]
_global_field_lookup: _FieldLookUpType = {
    dm.SmallIntegerField: integers_for_field(-32768, 32767),
    dm.IntegerField: integers_for_field(-2147483648, 2147483647),
    dm.BigIntegerField: integers_for_field(-9223372036854775808, 9223372036854775807),
    dm.PositiveIntegerField: integers_for_field(0, 2147483647),
    dm.PositiveSmallIntegerField: integers_for_field(0, 32767),
    dm.BooleanField: st.booleans(),
    dm.DateField: st.dates(),
    dm.EmailField: emails(),
    dm.FloatField: st.floats(),
    dm.NullBooleanField: st.one_of(st.none(), st.booleans()),
    dm.URLField: urls(),
    dm.UUIDField: st.uuids(),
    df.DateField: st.dates(),
    df.DurationField: st.timedeltas(),
    df.EmailField: emails(),
    df.FloatField: lambda field: st.floats(
        *numeric_bounds_from_validators(field), allow_nan=False, allow_infinity=False
    ),
    df.IntegerField: integers_for_field(-2147483648, 2147483647),
    df.NullBooleanField: st.one_of(st.none(), st.booleans()),
    df.URLField: urls(),
    df.UUIDField: st.uuids(),
}

_ipv6_strings = st.one_of(
    st.ip_addresses(v=6).map(str),
    st.ip_addresses(v=6).map(lambda addr: addr.exploded),
)


def register_for(field_type):
    def inner(func):
        _global_field_lookup[field_type] = func
        return func

    return inner


@register_for(dm.DateTimeField)
@register_for(df.DateTimeField)
def _for_datetime(field):
    if getattr(django.conf.settings, "USE_TZ", False):
        # avoid https://code.djangoproject.com/ticket/35683
        return st.datetimes(
            min_value=datetime.min + timedelta(days=1),
            max_value=datetime.max - timedelta(days=1),
            timezones=timezones(),
        )
    return st.datetimes()


def using_sqlite():
    try:
        return (
            getattr(django.conf.settings, "DATABASES", {})
            .get("default", {})
            .get("ENGINE", "")
            .endswith(".sqlite3")
        )
    except django.core.exceptions.ImproperlyConfigured:
        return None


@register_for(dm.TimeField)
def _for_model_time(field):
    # SQLITE supports TZ-aware datetimes, but not TZ-aware times.
    if getattr(django.conf.settings, "USE_TZ", False) and not using_sqlite():
        return st.times(timezones=timezones())
    return st.times()


@register_for(df.TimeField)
def _for_form_time(field):
    if getattr(django.conf.settings, "USE_TZ", False):
        return st.times(timezones=timezones())
    return st.times()


@register_for(dm.DurationField)
def _for_duration(field):
    # SQLite stores timedeltas as six bytes of microseconds
    if using_sqlite():
        delta = timedelta(microseconds=2**47 - 1)
        return st.timedeltas(-delta, delta)
    return st.timedeltas()


@register_for(dm.SlugField)
@register_for(df.SlugField)
def _for_slug(field):
    min_size = 1
    if getattr(field, "blank", False) or not getattr(field, "required", True):
        min_size = 0
    return st.text(
        alphabet=string.ascii_letters + string.digits,
        min_size=min_size,
        max_size=field.max_length,
    )


@register_for(dm.GenericIPAddressField)
def _for_model_ip(field):
    return {
        "ipv4": st.ip_addresses(v=4).map(str),
        "ipv6": _ipv6_strings,
        "both": st.ip_addresses(v=4).map(str) | _ipv6_strings,
    }[field.protocol.lower()]


@register_for(df.GenericIPAddressField)
def _for_form_ip(field):
    # the IP address form fields have no direct indication of which type
    #  of address they want, so direct comparison with the validator
    #  function has to be used instead. Sorry for the potato logic here
    if validate_ipv46_address in field.default_validators:
        return st.ip_addresses(v=4).map(str) | _ipv6_strings
    if validate_ipv4_address in field.default_validators:
        return st.ip_addresses(v=4).map(str)
    if validate_ipv6_address in field.default_validators:
        return _ipv6_strings
    raise ResolutionFailed(f"No IP version validator on {field=}")


@register_for(dm.DecimalField)
@register_for(df.DecimalField)
def _for_decimal(field):
    min_value, max_value = numeric_bounds_from_validators(field)
    bound = Decimal(10**field.max_digits - 1) / (10**field.decimal_places)
    return st.decimals(
        min_value=max(min_value, -bound),
        max_value=min(max_value, bound),
        places=field.decimal_places,
    )


def length_bounds_from_validators(field):
    min_size = 1
    max_size = field.max_length
    for v in field.validators:
        if isinstance(v, django.core.validators.MinLengthValidator):
            min_size = max(min_size, v.limit_value)
        elif isinstance(v, django.core.validators.MaxLengthValidator):
            max_size = min(max_size or v.limit_value, v.limit_value)
    return min_size, max_size


@register_for(dm.BinaryField)
def _for_binary(field):
    min_size, max_size = length_bounds_from_validators(field)
    if getattr(field, "blank", False) or not getattr(field, "required", True):
        return st.just(b"") | st.binary(min_size=min_size, max_size=max_size)
    return st.binary(min_size=min_size, max_size=max_size)


@register_for(dm.CharField)
@register_for(dm.TextField)
@register_for(df.CharField)
@register_for(df.RegexField)
@register_for(UsernameField)
def _for_text(field):
    # We can infer a vastly more precise strategy by considering the
    # validators as well as the field type.  This is a minimal proof of
    # concept, but we intend to leverage the idea much more heavily soon.
    # See https://github.com/HypothesisWorks/hypothesis-python/issues/1116
    regexes = [
        re.compile(v.regex, v.flags) if isinstance(v.regex, str) else v.regex
        for v in field.validators
        if isinstance(v, django.core.validators.RegexValidator) and not v.inverse_match
    ]
    if regexes:
        # This strategy generates according to one of the regexes, and
        # filters using the others.  It can therefore learn to generate
        # from the most restrictive and filter with permissive patterns.
        # Not maximally efficient, but it makes pathological cases rarer.
        # If you want a challenge: extend https://qntm.org/greenery to
        # compute intersections of the full Python regex language.
        return st.one_of(*(st.from_regex(r) for r in regexes))
    # If there are no (usable) regexes, we use a standard text strategy.
    min_size, max_size = length_bounds_from_validators(field)
    strategy = st.text(
        alphabet=st.characters(exclude_characters="\x00", exclude_categories=("Cs",)),
        min_size=min_size,
        max_size=max_size,
    ).filter(lambda s: min_size <= len(s.strip()))
    if getattr(field, "blank", False) or not getattr(field, "required", True):
        return st.just("") | strategy
    return strategy


@register_for(df.BooleanField)
def _for_form_boolean(field):
    if field.required:
        return st.just(True)
    return st.booleans()


def _model_choice_strategy(field):
    def _strategy():
        if field.choices is None:
            # The field was instantiated with queryset=None, and not
            # subsequently updated.
            raise InvalidArgument(
                "Cannot create strategy for ModelChoicesField with no choices"
            )
        elif hasattr(field, "_choices"):
            # The choices property was set manually.
            choices = field._choices
        else:
            # choices is not None, and was not set manually, so we
            # must have a QuerySet.
            choices = field.queryset

        if not choices.ordered:
            raise InvalidArgument(
                f"Cannot create strategy for {field.__class__.__name__} with a choices "
                "attribute derived from a QuerySet without an explicit ordering - this may "
                "cause Hypothesis to produce unstable results between runs."
            )

        return st.sampled_from(
            [
                (
                    choice.value
                    if isinstance(choice, df.models.ModelChoiceIteratorValue)
                    else choice  # Empty value, if included.
                )
                for choice, _ in field.choices
            ]
        )

    # Accessing field.choices causes database access, so defer the strategy.
    return st.deferred(_strategy)


@register_for(df.ModelChoiceField)
def _for_model_choice(field):
    return _model_choice_strategy(field)


@register_for(df.ModelMultipleChoiceField)
def _for_model_multiple_choice(field):
    min_size = 1 if field.required else 0
    return st.lists(_model_choice_strategy(field), min_size=min_size, unique=True)


def register_field_strategy(
    field_type: type[AnyField], strategy: st.SearchStrategy
) -> None:
    """Add an entry to the global field-to-strategy lookup used by
    :func:`~hypothesis.extra.django.from_field`.

    ``field_type`` must be a subtype of :class:`django.db.models.Field` or
    :class:`django.forms.Field`, which must not already be registered.
    ``strategy`` must be a :class:`~hypothesis.strategies.SearchStrategy`.
    """
    if not issubclass(field_type, (dm.Field, df.Field)):
        raise InvalidArgument(f"{field_type=} must be a subtype of Field")
    check_type(st.SearchStrategy, strategy, "strategy")
    if field_type in _global_field_lookup:
        raise InvalidArgument(
            f"{field_type=} already has a registered "
            f"strategy ({_global_field_lookup[field_type]!r})"
        )
    if issubclass(field_type, dm.AutoField):
        raise InvalidArgument("Cannot register a strategy for an AutoField")
    _global_field_lookup[field_type] = strategy


def from_field(field: F) -> st.SearchStrategy[Union[F, None]]:
    """Return a strategy for values that fit the given field.

    This function is used by :func:`~hypothesis.extra.django.from_form` and
    :func:`~hypothesis.extra.django.from_model` for any fields that require
    a value, or for which you passed ``...`` (:obj:`python:Ellipsis`) to infer
    a strategy from an annotation.

    It's pretty similar to the core :func:`~hypothesis.strategies.from_type`
    function, with a subtle but important difference: ``from_field`` takes a
    Field *instance*, rather than a Field *subtype*, so that it has access to
    instance attributes such as string length and validators.
    """
    check_type((dm.Field, df.Field), field, "field")

    # The following isinstance check must occur *before* the getattr
    # check. In the case of ModelChoicesField, evaluating
    # field.choices causes database access, which we want to avoid if
    # we don't have a connection (the generated strategies for
    # ModelChoicesField defer evaluation of `choices').
    if not isinstance(field, df.ModelChoiceField) and getattr(field, "choices", False):
        choices: list = []
        for value, name_or_optgroup in field.choices:
            if isinstance(name_or_optgroup, (list, tuple)):
                choices.extend(key for key, _ in name_or_optgroup)
            else:
                choices.append(value)
        # form fields automatically include an empty choice, strip it out
        if "" in choices:
            choices.remove("")
        min_size = 1
        if isinstance(field, (dm.CharField, dm.TextField)) and field.blank:
            choices.insert(0, "")
        elif isinstance(field, (df.Field)) and not field.required:
            choices.insert(0, "")
            min_size = 0
        strategy = st.sampled_from(choices)
        if isinstance(field, (df.MultipleChoiceField, df.TypedMultipleChoiceField)):
            strategy = st.lists(st.sampled_from(choices), min_size=min_size)
    else:
        if type(field) not in _global_field_lookup:
            if getattr(field, "null", False):
                return st.none()
            raise ResolutionFailed(f"Could not infer a strategy for {field!r}")
        strategy = _global_field_lookup[type(field)]  # type: ignore
        if not isinstance(strategy, st.SearchStrategy):
            strategy = strategy(field)
    assert isinstance(strategy, st.SearchStrategy)
    if field.validators:

        def validate(value):
            try:
                field.run_validators(value)
                return True
            except django.core.exceptions.ValidationError:
                return False

        strategy = strategy.filter(validate)

    if getattr(field, "null", False):
        return st.none() | strategy
    return strategy
