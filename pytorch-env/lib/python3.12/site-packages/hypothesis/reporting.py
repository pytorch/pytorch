# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from hypothesis._settings import Verbosity, settings
from hypothesis.internal.compat import escape_unicode_characters
from hypothesis.utils.dynamicvariables import DynamicVariable


def default(value):
    try:
        print(value)
    except UnicodeEncodeError:
        print(escape_unicode_characters(value))


reporter = DynamicVariable(default)


def current_reporter():
    return reporter.value


def with_reporter(new_reporter):
    return reporter.with_value(new_reporter)


def current_verbosity():
    return settings.default.verbosity


def verbose_report(text):
    if current_verbosity() >= Verbosity.verbose:
        base_report(text)


def debug_report(text):
    if current_verbosity() >= Verbosity.debug:
        base_report(text)


def report(text):
    if current_verbosity() >= Verbosity.normal:
        base_report(text)


def base_report(text):
    assert isinstance(text, str), f"unexpected non-str {text=}"
    current_reporter()(text)
