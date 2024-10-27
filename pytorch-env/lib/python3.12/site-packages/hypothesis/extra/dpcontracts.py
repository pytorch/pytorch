# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""
-----------------------
hypothesis[dpcontracts]
-----------------------

This module provides tools for working with the :pypi:`dpcontracts` library,
because `combining contracts and property-based testing works really well
<https://hillelwayne.com/talks/beyond-unit-tests/>`_.

It requires ``dpcontracts >= 0.4``.
"""

from dpcontracts import PreconditionError

from hypothesis import reject
from hypothesis.errors import InvalidArgument
from hypothesis.internal.reflection import proxies


def fulfill(contract_func):
    """Decorate ``contract_func`` to reject calls which violate preconditions,
    and retry them with different arguments.

    This is a convenience function for testing internal code that uses
    :pypi:`dpcontracts`, to automatically filter out arguments that would be
    rejected by the public interface before triggering a contract error.

    This can be used as ``builds(fulfill(func), ...)`` or in the body of the
    test e.g. ``assert fulfill(func)(*args)``.
    """
    if not hasattr(contract_func, "__contract_wrapped_func__"):
        raise InvalidArgument(
            f"{contract_func.__name__} has no dpcontracts preconditions"
        )

    @proxies(contract_func)
    def inner(*args, **kwargs):
        try:
            return contract_func(*args, **kwargs)
        except PreconditionError:
            reject()

    return inner
