# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from types import MethodType


def is_hypothesis_test(f: object) -> bool:
    """
    Returns ``True`` if ``f`` represents a test function that has been defined
    with Hypothesis. This is true for:

    * Functions decorated with |@given|
    * The ``runTest`` method of stateful tests

    For example:

    .. code-block:: python

        @given(st.integers())
        def f(n): ...

        class MyStateMachine(RuleBasedStateMachine): ...

        assert is_hypothesis_test(f)
        assert is_hypothesis_test(MyStateMachine.TestCase().runTest)

    .. seealso::

        See also the :doc:`Detect Hypothesis tests
        </how-to/detect-hypothesis-tests>` how-to.
    """
    if isinstance(f, MethodType):
        return is_hypothesis_test(f.__func__)
    return getattr(f, "is_hypothesis_test", False)
