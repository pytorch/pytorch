# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from hypothesis.internal.conjecture import utils as cu
from hypothesis.strategies._internal.strategies import SearchStrategy

FEATURE_LABEL = cu.calc_label_from_name("feature flag")


class FeatureFlags:
    """Object that can be used to control a number of feature flags for a
    given test run.

    This enables an approach to data generation called swarm testing (
    see Groce, Alex, et al. "Swarm testing." Proceedings of the 2012
    International Symposium on Software Testing and Analysis. ACM, 2012), in
    which generation is biased by selectively turning some features off for
    each test case generated. When there are many interacting features this can
    find bugs that a pure generation strategy would otherwise have missed.

    FeatureFlags are designed to "shrink open", so that during shrinking they
    become less restrictive. This allows us to potentially shrink to smaller
    test cases that were forbidden during the generation phase because they
    required disabled features.
    """

    def __init__(self, data=None, enabled=(), disabled=(), at_least_one_of=()):
        self.__data = data
        self.__is_disabled = {}

        for f in enabled:
            self.__is_disabled[f] = False

        for f in disabled:
            self.__is_disabled[f] = True

        # In the original swarm testing paper they turn features on or off
        # uniformly at random. Instead we decide the probability with which to
        # enable features up front. This can allow for scenarios where all or
        # no features are enabled, which are vanishingly unlikely in the
        # original model.
        #
        # We implement this as a single 8-bit integer and enable features which
        # score >= that value. In particular when self.__baseline is 0, all
        # features will be enabled. This is so that we shrink in the direction
        # of more features being enabled.
        if self.__data is not None:
            self.__p_disabled = data.draw_integer(0, 254) / 255
        else:
            # If data is None we're in example mode so all that matters is the
            # enabled/disabled lists above. We set this up so that everything
            # else is enabled by default.
            self.__p_disabled = 0.0

        # The naive approach can lead to disabling e.g. every single rule on a
        # RuleBasedStateMachine, which aborts the test as unable to make progress.
        # Track the set of possible names, and ensure that at least one is enabled.
        self.__at_least_one_of = set(at_least_one_of)

    def is_enabled(self, name):
        """Tests whether the feature named ``name`` should be enabled on this
        test run."""
        if self.__data is None or self.__data.frozen:
            # Feature set objects might hang around after data generation has
            # finished. If this happens then we just report all new features as
            # enabled, because that's our shrinking direction and they have no
            # impact on data generation if they weren't used while it was
            # running.
            return not self.__is_disabled.get(name, False)

        data = self.__data

        data.start_example(label=FEATURE_LABEL)

        # If we've already decided on this feature then we don't actually
        # need to draw anything, but we do write the same decision to the
        # input stream. This allows us to lazily decide whether a feature
        # is enabled, because it means that if we happen to delete the part
        # of the test case where we originally decided, the next point at
        # which we make this decision just makes the decision it previously
        # made.
        oneof = self.__at_least_one_of
        is_disabled = self.__data.draw_boolean(
            self.__p_disabled,
            forced=(
                False
                if len(oneof) == 1 and name in oneof
                else self.__is_disabled.get(name)
            ),
        )
        self.__is_disabled[name] = is_disabled
        if name in oneof and not is_disabled:
            oneof.clear()
        oneof.discard(name)
        data.stop_example()
        return not is_disabled

    def __repr__(self):
        enabled = []
        disabled = []
        for name, is_disabled in self.__is_disabled.items():
            if is_disabled:
                disabled.append(name)
            else:
                enabled.append(name)
        return f"FeatureFlags({enabled=}, {disabled=})"


class FeatureStrategy(SearchStrategy):
    def __init__(self, at_least_one_of=()):
        super().__init__()
        self._at_least_one_of = frozenset(at_least_one_of)

    def do_draw(self, data):
        return FeatureFlags(data, at_least_one_of=self._at_least_one_of)
