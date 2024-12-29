# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from hypothesis.errors import FailedHealthCheck


def fail_health_check(settings, message, label):
    # Tell pytest to omit the body of this function from tracebacks
    # https://docs.pytest.org/en/latest/example/simple.html#writing-well-integrated-assertion-helpers
    __tracebackhide__ = True

    if label in settings.suppress_health_check:
        return
    message += (
        "\nSee https://hypothesis.readthedocs.io/en/latest/health"
        "checks.html for more information about this. "
        f"If you want to disable just this health check, add {label} "
        "to the suppress_health_check settings for this test."
    )
    raise FailedHealthCheck(message)
