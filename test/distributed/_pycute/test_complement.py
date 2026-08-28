# ruff: noqa: PGH004, G004, F403
# flake8: noqa
# Owner(s): ["oncall: distributed"]
#################################################################################################
#
# Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

"""
Unit tests for _pycute.complement
"""

import logging

from torch.distributed._pycute import *
from torch.testing._internal.common_utils import run_tests, TestCase


_LOGGER = logging.getLogger(__name__)


class TestComplement(TestCase):
    def helper_test_complement(self, layout):
        layoutR = complement(layout)

        _LOGGER.debug(f"{layout}  =>  {layoutR}")

        # Post-condition: test disjointedness of the codomains
        for a in range(size(layout)):
            for b in range(size(layoutR)):
                if not (
                    (layout(a) != layoutR(b)) or (layout(a) == 0 and layoutR(b) == 0)
                ):
                    raise AssertionError(
                        f"Invariant violated at a={a}, b={b}: layout(a)={layout(a)}, layoutR(b)={layoutR(b)}"
                    )

    def test_complement(self):
        test = Layout(1, 0)
        self.helper_test_complement(test)

        test = Layout(1, 1)
        self.helper_test_complement(test)

        test = Layout(4, 0)
        self.helper_test_complement(test)

        test = Layout((2, 4), (1, 2))
        self.helper_test_complement(test)

        test = Layout((2, 3), (1, 2))
        self.helper_test_complement(test)

        test = Layout((2, 4), (1, 4))
        self.helper_test_complement(test)

        test = Layout((2, 4, 8), (8, 1, 64))
        self.helper_test_complement(test)

        test = Layout(((2, 2), (2, 2)), ((1, 4), (8, 32)))
        self.helper_test_complement(test)

        test = Layout((2, (3, 4)), (3, (1, 6)))
        self.helper_test_complement(test)

        test = Layout((4, 6), (1, 6))
        self.helper_test_complement(test)

        test = Layout((4, 10), (1, 10))
        self.helper_test_complement(test)


if __name__ == "__main__":
    run_tests()
